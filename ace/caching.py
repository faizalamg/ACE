"""Response caching for efficient multi-epoch training.

This module provides caching mechanisms to avoid redundant LLM calls
during multi-epoch training, saving time and costs.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLMClient


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""

    response: str
    created_at: float
    last_accessed: float
    access_count: int = 1


class ResponseCache:
    """LRU cache for LLM responses with TTL support.

    Provides efficient caching of LLM responses to avoid redundant calls
    during multi-epoch training. Supports:
    - TTL (time-to-live) for automatic expiration
    - LRU eviction when max size is reached
    - Context-aware caching (same prompt, different context = different entry)
    - Persistence to disk
    - Hit rate metrics

    Example:
        >>> cache = ResponseCache(max_size=1000, ttl_seconds=3600)
        >>> cache.set("What is 2+2?", "4")
        >>> result = cache.get("What is 2+2?")  # Returns "4"
        >>>
        >>> # With context (e.g., different playbook state)
        >>> cache.set("How?", "Method A", context="playbook_v1")
        >>> cache.set("How?", "Method B", context="playbook_v2")
    """

    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: Optional[float] = None,
        refresh_on_access: bool = False,
        persist_path: Optional[Path] = None,
    ) -> None:
        """Initialize the response cache.

        Args:
            max_size: Maximum number of entries before LRU eviction
            ttl_seconds: Optional TTL in seconds (None = no expiration)
            refresh_on_access: If True, accessing an entry resets its TTL
            persist_path: Optional path for automatic persistence
        """
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._refresh_on_access = refresh_on_access
        self._persist_path = persist_path

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()

        # Metrics
        self._hits = 0
        self._misses = 0

        # Load from persist path if exists
        if persist_path and persist_path.exists():
            self._load_from_file(persist_path)

    def __len__(self) -> int:
        """Return number of entries in cache."""
        return len(self._cache)

    def _make_key(self, prompt: str, context: Optional[str] = None) -> str:
        """Create cache key from prompt and optional context.

        Args:
            prompt: The prompt text
            context: Optional context (e.g., playbook hash)

        Returns:
            SHA256 hash key
        """
        key_data = prompt
        if context:
            key_data = f"{prompt}|||{context}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    def get(
        self,
        prompt: str,
        context: Optional[str] = None,
    ) -> Optional[str]:
        """Get cached response for a prompt.

        Args:
            prompt: The prompt to look up
            context: Optional context for differentiation

        Returns:
            Cached response or None if not found/expired
        """
        key = self._make_key(prompt, context)

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            # Check TTL
            if self._ttl_seconds is not None:
                age = time.time() - entry.created_at
                if age > self._ttl_seconds:
                    # Expired, remove and return None
                    del self._cache[key]
                    self._misses += 1
                    return None

            # Hit - update access info
            self._hits += 1
            entry.access_count += 1
            entry.last_accessed = time.time()

            if self._refresh_on_access and self._ttl_seconds:
                entry.created_at = time.time()

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            return entry.response

    def set(
        self,
        prompt: str,
        response: str,
        context: Optional[str] = None,
    ) -> None:
        """Cache a response for a prompt.

        Args:
            prompt: The prompt text
            response: The response to cache
            context: Optional context for differentiation
        """
        key = self._make_key(prompt, context)
        now = time.time()

        with self._lock:
            # Check if we need to evict
            if key not in self._cache and len(self._cache) >= self._max_size:
                # Remove oldest (first item in OrderedDict)
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(
                response=response,
                created_at=now,
                last_accessed=now,
            )

            # Move to end
            self._cache.move_to_end(key)

        # Auto-persist if configured
        if self._persist_path:
            self._save_to_file(self._persist_path)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics.

        Returns:
            Dict with hits, misses, hit_rate, size
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "max_size": self._max_size,
        }

    def save(self, path: Path) -> None:
        """Save cache to disk.

        Args:
            path: File path to save to
        """
        self._save_to_file(path)

    @classmethod
    def load(cls, path: Path) -> "ResponseCache":
        """Load cache from disk.

        Args:
            path: File path to load from

        Returns:
            ResponseCache instance with loaded entries
        """
        cache = cls()
        cache._load_from_file(path)
        return cache

    def _save_to_file(self, path: Path) -> None:
        """Internal save implementation."""
        data = {
            "entries": {
                key: {
                    "response": entry.response,
                    "created_at": entry.created_at,
                    "last_accessed": entry.last_accessed,
                    "access_count": entry.access_count,
                }
                for key, entry in self._cache.items()
            },
            "metrics": {
                "hits": self._hits,
                "misses": self._misses,
            },
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def _load_from_file(self, path: Path) -> None:
        """Internal load implementation."""
        with open(path, "r") as f:
            data = json.load(f)

        for key, entry_data in data.get("entries", {}).items():
            self._cache[key] = CacheEntry(
                response=entry_data["response"],
                created_at=entry_data["created_at"],
                last_accessed=entry_data["last_accessed"],
                access_count=entry_data.get("access_count", 1),
            )

        metrics = data.get("metrics", {})
        self._hits = metrics.get("hits", 0)
        self._misses = metrics.get("misses", 0)


class CachedLLMClient:
    """LLM client wrapper that adds response caching.

    Wraps any LLMClient implementation to add transparent caching.
    Useful for multi-epoch training where the same prompts may be
    processed multiple times.

    Example:
        >>> from ace.llm_providers import LiteLLMClient
        >>> from ace.caching import CachedLLMClient
        >>>
        >>> base_client = LiteLLMClient(model="gpt-4")
        >>> client = CachedLLMClient(base_client)
        >>>
        >>> # First call hits LLM
        >>> result1 = client.complete("Hello")
        >>>
        >>> # Second call returns cached response
        >>> result2 = client.complete("Hello")
    """

    def __init__(
        self,
        client: "LLMClient",
        cache: Optional[ResponseCache] = None,
        context_fn: Optional[callable] = None,
    ) -> None:
        """Initialize cached LLM client.

        Args:
            client: Base LLM client to wrap
            cache: Optional custom cache (default created if None)
            context_fn: Optional function to generate context from prompt
        """
        self._client = client
        self._cache = cache or ResponseCache()
        self._context_fn = context_fn

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Complete prompt with caching.

        Args:
            prompt: The prompt to complete
            **kwargs: Additional arguments passed to base client

        Returns:
            Response (from cache or LLM)
        """
        # Generate context for cache key
        context = None
        if self._context_fn:
            context = self._context_fn(prompt, **kwargs)

        # Check cache
        cached = self._cache.get(prompt, context=context)
        if cached is not None:
            return cached

        # Call base client
        response = self._client.complete(prompt, **kwargs)

        # Cache response
        self._cache.set(prompt, response, context=context)

        return response

    def get_cache_metrics(self) -> Dict[str, Any]:
        """Get cache metrics."""
        return self._cache.get_metrics()

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()

    @property
    def cache(self) -> ResponseCache:
        """Access the underlying cache."""
        return self._cache
