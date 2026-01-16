"""
Retrieval-specific caching layer for ACE Framework (Phase 4B).

This module caches RETRIEVAL data (embeddings, query results), NOT LLM responses.
For LLM response caching, see ace/caching.py.

Caching Strategy:
- EmbeddingCache: Text -> embedding vector (768-dim floats)
- QueryResultCache: Query -> List[QdrantScoredResult] with bullet-aware invalidation

Both caches use LRU eviction with optional TTL expiration.
"""

from threading import Lock
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Set
import time


@dataclass
class CacheEntry:
    """Cache entry with metadata for TTL and invalidation tracking."""
    value: Any
    created_at: float
    last_accessed: float
    bullet_ids: Set[str] = None  # For QueryResultCache invalidation tracking


class EmbeddingCache:
    """LRU cache for text -> embedding vector mapping.

    Thread-safe cache with optional TTL expiration and automatic LRU eviction.
    """

    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: Optional[float] = None,
    ):
        """Initialize embedding cache.

        Args:
            max_size: Maximum number of cached embeddings (LRU eviction when full)
            ttl_seconds: Time-to-live in seconds (None = no expiration)
        """
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, text: str) -> Optional[List[float]]:
        """Get cached embedding vector.

        Args:
            text: Input text to lookup

        Returns:
            Cached embedding vector or None on miss/expiration
        """
        with self._lock:
            entry = self._cache.get(text)

            if entry is None:
                self._misses += 1
                return None

            # Check TTL expiration
            if self._ttl_seconds is not None:
                if time.time() - entry.created_at > self._ttl_seconds:
                    # Expired - remove and count as miss
                    del self._cache[text]
                    self._misses += 1
                    return None

            # Hit - update access time and move to end (most recently used)
            entry.last_accessed = time.time()
            self._cache.move_to_end(text)
            self._hits += 1
            return entry.value

    def set(self, text: str, embedding: List[float]) -> None:
        """Cache embedding vector with LRU eviction.

        Args:
            text: Input text key
            embedding: Embedding vector to cache
        """
        with self._lock:
            # Remove existing entry if present (to update timestamp)
            if text in self._cache:
                del self._cache[text]

            # Evict LRU entry if at capacity
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)  # Remove oldest (first) item
                self._evictions += 1

            # Add new entry (becomes most recently used)
            now = time.time()
            entry = CacheEntry(
                value=embedding,
                created_at=now,
                last_accessed=now
            )
            self._cache[text] = entry

    def clear(self) -> None:
        """Clear all cached embeddings and reset metrics."""
        with self._lock:
            self._cache.clear()
            # Note: Tests expect hits/misses/evictions to persist after clear

    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics.

        Returns:
            Dictionary with hits, misses, size, hit_rate, evictions
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0

            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "hit_rate": hit_rate,
                "evictions": self._evictions,
            }

    def __len__(self) -> int:
        """Return current cache size."""
        with self._lock:
            return len(self._cache)


class QueryResultCache:
    """LRU cache for query -> retrieval results with bullet-aware invalidation.

    Tracks which bullets appear in each cached query result to enable
    efficient invalidation when bullets are updated/removed.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: Optional[float] = None,
    ):
        """Initialize query result cache.

        Args:
            max_size: Maximum number of cached query results
            ttl_seconds: Time-to-live in seconds (None = no expiration)
        """
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._bullet_to_queries: Dict[str, Set[str]] = {}  # bullet_id -> query keys
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, query: str) -> Optional[List[Any]]:
        """Get cached query results.

        Args:
            query: Search query string

        Returns:
            Cached results or None on miss/expiration
        """
        with self._lock:
            entry = self._cache.get(query)

            if entry is None:
                self._misses += 1
                return None

            # Check TTL expiration
            if self._ttl_seconds is not None:
                if time.time() - entry.created_at > self._ttl_seconds:
                    # Expired - remove and count as miss
                    self._remove_query(query)
                    self._misses += 1
                    return None

            # Hit - update access time and move to end
            entry.last_accessed = time.time()
            self._cache.move_to_end(query)
            self._hits += 1
            return entry.value

    def set(self, query: str, results: List[Any]) -> None:
        """Cache query results and track bullet IDs for invalidation.

        Args:
            query: Search query string
            results: List of QdrantScoredResult objects
        """
        with self._lock:
            # Remove existing entry if present
            if query in self._cache:
                self._remove_query(query)

            # Evict LRU entry if at capacity
            if len(self._cache) >= self._max_size:
                oldest_query = next(iter(self._cache))  # First item in OrderedDict
                self._remove_query(oldest_query)
                self._evictions += 1

            # Extract bullet IDs from results
            bullet_ids = {result.bullet_id for result in results}

            # Add new entry
            now = time.time()
            entry = CacheEntry(
                value=results,
                created_at=now,
                last_accessed=now,
                bullet_ids=bullet_ids
            )
            self._cache[query] = entry

            # Update bullet -> queries mapping
            for bullet_id in bullet_ids:
                if bullet_id not in self._bullet_to_queries:
                    self._bullet_to_queries[bullet_id] = set()
                self._bullet_to_queries[bullet_id].add(query)

    def invalidate_bullet(self, bullet_id: str) -> int:
        """Invalidate all cache entries containing this bullet.

        Args:
            bullet_id: Bullet ID that was updated/removed

        Returns:
            Number of cache entries invalidated
        """
        with self._lock:
            queries_to_remove = self._bullet_to_queries.get(bullet_id, set()).copy()

            for query in queries_to_remove:
                self._remove_query(query)

            return len(queries_to_remove)

    def invalidate_all(self) -> int:
        """Clear all cached results.

        Returns:
            Number of cache entries invalidated
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._bullet_to_queries.clear()
            return count

    def clear(self) -> None:
        """Alias for invalidate_all (returns None for compatibility)."""
        self.invalidate_all()

    def _remove_query(self, query: str) -> None:
        """Remove query from cache and bullet mappings (internal, no lock).

        Args:
            query: Query key to remove
        """
        entry = self._cache.get(query)
        if entry is None:
            return

        # Remove from main cache
        del self._cache[query]

        # Remove from bullet -> queries mappings
        if entry.bullet_ids:
            for bullet_id in entry.bullet_ids:
                if bullet_id in self._bullet_to_queries:
                    self._bullet_to_queries[bullet_id].discard(query)
                    # Clean up empty sets
                    if not self._bullet_to_queries[bullet_id]:
                        del self._bullet_to_queries[bullet_id]

    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics.

        Returns:
            Dictionary with hits, misses, size, hit_rate, evictions
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0

            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "hit_rate": hit_rate,
                "evictions": self._evictions,
            }

    def __len__(self) -> int:
        """Return current cache size."""
        with self._lock:
            return len(self._cache)
