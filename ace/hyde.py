"""HyDE (Hypothetical Document Embeddings) implementation.

This module implements HyDE for bridging the semantic gap between short queries
and detailed memory documents. HyDE transforms queries into hypothetical documents
that would answer the query, then uses their embeddings for more accurate retrieval.

Key Features:
- LLM-based hypothetical document generation (Z.ai GLM-4.6 by default)
- Configurable number of hypothetical documents (default: 3-5)
- Caching for repeated queries
- Async support for batch processing
- Optimized for memory retrieval domain

Reference: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
(Gao et al., 2022) - https://arxiv.org/abs/2212.10496
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import asyncio

if TYPE_CHECKING:
    from .llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class HyDEConfig:
    """Configuration for HyDE hypothetical document generation."""

    # Generation parameters
    num_hypotheticals: int = 3  # Number of hypothetical documents to generate
    max_tokens: int = 150  # Max tokens per hypothetical document
    temperature: float = 0.7  # Higher temperature for diversity

    # LLM configuration
    model: str = "openai/glm-4.6"  # Z.ai GLM-4.6 (default ACE model)
    api_key: Optional[str] = None  # Auto-detected from ZAI_API_KEY or OPENAI_API_KEY
    api_base: Optional[str] = None  # Auto-configured for Z.ai

    # Cache configuration
    cache_enabled: bool = True
    max_cache_size: int = 1000  # Maximum cached queries

    # Domain-specific prompt template
    prompt_template: str = (
        'Given the query: "{query}"\n\n'
        "Write a detailed passage that would be stored as a learned lesson or "
        "best practice that answers this query. Include specific technical details, "
        "error scenarios, and actionable recommendations. Write 2-3 sentences.\n\n"
        "Passage:"
    )


class HyDEGenerator:
    """Generate hypothetical documents for query expansion using LLM.

    Uses Z.ai GLM-4.6 by default (ACE framework standard) with fallback to OpenAI.

    Example:
        >>> from ace.hyde import HyDEGenerator, HyDEConfig
        >>> from ace.llm_providers.litellm_client import LiteLLMClient
        >>>
        >>> # Uses Z.ai GLM-4.6 by default (requires ZAI_API_KEY in .env)
        >>> llm = LiteLLMClient(model="openai/glm-4.6")
        >>> config = HyDEConfig(num_hypotheticals=3)
        >>> hyde = HyDEGenerator(llm, config)
        >>>
        >>> # Generate hypothetical documents
        >>> query = "How to fix authentication errors?"
        >>> hypotheticals = hyde.generate_hypotheticals(query)
        >>> for hyp in hypotheticals:
        ...     print(hyp)
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[HyDEConfig] = None
    ) -> None:
        """Initialize HyDE generator.

        Args:
            llm_client: LLM client for generation (uses LiteLLMClient if None)
            config: HyDE configuration (uses defaults if None)
        """
        self.config = config or HyDEConfig()

        # Initialize LLM client if not provided
        if llm_client is None:
            self._initialize_default_llm()
        else:
            self.llm_client = llm_client

        # Initialize cache
        self._cache: Dict[str, List[str]] = {}
        self._cache_order: List[str] = []  # LRU tracking

        logger.info(
            f"HyDE initialized with {self.config.num_hypotheticals} hypotheticals, "
            f"cache_enabled={self.config.cache_enabled}"
        )

    def _initialize_default_llm(self) -> None:
        """Initialize default LLM client (Z.ai GLM-4.6)."""
        try:
            from .llm_providers.litellm_client import LiteLLMClient

            # Auto-detect API key from environment
            api_key = self.config.api_key or os.getenv("ZAI_API_KEY") or os.getenv("OPENAI_API_KEY")

            if not api_key:
                raise ValueError(
                    "No API key found. Set ZAI_API_KEY (for Z.ai GLM) or "
                    "OPENAI_API_KEY in .env file, or pass api_key in HyDEConfig"
                )

            # Configure for Z.ai GLM-4.6 (ACE default)
            self.llm_client = LiteLLMClient(
                model=self.config.model,
                api_key=api_key,
                api_base=self.config.api_base,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            logger.info(f"Initialized LiteLLM client with model: {self.config.model}")

        except ImportError:
            raise ImportError(
                "LiteLLM not available. Install with: pip install litellm\n"
                "Or provide a custom LLMClient instance."
            )

    def _get_cache_key(self, query: str, num_docs: int) -> str:
        """Generate cache key for query."""
        key_input = f"{query}|{num_docs}|{self.config.prompt_template}"
        return hashlib.sha256(key_input.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[List[str]]:
        """Get hypotheticals from cache if available."""
        if not self.config.cache_enabled:
            return None

        if cache_key in self._cache:
            logger.debug(f"Cache HIT for key: {cache_key[:8]}...")
            return self._cache[cache_key]

        logger.debug(f"Cache MISS for key: {cache_key[:8]}...")
        return None

    def _add_to_cache(self, cache_key: str, hypotheticals: List[str]) -> None:
        """Add hypotheticals to cache with LRU eviction."""
        if not self.config.cache_enabled:
            return

        # Evict oldest if cache full
        if len(self._cache) >= self.config.max_cache_size:
            oldest_key = self._cache_order.pop(0)
            del self._cache[oldest_key]
            logger.debug(f"Cache evicted: {oldest_key[:8]}...")

        # Add to cache
        self._cache[cache_key] = hypotheticals
        self._cache_order.append(cache_key)
        logger.debug(f"Cache added: {cache_key[:8]}... (size: {len(self._cache)})")

    def _generate_single_hypothetical(self, query: str) -> str:
        """Generate a single hypothetical document for the query.

        Args:
            query: User query to expand

        Returns:
            Generated hypothetical document
        """
        # Format prompt with query
        prompt = self.config.prompt_template.format(query=query)

        # Generate hypothetical using LLM
        response = self.llm_client.complete(prompt)

        # Extract and clean response - handle None gracefully
        raw_text = response.text if response.text else ""
        hypothetical = raw_text.strip()

        if not hypothetical:
            logger.warning(f"LLM returned empty hypothetical for query: {query[:50]}...")
        else:
            logger.debug(f"Generated hypothetical: {hypothetical[:100]}...")

        return hypothetical

    def generate_hypotheticals(
        self,
        query: str,
        num_docs: Optional[int] = None
    ) -> List[str]:
        """Generate multiple hypothetical documents for the query.

        Args:
            query: User query to expand
            num_docs: Number of hypotheticals (defaults to config value)

        Returns:
            List of generated hypothetical documents
        """
        num_docs = num_docs or self.config.num_hypotheticals

        # Check cache first
        cache_key = self._get_cache_key(query, num_docs)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # Generate hypotheticals
        logger.info(f"Generating {num_docs} hypotheticals for query: {query[:50]}...")
        hypotheticals = []

        for i in range(num_docs):
            try:
                hypothetical = self._generate_single_hypothetical(query)
                hypotheticals.append(hypothetical)
                logger.debug(f"Generated hypothetical {i+1}/{num_docs}")
            except Exception as e:
                logger.warning(f"Failed to generate hypothetical {i+1}/{num_docs}: {e}")
                # Continue with other hypotheticals

        if not hypotheticals:
            logger.error("Failed to generate any hypotheticals, falling back to query")
            hypotheticals = [query]  # Fallback to original query

        # Cache results
        self._add_to_cache(cache_key, hypotheticals)

        return hypotheticals

    async def agenerate_hypotheticals(
        self,
        query: str,
        num_docs: Optional[int] = None
    ) -> List[str]:
        """Async version of generate_hypotheticals for batch processing.

        Args:
            query: User query to expand
            num_docs: Number of hypotheticals (defaults to config value)

        Returns:
            List of generated hypothetical documents
        """
        num_docs = num_docs or self.config.num_hypotheticals

        # Check cache first
        cache_key = self._get_cache_key(query, num_docs)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # Generate hypotheticals concurrently
        logger.info(f"Async generating {num_docs} hypotheticals for query: {query[:50]}...")

        async def _generate_async(idx: int) -> Optional[str]:
            """Async wrapper for single hypothetical generation."""
            try:
                # Run synchronous LLM call in thread pool
                loop = asyncio.get_event_loop()
                hypothetical = await loop.run_in_executor(
                    None,
                    self._generate_single_hypothetical,
                    query
                )
                logger.debug(f"Async generated hypothetical {idx+1}/{num_docs}")
                return hypothetical
            except Exception as e:
                logger.warning(f"Async failed hypothetical {idx+1}/{num_docs}: {e}")
                return None

        # Generate all hypotheticals concurrently
        tasks = [_generate_async(i) for i in range(num_docs)]
        results = await asyncio.gather(*tasks)

        # Filter out failures
        hypotheticals = [h for h in results if h is not None]

        if not hypotheticals:
            logger.error("Failed to generate any hypotheticals async, falling back to query")
            hypotheticals = [query]

        # Cache results
        self._add_to_cache(cache_key, hypotheticals)

        return hypotheticals

    def clear_cache(self) -> None:
        """Clear the hypothetical document cache."""
        self._cache.clear()
        self._cache_order.clear()
        logger.info("HyDE cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache size, hit rate, etc.
        """
        return {
            "cache_enabled": self.config.cache_enabled,
            "cache_size": len(self._cache),
            "max_cache_size": self.config.max_cache_size,
            "num_hypotheticals": self.config.num_hypotheticals,
        }
