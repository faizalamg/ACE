"""
Tests for ACE retrieval caching layer (Phase 4B).

This tests EMBEDDING and QUERY RESULT caching for Qdrant retrieval operations.
NOT to be confused with ace/caching.py which caches LLM responses.

Test Coverage:
- 4B.1: Embedding cache hit/miss/eviction
- 4B.3: Query result cache hit/miss/TTL
- 4B.5: Cache invalidation on bullet updates
- Cache metrics and monitoring
- Thread safety for concurrent access
"""

import unittest
import time
from typing import List
from dataclasses import dataclass


# Mock types until implementation exists
@dataclass
class QdrantScoredResult:
    """Mock scored result from Qdrant retrieval."""
    bullet_id: str
    score: float
    text: str
    metadata: dict


class TestEmbeddingCache(unittest.TestCase):
    """Test suite for EmbeddingCache - caches text -> embedding vector mappings."""

    def test_embedding_cache_hit(self):
        """4B.1: Test embedding cache returns cached vector on hit."""
        from ace.retrieval_caching import EmbeddingCache

        cache = EmbeddingCache(max_size=1000, ttl_seconds=3600)

        # Cache an embedding vector (768-dim typical for sentence transformers)
        embedding = [0.1] * 768
        cache.set("test text for embedding", embedding)

        # Cache hit should return exact vector
        result = cache.get("test text for embedding")
        self.assertEqual(result, embedding, "Cache hit should return exact embedding")

        # Metrics should track hit
        metrics = cache.get_metrics()
        self.assertEqual(metrics["hits"], 1, "Should record 1 cache hit")
        self.assertEqual(metrics["misses"], 0, "Should record 0 misses")

    def test_embedding_cache_miss(self):
        """Test embedding cache returns None on miss."""
        from ace.retrieval_caching import EmbeddingCache

        cache = EmbeddingCache(max_size=1000, ttl_seconds=3600)

        # No cached value exists
        result = cache.get("uncached text")
        self.assertIsNone(result, "Cache miss should return None")

        # Metrics should track miss
        metrics = cache.get_metrics()
        self.assertEqual(metrics["misses"], 1, "Should record 1 cache miss")
        self.assertEqual(metrics["hits"], 0, "Should record 0 hits")

    def test_embedding_cache_eviction_lru(self):
        """Test LRU eviction when cache exceeds max_size."""
        from ace.retrieval_caching import EmbeddingCache

        cache = EmbeddingCache(max_size=3, ttl_seconds=3600)

        # Fill cache to capacity
        cache.set("text1", [0.1] * 768)
        cache.set("text2", [0.2] * 768)
        cache.set("text3", [0.3] * 768)

        # Access text1 to make it recently used
        cache.get("text1")

        # Add new entry - should evict least recently used (text2)
        cache.set("text4", [0.4] * 768)

        # text2 should be evicted
        self.assertIsNone(cache.get("text2"), "LRU entry should be evicted")
        self.assertIsNotNone(cache.get("text1"), "Recently used entry should remain")
        self.assertIsNotNone(cache.get("text4"), "New entry should exist")

        # Metrics should track evictions
        metrics = cache.get_metrics()
        self.assertEqual(metrics["evictions"], 1, "Should record 1 eviction")

    def test_embedding_cache_ttl_expiration(self):
        """Test embeddings expire after TTL seconds."""
        from ace.retrieval_caching import EmbeddingCache

        cache = EmbeddingCache(max_size=1000, ttl_seconds=1)  # 1 second TTL

        embedding = [0.5] * 768
        cache.set("expiring text", embedding)

        # Immediate access should hit
        self.assertIsNotNone(cache.get("expiring text"))

        # Wait for TTL expiration
        time.sleep(1.5)

        # Access after TTL should miss
        result = cache.get("expiring text")
        self.assertIsNone(result, "Entry should expire after TTL")

        # Metrics should track expiration as miss
        metrics = cache.get_metrics()
        self.assertGreater(metrics["misses"], 0, "Expired entry should count as miss")

    def test_embedding_cache_clear(self):
        """Test clearing all cached embeddings."""
        from ace.retrieval_caching import EmbeddingCache

        cache = EmbeddingCache(max_size=1000, ttl_seconds=3600)

        # Add multiple entries
        cache.set("text1", [0.1] * 768)
        cache.set("text2", [0.2] * 768)
        cache.set("text3", [0.3] * 768)

        # Clear cache
        cache.clear()

        # All entries should be gone
        self.assertIsNone(cache.get("text1"))
        self.assertIsNone(cache.get("text2"))
        self.assertIsNone(cache.get("text3"))

        # Metrics should reset
        metrics = cache.get_metrics()
        self.assertEqual(len(cache), 0, "Cache should be empty after clear")


class TestQueryResultCache(unittest.TestCase):
    """Test suite for QueryResultCache - caches query -> List[QdrantScoredResult]."""

    def test_query_result_cache_hit(self):
        """4B.3: Test query result cache returns cached results on hit."""
        from ace.retrieval_caching import QueryResultCache

        cache = QueryResultCache(max_size=500, ttl_seconds=600)

        # Cache query results
        results = [
            QdrantScoredResult(
                bullet_id="bullet-1",
                score=0.95,
                text="Fix authentication bug",
                metadata={"category": "debugging"}
            ),
            QdrantScoredResult(
                bullet_id="bullet-2",
                score=0.87,
                text="Use async/await pattern",
                metadata={"category": "code_quality"}
            )
        ]

        cache.set("how to debug authentication", results)

        # Cache hit should return exact results
        cached = cache.get("how to debug authentication")
        self.assertEqual(cached, results, "Cache hit should return exact results")
        self.assertEqual(len(cached), 2, "Should return all cached results")

        # Metrics should track hit
        metrics = cache.get_metrics()
        self.assertEqual(metrics["hits"], 1, "Should record 1 cache hit")

    def test_query_result_cache_miss(self):
        """Test query result cache returns None on miss."""
        from ace.retrieval_caching import QueryResultCache

        cache = QueryResultCache(max_size=500, ttl_seconds=600)

        # No cached value exists
        result = cache.get("uncached query")
        self.assertIsNone(result, "Cache miss should return None")

        # Metrics should track miss
        metrics = cache.get_metrics()
        self.assertEqual(metrics["misses"], 1, "Should record 1 cache miss")

    def test_query_result_cache_ttl(self):
        """Test query results expire after TTL."""
        from ace.retrieval_caching import QueryResultCache

        cache = QueryResultCache(max_size=500, ttl_seconds=1)  # 1 second TTL

        results = [
            QdrantScoredResult(
                bullet_id="bullet-1",
                score=0.9,
                text="Test result",
                metadata={}
            )
        ]

        cache.set("test query", results)

        # Immediate access should hit
        self.assertIsNotNone(cache.get("test query"))

        # Wait for TTL expiration
        time.sleep(1.5)

        # Access after TTL should miss
        result = cache.get("test query")
        self.assertIsNone(result, "Query results should expire after TTL")

    def test_query_result_cache_invalidate_bullet(self):
        """4B.5: Test cache invalidation when bullet is updated."""
        from ace.retrieval_caching import QueryResultCache

        cache = QueryResultCache(max_size=500, ttl_seconds=600)

        # Cache multiple queries with overlapping bullets
        results_1 = [
            QdrantScoredResult(bullet_id="bullet-1", score=0.9, text="A", metadata={}),
            QdrantScoredResult(bullet_id="bullet-2", score=0.8, text="B", metadata={})
        ]
        results_2 = [
            QdrantScoredResult(bullet_id="bullet-2", score=0.85, text="B", metadata={}),
            QdrantScoredResult(bullet_id="bullet-3", score=0.75, text="C", metadata={})
        ]
        results_3 = [
            QdrantScoredResult(bullet_id="bullet-4", score=0.7, text="D", metadata={})
        ]

        cache.set("query1", results_1)
        cache.set("query2", results_2)
        cache.set("query3", results_3)

        # Invalidate bullet-2
        invalidated_count = cache.invalidate_bullet("bullet-2")

        # Queries containing bullet-2 should be invalidated
        self.assertIsNone(cache.get("query1"), "query1 should be invalidated (contains bullet-2)")
        self.assertIsNone(cache.get("query2"), "query2 should be invalidated (contains bullet-2)")
        self.assertIsNotNone(cache.get("query3"), "query3 should remain (doesn't contain bullet-2)")

        # Should report 2 invalidations
        self.assertEqual(invalidated_count, 2, "Should invalidate 2 cache entries")

    def test_query_result_cache_invalidate_all(self):
        """Test invalidating all cached query results."""
        from ace.retrieval_caching import QueryResultCache

        cache = QueryResultCache(max_size=500, ttl_seconds=600)

        # Cache multiple queries
        cache.set("query1", [QdrantScoredResult("b1", 0.9, "A", {})])
        cache.set("query2", [QdrantScoredResult("b2", 0.8, "B", {})])
        cache.set("query3", [QdrantScoredResult("b3", 0.7, "C", {})])

        # Invalidate all
        invalidated_count = cache.invalidate_all()

        # All queries should be invalidated
        self.assertIsNone(cache.get("query1"))
        self.assertIsNone(cache.get("query2"))
        self.assertIsNone(cache.get("query3"))

        # Should report 3 invalidations
        self.assertEqual(invalidated_count, 3, "Should invalidate all 3 cache entries")

    def test_query_result_cache_lru_eviction(self):
        """Test LRU eviction when query cache exceeds max_size."""
        from ace.retrieval_caching import QueryResultCache

        cache = QueryResultCache(max_size=3, ttl_seconds=600)

        # Fill cache to capacity
        cache.set("query1", [QdrantScoredResult("b1", 0.9, "A", {})])
        cache.set("query2", [QdrantScoredResult("b2", 0.8, "B", {})])
        cache.set("query3", [QdrantScoredResult("b3", 0.7, "C", {})])

        # Access query1 to make it recently used
        cache.get("query1")

        # Add new query - should evict LRU (query2)
        cache.set("query4", [QdrantScoredResult("b4", 0.6, "D", {})])

        # query2 should be evicted
        self.assertIsNone(cache.get("query2"), "LRU query should be evicted")
        self.assertIsNotNone(cache.get("query1"), "Recently used query should remain")
        self.assertIsNotNone(cache.get("query4"), "New query should exist")


class TestCacheMetrics(unittest.TestCase):
    """Test cache metrics and monitoring functionality."""

    def test_cache_hit_rate_calculation(self):
        """Test hit rate percentage calculation."""
        from ace.retrieval_caching import EmbeddingCache

        cache = EmbeddingCache(max_size=1000, ttl_seconds=3600)

        # 3 hits, 1 miss = 75% hit rate
        cache.set("text1", [0.1] * 768)
        cache.set("text2", [0.2] * 768)

        cache.get("text1")  # hit
        cache.get("text2")  # hit
        cache.get("text1")  # hit
        cache.get("nonexistent")  # miss

        metrics = cache.get_metrics()
        self.assertEqual(metrics["hits"], 3)
        self.assertEqual(metrics["misses"], 1)
        self.assertAlmostEqual(metrics["hit_rate"], 0.75, places=2)

    def test_cache_size_tracking(self):
        """Test cache size is accurately tracked."""
        from ace.retrieval_caching import QueryResultCache

        cache = QueryResultCache(max_size=100, ttl_seconds=600)

        # Add entries and track size
        cache.set("q1", [QdrantScoredResult("b1", 0.9, "A", {})])
        self.assertEqual(len(cache), 1)

        cache.set("q2", [QdrantScoredResult("b2", 0.8, "B", {})])
        self.assertEqual(len(cache), 2)

        # Invalidate and verify size decreases
        cache.invalidate_all()
        self.assertEqual(len(cache), 0)


class TestCacheThreadSafety(unittest.TestCase):
    """Test thread safety for concurrent cache access."""

    def test_concurrent_embedding_cache_access(self):
        """Test embedding cache handles concurrent reads/writes safely."""
        from ace.retrieval_caching import EmbeddingCache
        import threading

        cache = EmbeddingCache(max_size=1000, ttl_seconds=3600)
        errors = []

        def writer_thread(text_id: int):
            try:
                embedding = [float(text_id)] * 768
                cache.set(f"text_{text_id}", embedding)
            except Exception as e:
                errors.append(e)

        def reader_thread(text_id: int):
            try:
                cache.get(f"text_{text_id}")
            except Exception as e:
                errors.append(e)

        # Spawn 20 concurrent threads (10 writers, 10 readers)
        threads = []
        for i in range(10):
            threads.append(threading.Thread(target=writer_thread, args=(i,)))
            threads.append(threading.Thread(target=reader_thread, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No exceptions should occur
        self.assertEqual(len(errors), 0, f"Thread safety violations: {errors}")

    def test_concurrent_query_cache_invalidation(self):
        """Test query cache handles concurrent invalidations safely."""
        from ace.retrieval_caching import QueryResultCache
        import threading

        cache = QueryResultCache(max_size=500, ttl_seconds=600)
        errors = []

        # Pre-populate cache
        for i in range(50):
            cache.set(f"query_{i}", [QdrantScoredResult(f"b{i}", 0.9, "text", {})])

        def invalidate_thread(bullet_id: str):
            try:
                cache.invalidate_bullet(bullet_id)
            except Exception as e:
                errors.append(e)

        # Spawn 10 concurrent invalidations
        threads = [
            threading.Thread(target=invalidate_thread, args=(f"b{i}",))
            for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No exceptions should occur
        self.assertEqual(len(errors), 0, f"Thread safety violations: {errors}")


class TestCacheIntegrationWithQdrant(unittest.TestCase):
    """Test cache integration with Qdrant retrieval operations."""

    def test_qdrant_retrieval_with_embedding_cache(self):
        """Test Qdrant retrieval uses embedding cache when available."""
        # This test will be implemented when we integrate caching into QdrantRetrieval
        # For now, verify the interface exists
        from ace.retrieval_caching import EmbeddingCache

        cache = EmbeddingCache(max_size=1000, ttl_seconds=3600)

        # Verify cache can store/retrieve embeddings
        embedding = [0.5] * 768
        cache.set("test query", embedding)
        cached_embedding = cache.get("test query")

        self.assertEqual(cached_embedding, embedding, "Cache should support Qdrant integration")

    def test_qdrant_retrieval_with_result_cache(self):
        """Test Qdrant retrieval uses result cache when available."""
        # This test will be implemented when we integrate caching into QdrantRetrieval
        # For now, verify the interface exists
        from ace.retrieval_caching import QueryResultCache

        cache = QueryResultCache(max_size=500, ttl_seconds=600)

        # Verify cache can store/retrieve results
        results = [QdrantScoredResult("b1", 0.9, "text", {})]
        cache.set("test query", results)
        cached_results = cache.get("test query")

        self.assertEqual(cached_results, results, "Cache should support Qdrant integration")


if __name__ == "__main__":
    unittest.main()
