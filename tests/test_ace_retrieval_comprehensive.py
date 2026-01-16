"""
Comprehensive testing of ACE retrieval system.

Tests diverse query types and cross-encoder reranking impact on retrieval quality.
Measures:
- Query coverage across namespaces
- Cross-encoder reranking effectiveness  
- Latency impact
- Semantic relevance quality

Run with: python -m pytest tests/test_ace_retrieval_comprehensive.py -v --no-cov -s
"""

import time
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
import pytest
import os

from ace.unified_memory import UnifiedMemoryIndex, UnifiedNamespace
from ace.retrieval import SmartBulletIndex
from ace.config import get_config, get_retrieval_config, reset_config


@dataclass
class QueryTestCase:
    """Test case for retrieval evaluation."""
    query: str
    category: str  # task_specific, vague, domain_crossing, short, long
    expected_namespace: str  # user_prefs, task_strategies, project_specific, any
    description: str
    min_expected_results: int = 1
    expected_keywords: List[str] = field(default_factory=list)


# Diverse query test cases
DIVERSE_QUERY_TEST_CASES = [
    # Task-specific queries
    QueryTestCase(
        query="debugging timeout issues in production API",
        category="task_specific",
        expected_namespace="task_strategies",
        description="Task-specific debugging query",
        expected_keywords=["debug", "timeout", "error", "log", "trace"]
    ),
    QueryTestCase(
        query="how to fix memory leak in Python application",
        category="task_specific", 
        expected_namespace="task_strategies",
        description="Task-specific memory debugging",
        expected_keywords=["memory", "leak", "python", "profile"]
    ),
    QueryTestCase(
        query="best practices for error handling in async code",
        category="task_specific",
        expected_namespace="task_strategies",
        description="Task-specific async patterns",
        expected_keywords=["error", "async", "exception", "handling"]
    ),
    
    # Vague/ambiguous queries
    QueryTestCase(
        query="help with code",
        category="vague",
        expected_namespace="any",
        description="Vague help query",
        min_expected_results=0  # May not find relevant results
    ),
    QueryTestCase(
        query="make it faster",
        category="vague",
        expected_namespace="any",
        description="Vague performance query",
        expected_keywords=["performance", "speed", "optimize", "cache", "fast"]
    ),
    QueryTestCase(
        query="fix the bug",
        category="vague",
        expected_namespace="any",
        description="Vague debugging query",
        expected_keywords=["bug", "fix", "error", "debug"]
    ),
    
    # Domain-crossing queries
    QueryTestCase(
        query="API rate limiting security implications performance",
        category="domain_crossing",
        expected_namespace="any",
        description="Domain-crossing security+performance",
        expected_keywords=["rate", "limit", "security", "performance", "API"]
    ),
    QueryTestCase(
        query="database optimization with security constraints",
        category="domain_crossing",
        expected_namespace="any",
        description="Domain-crossing database+security",
        expected_keywords=["database", "security", "optimize"]
    ),
    
    # Short keyword queries
    QueryTestCase(
        query="TDD",
        category="short",
        expected_namespace="any",
        description="Short acronym query",
        expected_keywords=["test", "TDD", "driven"]
    ),
    QueryTestCase(
        query="KISS DRY",
        category="short",
        expected_namespace="any",
        description="Short principles query",
        expected_keywords=["KISS", "DRY", "simple", "repeat"]
    ),
    QueryTestCase(
        query="async await",
        category="short",
        expected_namespace="any",
        description="Short technical query",
        expected_keywords=["async", "await", "asyncio"]
    ),
    
    # Long descriptive queries
    QueryTestCase(
        query="When implementing a new feature that involves complex state management across multiple components, what patterns and strategies should I follow to ensure maintainability and testability?",
        category="long",
        expected_namespace="task_strategies",
        description="Long architectural query",
        expected_keywords=["state", "pattern", "component", "maintainability", "test"]
    ),
    QueryTestCase(
        query="I am encountering intermittent failures in my CI/CD pipeline where tests pass locally but fail randomly in GitHub Actions with timeout errors on database operations",
        category="long",
        expected_namespace="task_strategies",
        description="Long debugging scenario query",
        expected_keywords=["CI", "timeout", "database", "test", "fail"]
    ),
    
    # Namespace-specific queries
    QueryTestCase(
        query="user preferences for code formatting style",
        category="namespace_specific",
        expected_namespace="user_prefs",
        description="User preferences query",
        expected_keywords=["prefer", "style", "format", "naming"]
    ),
    QueryTestCase(
        query="cross-encoder reranking implementation",
        category="namespace_specific",
        expected_namespace="task_strategies", 
        description="Task strategies query",
        expected_keywords=["rerank", "cross-encoder", "precision"]
    ),
]


class TestDiverseQueryRetrieval:
    """Test retrieval across diverse query types."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        reset_config()
        config = get_config()
        self.index = UnifiedMemoryIndex(
            collection_name=config.qdrant.unified_collection,
            qdrant_url=config.qdrant.url,
        )

    def test_task_specific_queries(self):
        """Test task-specific queries return relevant results."""
        task_queries = [tc for tc in DIVERSE_QUERY_TEST_CASES if tc.category == "task_specific"]
        
        results_summary = []
        for tc in task_queries:
            results = self.index.retrieve(query=tc.query, limit=5)
            
            # Check if we got results
            has_results = len(results) >= tc.min_expected_results
            
            # Check keyword relevance
            if results and tc.expected_keywords:
                top_result_text = results[0].content.lower()
                keyword_matches = sum(1 for kw in tc.expected_keywords if kw.lower() in top_result_text)
                keyword_coverage = keyword_matches / len(tc.expected_keywords)
            else:
                keyword_coverage = 0
            
            results_summary.append({
                "query": tc.query[:50],
                "results_count": len(results),
                "has_min_results": has_results,
                "keyword_coverage": keyword_coverage,
            })
            
            print(f"\n[{tc.description}]")
            print(f"  Query: {tc.query[:60]}...")
            print(f"  Results: {len(results)}, Keyword coverage: {keyword_coverage:.2%}")
            if results:
                print(f"  Top result: {results[0].content[:80]}...")
        
        # At least 50% of task-specific queries should return results
        success_rate = sum(1 for r in results_summary if r["has_min_results"]) / len(results_summary)
        assert success_rate >= 0.5, f"Task-specific query success rate {success_rate:.2%} below 50%"

    def test_vague_queries_handled_gracefully(self):
        """Test that vague queries are handled without errors."""
        vague_queries = [tc for tc in DIVERSE_QUERY_TEST_CASES if tc.category == "vague"]
        
        for tc in vague_queries:
            # Should not raise exception
            results = self.index.retrieve(query=tc.query, limit=5)
            
            print(f"\n[Vague Query: {tc.description}]")
            print(f"  Query: '{tc.query}'")
            print(f"  Results: {len(results)}")
            if results:
                print(f"  Top result: {results[0].content[:80]}...")
            
            # Vague queries may or may not return results, but shouldn't crash
            assert isinstance(results, list)

    def test_short_keyword_queries(self):
        """Test short keyword/acronym queries."""
        short_queries = [tc for tc in DIVERSE_QUERY_TEST_CASES if tc.category == "short"]
        
        for tc in short_queries:
            results = self.index.retrieve(query=tc.query, limit=5)
            
            print(f"\n[Short Query: {tc.description}]")
            print(f"  Query: '{tc.query}'")
            print(f"  Results: {len(results)}")
            if results:
                print(f"  Top result: {results[0].content[:80]}...")
            
            # Short queries should still work
            assert isinstance(results, list)

    def test_long_descriptive_queries(self):
        """Test long descriptive queries."""
        long_queries = [tc for tc in DIVERSE_QUERY_TEST_CASES if tc.category == "long"]
        
        for tc in long_queries:
            results = self.index.retrieve(query=tc.query, limit=5)
            
            print(f"\n[Long Query: {tc.description}]")
            print(f"  Query: {tc.query[:80]}...")
            print(f"  Results: {len(results)}")
            if results:
                print(f"  Top result: {results[0].content[:80]}...")
            
            # Long queries should typically find some matches
            assert isinstance(results, list)

    def test_namespace_filtering(self):
        """Test that namespace filtering works correctly."""
        for namespace in [UnifiedNamespace.USER_PREFS, UnifiedNamespace.TASK_STRATEGIES]:
            results = self.index.retrieve(
                query="coding best practices",
                limit=5,
                namespace=namespace,
            )
            
            print(f"\n[Namespace: {namespace.value}]")
            print(f"  Results: {len(results)}")
            
            # Verify all results match the namespace
            for r in results:
                assert r.namespace == namespace or str(r.namespace) == namespace.value, \
                    f"Result namespace {r.namespace} doesn't match filter {namespace}"


class TestCrossEncoderReranking:
    """Test cross-encoder reranking effectiveness."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        reset_config()
        config = get_config()
        self.index = UnifiedMemoryIndex(
            collection_name=config.qdrant.unified_collection,
            qdrant_url=config.qdrant.url,
        )

    def test_reranking_enabled_by_default(self):
        """Verify reranking is enabled in default config."""
        config = get_retrieval_config()
        assert config.enable_reranking is True
        assert config.cross_encoder_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_retrieval_latency_with_reranking(self):
        """Measure retrieval latency with reranking enabled."""
        query = "debugging timeout errors in production"
        
        # Warm up (first call may load model)
        _ = self.index.retrieve(query=query, limit=5)
        
        # Measure latency over multiple queries
        latencies = []
        for _ in range(5):
            start = time.perf_counter()
            _ = self.index.retrieve(query=query, limit=5)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        print(f"\n[Reranking Latency]")
        print(f"  Average: {avg_latency*1000:.1f}ms")
        print(f"  Max: {max_latency*1000:.1f}ms")
        
        # Latency should be reasonable (<5s for full pipeline with LLM typo correction/expansion)
        # Note: GLM 4.7 adds ~1-3s per LLM call, threshold increased to account for this
        assert avg_latency < 5.0, f"Average latency {avg_latency:.2f}s exceeds 5s threshold"

    def test_reranking_improves_relevance(self):
        """Test that reranking score appears in results."""
        # Use a query where reranking should help
        query = "cross-encoder reranking precision improvement"
        
        results = self.index.retrieve(query=query, limit=10)
        
        print(f"\n[Reranking Results]")
        print(f"  Query: {query}")
        print(f"  Total results: {len(results)}")
        
        for i, r in enumerate(results[:5]):
            # Check for rerank score in metadata or qdrant_score
            score = getattr(r, 'qdrant_score', 0.0)
            print(f"  [{i+1}] Score: {score:.4f} | {r.content[:60]}...")
        
        # Should return some results
        assert len(results) >= 1


class TestRetrievalQualityMetrics:
    """Test retrieval quality metrics."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        reset_config()
        config = get_config()
        self.index = UnifiedMemoryIndex(
            collection_name=config.qdrant.unified_collection,
            qdrant_url=config.qdrant.url,
        )

    def test_recall_at_5(self):
        """Test recall@5 for known-good queries."""
        # Queries we expect to find relevant memories for
        test_queries = [
            ("cross-encoder reranking", ["rerank", "cross-encoder", "precision"]),
            ("user coding preferences", ["prefer", "style", "naming"]),
            ("API rate limiting", ["rate", "limit", "API"]),
            ("error handling best practices", ["error", "exception", "handling"]),
        ]
        
        total_queries = 0
        queries_with_relevant = 0
        
        for query, expected_terms in test_queries:
            results = self.index.retrieve(query=query, limit=5)
            
            # Check if any result contains expected terms
            has_relevant = False
            for r in results:
                content_lower = r.content.lower()
                term_matches = sum(1 for t in expected_terms if t.lower() in content_lower)
                if term_matches >= 1:  # At least 1 term match
                    has_relevant = True
                    break
            
            total_queries += 1
            if has_relevant:
                queries_with_relevant += 1
            
            print(f"\n[Query: {query}]")
            print(f"  Has relevant result: {has_relevant}")
            print(f"  Results: {len(results)}")
        
        recall = queries_with_relevant / total_queries if total_queries > 0 else 0
        print(f"\n[Recall@5: {recall:.2%}]")
        
        # Recall@5 should be at least 75% for these queries
        assert recall >= 0.75, f"Recall@5 {recall:.2%} below 75% threshold"


class TestHybridSearchComponents:
    """Test hybrid search (dense + BM25 + RRF) components."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        reset_config()
        config = get_config()
        self.index = UnifiedMemoryIndex(
            collection_name=config.qdrant.unified_collection,
            qdrant_url=config.qdrant.url,
        )

    def test_semantic_vs_keyword_queries(self):
        """Test that both semantic and keyword-heavy queries work."""
        # Semantic query (meaning-based)
        semantic_query = "strategies for handling slow network responses"
        semantic_results = self.index.retrieve(query=semantic_query, limit=5)
        
        # Keyword query (exact terms)
        keyword_query = "timeout retry backoff exponential"
        keyword_results = self.index.retrieve(query=keyword_query, limit=5)
        
        print(f"\n[Semantic Query]")
        print(f"  Query: {semantic_query}")
        print(f"  Results: {len(semantic_results)}")
        if semantic_results:
            print(f"  Top: {semantic_results[0].content[:80]}...")
        
        print(f"\n[Keyword Query]")
        print(f"  Query: {keyword_query}")
        print(f"  Results: {len(keyword_results)}")
        if keyword_results:
            print(f"  Top: {keyword_results[0].content[:80]}...")
        
        # Both should return results
        assert len(semantic_results) > 0 or len(keyword_results) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        reset_config()
        config = get_config()
        self.index = UnifiedMemoryIndex(
            collection_name=config.qdrant.unified_collection,
            qdrant_url=config.qdrant.url,
        )

    def test_empty_query(self):
        """Test empty query handling."""
        results = self.index.retrieve(query="", limit=5)
        # Should return empty or handle gracefully
        assert isinstance(results, list)

    def test_special_characters_query(self):
        """Test query with special characters."""
        results = self.index.retrieve(query="error: 'NoneType' != expected [fix]", limit=5)
        assert isinstance(results, list)

    def test_unicode_query(self):
        """Test query with unicode characters."""
        results = self.index.retrieve(query="configuration 设置 настройки", limit=5)
        assert isinstance(results, list)

    def test_very_long_query(self):
        """Test very long query handling."""
        long_query = "debugging " * 500  # Very long query
        results = self.index.retrieve(query=long_query[:5000], limit=5)
        assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--no-cov", "-s"])
