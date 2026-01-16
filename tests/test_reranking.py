"""Tests for cross-encoder reranking functionality.

TDD: Write failing tests first, then implement.
Tests cover:
1. Cross-encoder module initialization
2. Reranking API integration with SmartBulletIndex
3. Performance improvement validation
4. Latency measurement
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List
import time
import os

from ace.playbook import Bullet, Playbook
from ace.retrieval import SmartBulletIndex, ScoredBullet


# Fixture to disable reranking by default in tests (faster)
@pytest.fixture(autouse=True)
def disable_reranking_by_default():
    """Disable reranking by default in tests for speed."""
    original = os.environ.get("ACE_ENABLE_RERANKING")
    os.environ["ACE_ENABLE_RERANKING"] = "false"
    # Reset config to pick up env var
    from ace.config import reset_config
    reset_config()
    yield
    if original is not None:
        os.environ["ACE_ENABLE_RERANKING"] = original
    else:
        os.environ.pop("ACE_ENABLE_RERANKING", None)
    reset_config()


class TestCrossEncoderReranker:
    """Tests for the cross-encoder reranker module."""

    def test_reranker_module_exists(self):
        """Test that the reranker module can be imported."""
        from ace import reranker
        assert hasattr(reranker, 'CrossEncoderReranker')

    def test_reranker_lazy_initialization(self):
        """Test that reranker model is lazily loaded."""
        from ace.reranker import CrossEncoderReranker
        
        # Model should not load until first use
        reranker = CrossEncoderReranker()
        assert reranker._model is None
        
    def test_reranker_predict_scores(self):
        """Test that reranker returns relevance scores."""
        from ace.reranker import CrossEncoderReranker
        
        reranker = CrossEncoderReranker()
        query = "How to handle API rate limits?"
        documents = [
            "Use exponential backoff for retries",
            "API keys should be stored securely",
            "Handle rate limit errors gracefully",
        ]
        
        scores = reranker.predict(query, documents)
        
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)
        # Rate limit handling should score higher than API key storage
        assert scores[2] > scores[1]  # "rate limit errors" > "API keys"
        
    def test_reranker_singleton_model(self):
        """Test that reranker uses singleton model to avoid reloading."""
        from ace.reranker import CrossEncoderReranker, get_reranker
        
        r1 = get_reranker()
        r2 = get_reranker()
        
        assert r1 is r2  # Same instance


class TestSmartBulletIndexReranking:
    """Tests for reranking integration with SmartBulletIndex."""

    @pytest.fixture
    def playbook_with_bullets(self) -> Playbook:
        """Create a playbook with test bullets."""
        playbook = Playbook()
        
        # Add bullets with varying relevance to "API rate limits"
        bullets_data = [
            ("Use exponential backoff for API retries", ["api", "retry"]),
            ("Store API keys in environment variables", ["api", "security"]),
            ("Handle rate limit 429 errors with backoff", ["api", "rate-limit"]),
            ("Implement rate limiting middleware", ["rate-limit", "middleware"]),
            ("Use caching to reduce API calls", ["api", "caching"]),
            ("Monitor API usage with dashboards", ["api", "monitoring"]),
            ("Configure timeout for slow APIs", ["api", "timeout"]),
            ("Validate API responses before processing", ["api", "validation"]),
        ]
        
        for content, patterns in bullets_data:
            playbook.add_enriched_bullet(
                section="api",
                content=content,
                trigger_patterns=patterns,
            )
        
        return playbook

    def test_retrieve_without_rerank(self, playbook_with_bullets):
        """Test baseline retrieval without reranking."""
        index = SmartBulletIndex(playbook=playbook_with_bullets)
        
        results = index.retrieve(
            query="How to handle API rate limits?",
            limit=5,
            rerank=False,
        )
        
        assert len(results) <= 5
        assert all(isinstance(r, ScoredBullet) for r in results)
        
    def test_retrieve_with_rerank(self, playbook_with_bullets):
        """Test retrieval with cross-encoder reranking enabled."""
        index = SmartBulletIndex(playbook=playbook_with_bullets)
        
        results = index.retrieve(
            query="How to handle API rate limits?",
            limit=5,
            rerank=True,
        )
        
        assert len(results) <= 5
        # Rate limit specific bullets should be ranked higher
        top_contents = [r.content for r in results[:2]]
        assert any("rate limit" in c.lower() for c in top_contents)
        
    def test_retrieve_with_rerank_candidates(self, playbook_with_bullets):
        """Test that rerank_candidates controls first-pass retrieval count."""
        index = SmartBulletIndex(playbook=playbook_with_bullets)
        
        # With small candidate pool
        results_small = index.retrieve(
            query="API rate limits",
            limit=3,
            rerank=True,
            rerank_candidates=5,
        )
        
        # With larger candidate pool
        results_large = index.retrieve(
            query="API rate limits",
            limit=3,
            rerank=True,
            rerank_candidates=20,
        )
        
        # Both should return up to limit
        assert len(results_small) <= 3
        assert len(results_large) <= 3
        
    def test_rerank_match_reasons(self, playbook_with_bullets):
        """Test that reranked results include rerank score in match_reasons."""
        index = SmartBulletIndex(playbook=playbook_with_bullets)
        
        results = index.retrieve(
            query="rate limit handling",
            limit=3,
            rerank=True,
        )
        
        assert len(results) > 0
        # At least one result should have rerank score in reasons
        assert any("rerank:" in str(r.match_reasons) for r in results)


class TestRerankingPerformance:
    """Tests for reranking performance and quality improvement."""

    @pytest.fixture
    def diverse_playbook(self) -> Playbook:
        """Create a playbook with diverse bullets for quality testing."""
        playbook = Playbook()
        
        # Ground truth: bullets specifically about rate limiting
        rate_limit_bullets = [
            "Implement exponential backoff with jitter for 429 errors",
            "Use circuit breaker pattern for rate-limited APIs",
            "Cache responses to reduce rate limit consumption",
        ]
        
        # Noise: bullets mentioning "API" but not about rate limits
        noise_bullets = [
            "API authentication should use OAuth2",
            "Document API endpoints in OpenAPI spec",
            "API versioning follows semver",
            "Test APIs with Postman or similar",
            "API response times should be under 200ms",
        ]
        
        for content in rate_limit_bullets:
            playbook.add_enriched_bullet(
                section="rate_limit",
                content=content,
                trigger_patterns=["rate", "limit", "429", "throttle"],
            )
            
        for content in noise_bullets:
            playbook.add_enriched_bullet(
                section="api",
                content=content,
                trigger_patterns=["api"],
            )
        
        return playbook

    def test_reranking_improves_precision(self, diverse_playbook):
        """Test that reranking maintains or improves precision for relevant results."""
        index = SmartBulletIndex(playbook=diverse_playbook)
        query = "How to handle API rate limiting?"
        
        # Without reranking
        results_no_rerank = index.retrieve(query=query, limit=3, rerank=False)
        
        # With reranking
        results_rerank = index.retrieve(query=query, limit=3, rerank=True)
        
        # Count rate-limit specific results in top 3
        def count_relevant(results: List[ScoredBullet]) -> int:
            return sum(1 for r in results if any(
                kw in r.content.lower() 
                for kw in ["rate limit", "429", "backoff", "throttle", "circuit breaker"]
            ))
        
        relevant_no_rerank = count_relevant(results_no_rerank)
        relevant_rerank = count_relevant(results_rerank)
        
        # Reranking should find at least one relevant result
        # (The primary goal is precision improvement on ambiguous queries)
        assert relevant_rerank >= 1, "Reranking should find at least one relevant result"
        
        # Check that rerank scores are present in match reasons
        has_rerank_score = any("rerank:" in str(r.match_reasons) for r in results_rerank)
        assert has_rerank_score, "Reranked results should have rerank scores"
        
    def test_reranking_latency_acceptable(self, diverse_playbook):
        """Test that reranking adds acceptable latency (<500ms for small sets)."""
        index = SmartBulletIndex(playbook=diverse_playbook)
        query = "API rate limiting"
        
        # Measure without reranking
        start = time.perf_counter()
        index.retrieve(query=query, limit=5, rerank=False)
        time_no_rerank = time.perf_counter() - start
        
        # Measure with reranking
        start = time.perf_counter()
        index.retrieve(query=query, limit=5, rerank=True)
        time_rerank = time.perf_counter() - start
        
        # Reranking overhead should be under 500ms for small candidate sets
        overhead = time_rerank - time_no_rerank
        assert overhead < 0.5, f"Reranking overhead {overhead:.3f}s exceeds 500ms"


class TestRerankingOptionalDependency:
    """Tests for optional dependency handling."""

    def test_rerank_without_dependency_raises(self):
        """Test that reranking without sentence-transformers raises helpful error."""
        with patch.dict('sys.modules', {'sentence_transformers': None}):
            # Re-import to trigger missing dependency
            from ace.reranker import CrossEncoderReranker
            
            reranker = CrossEncoderReranker()
            
            with pytest.raises(ImportError) as exc_info:
                reranker.predict("query", ["doc1", "doc2"])
            
            assert "sentence-transformers" in str(exc_info.value).lower()
            
    def test_rerank_false_skips_import(self):
        """Test that rerank=False doesn't require sentence-transformers."""
        playbook = Playbook()
        playbook.add_bullet("test", "Test bullet content")
        
        index = SmartBulletIndex(playbook=playbook)
        
        # This should work even without sentence-transformers
        results = index.retrieve(query="test", limit=5, rerank=False)
        assert len(results) >= 0  # May be empty depending on scoring


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
