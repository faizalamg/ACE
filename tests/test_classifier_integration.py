"""
Integration tests for QueryComplexityClassifier with OptimizedRetriever.

Tests that the classifier correctly routes queries in the full retrieval pipeline.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
from ace.retrieval_optimized import OptimizedRetriever, QueryComplexityClassifier
from ace.config import ELFConfig, LLMConfig


class TestClassifierIntegration(unittest.TestCase):
    """Test classifier integration with retrieval pipeline."""

    def test_classifier_initialization(self):
        """Verify classifier is initialized in OptimizedRetriever."""
        with patch('ace.retrieval_optimized.httpx.Client'):
            retriever = OptimizedRetriever()

            self.assertIsNotNone(retriever.classifier)
            self.assertIsInstance(retriever.classifier, QueryComplexityClassifier)

    def test_technical_query_bypasses_llm(self):
        """Technical queries should skip LLM rewriting in search."""
        with patch('ace.retrieval_optimized.httpx.Client') as mock_client:
            # Mock HTTP client for embedding and search
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            # Mock embedding response
            mock_instance.post.return_value.status_code = 200
            mock_instance.post.return_value.json.return_value = {
                "data": [{"embedding": [0.1] * 768}],
                "result": {"points": []}
            }

            retriever = OptimizedRetriever()

            # Mock LLM rewriter to track calls
            retriever.llm_rewriter = Mock()
            retriever.llm_rewriter.rewrite.return_value = ["api error", "endpoint failure"]

            # Technical query should NOT call LLM rewriter
            retriever.search("api error", limit=5)

            # Verify LLM rewriter was NOT called (technical query bypasses LLM)
            retriever.llm_rewriter.rewrite.assert_not_called()

    def test_nontechnical_query_uses_llm(self):
        """Non-technical short queries should use LLM rewriting."""
        with patch('ace.retrieval_optimized.httpx.Client') as mock_client:
            # Mock HTTP client
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            # Mock embedding response
            mock_instance.post.return_value.status_code = 200
            mock_instance.post.return_value.json.return_value = {
                "data": [{"embedding": [0.1] * 768}],
                "result": {"points": []}
            }

            retriever = OptimizedRetriever()

            # Mock LLM rewriter to track calls
            retriever.llm_rewriter = Mock()
            retriever.llm_rewriter.rewrite.return_value = [
                "user preferences",
                "user settings and customization",
                "personal configuration options"
            ]

            # Non-technical query SHOULD call LLM rewriter
            retriever.search("user preferences", limit=5)

            # Verify LLM rewriter WAS called
            retriever.llm_rewriter.rewrite.assert_called_once_with("user preferences")

    def test_classifier_disabled_always_uses_llm(self):
        """When classifier is disabled, all queries should attempt LLM rewriting."""
        with patch('ace.retrieval_optimized.httpx.Client') as mock_client:
            # Mock HTTP client
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            # Mock embedding response
            mock_instance.post.return_value.status_code = 200
            mock_instance.post.return_value.json.return_value = {
                "data": [{"embedding": [0.1] * 768}],
                "result": {"points": []}
            }

            retriever = OptimizedRetriever()

            # Disable classifier
            retriever.classifier.config.enable_query_classifier = False

            # Mock LLM rewriter
            retriever.llm_rewriter = Mock()
            retriever.llm_rewriter.rewrite.return_value = ["api error", "endpoint failure"]

            # Even technical query should call LLM when classifier disabled
            retriever.search("api error", limit=5)

            # Verify LLM rewriter WAS called (classifier disabled)
            retriever.llm_rewriter.rewrite.assert_called_once_with("api error")

    def test_long_query_skips_llm(self):
        """Long queries should skip LLM rewriting regardless of content."""
        with patch('ace.retrieval_optimized.httpx.Client') as mock_client:
            # Mock HTTP client
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            # Mock embedding response
            mock_instance.post.return_value.status_code = 200
            mock_instance.post.return_value.json.return_value = {
                "data": [{"embedding": [0.1] * 768}],
                "result": {"points": []}
            }

            retriever = OptimizedRetriever()

            # Mock LLM rewriter
            retriever.llm_rewriter = Mock()
            retriever.llm_rewriter.rewrite.return_value = ["query"]

            # Long query (>3 words) should skip LLM
            retriever.search("how to implement user authentication flow", limit=5)

            # Verify LLM rewriter was NOT called
            retriever.llm_rewriter.rewrite.assert_not_called()


class TestClassifierPerformance(unittest.TestCase):
    """Test performance characteristics of classifier."""

    def test_technical_term_detection_is_fast(self):
        """Classifier should make decisions in < 1ms."""
        import time

        classifier = QueryComplexityClassifier()

        test_queries = [
            "api error",
            "database query",
            "config file",
            "user preferences",
            "best practices",
        ]

        start = time.perf_counter()

        for query in test_queries * 100:  # 500 total classifications
            classifier.needs_llm_rewrite(query)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should be able to classify 500 queries in < 50ms (< 0.1ms per query)
        self.assertLess(
            elapsed_ms,
            50,
            f"Classifier too slow: {elapsed_ms:.2f}ms for 500 queries"
        )

    def test_classifier_decision_consistency(self):
        """Same query should always get same classification."""
        classifier = QueryComplexityClassifier()

        queries = [
            "api error",
            "user preferences",
            "how to implement authentication",
        ]

        for query in queries:
            results = [classifier.needs_llm_rewrite(query) for _ in range(10)]

            # All results should be identical
            self.assertEqual(
                len(set(results)),
                1,
                f"Inconsistent classification for '{query}': {results}"
            )


if __name__ == '__main__':
    unittest.main()
