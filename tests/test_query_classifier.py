"""
Tests for QueryComplexityClassifier in retrieval_optimized.py

Tests the decision logic for when to use LLM rewriting vs simple keyword expansion.
"""
import unittest
from unittest.mock import MagicMock
from ace.retrieval_optimized import QueryComplexityClassifier
from ace.config import ELFConfig


class TestQueryComplexityClassifier(unittest.TestCase):
    """Test suite for query complexity classification."""

    def setUp(self):
        """Initialize classifier with default config."""
        self.config = ELFConfig()
        self.config.enable_query_classifier = True
        self.config.technical_terms_bypass_llm = True
        self.classifier = QueryComplexityClassifier(self.config)

    def test_technical_query_skips_llm(self):
        """Technical queries with clear intent should skip LLM rewriting."""
        technical_queries = [
            "api error",
            "config file",
            "database query",
            "async await",
            "test validation",
            "git commit",
            "docker deploy",
            "auth token",
        ]

        for query in technical_queries:
            with self.subTest(query=query):
                result = self.classifier.needs_llm_rewrite(query)
                self.assertFalse(
                    result,
                    f"Query '{query}' should skip LLM (contains technical terms)"
                )

    def test_short_nontechnical_query_needs_llm(self):
        """Short queries without technical terms need LLM for semantic expansion."""
        vague_queries = [
            "user preferences",
            "best practices",
            "common patterns",
            "recent changes",
            "important notes",
        ]

        for query in vague_queries:
            with self.subTest(query=query):
                result = self.classifier.needs_llm_rewrite(query)
                self.assertTrue(
                    result,
                    f"Query '{query}' should use LLM (short, no technical terms)"
                )

    def test_long_query_skips_llm(self):
        """Longer queries (>3 words) generally don't need LLM rewriting."""
        long_queries = [
            "how to implement user authentication flow",
            "best practices for error handling in production",
            "common patterns for database connection pooling",
        ]

        for query in long_queries:
            with self.subTest(query=query):
                result = self.classifier.needs_llm_rewrite(query)
                self.assertFalse(
                    result,
                    f"Query '{query}' should skip LLM (longer query with sufficient context)"
                )

    def test_classifier_disabled_always_returns_true(self):
        """When classifier is disabled, should always return True."""
        config = ELFConfig()
        config.enable_query_classifier = False
        classifier = QueryComplexityClassifier(config)

        test_queries = [
            "api error",
            "user preferences",
            "how to implement authentication",
        ]

        for query in test_queries:
            with self.subTest(query=query):
                result = classifier.needs_llm_rewrite(query)
                self.assertTrue(
                    result,
                    f"Query '{query}' should always need LLM when classifier is disabled"
                )

    def test_technical_terms_bypass_disabled(self):
        """When technical terms bypass is disabled, behavior should change."""
        config = ELFConfig()
        config.enable_query_classifier = True
        config.technical_terms_bypass_llm = False
        classifier = QueryComplexityClassifier(config)

        # Short technical query should now need LLM (bypass disabled)
        result = classifier.needs_llm_rewrite("api error")
        self.assertTrue(
            result,
            "Short technical query should need LLM when bypass is disabled"
        )

    def test_edge_case_three_word_queries(self):
        """Test boundary case of exactly 3-word queries."""
        # Technical 3-word query - should skip LLM
        result = self.classifier.needs_llm_rewrite("api error handling")
        self.assertFalse(
            result,
            "3-word technical query should skip LLM"
        )

        # Non-technical 3-word query - should use LLM
        result = self.classifier.needs_llm_rewrite("user preference settings")
        # Contains 'settings' which is technical, so should skip
        self.assertFalse(result)

    def test_case_insensitivity(self):
        """Classifier should be case-insensitive for technical terms."""
        queries = [
            "API Error",
            "CONFIG File",
            "Database QUERY",
            "ASYNC await",
        ]

        for query in queries:
            with self.subTest(query=query):
                result = self.classifier.needs_llm_rewrite(query)
                self.assertFalse(
                    result,
                    f"Query '{query}' should skip LLM (case-insensitive match)"
                )

    def test_technical_terms_coverage(self):
        """Verify comprehensive technical terms coverage."""
        # Sample from different categories
        categories = {
            'api_web': ['api', 'endpoint', 'http', 'https', 'request', 'response'],
            'config': ['config', 'configuration', 'settings'],
            'error': ['error', 'exception', 'bug', 'fix', 'debug'],
            'async': ['async', 'await', 'promise', 'callback'],
            'security': ['auth', 'authentication', 'token', 'jwt', 'encrypt', 'decrypt'],
            'database': ['database', 'query', 'sql', 'cache'],
            'testing': ['test', 'mock', 'spec', 'unittest'],
            'code': ['class', 'function', 'method', 'variable', 'import', 'export'],
            'devops': ['git', 'docker', 'deploy', 'ci', 'cd', 'build'],
        }

        for category, terms in categories.items():
            for term in terms:
                with self.subTest(category=category, term=term):
                    self.assertIn(
                        term,
                        self.classifier.TECHNICAL_TERMS,
                        f"Term '{term}' from category '{category}' should be in TECHNICAL_TERMS"
                    )


class TestQueryClassifierIntegration(unittest.TestCase):
    """Integration tests for classifier in retrieval pipeline."""

    def test_classifier_decision_logic_flow(self):
        """Test the complete decision flow for various query types."""
        config = ELFConfig()
        config.enable_query_classifier = True
        config.technical_terms_bypass_llm = True
        classifier = QueryComplexityClassifier(config)

        test_cases = [
            # (query, expected_needs_llm, reason)
            ("api", True, "Single word, even if technical, should use LLM"),
            ("api error", False, "Technical short query should skip LLM"),
            ("user says", True, "Non-technical short query needs LLM"),
            ("how to validate input", False, "Long query with technical term skips LLM"),
            ("best coding practices", False, "Longer query skips LLM"),
            ("validate input data", False, "Technical terms present, skip LLM"),
        ]

        for query, expected_needs_llm, reason in test_cases:
            with self.subTest(query=query, reason=reason):
                result = classifier.needs_llm_rewrite(query)
                word_count = len(query.split())

                # Adjust expectation for single-word queries
                if word_count <= 3:
                    has_technical = any(w in classifier.TECHNICAL_TERMS for w in query.lower().split())
                    expected = not has_technical
                else:
                    expected = False

                self.assertEqual(
                    result,
                    expected,
                    f"{reason} (Query: '{query}', words: {word_count})"
                )


if __name__ == '__main__':
    unittest.main()
