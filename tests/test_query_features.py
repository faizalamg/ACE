"""
Test suite for P7.2 Query Feature Extractor (10-dimension feature vector).

TDD RED PHASE: All tests written to FAIL initially.
Expected to fail until QueryFeatureExtractor is implemented.

Target Requirements:
- 10-dimension feature vector
- <5ms extraction latency
- >90% accuracy for feature detection
- All features normalized to [0, 1]
"""

import unittest
import time
from typing import List


class TestQueryFeatureExtractor(unittest.TestCase):
    """Test suite for QueryFeatureExtractor class and 10-dimension feature extraction."""

    def test_query_feature_extractor_class_exists(self):
        """Test that QueryFeatureExtractor class exists and is importable."""
        try:
            from ace.query_features import QueryFeatureExtractor
            extractor = QueryFeatureExtractor()
            self.assertIsNotNone(extractor)
        except ImportError:
            self.fail("QueryFeatureExtractor class does not exist in ace.query_features")

    def test_extract_returns_10_dimensions(self):
        """Test that extract() returns exactly 10 feature dimensions."""
        from ace.query_features import QueryFeatureExtractor

        extractor = QueryFeatureExtractor()
        query = "How do I fix authentication errors in production?"
        features = extractor.extract(query)

        self.assertEqual(len(features), 10,
                        f"Expected 10 dimensions, got {len(features)}")
        self.assertIsInstance(features, list)

    def test_length_normalized_calculation(self):
        """Test length_normalized feature (dimension 0) for short query."""
        from ace.query_features import QueryFeatureExtractor

        extractor = QueryFeatureExtractor()
        query = "fix auth"  # 8 characters
        features = extractor.extract(query)

        # Expect normalized length between 0.05-0.10 for very short query
        length_norm = features[0]
        self.assertGreaterEqual(length_norm, 0.05,
                               f"length_normalized too low: {length_norm}")
        self.assertLessEqual(length_norm, 0.15,
                            f"length_normalized too high: {length_norm}")

    def test_complexity_score_high_for_complex_query(self):
        """Test complexity_score (dimension 1) for syntactically complex query."""
        from ace.query_features import QueryFeatureExtractor

        extractor = QueryFeatureExtractor()
        complex_query = (
            "How do I implement a thread-safe singleton pattern in Python "
            "using metaclasses while ensuring compatibility with async/await?"
        )
        simple_query = "fix bug"

        complex_features = extractor.extract(complex_query)
        simple_features = extractor.extract(simple_query)

        complexity_complex = complex_features[1]
        complexity_simple = simple_features[1]

        self.assertGreater(complexity_complex, complexity_simple,
                          "Complex query should have higher complexity score")
        self.assertGreater(complexity_complex, 0.6,
                          f"Complex query score too low: {complexity_complex}")

    def test_has_code_detection(self):
        """Test has_code feature (dimension 4) detects code snippets."""
        from ace.query_features import QueryFeatureExtractor

        extractor = QueryFeatureExtractor()

        # Queries with code
        code_queries = [
            "def foo(): pass",
            "SELECT * FROM users WHERE id = 1",
            "const x = 42;",
            "import numpy as np"
        ]

        # Queries without code
        no_code_queries = [
            "How do I fix this error?",
            "Explain authentication flow"
        ]

        for query in code_queries:
            features = extractor.extract(query)
            has_code = features[4]
            self.assertEqual(has_code, 1.0,
                           f"Failed to detect code in: {query}")

        for query in no_code_queries:
            features = extractor.extract(query)
            has_code = features[4]
            self.assertEqual(has_code, 0.0,
                           f"False positive code detection in: {query}")

    def test_is_question_detection(self):
        """Test is_question feature (dimension 5) detects interrogative queries."""
        from ace.query_features import QueryFeatureExtractor

        extractor = QueryFeatureExtractor()

        # Questions
        questions = [
            "How do I fix this?",
            "What is the best approach?",
            "Why does this fail?",
            "Where should I place this code?"
        ]

        # Non-questions
        statements = [
            "Fix authentication error",
            "Implement caching layer",
            "Refactor database queries"
        ]

        for query in questions:
            features = extractor.extract(query)
            is_question = features[5]
            self.assertEqual(is_question, 1.0,
                           f"Failed to detect question: {query}")

        for query in statements:
            features = extractor.extract(query)
            is_question = features[5]
            self.assertEqual(is_question, 0.0,
                           f"False positive question detection: {query}")

    def test_domain_signal_technical_terms(self):
        """Test domain_signal (dimension 2) detects technical terminology."""
        from ace.query_features import QueryFeatureExtractor

        extractor = QueryFeatureExtractor()

        technical_query = "Optimize PostgreSQL query using EXPLAIN ANALYZE"
        generic_query = "Make it faster"

        technical_features = extractor.extract(technical_query)
        generic_features = extractor.extract(generic_query)

        domain_technical = technical_features[2]
        domain_generic = generic_features[2]

        self.assertGreater(domain_technical, domain_generic,
                          "Technical query should have higher domain signal")
        self.assertGreater(domain_technical, 0.5,
                          f"Domain signal too low for technical query: {domain_technical}")

    def test_intent_procedural_detection(self):
        """Test intent_procedural (dimension 3) detects how-to queries."""
        from ace.query_features import QueryFeatureExtractor

        extractor = QueryFeatureExtractor()

        procedural_queries = [
            "how to implement authentication",
            "steps to deploy to production",
            "guide for setting up database"
        ]

        non_procedural_queries = [
            "what is authentication",
            "authentication definition",
            "explain OAuth"
        ]

        for query in procedural_queries:
            features = extractor.extract(query)
            intent_proc = features[3]
            self.assertGreater(intent_proc, 0.7,
                             f"Failed to detect procedural intent: {query}")

        for query in non_procedural_queries:
            features = extractor.extract(query)
            intent_proc = features[3]
            self.assertLess(intent_proc, 0.3,
                          f"False procedural detection: {query}")

    def test_has_negation_detection(self):
        """Test has_negation (dimension 8) detects negation keywords."""
        from ace.query_features import QueryFeatureExtractor

        extractor = QueryFeatureExtractor()

        negation_queries = [
            "How to fix errors without breaking production",
            "Implement caching but not in memory",
            "Avoid race conditions"
        ]

        no_negation_queries = [
            "Fix authentication error",
            "Implement caching layer"
        ]

        for query in negation_queries:
            features = extractor.extract(query)
            has_negation = features[8]
            self.assertEqual(has_negation, 1.0,
                           f"Failed to detect negation in: {query}")

        for query in no_negation_queries:
            features = extractor.extract(query)
            has_negation = features[8]
            self.assertEqual(has_negation, 0.0,
                           f"False negation detection: {query}")

    def test_entity_density_calculation(self):
        """Test entity_density (dimension 9) measures named entity concentration."""
        from ace.query_features import QueryFeatureExtractor

        extractor = QueryFeatureExtractor()

        high_entity_query = "Deploy FastAPI app to AWS Lambda using Docker"
        low_entity_query = "make it work better"

        high_features = extractor.extract(high_entity_query)
        low_features = extractor.extract(low_entity_query)

        entity_high = high_features[9]
        entity_low = low_features[9]

        self.assertGreater(entity_high, entity_low,
                          "High-entity query should have higher density")
        self.assertGreater(entity_high, 0.3,
                          f"Entity density too low for technical query: {entity_high}")

    def test_temporal_signal_detection(self):
        """Test temporal_signal (dimension 7) detects time-related keywords."""
        from ace.query_features import QueryFeatureExtractor

        extractor = QueryFeatureExtractor()

        temporal_queries = [
            "errors that occurred yesterday",
            "last week's deployment failures",
            "recent authentication issues",
            "current production problems"
        ]

        non_temporal_queries = [
            "fix authentication error",
            "implement caching"
        ]

        for query in temporal_queries:
            features = extractor.extract(query)
            temporal = features[7]
            self.assertGreater(temporal, 0.5,
                             f"Failed to detect temporal signal: {query}")

        for query in non_temporal_queries:
            features = extractor.extract(query)
            temporal = features[7]
            self.assertLess(temporal, 0.3,
                          f"False temporal detection: {query}")

    def test_extraction_latency_under_5ms(self):
        """Test that feature extraction completes in <5ms (performance requirement)."""
        from ace.query_features import QueryFeatureExtractor

        extractor = QueryFeatureExtractor()
        query = "How do I implement OAuth2 authentication in FastAPI production?"

        # Warm-up run (exclude cold start)
        extractor.extract(query)

        # Measure 100 iterations
        start = time.perf_counter()
        for _ in range(100):
            extractor.extract(query)
        end = time.perf_counter()

        avg_latency_ms = ((end - start) / 100) * 1000

        self.assertLess(avg_latency_ms, 5.0,
                       f"Extraction latency {avg_latency_ms:.2f}ms exceeds 5ms target")

    def test_feature_vector_all_normalized_0_to_1(self):
        """Test that all features are normalized to [0, 1] range."""
        from ace.query_features import QueryFeatureExtractor

        extractor = QueryFeatureExtractor()

        test_queries = [
            "fix bug",
            "How do I implement a distributed cache using Redis cluster?",
            "def authenticate(user): return token",
            "errors yesterday",
            "What is the best approach without breaking production?"
        ]

        for query in test_queries:
            features = extractor.extract(query)

            for i, value in enumerate(features):
                self.assertGreaterEqual(value, 0.0,
                                       f"Feature {i} below 0.0: {value} for query: {query}")
                self.assertLessEqual(value, 1.0,
                                    f"Feature {i} above 1.0: {value} for query: {query}")


    def test_get_contextual_stopwords_conversational(self):
        """Test get_contextual_stopwords returns aggressive stopwords for conversational queries."""
        from ace.query_features import QueryFeatureExtractor

        extractor = QueryFeatureExtractor()
        stopwords = extractor.get_contextual_stopwords(query_type="conversational")

        # Conversational queries should have aggressive stopword removal
        self.assertIsInstance(stopwords, set, "Stopwords should be a set")
        self.assertGreater(len(stopwords), 30,
                          f"Conversational stopwords too few: {len(stopwords)}")

        # Must include common conversational filler words
        conversational_fillers = {'how', 'does', 'this', 'what', 'is', 'that', 'the', 'a', 'an'}
        for word in conversational_fillers:
            self.assertIn(word, stopwords,
                         f"Conversational stopwords missing '{word}'")

    def test_get_contextual_stopwords_technical(self):
        """Test get_contextual_stopwords returns minimal stopwords for technical queries."""
        from ace.query_features import QueryFeatureExtractor

        extractor = QueryFeatureExtractor()
        stopwords = extractor.get_contextual_stopwords(query_type="technical")

        # Technical queries should have minimal stopword removal
        self.assertIsInstance(stopwords, set, "Stopwords should be a set")
        self.assertLess(len(stopwords), 20,
                       f"Technical stopwords too many: {len(stopwords)}")

        # Must NOT remove technical action words
        technical_preserves = {'configure', 'setup', 'install', 'implement', 'deploy'}
        for word in technical_preserves:
            self.assertNotIn(word, stopwords,
                           f"Technical stopwords incorrectly includes '{word}'")

    def test_filter_stopwords_conversational_query(self):
        """Test filter_stopwords removes filler words from conversational queries."""
        from ace.query_features import QueryFeatureExtractor

        extractor = QueryFeatureExtractor()

        conversational_query = "How does this authentication thing work exactly?"
        filtered = extractor.filter_stopwords(conversational_query, query_type="conversational")

        # Should remove: how, does, this, thing, exactly
        # Should keep: authentication, work
        self.assertIsInstance(filtered, str, "Filtered result should be a string")
        self.assertIn("authentication", filtered,
                     "Should preserve technical term 'authentication'")
        self.assertIn("work", filtered,
                     "Should preserve action word 'work'")

        # Verify aggressive removal of fillers
        self.assertNotIn("how", filtered.lower(),
                        "Should remove filler word 'how'")
        self.assertNotIn("does", filtered.lower(),
                        "Should remove filler word 'does'")
        self.assertNotIn("this", filtered.lower(),
                        "Should remove filler word 'this'")

    def test_filter_stopwords_technical_query(self):
        """Test filter_stopwords preserves technical action words."""
        from ace.query_features import QueryFeatureExtractor

        extractor = QueryFeatureExtractor()

        technical_query = "configure setup install nginx server production"
        filtered = extractor.filter_stopwords(technical_query, query_type="technical")

        # Should keep ALL technical terms (minimal stopword removal)
        technical_terms = ['configure', 'setup', 'install', 'nginx', 'server', 'production']
        for term in technical_terms:
            self.assertIn(term, filtered.lower(),
                         f"Should preserve technical term '{term}'")

    def test_filter_stopwords_default_type(self):
        """Test filter_stopwords defaults to conversational type when not specified."""
        from ace.query_features import QueryFeatureExtractor

        extractor = QueryFeatureExtractor()

        query = "what is the best approach"

        # Default should match conversational behavior
        filtered_default = extractor.filter_stopwords(query)
        filtered_conversational = extractor.filter_stopwords(query, query_type="conversational")

        self.assertEqual(filtered_default, filtered_conversational,
                        "Default filter should match conversational behavior")

    def test_filter_stopwords_preserves_word_order(self):
        """Test filter_stopwords maintains original word order."""
        from ace.query_features import QueryFeatureExtractor

        extractor = QueryFeatureExtractor()

        query = "authentication database production server"
        filtered = extractor.filter_stopwords(query, query_type="technical")

        # All words should be preserved in original order
        self.assertEqual(filtered.lower(), query.lower(),
                        "Technical filtering should preserve all technical terms in order")

    def test_filter_stopwords_empty_query(self):
        """Test filter_stopwords handles empty query gracefully."""
        from ace.query_features import QueryFeatureExtractor

        extractor = QueryFeatureExtractor()

        filtered = extractor.filter_stopwords("", query_type="conversational")

        self.assertEqual(filtered, "",
                        "Empty query should return empty string")

    def test_filter_stopwords_only_stopwords(self):
        """Test filter_stopwords handles query with only stopwords."""
        from ace.query_features import QueryFeatureExtractor

        extractor = QueryFeatureExtractor()

        # Query with only conversational filler words
        query = "how does this that the a an"
        filtered = extractor.filter_stopwords(query, query_type="conversational")

        # Should return empty or minimal result
        self.assertLess(len(filtered.split()), 2,
                       "Query with only stopwords should be heavily filtered")


if __name__ == '__main__':
    unittest.main()
