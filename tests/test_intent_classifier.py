"""Tests for Intent Classifier that routes queries to appropriate retrieval strategies.

TDD: These tests define the expected behavior BEFORE implementation.
"""

import unittest

import pytest


@pytest.mark.unit
class TestIntentClassifierBasic(unittest.TestCase):
    """Test basic intent classification functionality."""

    def test_intent_classifier_exists(self):
        """Test that IntentClassifier class exists."""
        from ace.retrieval import IntentClassifier

        classifier = IntentClassifier()
        self.assertIsNotNone(classifier)

    def test_classify_returns_valid_intent(self):
        """Test that classify returns a valid intent type."""
        from ace.retrieval import IntentClassifier, IntentType

        classifier = IntentClassifier()
        result = classifier.classify("How do I debug this error?")

        self.assertIn(result, ["analytical", "factual", "procedural", "general"])


@pytest.mark.unit
class TestIntentClassifierAnalytical(unittest.TestCase):
    """Test classification of analytical queries."""

    def setUp(self):
        """Set up classifier."""
        from ace.retrieval import IntentClassifier

        self.classifier = IntentClassifier()

    def test_classify_comparison_query(self):
        """Test classification of comparison queries."""
        queries = [
            "Which database should I choose between PostgreSQL and MongoDB?",
            "What are the trade-offs between REST and GraphQL?",
            "Compare React vs Vue for this use case",
            "What are the pros and cons of microservices?",
        ]

        for query in queries:
            result = self.classifier.classify(query)
            self.assertEqual(result, "analytical", f"Failed for: {query}")

    def test_classify_decision_query(self):
        """Test classification of decision-making queries."""
        queries = [
            "Should I use a cache here?",
            "Is it better to normalize or denormalize this data?",
            "What's the best approach for handling this?",
        ]

        for query in queries:
            result = self.classifier.classify(query)
            self.assertEqual(result, "analytical", f"Failed for: {query}")

    def test_classify_why_query(self):
        """Test classification of 'why' analytical queries."""
        queries = [
            "Why is this architecture preferred?",
            "Why does this pattern work better?",
        ]

        for query in queries:
            result = self.classifier.classify(query)
            self.assertEqual(result, "analytical", f"Failed for: {query}")


@pytest.mark.unit
class TestIntentClassifierFactual(unittest.TestCase):
    """Test classification of factual queries."""

    def setUp(self):
        """Set up classifier."""
        from ace.retrieval import IntentClassifier

        self.classifier = IntentClassifier()

    def test_classify_what_is_query(self):
        """Test classification of 'what is' queries."""
        queries = [
            "What is the syntax for async/await?",
            "What are the required parameters for this function?",
            "What version of Python supports this feature?",
        ]

        for query in queries:
            result = self.classifier.classify(query)
            self.assertEqual(result, "factual", f"Failed for: {query}")

    def test_classify_lookup_query(self):
        """Test classification of lookup/reference queries."""
        queries = [
            "Show me the configuration options",
            "List all available methods",
            "What does this error code mean?",
        ]

        for query in queries:
            result = self.classifier.classify(query)
            self.assertEqual(result, "factual", f"Failed for: {query}")

    def test_classify_definition_query(self):
        """Test classification of definition queries."""
        queries = [
            "Define idempotency",
            "What does SOLID stand for?",
        ]

        for query in queries:
            result = self.classifier.classify(query)
            self.assertEqual(result, "factual", f"Failed for: {query}")


@pytest.mark.unit
class TestIntentClassifierProcedural(unittest.TestCase):
    """Test classification of procedural queries."""

    def setUp(self):
        """Set up classifier."""
        from ace.retrieval import IntentClassifier

        self.classifier = IntentClassifier()

    def test_classify_how_to_query(self):
        """Test classification of 'how to' queries."""
        queries = [
            "How do I deploy this application?",
            "How to set up the database connection?",
            "How can I implement authentication?",
        ]

        for query in queries:
            result = self.classifier.classify(query)
            self.assertEqual(result, "procedural", f"Failed for: {query}")

    def test_classify_step_by_step_query(self):
        """Test classification of step-by-step queries."""
        queries = [
            "Walk me through the setup process",
            "Guide me through debugging this",
            "Steps to migrate the database",
        ]

        for query in queries:
            result = self.classifier.classify(query)
            self.assertEqual(result, "procedural", f"Failed for: {query}")

    def test_classify_action_query(self):
        """Test classification of action-oriented queries."""
        queries = [
            "Fix the broken tests",
            "Implement a new endpoint",
            "Create a backup of the database",
        ]

        for query in queries:
            result = self.classifier.classify(query)
            self.assertEqual(result, "procedural", f"Failed for: {query}")


@pytest.mark.unit
class TestIntentClassifierGeneral(unittest.TestCase):
    """Test classification of general/ambiguous queries."""

    def setUp(self):
        """Set up classifier."""
        from ace.retrieval import IntentClassifier

        self.classifier = IntentClassifier()

    def test_classify_ambiguous_query(self):
        """Test classification of ambiguous queries."""
        queries = [
            "Hello",
            "Thanks",
            "OK",
        ]

        for query in queries:
            result = self.classifier.classify(query)
            self.assertEqual(result, "general", f"Failed for: {query}")


@pytest.mark.unit
class TestIntentClassifierConfidence(unittest.TestCase):
    """Test confidence scores for classification."""

    def setUp(self):
        """Set up classifier."""
        from ace.retrieval import IntentClassifier

        self.classifier = IntentClassifier()

    def test_classify_with_confidence_returns_tuple(self):
        """Test that classify_with_confidence returns intent and score."""
        result = self.classifier.classify_with_confidence("How do I deploy?")

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        intent, confidence = result
        self.assertIn(intent, ["analytical", "factual", "procedural", "general"])
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_high_confidence_for_clear_query(self):
        """Test high confidence for clearly procedural query."""
        intent, confidence = self.classifier.classify_with_confidence(
            "How do I step-by-step deploy the application to production?"
        )

        self.assertEqual(intent, "procedural")
        self.assertGreater(confidence, 0.7)

    def test_lower_confidence_for_ambiguous_query(self):
        """Test that truly ambiguous queries get general classification."""
        # "OK" is actually a clear general/greeting pattern with high confidence
        # Test with a truly ambiguous query instead
        intent, confidence = self.classifier.classify_with_confidence("maybe tomorrow")

        # Should be general but with low confidence since no clear pattern
        self.assertEqual(intent, "general")
        self.assertLess(confidence, 0.5)


@pytest.mark.unit
class TestIntentClassifierIntegration(unittest.TestCase):
    """Test integration with SmartBulletIndex."""

    def test_smart_bullet_index_uses_intent_classifier(self):
        """Test that SmartBulletIndex can use IntentClassifier for auto-routing."""
        from ace import Playbook
        from ace.retrieval import SmartBulletIndex, IntentClassifier

        playbook = Playbook()
        playbook.add_enriched_bullet(
            section="process",
            content="Run tests then deploy",
            task_types=["deployment"],
            retrieval_type="procedural",
        )

        index = SmartBulletIndex(playbook=playbook)
        classifier = IntentClassifier()

        # Auto-detect intent from query
        query = "How do I deploy?"
        intent = classifier.classify(query)

        results = index.retrieve(query=query, intent=intent)

        self.assertGreater(len(results), 0)
        self.assertIn("deploy", results[0].content.lower())


if __name__ == "__main__":
    unittest.main()
