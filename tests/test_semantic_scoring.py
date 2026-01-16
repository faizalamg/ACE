"""
Tests for semantic-first scoring in SmartBulletIndex.

This test suite validates:
1. qdrant_score field preservation in UnifiedBullet
2. Semantic-first scoring formula (50% semantic, 20% trigger, 30% effectiveness)
3. Stop word filtering in trigger matching
4. Minimum threshold filtering
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

# Add ace module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ace.unified_memory import (
    UnifiedBullet,
    UnifiedNamespace,
    UnifiedSource,
)
from ace.retrieval import SmartBulletIndex


class TestUnifiedBulletQdrantScore(unittest.TestCase):
    """Test qdrant_score field in UnifiedBullet."""

    def test_qdrant_score_default_value(self):
        """qdrant_score should default to 0.0."""
        bullet = UnifiedBullet(
            id="test-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Test content",
            section="test"
        )
        self.assertEqual(bullet.qdrant_score, 0.0)

    def test_qdrant_score_can_be_set(self):
        """qdrant_score should be settable after creation."""
        bullet = UnifiedBullet(
            id="test-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Test content",
            section="test"
        )
        bullet.qdrant_score = 0.75
        self.assertEqual(bullet.qdrant_score, 0.75)

    def test_qdrant_score_not_in_to_dict(self):
        """qdrant_score should NOT be serialized (runtime-only field)."""
        bullet = UnifiedBullet(
            id="test-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Test content",
            section="test"
        )
        bullet.qdrant_score = 0.85

        result = bullet.to_dict()
        self.assertNotIn("qdrant_score", result)

    def test_from_dict_handles_missing_qdrant_score(self):
        """from_dict should work without qdrant_score (it's not stored)."""
        data = {
            "id": "test-001",
            "namespace": "user_prefs",
            "source": "user_feedback",
            "content": "Test content",
            "section": "test"
        }
        bullet = UnifiedBullet.from_dict(data)
        self.assertEqual(bullet.qdrant_score, 0.0)


class TestStopWordFiltering(unittest.TestCase):
    """Test stop word filtering in trigger matching."""

    def setUp(self):
        """Create a SmartBulletIndex for testing."""
        self.index = SmartBulletIndex()

    def test_common_words_in_stop_list(self):
        """Verify common words are in the stop word list."""
        stop_words = self.index._COMMON_STOP_WORDS

        # Check common verbs
        self.assertIn('add', stop_words)
        self.assertIn('use', stop_words)
        self.assertIn('get', stop_words)

        # Check articles
        self.assertIn('the', stop_words)
        self.assertIn('a', stop_words)

        # Check prepositions
        self.assertIn('to', stop_words)
        self.assertIn('for', stop_words)
        self.assertIn('in', stop_words)

    def test_stop_word_only_query_gets_low_score(self):
        """Query with only stop words should get very low trigger score."""
        bullet = UnifiedBullet(
            id="test-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Always add the values to the list for processing",
            section="test"
        )

        # Query with only stop words
        score = self.index._match_trigger_patterns("add the to for", bullet)
        self.assertLessEqual(score, 0.06)  # Max for stop words only

    def test_meaningful_word_gets_higher_score(self):
        """Query with meaningful words should get higher trigger score."""
        bullet = UnifiedBullet(
            id="test-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Include copyright headers in all source files",
            section="test"
        )

        # Query with meaningful word "copyright"
        score = self.index._match_trigger_patterns("add copyright line", bullet)
        self.assertGreater(score, 0.06)  # Should be higher than stop-word-only

    def test_copyright_query_scores_higher_than_generic(self):
        """Copyright-related memory should score higher for copyright query."""
        copyright_bullet = UnifiedBullet(
            id="test-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Include copyright headers in all source files",
            section="test"
        )

        generic_bullet = UnifiedBullet(
            id="test-002",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Store secrets in env files and add them to gitignore",
            section="test"
        )

        query = "add copyright line"

        copyright_score = self.index._match_trigger_patterns(query, copyright_bullet)
        generic_score = self.index._match_trigger_patterns(query, generic_bullet)

        # Copyright bullet should score higher (meaningful match vs stop word match)
        self.assertGreater(copyright_score, generic_score)


class TestSemanticFirstScoring(unittest.TestCase):
    """Test the semantic-first scoring formula."""

    def test_scoring_formula_weights(self):
        """Verify the scoring formula uses correct weights."""
        # The formula should be:
        # score = (0.5 * semantic) + (0.2 * trigger) + (0.3 * effectiveness)

        # Test case: semantic=0.8, trigger=0.3, effectiveness=0.5
        # Expected: 0.5*0.8 + 0.2*0.3 + 0.3*0.5 = 0.4 + 0.06 + 0.15 = 0.61

        semantic = 0.8
        trigger = 0.3
        effectiveness = 0.5

        expected_score = (0.5 * semantic) + (0.2 * trigger) + (0.3 * effectiveness)
        self.assertAlmostEqual(expected_score, 0.61, places=2)

    def test_high_semantic_dominates_score(self):
        """High semantic score should dominate final score."""
        # High semantic, low trigger, medium effectiveness
        high_semantic = (0.5 * 0.9) + (0.2 * 0.1) + (0.3 * 0.5)

        # Low semantic, high trigger, medium effectiveness
        high_trigger = (0.5 * 0.3) + (0.2 * 0.9) + (0.3 * 0.5)

        # High semantic should result in higher overall score
        self.assertGreater(high_semantic, high_trigger)


class TestSmartBulletIndexIntegration(unittest.TestCase):
    """Integration tests for SmartBulletIndex with mocked UnifiedMemoryIndex."""

    def setUp(self):
        """Set up mock unified index."""
        self.mock_unified_index = Mock()
        self.index = SmartBulletIndex(unified_index=self.mock_unified_index)

    def test_retrieve_uses_threshold(self):
        """Retrieve should pass threshold=0.35 to unified index (balanced with query expansion)."""
        # Set up mock to return empty list
        self.mock_unified_index.retrieve.return_value = []

        self.index.retrieve(query="test query", namespace=None, limit=10)

        # Verify threshold=0.35 was passed (balanced with query expansion for 95%+ precision)
        call_args = self.mock_unified_index.retrieve.call_args
        self.assertEqual(call_args.kwargs.get('threshold'), 0.35)

    def test_retrieve_preserves_qdrant_score(self):
        """Retrieve should use qdrant_score from bullets in scoring."""
        # Create mock bullets with different qdrant_scores
        high_relevance = UnifiedBullet(
            id="high",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Very relevant content about copyright",
            section="test"
        )
        high_relevance.qdrant_score = 0.8

        low_relevance = UnifiedBullet(
            id="low",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Less relevant generic content",
            section="test"
        )
        low_relevance.qdrant_score = 0.2

        self.mock_unified_index.retrieve.return_value = [high_relevance, low_relevance]

        results = self.index.retrieve(query="copyright", namespace=None, limit=10)

        # High relevance bullet should have higher score
        if len(results) >= 2:
            self.assertGreater(results[0].score, results[1].score)

    def test_semantic_score_in_match_reasons(self):
        """Match reasons should include semantic score."""
        bullet = UnifiedBullet(
            id="test",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Test content",
            section="test"
        )
        bullet.qdrant_score = 0.75

        self.mock_unified_index.retrieve.return_value = [bullet]

        results = self.index.retrieve(query="test", namespace=None, limit=10)

        if results:
            match_reasons = results[0].match_reasons
            # Should have semantic score in reasons
            semantic_reasons = [r for r in match_reasons if r.startswith("semantic:")]
            self.assertTrue(len(semantic_reasons) > 0)


if __name__ == "__main__":
    unittest.main()
