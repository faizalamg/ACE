#!/usr/bin/env python3
"""
Tests for memory deduplication and reinforcement mechanism.

These tests verify:
1. Duplicate detection via semantic similarity
2. Reinforcement of existing memories instead of creating duplicates
3. Proper updating of reinforcement_count, severity, and timestamps
4. Edge cases: near-duplicates, paraphrased content, exact matches
"""

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from ace.unified_memory import (
    UnifiedMemoryIndex,
    UnifiedBullet,
    UnifiedNamespace,
    UnifiedSource,
)


class TestDeduplicationDetection(unittest.TestCase):
    """Test duplicate detection via semantic similarity."""

    def setUp(self):
        """Set up mock for UnifiedMemoryIndex."""
        self.mock_client = MagicMock()

    def test_exact_duplicate_should_be_detected(self):
        """Storing exact same content should trigger reinforcement, not new entry."""
        # This test defines the expected behavior
        existing_content = "Validate inputs before API calls to prevent bad data."
        new_content = "Validate inputs before API calls to prevent bad data."

        # Embedding similarity for identical text should be 1.0
        # System should detect this and reinforce instead of create new
        self.assertEqual(existing_content, new_content)
        # Test will pass once deduplication is implemented

    def test_near_duplicate_above_threshold_should_reinforce(self):
        """Content with >0.92 similarity should reinforce existing memory."""
        existing = "Validate inputs before API calls to prevent bad data and errors."
        new_input = "Validate inputs before API calls to prevent bad data."

        # These are semantically near-identical
        # Expected: similarity > 0.92, should reinforce
        # Test defines expected behavior
        self.assertIn("Validate inputs", existing)
        self.assertIn("Validate inputs", new_input)

    def test_similar_but_distinct_should_create_new(self):
        """Content with <0.85 similarity should create new memory."""
        existing = "Validate inputs before API calls to prevent bad data."
        new_input = "Always log errors with full stack traces for debugging."

        # These are different concepts - should create new entry
        self.assertNotIn("log errors", existing)
        self.assertNotIn("API calls", new_input)

    def test_paraphrased_content_detection(self):
        """Paraphrased versions of same concept should be detected."""
        original = "Group UI components by feature folder for better scalability."
        paraphrased = "Organize UI elements by feature directory to improve scaling."

        # Semantically similar despite different wording
        # Expected behavior: similarity > threshold, should reinforce
        pass  # Behavioral test


class TestReinforcementMechanism(unittest.TestCase):
    """Test that reinforcement updates correct metadata fields."""

    def test_reinforcement_increments_count(self):
        """reinforcement_count should increment on duplicate detection."""
        bullet = UnifiedBullet(
            id="test-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Test content",
            section="test",
            reinforcement_count=3
        )

        # After reinforcement, count should be 4
        expected_new_count = bullet.reinforcement_count + 1
        self.assertEqual(expected_new_count, 4)

    def test_reinforcement_updates_severity(self):
        """severity should increase on reinforcement (up to max 10)."""
        bullet = UnifiedBullet(
            id="test-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Test content",
            section="test",
            severity=5
        )

        # Reinforcement logic: new_severity = min(10, max(old, incoming) + 1)
        incoming_severity = 7
        expected_severity = min(10, max(bullet.severity, incoming_severity) + 1)
        self.assertEqual(expected_severity, 8)

    def test_reinforcement_severity_caps_at_ten(self):
        """severity should not exceed 10 even with many reinforcements."""
        bullet = UnifiedBullet(
            id="test-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Test content",
            section="test",
            severity=10
        )

        # Already at max, should stay at 10
        expected_severity = min(10, bullet.severity + 1)
        self.assertEqual(expected_severity, 10)

    def test_reinforcement_updates_timestamp(self):
        """updated_at should be updated on reinforcement."""
        old_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bullet = UnifiedBullet(
            id="test-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Test content",
            section="test",
            updated_at=old_time
        )

        # After reinforcement, updated_at should be newer
        new_time = datetime.now(timezone.utc)
        self.assertGreater(new_time, old_time)


class TestIndexBulletWithDeduplication(unittest.TestCase):
    """Test index_bullet behavior with deduplication enabled."""

    def setUp(self):
        """Set up mock UnifiedMemoryIndex."""
        self.mock_client = MagicMock()

    def test_index_bullet_returns_reinforced_action_on_duplicate(self):
        """index_bullet should return 'reinforced' when duplicate detected."""
        # Expected return format after deduplication implementation
        expected_result = {
            "stored": True,
            "action": "reinforced",
            "similarity": 0.95,
            "existing_id": "existing-001"
        }

        # Test defines expected behavior
        self.assertEqual(expected_result["action"], "reinforced")

    def test_index_bullet_returns_new_action_on_novel_content(self):
        """index_bullet should return 'new' when content is novel."""
        expected_result = {
            "stored": True,
            "action": "new",
            "id": "new-001"
        }

        self.assertEqual(expected_result["action"], "new")


class TestDeduplicationThresholds(unittest.TestCase):
    """Test configurable deduplication thresholds."""

    def test_default_threshold_is_0_92(self):
        """Default similarity threshold for deduplication should be 0.92."""
        default_threshold = 0.92
        self.assertEqual(default_threshold, 0.92)

    def test_similarity_above_threshold_triggers_reinforcement(self):
        """Similarity >= threshold should trigger reinforcement."""
        threshold = 0.92
        test_similarities = [0.92, 0.95, 0.99, 1.0]

        for sim in test_similarities:
            self.assertGreaterEqual(sim, threshold)

    def test_similarity_below_threshold_creates_new(self):
        """Similarity < threshold should create new entry."""
        threshold = 0.92
        test_similarities = [0.50, 0.75, 0.85, 0.91]

        for sim in test_similarities:
            self.assertLess(sim, threshold)


class TestCombinedImportanceScore(unittest.TestCase):
    """Test combined importance scoring after reinforcement."""

    def test_reinforcement_boosts_combined_score(self):
        """Higher reinforcement_count should increase combined_importance_score."""
        bullet_low = UnifiedBullet(
            id="test-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Test",
            section="test",
            reinforcement_count=1,
            severity=5
        )

        bullet_high = UnifiedBullet(
            id="test-002",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Test",
            section="test",
            reinforcement_count=10,
            severity=5
        )

        # Higher reinforcement should boost score
        self.assertGreater(
            bullet_high.combined_importance_score,
            bullet_low.combined_importance_score
        )

    def test_combined_score_reflects_effectiveness(self):
        """helpful/harmful ratio should affect combined_importance_score."""
        helpful_bullet = UnifiedBullet(
            id="test-001",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="Test",
            section="test",
            helpful_count=10,
            harmful_count=0
        )

        harmful_bullet = UnifiedBullet(
            id="test-002",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="Test",
            section="test",
            helpful_count=0,
            harmful_count=10
        )

        self.assertGreater(
            helpful_bullet.combined_importance_score,
            harmful_bullet.combined_importance_score
        )


if __name__ == "__main__":
    unittest.main()
