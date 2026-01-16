"""Tests for retrieval filter logic - trigger vs effectiveness priority.

TDD: Phase 1B - Fix min_effectiveness filter to respect strong trigger matches.

These tests validate that bullets with strong trigger matches (>0.3) should NOT
be excluded by the effectiveness filter, even if their effectiveness is low.
"""

import unittest
from typing import List

import pytest
from ace import Playbook
from ace.retrieval import SmartBulletIndex


@pytest.mark.unit
class TestTriggerOverridesEffectiveness(unittest.TestCase):
    """Test that strong trigger matches override effectiveness filtering."""

    def test_strong_trigger_overrides_effectiveness_filter(self):
        """
        FAILING TEST (TDD Phase 1: RED)

        BUG: Current code at retrieval.py:202-203 excludes bullets based on
        effectiveness without considering trigger score.

        EXPECTED: Bullet with high trigger match should be included even if
        effectiveness is low, because the trigger match is strong (>0.3).

        ACTUAL (before fix): Bullet is excluded by effectiveness filter.

        NOTE: _match_trigger_patterns gives 0.15 per pattern match, max 0.45.
        Need 3+ patterns to match to get >0.3 trigger_score.
        """
        # Setup: Create playbook with enriched bullet with multiple patterns
        playbook = Playbook()
        bullet = playbook.add_enriched_bullet(
            section="error_handling",
            content="When seeing 'NoneType' error, check for uninitialized variables",
            trigger_patterns=["NoneType", "error", "uninitialized", "AttributeError"]
        )

        # Simulate low effectiveness (e.g., 1 helpful, 9 harmful)
        bullet.helpful = 1
        bullet.harmful = 9
        bullet.neutral = 0

        # Create index and retrieve with effectiveness filter
        index = SmartBulletIndex(playbook=playbook)

        # Query matches 3 patterns: "NoneType" + "error" + "uninitialized"
        # Trigger score = 3 * 0.15 = 0.45 (above 0.3 threshold)
        results = index.retrieve(
            query="I'm getting a NoneType error with uninitialized variable",
            min_effectiveness=0.5,  # Effectiveness filter that would exclude bullet
            limit=5
        )

        # ASSERTION: Bullet should be included due to strong trigger match
        self.assertEqual(len(results), 1,
                        "Strong trigger match (3+ patterns = 0.45 score) should override effectiveness filter")
        self.assertEqual(results[0].bullet.id, bullet.id)
        self.assertTrue(any("trigger" in reason for reason in results[0].match_reasons),
                       "Should have trigger match reason")

    def test_weak_trigger_respects_effectiveness_filter(self):
        """
        Test that weak trigger matches ARE excluded by effectiveness filter.

        This ensures we don't break the effectiveness filter entirely - only
        strong trigger matches (>0.3) should override it.
        """
        playbook = Playbook()
        bullet = playbook.add_enriched_bullet(
            section="error_handling",
            content="Check for null pointers in C code",
            trigger_patterns=["null pointer", "segfault"]
        )

        bullet.helpful = 1
        bullet.harmful = 9
        bullet.neutral = 0

        index = SmartBulletIndex(playbook=playbook)

        # Weak trigger match should NOT override effectiveness filter
        # Query has weak semantic match but no strong trigger pattern hit
        results = index.retrieve(
            query="My code crashed unexpectedly",  # Vague match, not strong trigger
            min_effectiveness=0.5,
            limit=5
        )

        # Should be excluded (weak/no trigger + low effectiveness)
        self.assertEqual(len(results), 0,
                        "Weak trigger match should respect effectiveness filter")

    def test_edge_case_trigger_threshold_boundary(self):
        """
        Test behavior at the trigger threshold boundary (0.3).

        This test uses two bullets with different trigger patterns to verify
        the threshold logic works correctly.

        NOTE: 0.15 per match, so need 3 matches for 0.45 score > 0.3 threshold.
        """
        playbook = Playbook()

        # Bullet 1: Weak trigger (1 match = 0.15 < 0.3 threshold)
        bullet_weak = playbook.add_enriched_bullet(
            section="weak_trigger",
            content="This bullet has a weak trigger pattern match",
            trigger_patterns=["weak"]  # Only 1 pattern will match
        )
        bullet_weak.helpful = 1
        bullet_weak.harmful = 9

        # Bullet 2: Strong trigger (3+ matches = 0.45 > 0.3 threshold)
        bullet_strong = playbook.add_enriched_bullet(
            section="strong_trigger",
            content="This bullet has strong trigger patterns",
            trigger_patterns=["STRONGTRIGGER", "exact", "match", "keyword"]
        )
        bullet_strong.helpful = 1
        bullet_strong.harmful = 9

        index = SmartBulletIndex(playbook=playbook)

        # Query matching 3+ patterns in bullet_strong
        results = index.retrieve(
            query="I need STRONGTRIGGER exact match keyword help",
            min_effectiveness=0.5,
            limit=10
        )

        # Only the strong trigger bullet should be included
        self.assertGreater(len(results), 0,
                          "Strong trigger match (3+ patterns) should override effectiveness")
        self.assertEqual(results[0].bullet.id, bullet_strong.id)


@pytest.mark.unit
class TestConfigurableTriggerThreshold(unittest.TestCase):
    """Test that trigger override threshold is configurable."""

    def test_custom_trigger_threshold(self):
        """
        Test that trigger_override_threshold parameter works.

        This allows tuning the threshold for different use cases:
        - Strict filtering: higher threshold (e.g., 0.5)
        - Loose filtering: lower threshold (e.g., 0.2)

        NOTE: 0.15 per match, so need 3 matches for 0.45 score.
        """
        playbook = Playbook()

        # Bullet with moderate trigger match (3 patterns = 0.45 score)
        bullet = playbook.add_enriched_bullet(
            section="test",
            content="Test bullet with moderate trigger",
            trigger_patterns=["MODERATE", "TRIGGER", "issue"]
        )
        bullet.helpful = 1
        bullet.harmful = 9

        index = SmartBulletIndex(playbook=playbook)

        # With default threshold (0.3), bullet should be included
        # Query matches all 3 patterns = 0.45 score > 0.3 threshold
        results_default = index.retrieve(
            query="I have a MODERATE TRIGGER issue with my code",
            min_effectiveness=0.5,
            limit=5
        )
        self.assertGreater(len(results_default), 0,
                          "Default threshold (0.3) should include bullet with 0.45 trigger score")

        # With very strict threshold (0.5), bullet should be excluded
        # 0.45 score < 0.5 threshold
        results_strict = index.retrieve(
            query="I have a MODERATE TRIGGER issue with my code",
            min_effectiveness=0.5,
            trigger_override_threshold=0.5,  # Higher than bullet's 0.45 score
            limit=5
        )
        self.assertEqual(len(results_strict), 0,
                        "Strict threshold (0.5) should exclude bullet with 0.45 score")


if __name__ == "__main__":
    unittest.main()
