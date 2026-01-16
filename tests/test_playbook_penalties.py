"""
Unit tests for asymmetric penalty weights in Bullet.tag() method.

Tests Phase 1C implementation from TUNINGPROJECT.md:
- Harmful tags penalized 2x (increment=2)
- Helpful tags rewarded 1x (increment=1)
- Neutral tags use default 1x (increment=1)
- Explicit increment parameter overrides default weights
"""

import unittest
from ace.playbook import Bullet, PENALTY_WEIGHTS


class TestAsymmetricPenalties(unittest.TestCase):
    """Test asymmetric penalty system for bullet tagging."""

    def setUp(self):
        """Create a fresh bullet for each test."""
        self.bullet = Bullet(
            id="test-001",
            section="test",
            content="Test bullet content",
            helpful=0,
            harmful=0,
            neutral=0
        )

    def test_harmful_penalized_more_than_helpful_rewarded(self):
        """CRITICAL: Harmful tags should increment by 2, helpful by 1."""
        # Helpful tag should increment by 1 (default reward)
        self.bullet.tag("helpful")
        self.assertEqual(self.bullet.helpful, 1, "Helpful should increment by 1")

        # Harmful tag should increment by 2 (asymmetric penalty)
        self.bullet.tag("harmful")
        self.assertEqual(self.bullet.harmful, 2, "Harmful should increment by 2 (asymmetric penalty)")

        # Neutral tag should increment by 1 (default)
        self.bullet.tag("neutral")
        self.assertEqual(self.bullet.neutral, 1, "Neutral should increment by 1")

    def test_penalty_weights_constant_exists(self):
        """Verify PENALTY_WEIGHTS constant is defined correctly."""
        self.assertIsNotNone(PENALTY_WEIGHTS, "PENALTY_WEIGHTS must be defined")
        self.assertEqual(PENALTY_WEIGHTS["helpful"], 1, "Helpful weight should be 1")
        self.assertEqual(PENALTY_WEIGHTS["harmful"], 2, "Harmful weight should be 2")
        self.assertEqual(PENALTY_WEIGHTS["neutral"], 1, "Neutral weight should be 1")

    def test_custom_weight_override(self):
        """Explicit increment parameter should override default weights."""
        # Override helpful with custom increment
        self.bullet.tag("helpful", increment=5)
        self.assertEqual(self.bullet.helpful, 5, "Explicit increment=5 should override default")

        # Override harmful with custom increment
        self.bullet.tag("harmful", increment=10)
        self.assertEqual(self.bullet.harmful, 10, "Explicit increment=10 should override default")

        # Override neutral with custom increment
        self.bullet.tag("neutral", increment=3)
        self.assertEqual(self.bullet.neutral, 3, "Explicit increment=3 should override default")

    def test_multiple_harmful_tags_accumulate_correctly(self):
        """Multiple harmful tags should accumulate with 2x penalty."""
        self.bullet.tag("harmful")  # +2 = 2
        self.bullet.tag("harmful")  # +2 = 4
        self.bullet.tag("harmful")  # +2 = 6

        self.assertEqual(self.bullet.harmful, 6, "Three harmful tags should total 6 (3 x 2)")

    def test_multiple_helpful_tags_accumulate_correctly(self):
        """Multiple helpful tags should accumulate with 1x reward."""
        self.bullet.tag("helpful")  # +1 = 1
        self.bullet.tag("helpful")  # +1 = 2
        self.bullet.tag("helpful")  # +1 = 3

        self.assertEqual(self.bullet.helpful, 3, "Three helpful tags should total 3 (3 x 1)")

    def test_mixed_tags_with_asymmetric_weights(self):
        """Verify harmful tags accumulate faster than helpful tags."""
        # Simulate: 2 helpful, 2 harmful
        self.bullet.tag("helpful")  # +1 = 1
        self.bullet.tag("harmful")  # +2 = 2
        self.bullet.tag("helpful")  # +1 = 2
        self.bullet.tag("harmful")  # +2 = 4

        self.assertEqual(self.bullet.helpful, 2, "Two helpful tags should total 2")
        self.assertEqual(self.bullet.harmful, 4, "Two harmful tags should total 4 (asymmetric)")
        self.assertGreater(self.bullet.harmful, self.bullet.helpful,
                          "Harmful should accumulate faster than helpful")

    def test_zero_increment_allowed(self):
        """Explicit increment=0 should be allowed (for testing/debugging)."""
        self.bullet.tag("harmful", increment=0)
        self.assertEqual(self.bullet.harmful, 0, "Zero increment should keep count at 0")

    def test_negative_increment_allowed(self):
        """Negative increments should be allowed (for corrections/rollbacks)."""
        self.bullet.harmful = 10  # Set initial value
        self.bullet.tag("harmful", increment=-3)
        self.assertEqual(self.bullet.harmful, 7, "Negative increment should subtract from count")

    def test_updated_at_timestamp_changes(self):
        """Tagging should update the updated_at timestamp."""
        original_timestamp = self.bullet.updated_at

        # Small delay to ensure timestamp differs
        import time
        time.sleep(0.01)

        self.bullet.tag("harmful")

        self.assertNotEqual(self.bullet.updated_at, original_timestamp,
                           "Timestamp should update after tagging")


if __name__ == "__main__":
    unittest.main()
