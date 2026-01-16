"""Tests for confidence decay feature in Bullet class."""

import json
import unittest
from datetime import datetime, timedelta, timezone

from ace.playbook import Bullet, EnrichedBullet, Playbook


class TestBulletConfidenceDecay(unittest.TestCase):
    """Test confidence decay functionality in Bullet class."""

    def test_bullet_has_last_validated_field(self):
        """Test that Bullet has last_validated field with default None."""
        bullet = Bullet(id="b1", section="test", content="test strategy")
        self.assertIsNone(bullet.last_validated)

    def test_effective_score_no_decay_when_not_validated(self):
        """Test that bullets without validation timestamp don't decay."""
        bullet = Bullet(id="b1", section="test", content="test strategy")
        bullet.helpful = 10
        bullet.harmful = 2

        # Should return base score without decay
        self.assertEqual(bullet.effective_score(), 8.0)

    def test_effective_score_no_decay_immediately_after_validation(self):
        """Test that freshly validated bullets have no decay."""
        bullet = Bullet(id="b1", section="test", content="test strategy")
        bullet.helpful = 10
        bullet.harmful = 2
        bullet.validate()

        # Should return full score (minimal decay)
        score = bullet.effective_score()
        self.assertAlmostEqual(score, 8.0, places=2)

    def test_effective_score_decays_over_time(self):
        """Test that scores decay exponentially over time."""
        bullet = Bullet(id="b1", section="test", content="test strategy")
        bullet.helpful = 10
        bullet.harmful = 2

        # Set validation to 4 weeks ago
        four_weeks_ago = datetime.now(timezone.utc) - timedelta(weeks=4)
        bullet.last_validated = four_weeks_ago.isoformat()

        # Calculate expected decay: 8 * (0.95^4) ≈ 6.5
        score = bullet.effective_score(decay_rate=0.95)
        expected = 8.0 * (0.95 ** 4)
        self.assertAlmostEqual(score, expected, places=2)
        self.assertLess(score, 8.0)  # Should be less than base score

    def test_effective_score_with_custom_decay_rate(self):
        """Test custom decay rate parameter."""
        bullet = Bullet(id="b1", section="test", content="test strategy")
        bullet.helpful = 10
        bullet.harmful = 2

        # Set validation to 2 weeks ago
        two_weeks_ago = datetime.now(timezone.utc) - timedelta(weeks=2)
        bullet.last_validated = two_weeks_ago.isoformat()

        # Test faster decay (0.90 = 10% weekly)
        score_fast = bullet.effective_score(decay_rate=0.90)
        expected_fast = 8.0 * (0.90 ** 2)
        self.assertAlmostEqual(score_fast, expected_fast, places=2)

        # Test slower decay (0.98 = 2% weekly)
        score_slow = bullet.effective_score(decay_rate=0.98)
        expected_slow = 8.0 * (0.98 ** 2)
        self.assertAlmostEqual(score_slow, expected_slow, places=2)

        # Faster decay should result in lower score
        self.assertLess(score_fast, score_slow)

    def test_effective_score_with_negative_base_score(self):
        """Test decay works correctly with harmful bullets (negative scores)."""
        bullet = Bullet(id="b1", section="test", content="bad strategy")
        bullet.helpful = 2
        bullet.harmful = 10

        # Set validation to 3 weeks ago
        three_weeks_ago = datetime.now(timezone.utc) - timedelta(weeks=3)
        bullet.last_validated = three_weeks_ago.isoformat()

        # Negative score should also decay (become less negative)
        base_score = -8.0
        score = bullet.effective_score(decay_rate=0.95)
        expected = base_score * (0.95 ** 3)
        self.assertAlmostEqual(score, expected, places=2)

        # Decayed negative score should be closer to zero
        self.assertGreater(score, base_score)
        self.assertLess(score, 0)

    def test_validate_method_sets_timestamp(self):
        """Test that validate() method sets current timestamp."""
        bullet = Bullet(id="b1", section="test", content="test strategy")
        self.assertIsNone(bullet.last_validated)

        before = datetime.now(timezone.utc)
        bullet.validate()
        after = datetime.now(timezone.utc)

        self.assertIsNotNone(bullet.last_validated)

        # Parse and verify timestamp is between before and after
        validated_dt = datetime.fromisoformat(bullet.last_validated)
        self.assertGreaterEqual(validated_dt, before)
        self.assertLessEqual(validated_dt, after)

    def test_validate_method_resets_decay(self):
        """Test that validate() resets decay for stale bullets."""
        bullet = Bullet(id="b1", section="test", content="test strategy")
        bullet.helpful = 10
        bullet.harmful = 2

        # Set stale validation
        old_date = datetime.now(timezone.utc) - timedelta(weeks=10)
        bullet.last_validated = old_date.isoformat()

        # Score should be heavily decayed
        old_score = bullet.effective_score()
        self.assertLess(old_score, 6.0)

        # Validate to reset
        bullet.validate()

        # Score should be back to near-full
        new_score = bullet.effective_score()
        self.assertAlmostEqual(new_score, 8.0, places=1)
        self.assertGreater(new_score, old_score)

    def test_effective_score_handles_invalid_timestamp(self):
        """Test graceful handling of invalid timestamps."""
        bullet = Bullet(id="b1", section="test", content="test strategy")
        bullet.helpful = 10
        bullet.harmful = 2

        # Set invalid timestamp
        bullet.last_validated = "invalid-timestamp"

        # Should return base score without crashing
        score = bullet.effective_score()
        self.assertEqual(score, 8.0)

    def test_effective_score_handles_zero_base_score(self):
        """Test decay with neutral bullets (zero score)."""
        bullet = Bullet(id="b1", section="test", content="neutral strategy")
        bullet.helpful = 5
        bullet.harmful = 5

        # Set validation to past
        old_date = datetime.now(timezone.utc) - timedelta(weeks=5)
        bullet.last_validated = old_date.isoformat()

        # Zero should remain zero after decay
        score = bullet.effective_score()
        self.assertEqual(score, 0.0)

    def test_serialization_preserves_last_validated(self):
        """Test that last_validated is preserved in JSON serialization."""
        playbook = Playbook()
        bullet = playbook.add_bullet(
            section="test",
            content="test strategy"
        )
        bullet.helpful = 10
        bullet.validate()

        # Serialize to JSON
        json_str = playbook.dumps()
        data = json.loads(json_str)

        # Verify last_validated is in JSON
        bullet_data = data["bullets"][bullet.id]
        self.assertIn("last_validated", bullet_data)
        self.assertIsNotNone(bullet_data["last_validated"])

    def test_deserialization_loads_last_validated(self):
        """Test that last_validated is loaded from JSON."""
        # Create playbook with validated bullet
        playbook = Playbook()
        bullet = playbook.add_bullet(section="test", content="test strategy")
        bullet.helpful = 10
        validation_time = datetime.now(timezone.utc).isoformat()
        bullet.last_validated = validation_time

        # Serialize and deserialize
        json_str = playbook.dumps()
        loaded_playbook = Playbook.loads(json_str)

        # Verify last_validated was loaded
        loaded_bullet = loaded_playbook.get_bullet(bullet.id)
        self.assertEqual(loaded_bullet.last_validated, validation_time)

    def test_backward_compatibility_with_old_bullets(self):
        """Test that bullets without last_validated still work."""
        # Simulate old bullet JSON without last_validated field
        old_json = {
            "bullets": {
                "test-00001": {
                    "id": "test-00001",
                    "section": "test",
                    "content": "old bullet",
                    "helpful": 5,
                    "harmful": 1,
                    "neutral": 0,
                    "created_at": "2024-01-01T00:00:00+00:00",
                    "updated_at": "2024-01-01T00:00:00+00:00"
                }
            },
            "sections": {"test": ["test-00001"]},
            "next_id": 1
        }

        playbook = Playbook.from_dict(old_json)
        bullet = playbook.get_bullet("test-00001")

        # Should load successfully with None last_validated
        self.assertIsNone(bullet.last_validated)

        # effective_score should work without decay
        self.assertEqual(bullet.effective_score(), 4.0)

    def test_enriched_bullet_inherits_decay(self):
        """Test that EnrichedBullet inherits decay functionality."""
        playbook = Playbook()
        bullet = playbook.add_enriched_bullet(
            section="test",
            content="enriched strategy",
            task_types=["reasoning"],
            domains=["python"]
        )
        bullet.helpful = 10
        bullet.harmful = 2

        # Set validation to past
        old_date = datetime.now(timezone.utc) - timedelta(weeks=4)
        bullet.last_validated = old_date.isoformat()

        # Should decay like regular bullet
        score = bullet.effective_score()
        expected = 8.0 * (0.95 ** 4)
        self.assertAlmostEqual(score, expected, places=2)

    def test_enriched_bullet_serialization_with_last_validated(self):
        """Test EnrichedBullet serialization preserves last_validated."""
        playbook = Playbook()
        bullet = playbook.add_enriched_bullet(
            section="test",
            content="enriched strategy",
            task_types=["debugging"]
        )
        bullet.helpful = 8
        bullet.validate()

        # Serialize and deserialize
        json_str = playbook.dumps()
        loaded_playbook = Playbook.loads(json_str)

        loaded_bullet = loaded_playbook.get_bullet(bullet.id)
        self.assertIsNotNone(loaded_bullet.last_validated)
        self.assertIsInstance(loaded_bullet, EnrichedBullet)


class TestConfidenceDecayIntegration(unittest.TestCase):
    """Integration tests for confidence decay in real-world scenarios."""

    def test_playbook_with_mixed_validation_ages(self):
        """Test playbook with bullets of varying staleness."""
        playbook = Playbook()

        # Fresh bullet (just validated)
        fresh = playbook.add_bullet(section="strategies", content="fresh strategy")
        fresh.helpful = 10
        fresh.validate()

        # Stale bullet (4 weeks old)
        stale = playbook.add_bullet(section="strategies", content="stale strategy")
        stale.helpful = 10
        old_date = datetime.now(timezone.utc) - timedelta(weeks=4)
        stale.last_validated = old_date.isoformat()

        # Never validated bullet
        never = playbook.add_bullet(section="strategies", content="never validated")
        never.helpful = 10

        # Get effective scores
        fresh_score = fresh.effective_score()
        stale_score = stale.effective_score()
        never_score = never.effective_score()

        # Fresh should be highest, stale lowest
        self.assertGreater(fresh_score, stale_score)
        self.assertEqual(never_score, 10.0)  # No decay without validation

    def test_sorting_bullets_by_effective_score(self):
        """Test sorting bullets by decayed effectiveness."""
        playbook = Playbook()

        bullets_data = [
            ("b1", 10, 2, None),  # helpful=10, harmful=2, weeks_ago=None
            ("b2", 10, 2, 1),     # helpful=10, harmful=2, weeks_ago=1
            ("b3", 10, 2, 4),     # helpful=10, harmful=2, weeks_ago=4
            ("b4", 15, 5, 2),     # helpful=15, harmful=5, weeks_ago=2
        ]

        bullets = []
        for bid, helpful, harmful, weeks_ago in bullets_data:
            bullet = playbook.add_bullet(section="test", content=f"strategy {bid}")
            bullet.helpful = helpful
            bullet.harmful = harmful
            if weeks_ago is not None:
                old_date = datetime.now(timezone.utc) - timedelta(weeks=weeks_ago)
                bullet.last_validated = old_date.isoformat()
            bullets.append(bullet)

        # Sort by effective score (descending)
        sorted_bullets = sorted(
            bullets,
            key=lambda b: b.effective_score(),
            reverse=True
        )

        # b4 should be first (10 base * 0.95^2 ≈ 9.5)
        # b1 should be second (8 base, no decay)
        # b2 should be third (8 base * 0.95^1 ≈ 7.6)
        # b3 should be last (8 base * 0.95^4 ≈ 6.5)
        self.assertEqual(sorted_bullets[0].id, bullets[3].id)  # b4
        self.assertEqual(sorted_bullets[1].id, bullets[0].id)  # b1


if __name__ == "__main__":
    unittest.main()
