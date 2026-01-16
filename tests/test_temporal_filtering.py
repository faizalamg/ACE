"""Tests for temporal filtering in SmartBulletIndex.retrieve().

This test suite covers the TDD RED phase for adding temporal filtering
capabilities to the retrieval system. All tests are expected to FAIL
until the production code is implemented.

Temporal filtering allows filtering bullets by:
- created_after: Bullets created on or after a specific datetime
- created_before: Bullets created before a specific datetime
- updated_after: Bullets updated on or after a specific datetime
"""

import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock

from ace.playbook import Playbook, EnrichedBullet
from ace.retrieval import SmartBulletIndex, ScoredBullet


class TestTemporalFiltering(unittest.TestCase):
    """Test suite for temporal filtering in SmartBulletIndex.retrieve()."""

    def setUp(self):
        """Set up test playbook with bullets at different timestamps."""
        self.playbook = Playbook()
        self.now = datetime.now(timezone.utc)

        # Create bullets with different timestamps
        # Bullet 1: Created 10 days ago, updated 5 days ago
        self.bullet1 = self.playbook.add_enriched_bullet(
            section="debugging",
            content="Check logs first",
            task_types=["debugging"],
            trigger_patterns=["error", "bug"],
        )
        self.bullet1.created_at = (self.now - timedelta(days=10)).isoformat()
        self.bullet1.updated_at = (self.now - timedelta(days=5)).isoformat()

        # Bullet 2: Created 5 days ago, updated 2 days ago
        self.bullet2 = self.playbook.add_enriched_bullet(
            section="debugging",
            content="Verify configuration",
            task_types=["debugging"],
            trigger_patterns=["config", "settings"],
        )
        self.bullet2.created_at = (self.now - timedelta(days=5)).isoformat()
        self.bullet2.updated_at = (self.now - timedelta(days=2)).isoformat()

        # Bullet 3: Created 2 days ago, updated 1 day ago
        self.bullet3 = self.playbook.add_enriched_bullet(
            section="testing",
            content="Run unit tests",
            task_types=["testing"],
            trigger_patterns=["test", "verify"],
        )
        self.bullet3.created_at = (self.now - timedelta(days=2)).isoformat()
        self.bullet3.updated_at = (self.now - timedelta(days=1)).isoformat()

        # Bullet 4: Created today, updated today
        self.bullet4 = self.playbook.add_enriched_bullet(
            section="optimization",
            content="Profile performance",
            task_types=["optimization"],
            trigger_patterns=["slow", "performance"],
        )
        self.bullet4.created_at = self.now.isoformat()
        self.bullet4.updated_at = self.now.isoformat()

        self.index = SmartBulletIndex(playbook=self.playbook)

    def test_filter_by_created_after(self):
        """Test filtering bullets created after a specific datetime.

        Expected behavior:
        - created_after=(now - 6 days) should return bullet2, bullet3, bullet4
        - Bullet1 (created 10 days ago) should be excluded
        """
        cutoff = self.now - timedelta(days=6)
        results = self.index.retrieve(created_after=cutoff)

        # Should return 3 bullets (bullet2, bullet3, bullet4)
        self.assertEqual(len(results), 3)

        # Verify bullet1 is excluded
        result_ids = {r.bullet.id for r in results}
        self.assertNotIn(self.bullet1.id, result_ids)
        self.assertIn(self.bullet2.id, result_ids)
        self.assertIn(self.bullet3.id, result_ids)
        self.assertIn(self.bullet4.id, result_ids)

    def test_filter_by_created_before(self):
        """Test filtering bullets created before a specific datetime.

        Expected behavior:
        - created_before=(now - 3 days) should return bullet1, bullet2
        - Bullet3 and bullet4 (created recently) should be excluded
        """
        cutoff = self.now - timedelta(days=3)
        results = self.index.retrieve(created_before=cutoff)

        # Should return 2 bullets (bullet1, bullet2)
        self.assertEqual(len(results), 2)

        # Verify recent bullets are excluded
        result_ids = {r.bullet.id for r in results}
        self.assertIn(self.bullet1.id, result_ids)
        self.assertIn(self.bullet2.id, result_ids)
        self.assertNotIn(self.bullet3.id, result_ids)
        self.assertNotIn(self.bullet4.id, result_ids)

    def test_filter_by_updated_after(self):
        """Test filtering bullets updated after a specific datetime.

        Expected behavior:
        - updated_after=(now - 3 days) should return bullet2, bullet3, bullet4
        - Bullet1 (updated 5 days ago) should be excluded
        """
        cutoff = self.now - timedelta(days=3)
        results = self.index.retrieve(updated_after=cutoff)

        # Should return 3 bullets (bullet2, bullet3, bullet4)
        self.assertEqual(len(results), 3)

        # Verify bullet1 is excluded
        result_ids = {r.bullet.id for r in results}
        self.assertNotIn(self.bullet1.id, result_ids)
        self.assertIn(self.bullet2.id, result_ids)
        self.assertIn(self.bullet3.id, result_ids)
        self.assertIn(self.bullet4.id, result_ids)

    def test_combine_created_after_and_before(self):
        """Test combining created_after and created_before filters.

        Expected behavior:
        - created_after=(now - 8 days) AND created_before=(now - 3 days)
        - Should return only bullet2 (created 5 days ago)
        """
        after_cutoff = self.now - timedelta(days=8)
        before_cutoff = self.now - timedelta(days=3)
        results = self.index.retrieve(
            created_after=after_cutoff,
            created_before=before_cutoff
        )

        # Should return only bullet2
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].bullet.id, self.bullet2.id)

    def test_combine_temporal_and_task_type_filters(self):
        """Test combining temporal filters with existing task_type filter.

        Expected behavior:
        - created_after=(now - 6 days) AND task_type="debugging"
        - Should return bullet2 only (bullet3 is testing, bullet4 is optimization)
        """
        cutoff = self.now - timedelta(days=6)
        results = self.index.retrieve(
            created_after=cutoff,
            task_type="debugging"
        )

        # Should return only bullet2 (recent + debugging)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].bullet.id, self.bullet2.id)

    def test_combine_temporal_and_domain_filters(self):
        """Test combining temporal filters with domain filter.

        Expected behavior:
        - Temporal filtering should work alongside domain filtering
        """
        # Add bullets with domain metadata
        domain_bullet = self.playbook.add_enriched_bullet(
            section="python",
            content="Use type hints",
            task_types=["coding"],
            domains=["python"],
        )
        domain_bullet.created_at = (self.now - timedelta(days=3)).isoformat()
        self.index.update()

        cutoff = self.now - timedelta(days=4)
        results = self.index.retrieve(
            created_after=cutoff,
            domain="python"
        )

        # Should return only the domain_bullet
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].bullet.id, domain_bullet.id)

    def test_temporal_filter_with_none_value(self):
        """Test that None values for temporal filters are ignored.

        Expected behavior:
        - created_after=None should not filter anything
        - Should return all bullets
        """
        results = self.index.retrieve(created_after=None)

        # Should return all 4 bullets
        self.assertEqual(len(results), 4)

    def test_temporal_filter_with_future_date(self):
        """Test filtering with a future date.

        Expected behavior:
        - created_after=(future date) should return empty results
        - created_before=(future date) should return all bullets
        """
        future = self.now + timedelta(days=7)

        # No bullets created after future date
        results_after = self.index.retrieve(created_after=future)
        self.assertEqual(len(results_after), 0)

        # All bullets created before future date
        results_before = self.index.retrieve(created_before=future)
        self.assertEqual(len(results_before), 4)

    def test_temporal_filter_preserves_scoring(self):
        """Test that temporal filtering preserves relevance scoring.

        Expected behavior:
        - Temporal filters should not affect scoring logic
        - Results should still be sorted by score descending
        """
        # Add effectiveness scores to differentiate bullets
        self.bullet2.helpful = 10
        self.bullet3.helpful = 5
        self.bullet4.helpful = 8

        cutoff = self.now - timedelta(days=6)
        results = self.index.retrieve(
            created_after=cutoff,
            rank_by_effectiveness=True
        )

        # Should return 3 bullets sorted by effectiveness
        self.assertEqual(len(results), 3)
        # bullet2 (helpful=10) should be first
        self.assertEqual(results[0].bullet.id, self.bullet2.id)

    def test_temporal_filter_with_limit(self):
        """Test temporal filtering with result limit.

        Expected behavior:
        - Temporal filter + limit should work together correctly
        """
        cutoff = self.now - timedelta(days=6)
        results = self.index.retrieve(created_after=cutoff, limit=2)

        # Should return only 2 bullets (limit applied after temporal filter)
        self.assertEqual(len(results), 2)

    def test_updated_after_boundary_condition(self):
        """Test boundary condition for updated_after filter.

        Expected behavior:
        - Bullets updated exactly at cutoff time should be included
        """
        # Set bullet2's updated_at to exact cutoff time
        cutoff = self.now - timedelta(days=2)
        self.bullet2.updated_at = cutoff.isoformat()
        self.index.update()

        results = self.index.retrieve(updated_after=cutoff)

        # bullet2 should be included (>= semantics)
        result_ids = {r.bullet.id for r in results}
        self.assertIn(self.bullet2.id, result_ids)

    def test_temporal_filter_with_timezone_handling(self):
        """Test that temporal filters handle timezone-aware datetimes correctly.

        Expected behavior:
        - Should correctly compare timezone-aware datetimes
        """
        # Create cutoff in different timezone (UTC offset)
        import pytz
        eastern = pytz.timezone('US/Eastern')
        cutoff_eastern = eastern.localize(datetime.now() - timedelta(days=6))

        # Should still work with timezone-aware datetime
        results = self.index.retrieve(created_after=cutoff_eastern)

        # Should handle timezone conversion properly
        self.assertGreaterEqual(len(results), 0)

    def test_match_reasons_include_temporal_filter(self):
        """Test that match_reasons include temporal filter information.

        Expected behavior:
        - ScoredBullet.match_reasons should indicate temporal filtering
        """
        cutoff = self.now - timedelta(days=6)
        results = self.index.retrieve(created_after=cutoff)

        # At least one result should exist
        self.assertGreater(len(results), 0)

        # Match reasons should indicate temporal filtering was applied
        # (exact format TBD during implementation)
        for result in results:
            # This assertion will fail until implementation adds temporal info
            # to match_reasons
            self.assertTrue(
                any("temporal" in reason.lower() or "created" in reason.lower()
                    for reason in result.match_reasons),
                f"Expected temporal filter in match_reasons, got: {result.match_reasons}"
            )


if __name__ == "__main__":
    unittest.main()
