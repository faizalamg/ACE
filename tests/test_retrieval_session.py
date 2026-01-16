"""
Tests for Session-Aware Retrieval Integration (Phase 2B).

Tests the integration of SessionOutcomeTracker with SmartBulletIndex
to enable session-specific bullet effectiveness scoring.
"""

import unittest
from ace.playbook import Playbook
from ace.retrieval import SmartBulletIndex
from ace.session_tracking import SessionOutcomeTracker


class TestSessionAwareRetrieval(unittest.TestCase):
    """Test suite for session-aware retrieval functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.playbook = Playbook()

        # Add enriched bullets for testing
        self.playbook.add_enriched_bullet(
            section="browser_automation",
            content="Always wait for page load before interacting",
            task_types=["browser_automation"],
            trigger_patterns=["page", "load", "wait"],
        )
        self.playbook.add_enriched_bullet(
            section="browser_automation",
            content="Use explicit waits instead of sleep",
            task_types=["browser_automation"],
            trigger_patterns=["wait", "sleep"],
        )
        self.playbook.add_enriched_bullet(
            section="api_calls",
            content="Retry with exponential backoff on 5xx errors",
            task_types=["api_calls"],
            trigger_patterns=["retry", "error", "5xx"],
        )

        # Create session tracker
        self.tracker = SessionOutcomeTracker(ttl_hours=24)

    def test_retrieve_uses_session_effectiveness(self):
        """
        Test that retrieve() uses session-specific effectiveness
        when session_type is provided.

        EXPECTED BEHAVIOR:
        - When session_tracker and session_type are provided,
          retrieval should prefer bullets with high session effectiveness
        - Bullets with better session performance should rank higher
        """
        # Get bullet IDs from playbook
        bullets = list(self.playbook.bullets())
        bullet_1_id = bullets[0].id  # "Always wait for page load"
        bullet_2_id = bullets[1].id  # "Use explicit waits"

        # Simulate session outcomes:
        # Bullet 1: High success in browser_automation (8/10 = 0.8)
        for _ in range(8):
            self.tracker.track("browser_automation", bullet_1_id, "worked")
        for _ in range(2):
            self.tracker.track("browser_automation", bullet_1_id, "failed")

        # Bullet 2: Low success in browser_automation (2/10 = 0.2)
        for _ in range(2):
            self.tracker.track("browser_automation", bullet_2_id, "worked")
        for _ in range(8):
            self.tracker.track("browser_automation", bullet_2_id, "failed")

        # Create index with session tracker
        index = SmartBulletIndex(playbook=self.playbook, session_tracker=self.tracker)

        # Retrieve with session type
        results = index.retrieve(
            task_type="browser_automation",
            session_type="browser_automation",  # This parameter should trigger session-aware scoring
            limit=5
        )

        # Bullet 1 should rank higher due to better session effectiveness
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].bullet.id, bullet_1_id,
                        "Bullet with higher session effectiveness should rank first")
        self.assertEqual(results[1].bullet.id, bullet_2_id,
                        "Bullet with lower session effectiveness should rank second")

    def test_fallback_to_global_when_no_session_data(self):
        """
        Test that retrieve() falls back to global effectiveness
        when session_type is provided but no session data exists.

        EXPECTED BEHAVIOR:
        - If session_tracker is provided but bullet has no session-specific data,
          should use global effectiveness (bullet.helpful / total)
        - Should not crash or error
        """
        # Create index with session tracker (but no tracked data yet)
        index = SmartBulletIndex(playbook=self.playbook, session_tracker=self.tracker)

        # Retrieve with session type (no session data exists)
        results = index.retrieve(
            task_type="browser_automation",
            session_type="browser_automation",
            limit=5
        )

        # Should return results without error
        self.assertGreaterEqual(len(results), 1, "Should return results even without session data")

        # All results should have default scores (no session data to differentiate)
        for result in results:
            self.assertIsNotNone(result.score, "Bullets should have valid scores")

    def test_retrieve_without_session_type_uses_global(self):
        """
        Test that retrieve() uses global effectiveness when session_type is not provided.

        EXPECTED BEHAVIOR:
        - When session_type is None (default), ignore session tracker
        - Use global bullet effectiveness (backward compatibility)
        """
        bullets = list(self.playbook.bullets())
        bullet_1_id = bullets[0].id

        # Add session-specific data (should be ignored)
        self.tracker.track("browser_automation", bullet_1_id, "worked")

        # Create index with session tracker
        index = SmartBulletIndex(playbook=self.playbook, session_tracker=self.tracker)

        # Retrieve WITHOUT session_type (should use global effectiveness)
        results = index.retrieve(
            task_type="browser_automation",
            # session_type NOT provided
            limit=5
        )

        # Should return results using global effectiveness
        self.assertGreaterEqual(len(results), 1, "Should return results using global effectiveness")

    def test_index_without_session_tracker_works_normally(self):
        """
        Test that SmartBulletIndex works normally when session_tracker is not provided.

        EXPECTED BEHAVIOR:
        - Backward compatibility: Index should work without session_tracker
        - retrieve() should ignore session_type parameter if no tracker
        """
        # Create index WITHOUT session tracker
        index = SmartBulletIndex(playbook=self.playbook)  # No session_tracker parameter

        # Retrieve with session_type (should be ignored gracefully)
        results = index.retrieve(
            task_type="browser_automation",
            session_type="browser_automation",  # Should be ignored (no tracker)
            limit=5
        )

        # Should work normally
        self.assertGreaterEqual(len(results), 1, "Should work without session tracker")


if __name__ == "__main__":
    unittest.main()
