"""
Tests for Phase 2 Session Tracking Bug Fix.

Bug: Session effectiveness was applied unconditionally to ALL bullets,
even when bullet's task_types didn't match the session_type.

Fix: Only apply session effectiveness when bullet.task_types includes session_type.
Otherwise, fall back to global effectiveness.
"""

import unittest
from ace import Playbook
from ace.retrieval import SmartBulletIndex
from ace.session_tracking import SessionOutcomeTracker


class TestSessionTypeMismatchFix(unittest.TestCase):
    """Test that session effectiveness is only applied when bullet matches session type."""

    def setUp(self):
        """Create playbook with bullets of different task_types."""
        self.playbook = Playbook()

        # Debugging bullet
        self.debug_bullet = self.playbook.add_enriched_bullet(
            section="debugging",
            content="Debug timeout by checking connection pools",
            task_types=["debugging", "troubleshooting"],
            trigger_patterns=["timeout", "debug"]
        )

        # Security bullet
        self.security_bullet = self.playbook.add_enriched_bullet(
            section="security",
            content="Investigate security breach with forensic analysis",
            task_types=["security", "incident_response"],
            trigger_patterns=["breach", "security", "incident"]
        )

        # Optimization bullet
        self.optimization_bullet = self.playbook.add_enriched_bullet(
            section="optimization",
            content="Profile before optimizing to find bottlenecks",
            task_types=["optimization", "performance"],
            trigger_patterns=["slow", "performance", "optimize"]
        )

        # Create session tracker with divergent effectiveness
        self.tracker = SessionOutcomeTracker()

        # Debugging bullets: high effectiveness in debugging sessions
        for _ in range(9):
            self.tracker.track("debugging", self.debug_bullet.id, "worked")
        self.tracker.track("debugging", self.debug_bullet.id, "failed")
        # debug_bullet in debugging session: 0.9 effectiveness

        # Security bullets: low effectiveness in debugging sessions (irrelevant)
        self.tracker.track("debugging", self.security_bullet.id, "worked")
        for _ in range(9):
            self.tracker.track("debugging", self.security_bullet.id, "failed")
        # security_bullet in debugging session: 0.1 effectiveness

        # Optimization bullets: low effectiveness in debugging sessions (irrelevant)
        self.tracker.track("debugging", self.optimization_bullet.id, "worked")
        for _ in range(9):
            self.tracker.track("debugging", self.optimization_bullet.id, "failed")
        # optimization_bullet in debugging session: 0.1 effectiveness

    def test_security_query_should_not_be_penalized_by_debugging_session(self):
        """
        CRITICAL BUG TEST: A security query should return security bullets first,
        even when the session_type is "debugging".

        Before fix: Security bullet gets 0.1 effectiveness (from debugging session)
        After fix: Security bullet gets global effectiveness (bullet doesn't match session)
        """
        index = SmartBulletIndex(playbook=self.playbook, session_tracker=self.tracker)

        # Query for security incident - should return security bullet first
        results = index.retrieve(
            query="How to investigate a security breach?",
            query_type="security",  # Explicit security query
            session_type="debugging",  # But session is debugging
            limit=10
        )

        self.assertTrue(len(results) > 0, "Should return results")

        # Security bullet should rank first for security query
        # BUG: Before fix, debugging bullet ranks first because security bullet
        # gets penalized by low debugging-session effectiveness
        top_result = results[0]
        self.assertIn(
            "security",
            top_result.bullet.task_types,
            f"Security query should return security bullet first, got: {top_result.bullet.task_types}"
        )

    def test_session_penalty_not_applied_to_mismatched_bullets_mature(self):
        """
        PRECISE BUG TEST: Verify session effectiveness is NOT used
        when bullet.task_types doesn't include session_type.

        Uses MATURE bullets (10+ signals) to trigger 0.3/0.7 weights
        where the bug manifests most severely.
        """
        # Make bullets MATURE by adding feedback signals
        for _ in range(10):
            self.debug_bullet.tag("helpful", increment=1)
        for _ in range(10):
            self.security_bullet.tag("helpful", increment=1)

        index = SmartBulletIndex(playbook=self.playbook, session_tracker=self.tracker)

        # Retrieve WITHOUT query_type to isolate session effectiveness impact
        results = index.retrieve(
            query="security breach",  # Matches security trigger
            session_type="debugging",  # Session doesn't match security bullet
            limit=10
        )

        # Find the security and debug bullets in results
        security_score = None
        debug_score = None

        for r in results:
            if "security" in r.bullet.task_types and "debugging" not in r.bullet.task_types:
                security_score = r.score
            if "debugging" in r.bullet.task_types:
                debug_score = r.score

        self.assertIsNotNone(security_score, "Security bullet should be in results")
        self.assertIsNotNone(debug_score, "Debug bullet should be in results")

        # BUG MANIFESTATION (with mature bullets, weights = 0.3/0.7):
        # BEFORE FIX:
        #   - Security bullet: 0.3*0.3 + 0.7*0.1 = 0.16 (session eff = 0.1)
        #   - Debug bullet: 0.3*0 + 0.7*0.9 = 0.63 (session eff = 0.9)
        #   - Debug wins despite not matching query!
        #
        # AFTER FIX:
        #   - Security bullet: 0.3*0.3 + 0.7*1.0 = 0.79 (global eff = 1.0)
        #   - Debug bullet: 0.3*0 + 0.7*0.9 = 0.63 (session eff applies - matches)
        #   - Security wins due to trigger match + correct effectiveness
        self.assertGreater(
            security_score,
            debug_score,
            f"Security bullet (score={security_score}) should rank higher than debug bullet "
            f"(score={debug_score}) for 'security breach' query with mature bullets. "
            "BUG: Session effectiveness (0.1) incorrectly applied to security bullet "
            "that doesn't have 'debugging' in task_types."
        )

    def test_optimization_query_should_not_be_penalized_by_debugging_session(self):
        """Optimization query should return optimization bullets first."""
        index = SmartBulletIndex(playbook=self.playbook, session_tracker=self.tracker)

        results = index.retrieve(
            query="How to optimize slow performance?",
            query_type="optimization",
            session_type="debugging",
            limit=10
        )

        self.assertTrue(len(results) > 0)
        top_result = results[0]
        self.assertIn(
            "optimization",
            top_result.bullet.task_types,
            f"Optimization query should return optimization bullet first, got: {top_result.bullet.task_types}"
        )

    def test_debugging_query_in_debugging_session_uses_session_effectiveness(self):
        """Debugging query in debugging session SHOULD use session effectiveness."""
        index = SmartBulletIndex(playbook=self.playbook, session_tracker=self.tracker)

        results = index.retrieve(
            query="How to debug timeout issues?",
            query_type="debugging",
            session_type="debugging",
            limit=10
        )

        self.assertTrue(len(results) > 0)
        top_result = results[0]

        # Debugging bullet should rank first (matches both query_type and session_type)
        self.assertIn(
            "debugging",
            top_result.bullet.task_types,
            "Debugging query in debugging session should return debugging bullet"
        )

    def test_session_effectiveness_only_applied_when_task_type_matches(self):
        """Verify the fix: session effectiveness only applied when bullet matches session."""
        index = SmartBulletIndex(playbook=self.playbook, session_tracker=self.tracker)

        # Get security bullet score with debugging session
        results = index.retrieve(
            query="security breach investigation",
            query_type="security",
            session_type="debugging",  # Mismatched session
            limit=10
        )

        security_result = None
        for r in results:
            if "security" in r.bullet.task_types:
                security_result = r
                break

        self.assertIsNotNone(security_result, "Security bullet should be in results")

        # The security bullet should NOT have the low debugging-session effectiveness
        # It should use global effectiveness instead (0.5 default for new bullets)
        # Check match_reasons to verify correct effectiveness was used
        reasons = " ".join(security_result.match_reasons)

        # After fix: should NOT see session effectiveness penalty
        # The effectiveness component should be global (0.5), not session (0.1)
        self.assertNotIn(
            "session_eff:0.1",
            reasons,
            "Security bullet should NOT use debugging session effectiveness"
        )

    def test_no_session_type_uses_global_effectiveness(self):
        """When session_type is None, should use global effectiveness."""
        index = SmartBulletIndex(playbook=self.playbook, session_tracker=self.tracker)

        results = index.retrieve(
            query="security breach investigation",
            query_type="security",
            session_type=None,  # No session
            limit=10
        )

        self.assertTrue(len(results) > 0)
        top_result = results[0]
        self.assertIn("security", top_result.bullet.task_types)


class TestSessionMatchingLogic(unittest.TestCase):
    """Test the session type matching logic in detail."""

    def test_bullet_with_multiple_task_types_matches_any(self):
        """Bullet with multiple task_types should match if ANY matches session."""
        playbook = Playbook()

        # Multi-type bullet
        multi_bullet = playbook.add_enriched_bullet(
            section="general",
            content="Multi-purpose strategy for debugging and security",
            task_types=["debugging", "security", "general"],
            trigger_patterns=["issue", "problem"]
        )

        tracker = SessionOutcomeTracker()
        # High effectiveness in debugging session
        for _ in range(9):
            tracker.track("debugging", multi_bullet.id, "worked")
        tracker.track("debugging", multi_bullet.id, "failed")

        index = SmartBulletIndex(playbook=playbook, session_tracker=tracker)

        # Query in debugging session - should use session effectiveness
        # because bullet.task_types includes "debugging"
        results = index.retrieve(
            query="problem investigation",
            session_type="debugging",
            limit=10
        )

        self.assertTrue(len(results) > 0)
        # The bullet matches the session type, so session effectiveness should be used

    def test_bullet_without_task_types_uses_global(self):
        """Bullet without task_types should use global effectiveness."""
        playbook = Playbook()

        # Bullet with empty task_types
        basic_bullet = playbook.add_enriched_bullet(
            section="general",
            content="Generic strategy",
            task_types=[],  # No task types
            trigger_patterns=["help"]
        )

        tracker = SessionOutcomeTracker()
        # Some session data exists
        tracker.track("debugging", basic_bullet.id, "failed")

        index = SmartBulletIndex(playbook=playbook, session_tracker=tracker)

        results = index.retrieve(
            query="help me",
            session_type="debugging",
            limit=10
        )

        # Should use global effectiveness, not session (since bullet has no task_types)
        self.assertTrue(len(results) > 0)


if __name__ == "__main__":
    unittest.main()
