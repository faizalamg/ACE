"""
Tests for session outcome tracking infrastructure.

Tests SessionOutcomeTracker class for tracking bullet effectiveness
per session type with TTL-based cleanup.
"""

import unittest
import tempfile
import json
import os
from datetime import datetime, timedelta
from unittest.mock import patch
from ace.session_tracking import SessionOutcomeTracker, SessionOutcome


class TestSessionOutcomeTracker(unittest.TestCase):
    """Test cases for SessionOutcomeTracker class."""

    def test_initialization_default_ttl(self):
        """Test SessionOutcomeTracker initializes with default 24h TTL."""
        tracker = SessionOutcomeTracker()

        # Should have empty outcomes dictionary
        self.assertEqual(len(tracker._outcomes), 0)

        # Should have 24h TTL by default
        self.assertEqual(tracker._ttl, timedelta(hours=24))

    def test_initialization_custom_ttl(self):
        """Test SessionOutcomeTracker initializes with custom TTL."""
        tracker = SessionOutcomeTracker(ttl_hours=48)

        # Should have specified TTL
        self.assertEqual(tracker._ttl, timedelta(hours=48))

    def test_track_creates_new_session_outcome(self):
        """Test track() creates new SessionOutcome for first use."""
        tracker = SessionOutcomeTracker()

        # Track a "worked" outcome
        tracker.track("browser_automation", "bullet_001", "worked")

        # Should create entry with session_type:bullet_id key
        key = "browser_automation:bullet_001"
        self.assertIn(key, tracker._outcomes)

        outcome = tracker._outcomes[key]
        self.assertEqual(outcome.uses, 1)
        self.assertEqual(outcome.worked, 1)
        self.assertEqual(outcome.failed, 0)

    def test_track_increments_worked_count(self):
        """Test track() correctly increments worked counter."""
        tracker = SessionOutcomeTracker()

        # Track multiple "worked" outcomes
        tracker.track("api_calls", "bullet_002", "worked")
        tracker.track("api_calls", "bullet_002", "worked")
        tracker.track("api_calls", "bullet_002", "worked")

        key = "api_calls:bullet_002"
        outcome = tracker._outcomes[key]
        self.assertEqual(outcome.uses, 3)
        self.assertEqual(outcome.worked, 3)
        self.assertEqual(outcome.failed, 0)

    def test_track_increments_failed_count(self):
        """Test track() correctly increments failed counter."""
        tracker = SessionOutcomeTracker()

        # Track "failed" outcome
        tracker.track("data_processing", "bullet_003", "failed")

        key = "data_processing:bullet_003"
        outcome = tracker._outcomes[key]
        self.assertEqual(outcome.uses, 1)
        self.assertEqual(outcome.worked, 0)
        self.assertEqual(outcome.failed, 1)

    def test_track_mixed_outcomes(self):
        """Test track() handles mixed worked/failed outcomes."""
        tracker = SessionOutcomeTracker()

        # Track mixed outcomes
        tracker.track("testing", "bullet_004", "worked")
        tracker.track("testing", "bullet_004", "failed")
        tracker.track("testing", "bullet_004", "worked")
        tracker.track("testing", "bullet_004", "failed")
        tracker.track("testing", "bullet_004", "worked")

        key = "testing:bullet_004"
        outcome = tracker._outcomes[key]
        self.assertEqual(outcome.uses, 5)
        self.assertEqual(outcome.worked, 3)
        self.assertEqual(outcome.failed, 2)

    def test_track_updates_timestamp(self):
        """Test track() updates last_updated timestamp."""
        tracker = SessionOutcomeTracker()

        before = datetime.now()
        tracker.track("session_type", "bullet_005", "worked")
        after = datetime.now()

        key = "session_type:bullet_005"
        outcome = tracker._outcomes[key]

        # Timestamp should be between before and after
        self.assertGreaterEqual(outcome.last_updated, before)
        self.assertLessEqual(outcome.last_updated, after)

    def test_track_separate_session_types(self):
        """Test track() maintains separate counters per session type."""
        tracker = SessionOutcomeTracker()

        # Same bullet_id in different session types
        tracker.track("browser", "bullet_006", "worked")
        tracker.track("api", "bullet_006", "failed")

        browser_key = "browser:bullet_006"
        api_key = "api:bullet_006"

        # Should have separate entries
        self.assertIn(browser_key, tracker._outcomes)
        self.assertIn(api_key, tracker._outcomes)

        # Browser session
        browser_outcome = tracker._outcomes[browser_key]
        self.assertEqual(browser_outcome.worked, 1)
        self.assertEqual(browser_outcome.failed, 0)

        # API session
        api_outcome = tracker._outcomes[api_key]
        self.assertEqual(api_outcome.worked, 0)
        self.assertEqual(api_outcome.failed, 1)

    def test_get_session_effectiveness_no_data_returns_default(self):
        """Test get_session_effectiveness returns default when no data exists."""
        tracker = SessionOutcomeTracker()

        # No data tracked - should return default
        effectiveness = tracker.get_session_effectiveness("unknown", "bullet_999")
        self.assertEqual(effectiveness, 0.5)

        # Custom default
        effectiveness = tracker.get_session_effectiveness("unknown", "bullet_999", default=0.7)
        self.assertEqual(effectiveness, 0.7)

    def test_get_session_effectiveness_perfect_success(self):
        """Test get_session_effectiveness calculates 100% success rate."""
        tracker = SessionOutcomeTracker()

        # All worked outcomes
        tracker.track("perfect", "bullet_100", "worked")
        tracker.track("perfect", "bullet_100", "worked")
        tracker.track("perfect", "bullet_100", "worked")

        effectiveness = tracker.get_session_effectiveness("perfect", "bullet_100")
        self.assertEqual(effectiveness, 1.0)

    def test_get_session_effectiveness_total_failure(self):
        """Test get_session_effectiveness calculates 0% success rate."""
        tracker = SessionOutcomeTracker()

        # All failed outcomes
        tracker.track("failure", "bullet_101", "failed")
        tracker.track("failure", "bullet_101", "failed")
        tracker.track("failure", "bullet_101", "failed")

        effectiveness = tracker.get_session_effectiveness("failure", "bullet_101")
        self.assertEqual(effectiveness, 0.0)

    def test_get_session_effectiveness_partial_success(self):
        """Test get_session_effectiveness calculates partial success rate."""
        tracker = SessionOutcomeTracker()

        # 3 worked, 2 failed = 60% effectiveness
        tracker.track("partial", "bullet_102", "worked")
        tracker.track("partial", "bullet_102", "failed")
        tracker.track("partial", "bullet_102", "worked")
        tracker.track("partial", "bullet_102", "failed")
        tracker.track("partial", "bullet_102", "worked")

        effectiveness = tracker.get_session_effectiveness("partial", "bullet_102")
        self.assertAlmostEqual(effectiveness, 0.6, places=2)

    def test_get_session_effectiveness_only_uses_no_outcomes(self):
        """Test get_session_effectiveness returns default when only uses tracked but no outcomes."""
        tracker = SessionOutcomeTracker()

        # Manually create outcome with uses but no worked/failed
        # (This shouldn't happen in practice, but tests edge case)
        key = "edge:bullet_103"
        tracker._outcomes[key] = SessionOutcome()
        tracker._outcomes[key].uses = 5

        # Should return default since worked + failed = 0
        effectiveness = tracker.get_session_effectiveness("edge", "bullet_103")
        self.assertEqual(effectiveness, 0.5)

    def test_get_session_effectiveness_different_sessions(self):
        """Test get_session_effectiveness returns different values per session type."""
        tracker = SessionOutcomeTracker()

        # Same bullet_id, different session types with different effectiveness
        # Browser session: 3/4 = 75%
        tracker.track("browser", "bullet_104", "worked")
        tracker.track("browser", "bullet_104", "worked")
        tracker.track("browser", "bullet_104", "worked")
        tracker.track("browser", "bullet_104", "failed")

        # API session: 1/3 = 33%
        tracker.track("api", "bullet_104", "worked")
        tracker.track("api", "bullet_104", "failed")
        tracker.track("api", "bullet_104", "failed")

        browser_effectiveness = tracker.get_session_effectiveness("browser", "bullet_104")
        api_effectiveness = tracker.get_session_effectiveness("api", "bullet_104")

        self.assertAlmostEqual(browser_effectiveness, 0.75, places=2)
        self.assertAlmostEqual(api_effectiveness, 0.33, places=2)

    def test_cleanup_expired_removes_old_entries(self):
        """Test cleanup_expired removes entries older than TTL."""
        tracker = SessionOutcomeTracker(ttl_hours=1)

        # Create entries with different timestamps
        now = datetime.now()

        # Fresh entry (just created)
        tracker.track("fresh", "bullet_200", "worked")

        # Old entry (2 hours ago - beyond 1h TTL)
        with patch("ace.session_tracking.datetime") as mock_datetime:
            mock_datetime.now.return_value = now - timedelta(hours=2)
            tracker.track("old", "bullet_201", "worked")

        # Verify both entries exist before cleanup
        self.assertEqual(len(tracker._outcomes), 2)

        # Run cleanup
        tracker.cleanup_expired()

        # Should only have fresh entry
        self.assertEqual(len(tracker._outcomes), 1)
        self.assertIn("fresh:bullet_200", tracker._outcomes)
        self.assertNotIn("old:bullet_201", tracker._outcomes)

    def test_cleanup_expired_keeps_recent_entries(self):
        """Test cleanup_expired keeps entries within TTL."""
        tracker = SessionOutcomeTracker(ttl_hours=24)

        # Create entries within last 24 hours
        tracker.track("recent1", "bullet_202", "worked")
        tracker.track("recent2", "bullet_203", "failed")
        tracker.track("recent3", "bullet_204", "worked")

        # All should exist
        self.assertEqual(len(tracker._outcomes), 3)

        # Run cleanup
        tracker.cleanup_expired()

        # All should still exist (within TTL)
        self.assertEqual(len(tracker._outcomes), 3)

    def test_cleanup_expired_with_mixed_ages(self):
        """Test cleanup_expired handles mix of old and new entries."""
        tracker = SessionOutcomeTracker(ttl_hours=2)

        now = datetime.now()

        # Create entries at different times
        # Fresh: 0 hours ago
        tracker.track("fresh", "bullet_205", "worked")

        # Recent: 1 hour ago (within TTL)
        key_recent = "recent:bullet_206"
        tracker._outcomes[key_recent] = SessionOutcome()
        tracker._outcomes[key_recent].worked = 1
        tracker._outcomes[key_recent].uses = 1
        tracker._outcomes[key_recent].last_updated = now - timedelta(hours=1)

        # Old: 3 hours ago (beyond TTL)
        key_old1 = "old1:bullet_207"
        tracker._outcomes[key_old1] = SessionOutcome()
        tracker._outcomes[key_old1].worked = 1
        tracker._outcomes[key_old1].uses = 1
        tracker._outcomes[key_old1].last_updated = now - timedelta(hours=3)

        # Very old: 10 hours ago (beyond TTL)
        key_old2 = "old2:bullet_208"
        tracker._outcomes[key_old2] = SessionOutcome()
        tracker._outcomes[key_old2].failed = 1
        tracker._outcomes[key_old2].uses = 1
        tracker._outcomes[key_old2].last_updated = now - timedelta(hours=10)

        # Should have 4 entries before cleanup
        self.assertEqual(len(tracker._outcomes), 4)

        # Run cleanup
        tracker.cleanup_expired()

        # Should only have 2 entries (fresh + recent)
        self.assertEqual(len(tracker._outcomes), 2)
        self.assertIn("fresh:bullet_205", tracker._outcomes)
        self.assertIn(key_recent, tracker._outcomes)
        self.assertNotIn(key_old1, tracker._outcomes)
        self.assertNotIn(key_old2, tracker._outcomes)

    def test_cleanup_expired_empty_tracker(self):
        """Test cleanup_expired handles empty tracker gracefully."""
        tracker = SessionOutcomeTracker()

        # Should not raise error on empty tracker
        tracker.cleanup_expired()

        self.assertEqual(len(tracker._outcomes), 0)

    def test_automatic_cleanup_on_track(self):
        """Test that tracking automatically triggers cleanup if needed."""
        tracker = SessionOutcomeTracker(ttl_hours=1)

        now = datetime.now()

        # Create old entry manually
        key_old = "old:bullet_209"
        tracker._outcomes[key_old] = SessionOutcome()
        tracker._outcomes[key_old].worked = 1
        tracker._outcomes[key_old].uses = 1
        tracker._outcomes[key_old].last_updated = now - timedelta(hours=2)

        # Verify old entry exists
        self.assertIn(key_old, tracker._outcomes)

        # Track new entry (should trigger cleanup)
        tracker.track("new", "bullet_210", "worked")

        # Old entry should be cleaned up, new entry should exist
        self.assertNotIn(key_old, tracker._outcomes)
        self.assertIn("new:bullet_210", tracker._outcomes)

    def test_session_tracker_save_load(self):
        """Test SessionOutcomeTracker save_to_file and load_from_file."""
        tracker = SessionOutcomeTracker(ttl_hours=48)

        # Track various outcomes
        tracker.track("browser", "bullet_001", "worked")
        tracker.track("browser", "bullet_001", "worked")
        tracker.track("browser", "bullet_001", "failed")
        tracker.track("api", "bullet_002", "worked")
        tracker.track("api", "bullet_003", "failed")

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            # Save tracker data
            tracker.save_to_file(temp_path)

            # Verify file was created
            self.assertTrue(os.path.exists(temp_path))

            # Load from file
            loaded_tracker = SessionOutcomeTracker.load_from_file(temp_path)

            # Verify TTL was preserved
            self.assertEqual(loaded_tracker._ttl, timedelta(hours=48))

            # Verify all outcomes were preserved
            self.assertEqual(len(loaded_tracker._outcomes), 3)

            # Verify browser:bullet_001 data
            browser_key = "browser:bullet_001"
            self.assertIn(browser_key, loaded_tracker._outcomes)
            browser_outcome = loaded_tracker._outcomes[browser_key]
            self.assertEqual(browser_outcome.uses, 3)
            self.assertEqual(browser_outcome.worked, 2)
            self.assertEqual(browser_outcome.failed, 1)

            # Verify api:bullet_002 data
            api_key = "api:bullet_002"
            self.assertIn(api_key, loaded_tracker._outcomes)
            api_outcome = loaded_tracker._outcomes[api_key]
            self.assertEqual(api_outcome.uses, 1)
            self.assertEqual(api_outcome.worked, 1)
            self.assertEqual(api_outcome.failed, 0)

            # Verify api:bullet_003 data
            api_key2 = "api:bullet_003"
            self.assertIn(api_key2, loaded_tracker._outcomes)
            api_outcome2 = loaded_tracker._outcomes[api_key2]
            self.assertEqual(api_outcome2.uses, 1)
            self.assertEqual(api_outcome2.worked, 0)
            self.assertEqual(api_outcome2.failed, 1)

            # Verify timestamps were preserved (within 1 second tolerance)
            original_timestamp = tracker._outcomes[browser_key].last_updated
            loaded_timestamp = loaded_tracker._outcomes[browser_key].last_updated
            time_diff = abs((original_timestamp - loaded_timestamp).total_seconds())
            self.assertLess(time_diff, 1.0)

        finally:
            # Cleanup temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_to_file_creates_valid_json(self):
        """Test save_to_file creates valid JSON with expected structure."""
        tracker = SessionOutcomeTracker(ttl_hours=12)
        tracker.track("test", "bullet_100", "worked")

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            tracker.save_to_file(temp_path)

            # Read and parse JSON
            with open(temp_path, 'r') as f:
                data = json.load(f)

            # Verify structure
            self.assertIn("ttl_hours", data)
            self.assertEqual(data["ttl_hours"], 12)

            self.assertIn("outcomes", data)
            self.assertIsInstance(data["outcomes"], dict)

            # Verify outcome entry
            key = "test:bullet_100"
            self.assertIn(key, data["outcomes"])
            outcome_data = data["outcomes"][key]

            self.assertIn("uses", outcome_data)
            self.assertIn("worked", outcome_data)
            self.assertIn("failed", outcome_data)
            self.assertIn("last_updated", outcome_data)

            self.assertEqual(outcome_data["uses"], 1)
            self.assertEqual(outcome_data["worked"], 1)
            self.assertEqual(outcome_data["failed"], 0)

            # Verify timestamp is ISO format
            datetime.fromisoformat(outcome_data["last_updated"])

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_from_file_empty_outcomes(self):
        """Test load_from_file handles tracker with no outcomes."""
        tracker = SessionOutcomeTracker(ttl_hours=6)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            # Save empty tracker
            tracker.save_to_file(temp_path)

            # Load it back
            loaded_tracker = SessionOutcomeTracker.load_from_file(temp_path)

            # Verify empty state
            self.assertEqual(len(loaded_tracker._outcomes), 0)
            self.assertEqual(loaded_tracker._ttl, timedelta(hours=6))

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_incremental_persistence_pattern(self):
        """Test incremental save/load pattern for append-only workflow."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            # Session 1: Track some data and save
            tracker1 = SessionOutcomeTracker(ttl_hours=24)
            tracker1.track("session1", "bullet_A", "worked")
            tracker1.track("session1", "bullet_B", "failed")
            tracker1.save_to_file(temp_path)

            # Session 2: Load, add more data, save
            tracker2 = SessionOutcomeTracker.load_from_file(temp_path)
            tracker2.track("session2", "bullet_C", "worked")
            tracker2.track("session1", "bullet_A", "worked")  # Update existing
            tracker2.save_to_file(temp_path)

            # Session 3: Load and verify all data persisted
            tracker3 = SessionOutcomeTracker.load_from_file(temp_path)

            self.assertEqual(len(tracker3._outcomes), 3)

            # Verify session1:bullet_A has 2 worked outcomes
            key_a = "session1:bullet_A"
            self.assertIn(key_a, tracker3._outcomes)
            self.assertEqual(tracker3._outcomes[key_a].uses, 2)
            self.assertEqual(tracker3._outcomes[key_a].worked, 2)

            # Verify session1:bullet_B has 1 failed outcome
            key_b = "session1:bullet_B"
            self.assertIn(key_b, tracker3._outcomes)
            self.assertEqual(tracker3._outcomes[key_b].uses, 1)
            self.assertEqual(tracker3._outcomes[key_b].failed, 1)

            # Verify session2:bullet_C has 1 worked outcome
            key_c = "session2:bullet_C"
            self.assertIn(key_c, tracker3._outcomes)
            self.assertEqual(tracker3._outcomes[key_c].uses, 1)
            self.assertEqual(tracker3._outcomes[key_c].worked, 1)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
