#!/usr/bin/env python3
"""
Integration tests for complete migration + rollback workflow.

This test simulates a real migration scenario:
1. Check initial state
2. Verify rollback feasibility
3. Simulate migration (creates unified collection)
4. Execute rollback
5. Verify state restored

WARNING: These tests require Qdrant to be running and will create/delete collections.
Use with caution in production environments.
"""

import unittest
import sys
import os
import subprocess

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.rollback_unified_migration import (
    get_collection_info,
    check_rollback_feasibility,
    QDRANT_URL,
    OLD_COLLECTION,
    NEW_COLLECTION,
)


class TestRollbackIntegration(unittest.TestCase):
    """Integration tests for rollback workflow."""

    def setUp(self):
        """Setup test environment."""
        self.qdrant_url = QDRANT_URL
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_rollback_script_cli_status(self):
        """Test that CLI status command produces valid output."""
        result = subprocess.run(
            [sys.executable, "scripts/rollback_unified_migration.py", "--status"],
            capture_output=True,
            text=True,
            cwd=self.project_root
        )

        # Should succeed regardless of collection state
        self.assertIn("ACE Unified Memory Migration Status", result.stdout)
        self.assertIn(OLD_COLLECTION, result.stdout)
        self.assertIn(NEW_COLLECTION, result.stdout)
        self.assertIn("Migration State:", result.stdout)

    def test_rollback_script_cli_check(self):
        """Test that CLI check command produces valid output."""
        result = subprocess.run(
            [sys.executable, "scripts/rollback_unified_migration.py", "--check"],
            capture_output=True,
            text=True,
            cwd=self.project_root
        )

        # Should produce feasibility output
        self.assertIn("Checking rollback feasibility", result.stdout)
        # Exit code depends on whether old collection exists
        # So we just verify command runs without crashing

    def test_rollback_script_help(self):
        """Test that help command works."""
        result = subprocess.run(
            [sys.executable, "scripts/rollback_unified_migration.py", "--help"],
            capture_output=True,
            text=True,
            cwd=self.project_root
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Rollback ACE unified memory migration", result.stdout)
        self.assertIn("--check", result.stdout)
        self.assertIn("--status", result.stdout)
        self.assertIn("--rollback", result.stdout)

    def test_collection_states(self):
        """Test various collection state scenarios."""
        old_info = get_collection_info(OLD_COLLECTION, self.qdrant_url)
        new_info = get_collection_info(NEW_COLLECTION, self.qdrant_url)

        # Document current state for debugging
        if old_info:
            print(f"\nOld collection exists: {old_info['points_count']} points")
        else:
            print("\nOld collection does NOT exist")

        if new_info:
            print(f"New collection exists: {new_info['points_count']} points")
        else:
            print("New collection does NOT exist")

        # Verify get_collection_info returns expected structure
        if old_info:
            self.assertIn("name", old_info)
            self.assertIn("points_count", old_info)
            self.assertIn("status", old_info)
            self.assertEqual(old_info["name"], OLD_COLLECTION)

    def test_feasibility_check_logic(self):
        """Test feasibility check logic with current state."""
        old_exists = get_collection_info(OLD_COLLECTION, self.qdrant_url) is not None

        if old_exists:
            # Should be feasible when old collection exists
            feasible = check_rollback_feasibility(self.qdrant_url)
            self.assertTrue(feasible, "Rollback should be feasible when old collection exists")
        else:
            # Should NOT be feasible when old collection missing
            feasible = check_rollback_feasibility(self.qdrant_url)
            self.assertFalse(feasible, "Rollback should NOT be feasible when old collection missing")

    def test_status_output_format(self):
        """Test that status output contains all required sections."""
        result = subprocess.run(
            [sys.executable, "scripts/rollback_unified_migration.py", "--status"],
            capture_output=True,
            text=True,
            cwd=self.project_root
        )

        # Verify all required sections present
        required_sections = [
            "Old Collection:",
            "New Collection:",
            "Migration State:",
            "Action:",
        ]

        for section in required_sections:
            self.assertIn(section, result.stdout,
                         f"Status output missing required section: {section}")

    @unittest.skip("Destructive test - requires manual execution and confirmation")
    def test_full_migration_rollback_cycle(self):
        """
        DESTRUCTIVE TEST - Test complete migration + rollback cycle.

        This test:
        1. Runs migration to create unified collection
        2. Verifies migration succeeded
        3. Executes rollback
        4. Verifies rollback succeeded

        SKIP BY DEFAULT to prevent accidental data loss.
        """
        # Check pre-conditions
        old_info = get_collection_info(OLD_COLLECTION, self.qdrant_url)
        if not old_info:
            self.skipTest("Old collection doesn't exist - cannot test migration")

        # Run migration
        result = subprocess.run(
            [sys.executable, "scripts/migrate_memories_to_unified.py"],
            capture_output=True,
            text=True,
            cwd=self.project_root
        )
        self.assertEqual(result.returncode, 0, "Migration should succeed")

        # Verify unified collection created
        new_info = get_collection_info(NEW_COLLECTION, self.qdrant_url)
        self.assertIsNotNone(new_info, "Unified collection should exist after migration")

        # Execute rollback (with --no-confirm for automation)
        result = subprocess.run(
            [sys.executable, "scripts/rollback_unified_migration.py",
             "--rollback", "--no-confirm"],
            capture_output=True,
            text=True,
            cwd=self.project_root
        )
        self.assertEqual(result.returncode, 0, "Rollback should succeed")

        # Verify rollback succeeded
        verify_new = get_collection_info(NEW_COLLECTION, self.qdrant_url)
        verify_old = get_collection_info(OLD_COLLECTION, self.qdrant_url)

        self.assertIsNone(verify_new, "Unified collection should be deleted after rollback")
        self.assertIsNotNone(verify_old, "Old collection should still exist after rollback")


class TestRollbackSafety(unittest.TestCase):
    """Test rollback safety mechanisms."""

    def test_no_confirm_flag_prevents_prompt(self):
        """Test that --no-confirm skips confirmation prompt."""
        # This test verifies the flag exists and is accepted
        # Actually executing rollback would be destructive
        result = subprocess.run(
            [sys.executable, "scripts/rollback_unified_migration.py", "--help"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

        self.assertIn("--no-confirm", result.stdout)
        self.assertIn("Skip confirmation prompt", result.stdout)

    def test_mutual_exclusive_actions(self):
        """Test that actions are mutually exclusive."""
        # Cannot use --check and --status together
        result = subprocess.run(
            [sys.executable, "scripts/rollback_unified_migration.py",
             "--check", "--status"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

        # Should fail due to mutually exclusive group
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not allowed with argument", result.stderr)


if __name__ == "__main__":
    unittest.main()
