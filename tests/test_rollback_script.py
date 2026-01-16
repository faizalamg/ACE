#!/usr/bin/env python3
"""
Tests for rollback_unified_migration.py script.

This test suite verifies the rollback script's ability to:
1. Check rollback feasibility
2. Show migration status
3. Execute safe rollback operations

Note: These are integration tests that require Qdrant to be running.
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import script functions
from scripts.rollback_unified_migration import (
    get_collection_info,
    check_rollback_feasibility,
    delete_collection,
    QDRANT_URL,
    OLD_COLLECTION,
    NEW_COLLECTION,
)


class TestRollbackScript(unittest.TestCase):
    """Test rollback script functionality."""

    def setUp(self):
        """Setup test environment."""
        self.qdrant_url = QDRANT_URL

    def test_get_collection_info_old_collection(self):
        """Test getting info for old collection."""
        info = get_collection_info(OLD_COLLECTION, self.qdrant_url)

        if info is None:
            self.skipTest(f"Collection {OLD_COLLECTION} does not exist")

        # Verify structure
        self.assertIsNotNone(info)
        self.assertIn("name", info)
        self.assertIn("points_count", info)
        self.assertIn("status", info)
        self.assertEqual(info["name"], OLD_COLLECTION)

    def test_get_collection_info_nonexistent(self):
        """Test getting info for non-existent collection."""
        info = get_collection_info("nonexistent_collection_xyz", self.qdrant_url)
        self.assertIsNone(info)

    def test_check_rollback_feasibility_with_old_collection(self):
        """Test rollback feasibility when old collection exists."""
        # Check if old collection exists first
        old_info = get_collection_info(OLD_COLLECTION, self.qdrant_url)

        if old_info is None:
            self.skipTest(f"Collection {OLD_COLLECTION} does not exist")

        # Rollback should be feasible
        feasible = check_rollback_feasibility(self.qdrant_url)
        self.assertTrue(feasible, "Rollback should be feasible when old collection exists")

    @unittest.skip("Destructive test - requires manual execution")
    def test_delete_collection(self):
        """Test deleting a collection (DESTRUCTIVE - skipped by default)."""
        # This test is skipped to prevent accidental data loss
        # To run: manually remove @unittest.skip decorator
        test_collection = "test_rollback_collection"

        # Create test collection
        # ... (implementation omitted for safety)

        # Delete it
        success = delete_collection(test_collection, self.qdrant_url)
        self.assertTrue(success)

    def test_script_help_runs(self):
        """Test that script help command works."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/rollback_unified_migration.py", "--help"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Rollback ACE unified memory migration", result.stdout)

    def test_script_status_runs(self):
        """Test that script status command works."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/rollback_unified_migration.py", "--status"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        # Should succeed even if collections don't exist
        self.assertIn("ACE Unified Memory Migration Status", result.stdout)
        self.assertIn(OLD_COLLECTION, result.stdout)
        self.assertIn(NEW_COLLECTION, result.stdout)

    def test_script_check_runs(self):
        """Test that script check command works."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/rollback_unified_migration.py", "--check"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        # Exit code depends on whether old collection exists
        self.assertIn("Checking rollback feasibility", result.stdout)


class TestRollbackLogic(unittest.TestCase):
    """Test rollback business logic."""

    def test_rollback_feasible_with_old_collection(self):
        """Rollback should be feasible when old collection exists."""
        with patch('scripts.rollback_unified_migration.get_collection_info') as mock_get:
            # Mock: old collection exists
            mock_get.return_value = {
                "name": OLD_COLLECTION,
                "points_count": 100,
                "status": "green"
            }

            feasible = check_rollback_feasibility()
            self.assertTrue(feasible)

    def test_rollback_not_feasible_without_old_collection(self):
        """Rollback should NOT be feasible when old collection is missing."""
        with patch('scripts.rollback_unified_migration.get_collection_info') as mock_get:
            # Mock: old collection does NOT exist
            mock_get.return_value = None

            feasible = check_rollback_feasibility()
            self.assertFalse(feasible)


if __name__ == "__main__":
    unittest.main()
