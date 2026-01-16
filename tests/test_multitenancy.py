"""
Unit tests for multitenancy module.

These tests cover:
- TenantContext: Thread-local tenant tracking
- TenantManager: Tenant-scoped playbook/Qdrant operations
- TenantIsolationError: Cross-tenant access prevention
- Tenant-scoped collections and retrieval
"""

import unittest
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


@pytest.mark.unit
class TestTenantContext(unittest.TestCase):
    """Test TenantContext thread-local tenant tracking."""

    def test_tenant_context_creation(self):
        """TenantContext stores tenant_id correctly."""
        from ace.multitenancy import TenantContext

        ctx = TenantContext(tenant_id="tenant-001")
        self.assertEqual(ctx.tenant_id, "tenant-001")

    def test_tenant_context_active(self):
        """get_current_tenant returns active tenant within context."""
        from ace.multitenancy import TenantContext, get_current_tenant

        # No tenant active initially
        self.assertIsNone(get_current_tenant())

        # Tenant active within context
        with TenantContext(tenant_id="tenant-002"):
            self.assertEqual(get_current_tenant(), "tenant-002")

        # No tenant after context exits
        self.assertIsNone(get_current_tenant())

    def test_tenant_context_manager(self):
        """TenantContext works as context manager with proper cleanup."""
        from ace.multitenancy import TenantContext, get_current_tenant

        self.assertIsNone(get_current_tenant())

        with TenantContext(tenant_id="tenant-003"):
            self.assertEqual(get_current_tenant(), "tenant-003")

        # Context properly cleaned up
        self.assertIsNone(get_current_tenant())

    def test_tenant_context_nesting(self):
        """Nested TenantContext works correctly (inner overrides outer)."""
        from ace.multitenancy import TenantContext, get_current_tenant

        with TenantContext(tenant_id="outer-tenant"):
            self.assertEqual(get_current_tenant(), "outer-tenant")

            with TenantContext(tenant_id="inner-tenant"):
                self.assertEqual(get_current_tenant(), "inner-tenant")

            # Outer context restored after inner exits
            self.assertEqual(get_current_tenant(), "outer-tenant")

        # All contexts cleaned up
        self.assertIsNone(get_current_tenant())


@pytest.mark.unit
class TestTenantPlaybookIsolation(unittest.TestCase):
    """Test tenant-scoped playbook isolation."""

    def setUp(self):
        """Create mock TenantManager for each test."""
        from ace.playbook import Playbook

        self.playbook_tenant_a = Playbook()
        self.playbook_tenant_a.add_bullet(
            section="strategy", content="Tenant A strategy", metadata={"helpful": 7}
        )

        self.playbook_tenant_b = Playbook()
        self.playbook_tenant_b.add_bullet(
            section="strategy", content="Tenant B strategy", metadata={"helpful": 8}
        )

    def test_tenant_playbook_isolation(self):
        """Different tenants cannot see each other's playbooks."""
        from ace.multitenancy import TenantContext, TenantManager

        manager = TenantManager()

        # Save playbook for tenant A
        with TenantContext(tenant_id="tenant-a"):
            manager.save_playbook(self.playbook_tenant_a, "test_playbook")

        # Save playbook for tenant B
        with TenantContext(tenant_id="tenant-b"):
            manager.save_playbook(self.playbook_tenant_b, "test_playbook")

        # Tenant A can only see their own playbook
        with TenantContext(tenant_id="tenant-a"):
            loaded = manager.load_playbook("test_playbook")
            self.assertEqual(len(loaded.bullets()), 1)
            self.assertIn("Tenant A strategy", loaded.bullets()[0].content)
            self.assertNotIn("Tenant B strategy", loaded.bullets()[0].content)

        # Tenant B can only see their own playbook
        with TenantContext(tenant_id="tenant-b"):
            loaded = manager.load_playbook("test_playbook")
            self.assertEqual(len(loaded.bullets()), 1)
            self.assertIn("Tenant B strategy", loaded.bullets()[0].content)
            self.assertNotIn("Tenant A strategy", loaded.bullets()[0].content)

    def test_tenant_playbook_creation(self):
        """Playbook saved in tenant namespace."""
        from ace.multitenancy import TenantContext, TenantManager

        manager = TenantManager()

        with TenantContext(tenant_id="tenant-create"):
            manager.save_playbook(self.playbook_tenant_a, "new_playbook")

            # Verify saved in tenant-scoped path
            expected_path = (
                Path(manager.storage_dir) / "tenant-create" / "new_playbook.json"
            )
            self.assertTrue(expected_path.exists())

    def test_tenant_playbook_retrieval(self):
        """Only retrieves tenant's own playbooks."""
        from ace.multitenancy import TenantContext, TenantManager

        manager = TenantManager()

        # Save playbook for tenant X
        with TenantContext(tenant_id="tenant-x"):
            manager.save_playbook(self.playbook_tenant_a, "playbook_x")

        # Tenant Y cannot retrieve tenant X's playbook
        with TenantContext(tenant_id="tenant-y"):
            with self.assertRaises(FileNotFoundError):
                manager.load_playbook("playbook_x")

    def test_cross_tenant_access_denied(self):
        """Accessing other tenant's data raises TenantIsolationError."""
        from ace.multitenancy import (
            TenantContext,
            TenantManager,
            TenantIsolationError,
        )

        manager = TenantManager()

        with TenantContext(tenant_id="tenant-secure"):
            manager.save_playbook(self.playbook_tenant_a, "secure_playbook")

        # Attempt to access with different tenant context
        with TenantContext(tenant_id="tenant-attacker"):
            with self.assertRaises(TenantIsolationError):
                manager.load_playbook("secure_playbook", tenant_id="tenant-secure")


@pytest.mark.unit
class TestTenantScopedCollection(unittest.TestCase):
    """Test tenant-scoped Qdrant collection naming and isolation."""

    @patch("ace.multitenancy.QdrantClient")
    def test_tenant_scoped_collection_name(self, mock_qdrant):
        """Qdrant collection named {tenant_id}_bullets."""
        from ace.multitenancy import TenantContext, TenantManager

        manager = TenantManager()
        mock_client = MagicMock()
        manager.qdrant_client = mock_client

        with TenantContext(tenant_id="tenant-qdrant"):
            manager.get_qdrant_collection()

        # Verify collection name is tenant-scoped
        expected_collection = "tenant-qdrant_bullets"
        mock_client.get_collection.assert_called_with(expected_collection)

    @patch("ace.multitenancy.QdrantClient")
    def test_tenant_scoped_index_isolation(self, mock_qdrant):
        """Index operations isolated per tenant."""
        from ace.multitenancy import TenantContext, TenantManager
        from ace.playbook import Bullet

        manager = TenantManager()
        mock_client = MagicMock()
        manager.qdrant_client = mock_client

        bullet = Bullet(id="test-bullet-1", section="test", content="Test bullet")

        # Index for tenant 1
        with TenantContext(tenant_id="tenant-1"):
            manager.index_bullet(bullet)

        # Index for tenant 2
        with TenantContext(tenant_id="tenant-2"):
            manager.index_bullet(bullet)

        # Verify different collections used
        calls = mock_client.upsert.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertIn("tenant-1_bullets", str(calls[0]))
        self.assertIn("tenant-2_bullets", str(calls[1]))

    @patch("ace.multitenancy.QdrantClient")
    def test_tenant_scoped_retrieval(self, mock_qdrant):
        """Retrieval scoped to tenant collection."""
        from ace.multitenancy import TenantContext, TenantManager

        manager = TenantManager()
        mock_client = MagicMock()
        manager.qdrant_client = mock_client

        mock_client.search.return_value = []

        with TenantContext(tenant_id="tenant-search"):
            manager.search_bullets("test query")

        # Verify search only in tenant collection
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        self.assertEqual(call_args[1]["collection_name"], "tenant-search_bullets")


@pytest.mark.unit
class TestCrossTenantPrevention(unittest.TestCase):
    """Test cross-tenant access prevention mechanisms."""

    def test_cross_tenant_query_blocked(self):
        """Query with wrong tenant context blocked."""
        from ace.multitenancy import (
            TenantContext,
            TenantManager,
            TenantIsolationError,
        )

        manager = TenantManager()

        with TenantContext(tenant_id="tenant-owner"):
            manager.save_playbook(Mock(), "protected_playbook")

        # Attempt query from different tenant
        with TenantContext(tenant_id="tenant-intruder"):
            with self.assertRaises(TenantIsolationError):
                manager.load_playbook("protected_playbook", tenant_id="tenant-owner")

    def test_cross_tenant_write_blocked(self):
        """Write to other tenant's namespace blocked."""
        from ace.multitenancy import (
            TenantContext,
            TenantManager,
            TenantIsolationError,
        )

        manager = TenantManager()

        with TenantContext(tenant_id="tenant-writer"):
            # Attempt to write to different tenant's namespace
            with self.assertRaises(TenantIsolationError):
                manager.save_playbook(
                    Mock(), "malicious_playbook", tenant_id="tenant-victim"
                )

    def test_tenant_id_required(self):
        """Operations fail without tenant context."""
        from ace.multitenancy import TenantManager, TenantIsolationError

        manager = TenantManager()

        # No tenant context active
        with self.assertRaises(TenantIsolationError):
            manager.save_playbook(Mock(), "no_tenant_playbook")

        with self.assertRaises(TenantIsolationError):
            manager.load_playbook("some_playbook")

    def test_tenant_validation(self):
        """Invalid tenant IDs rejected."""
        from ace.multitenancy import TenantContext

        # Empty tenant ID
        with self.assertRaises(ValueError):
            TenantContext(tenant_id="")

        # None tenant ID
        with self.assertRaises(ValueError):
            TenantContext(tenant_id=None)

        # Invalid characters (e.g., path traversal)
        with self.assertRaises(ValueError):
            TenantContext(tenant_id="../../../etc/passwd")

        # Valid alphanumeric with hyphens/underscores
        try:
            TenantContext(tenant_id="tenant-123_abc")
        except ValueError:
            self.fail("Valid tenant ID rejected")


if __name__ == "__main__":
    unittest.main()
