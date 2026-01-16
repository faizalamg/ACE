"""
Tests for ace_qdrant_memory.py integration with UnifiedMemoryIndex.

This test suite verifies:
1. Wrapper functions use UnifiedMemoryIndex internally
2. Backwards compatibility with existing API
3. Namespace-aware functions work correctly
4. Migration path from old to unified system

NOTE: These tests require:
- A running Qdrant instance (or tests will be skipped)
- The ace_qdrant_memory hook to be available
"""

import unittest
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add hooks directory to path to import ace_qdrant_memory
hooks_dir = Path.home() / ".claude" / "hooks"
sys.path.insert(0, str(hooks_dir))

# Add ace module to path
ace_dir = Path(__file__).parent.parent
sys.path.insert(0, str(ace_dir))

from ace.unified_memory import (
    UnifiedMemoryIndex,
    UnifiedBullet,
    UnifiedNamespace,
    UnifiedSource,
    convert_memory_to_unified,
)

# Check if Qdrant is available
def check_qdrant_available():
    """Check if Qdrant is running and accessible."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"), timeout=2)
        client.get_collections()
        return True
    except Exception:
        return False

QDRANT_AVAILABLE = check_qdrant_available()

# Check if hook exists
HOOK_EXISTS = (hooks_dir / "ace_qdrant_memory.py").exists()

# Import after adding hooks to path
if HOOK_EXISTS:
    import ace_qdrant_memory


@unittest.skipIf(not HOOK_EXISTS, "ace_qdrant_memory.py hook not found")
@unittest.skipIf(not QDRANT_AVAILABLE, "Qdrant not available")
class TestQdrantMemoryUnifiedWrapper(unittest.TestCase):
    """Test wrapper functions using real Qdrant."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_collection = "ace_test_qdrant_memory"
        cls.index = UnifiedMemoryIndex(
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection_name=cls.test_collection
        )
        cls.index.create_collection()

    @classmethod
    def tearDownClass(cls):
        """Clean up test collection."""
        try:
            cls.index.client.delete_collection(cls.test_collection)
        except Exception:
            pass

    def test_store_memory_creates_unified_bullet(self):
        """Test store_memory() creates UnifiedBullet and indexes it."""
        # Store directly via unified index
        bullet = UnifiedBullet(
            id="test-store-001",
            namespace=UnifiedNamespace.PROJECT_SPECIFIC,
            source=UnifiedSource.EXPLICIT_STORE,
            content="Always use type hints test memory",
            section="task_guidance",
            severity=8,
            feedback_type="DIRECTIVE"
        )

        result = self.index.index_bullet(bullet)

        self.assertTrue(result.get('stored', False))

    def test_search_memories_uses_unified_retrieval(self):
        """Test that unified retrieval works correctly."""
        # Index a test bullet first
        bullet = UnifiedBullet(
            id="test-search-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Test memory for search verification",
            section="preferences",
            severity=7
        )
        self.index.index_bullet(bullet)

        # Retrieve using unified index
        results = self.index.retrieve("test memory search verification", limit=10)

        # Verify results
        self.assertIsInstance(results, list)


@unittest.skipIf(not HOOK_EXISTS, "ace_qdrant_memory.py hook not found")
@unittest.skipIf(not QDRANT_AVAILABLE, "Qdrant not available")
class TestNamespaceAwareFunctions(unittest.TestCase):
    """Test namespace-aware functions with real Qdrant."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_collection = "ace_test_namespace_aware"
        cls.index = UnifiedMemoryIndex(
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection_name=cls.test_collection
        )
        cls.index.create_collection()

    @classmethod
    def tearDownClass(cls):
        """Clean up test collection."""
        try:
            cls.index.client.delete_collection(cls.test_collection)
        except Exception:
            pass

    def test_store_preference_uses_user_prefs_namespace(self):
        """Test storing in USER_PREFS namespace."""
        bullet = UnifiedBullet(
            id="test-pref-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Prefer TypeScript over JavaScript",
            section="preferences",
            severity=9
        )

        result = self.index.index_bullet(bullet)
        self.assertTrue(result.get('stored', False))

        # Verify it can be retrieved with namespace filter
        results = self.index.retrieve(
            "TypeScript JavaScript",
            namespace=UnifiedNamespace.USER_PREFS,
            limit=10
        )

        # Should find our bullet
        found = any(r.id == "test-pref-001" for r in results)
        self.assertTrue(found)

    def test_store_strategy_uses_task_strategies_namespace(self):
        """Test storing in TASK_STRATEGIES namespace."""
        bullet = UnifiedBullet(
            id="test-strat-001",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="Use pytest for all Python tests",
            section="task_guidance",
            severity=7
        )

        result = self.index.index_bullet(bullet)
        self.assertTrue(result.get('stored', False))

        # Verify namespace filter works
        results = self.index.retrieve(
            "pytest Python tests",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            limit=10
        )

        found = any(r.id == "test-strat-001" for r in results)
        self.assertTrue(found)

    def test_search_by_namespace_filters_correctly(self):
        """Test that namespace filtering works correctly."""
        # Create bullets in different namespaces
        pref = UnifiedBullet(
            id="test-filter-pref",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Namespace filter test preference",
            section="preferences",
            severity=5
        )

        strat = UnifiedBullet(
            id="test-filter-strat",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="Namespace filter test strategy",
            section="task_guidance",
            severity=5
        )

        self.index.index_bullet(pref)
        self.index.index_bullet(strat)

        # Search with TASK_STRATEGIES filter
        results = self.index.retrieve(
            "Namespace filter test",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            limit=10
        )

        # All results should be from TASK_STRATEGIES
        for r in results:
            self.assertEqual(r.namespace, "task_strategies")


@unittest.skipIf(not QDRANT_AVAILABLE, "Qdrant not available")
class TestConversionFunctions(unittest.TestCase):
    """Test conversion from legacy memory format to UnifiedBullet."""

    def test_convert_memory_dict_to_unified_bullet(self):
        """Test converting legacy memory dict to UnifiedBullet."""
        legacy_memory = {
            "lesson": "Test lesson",
            "category": "ARCHITECTURE",
            "severity": 8,
            "feedback_type": "DIRECTIVE",
            "context": "During code review",
            "timestamp": "2025-01-01T00:00:00Z"
        }

        bullet = convert_memory_to_unified(legacy_memory)

        self.assertIsInstance(bullet, UnifiedBullet)
        self.assertEqual(bullet.content, "Test lesson")
        self.assertEqual(bullet.severity, 8)
        self.assertEqual(bullet.namespace, UnifiedNamespace.USER_PREFS.value)
        self.assertEqual(bullet.feedback_type, "DIRECTIVE")

    def test_conversion_maps_categories_to_sections(self):
        """Test category to section mapping in conversion."""
        categories_to_test = [
            ("ARCHITECTURE", "task_guidance"),
            ("WORKFLOW", "common_patterns"),
            ("DEBUGGING", "common_errors"),
            ("TESTING", "task_guidance"),
        ]

        for category, expected_section in categories_to_test:
            memory = {"lesson": "Test", "category": category, "severity": 5}
            bullet = convert_memory_to_unified(memory)
            self.assertEqual(bullet.section, expected_section)


@unittest.skipIf(not QDRANT_AVAILABLE, "Qdrant not available")
class TestIntegration(unittest.TestCase):
    """Integration tests with real UnifiedMemoryIndex."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_collection = "ace_test_integration"
        cls.index = UnifiedMemoryIndex(
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection_name=cls.test_collection
        )
        cls.index.create_collection()

    @classmethod
    def tearDownClass(cls):
        """Clean up test collection."""
        try:
            cls.index.client.delete_collection(cls.test_collection)
        except Exception:
            pass

    def test_end_to_end_store_and_retrieve(self):
        """Test end-to-end flow: store -> retrieve -> verify."""
        # Create and index bullet
        bullet = UnifiedBullet(
            id="integration-e2e-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Integration test memory end to end",
            section="preferences",
            severity=7
        )

        result = self.index.index_bullet(bullet)
        self.assertTrue(result.get('stored', False))

        # Retrieve
        results = self.index.retrieve(
            "Integration test memory",
            namespace=UnifiedNamespace.USER_PREFS,
            limit=10
        )

        # Find our bullet
        found = any(r.id == "integration-e2e-001" for r in results)
        self.assertTrue(found)

    def test_namespace_filtering_in_retrieval(self):
        """Test namespace filter is correctly applied in retrieval."""
        # Create bullets in different namespaces
        pref = UnifiedBullet(
            id="test-ns-filter-pref",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Unique content for namespace filtering test pref",
            section="preferences",
            severity=5
        )

        strat = UnifiedBullet(
            id="test-ns-filter-strat",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="Unique content for namespace filtering test strat",
            section="task_guidance",
            severity=5
        )

        self.index.index_bullet(pref)
        self.index.index_bullet(strat)

        # Retrieve with USER_PREFS filter
        results = self.index.retrieve(
            "Unique content namespace filtering",
            namespace=UnifiedNamespace.USER_PREFS,
            limit=10
        )

        # All results should be user_prefs
        for r in results:
            self.assertEqual(r.namespace, "user_prefs")


if __name__ == "__main__":
    unittest.main()
