#!/usr/bin/env python3
"""
Tests for ace_search_qdrant.py - Unified Memory Search Hook

Tests cover:
1. Search across all namespaces (search_all)
2. Search specific namespaces (preferences, strategies, project)
3. Namespace filtering with multiple namespaces
4. Output formatting using format_unified_context
5. Backwards compatibility
6. Error handling

NOTE: These tests require:
- The ace_search_qdrant.py hook file to exist
- A running Qdrant instance (or tests will be skipped)
"""

import unittest
from pathlib import Path
import sys
import os

# Add hooks directory to path
hooks_dir = Path.home() / ".claude" / "hooks"
sys.path.insert(0, str(hooks_dir))

# Add ace module to path
ace_dir = Path(__file__).parent.parent
sys.path.insert(0, str(ace_dir))

# Check if hook file exists
HOOK_FILE_EXISTS = (hooks_dir / "ace_search_qdrant.py").exists()

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

from ace.unified_memory import (
    UnifiedMemoryIndex,
    UnifiedBullet,
    UnifiedNamespace,
    UnifiedSource,
    format_unified_context
)

# Import hook module if available
if HOOK_FILE_EXISTS:
    import ace_search_qdrant


@unittest.skipIf(not HOOK_FILE_EXISTS, "ace_search_qdrant.py hook not found")
@unittest.skipIf(not QDRANT_AVAILABLE, "Qdrant not available")
class TestSearchQdrantUnified(unittest.TestCase):
    """Test suite for unified memory search functionality using real Qdrant."""

    @classmethod
    def setUpClass(cls):
        """Set up test collection with sample data."""
        cls.test_collection = "ace_test_search_unified"
        cls.index = UnifiedMemoryIndex(
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection_name=cls.test_collection
        )

        # Create test collection
        cls.index.create_collection()

        # Sample bullets for testing
        cls.user_pref_bullets = [
            UnifiedBullet(
                id="pref-001",
                namespace=UnifiedNamespace.USER_PREFS,
                source=UnifiedSource.USER_FEEDBACK,
                content="User prefers TypeScript over JavaScript",
                section="preferences",
                severity=8,
                reinforcement_count=3
            ),
            UnifiedBullet(
                id="pref-002",
                namespace=UnifiedNamespace.USER_PREFS,
                source=UnifiedSource.USER_FEEDBACK,
                content="Always use async/await instead of promises",
                section="preferences",
                severity=7,
                reinforcement_count=2
            )
        ]

        cls.strategy_bullets = [
            UnifiedBullet(
                id="strat-001",
                namespace=UnifiedNamespace.TASK_STRATEGIES,
                source=UnifiedSource.TASK_EXECUTION,
                content="Run tests before committing code changes",
                section="task_guidance",
                helpful_count=5,
                harmful_count=0,
                severity=9
            ),
            UnifiedBullet(
                id="strat-002",
                namespace=UnifiedNamespace.TASK_STRATEGIES,
                source=UnifiedSource.TASK_EXECUTION,
                content="Use dependency injection for testability",
                section="common_patterns",
                helpful_count=3,
                harmful_count=1,
                severity=7
            )
        ]

        cls.project_bullets = [
            UnifiedBullet(
                id="proj-001",
                namespace=UnifiedNamespace.PROJECT_SPECIFIC,
                source=UnifiedSource.EXPLICIT_STORE,
                content="This project uses UV for package management",
                section="project_config",
                severity=8
            )
        ]

        cls.all_bullets = (
            cls.user_pref_bullets +
            cls.strategy_bullets +
            cls.project_bullets
        )

        # Index all bullets
        for bullet in cls.all_bullets:
            cls.index.index_bullet(bullet)

    @classmethod
    def tearDownClass(cls):
        """Clean up test collection."""
        try:
            cls.index.client.delete_collection(cls.test_collection)
        except Exception:
            pass

    def test_search_all_namespaces(self):
        """Test search_all() returns results from all namespaces."""
        results = self.index.retrieve("TypeScript testing", limit=10)

        # Verify results contain items (may not be all 5 due to relevance)
        self.assertGreater(len(results), 0)

    def test_search_preferences_only(self):
        """Test search with USER_PREFS namespace filter."""
        results = self.index.retrieve(
            "TypeScript",
            namespace=UnifiedNamespace.USER_PREFS,
            limit=10
        )

        # Verify results are only preferences
        for bullet in results:
            self.assertEqual(bullet.namespace, "user_prefs")

    def test_search_strategies_only(self):
        """Test search with TASK_STRATEGIES namespace filter."""
        results = self.index.retrieve(
            "testing patterns",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            limit=10
        )

        # Verify results are only strategies
        for bullet in results:
            self.assertEqual(bullet.namespace, "task_strategies")

    def test_search_project_only(self):
        """Test search with PROJECT_SPECIFIC namespace filter."""
        results = self.index.retrieve(
            "UV package",
            namespace=UnifiedNamespace.PROJECT_SPECIFIC,
            limit=10
        )

        # Verify results are only project-specific
        for bullet in results:
            self.assertEqual(bullet.namespace, "project_specific")

    def test_format_output_uses_unified_format(self):
        """Test that output formatting uses format_unified_context()."""
        formatted = format_unified_context(self.all_bullets)

        # Verify format includes namespace sections
        self.assertIn("User Preferences:", formatted)
        self.assertIn("Task Strategies:", formatted)
        self.assertIn("Project Context:", formatted)

        # Verify importance indicators present
        self.assertIn("[!]", formatted)  # Critical
        self.assertIn("[*]", formatted)  # Important

        # Verify reinforcement counts shown
        self.assertIn("[x3]", formatted)  # pref-001 has 3 reinforcements

    def test_search_with_limit(self):
        """Test that limit parameter is passed through correctly."""
        results = self.index.retrieve("test", limit=2)

        # Verify limit is respected
        self.assertLessEqual(len(results), 2)

    def test_search_with_threshold(self):
        """Test that score threshold parameter works."""
        results = self.index.retrieve("test", threshold=0.5, limit=10)

        # Should return results (if any match threshold)
        # The actual count depends on similarity scores
        self.assertIsInstance(results, list)

    def test_empty_results_handling(self):
        """Test handling when no results found."""
        results = self.index.retrieve(
            "xyzzy nonexistent query 12345",
            threshold=0.99,  # Very high threshold
            limit=10
        )
        formatted = format_unified_context(results)

        # Empty results should produce empty string
        if len(results) == 0:
            self.assertEqual(formatted, "")

    def test_importance_scoring_integrated(self):
        """Test that bullets are sorted by combined importance score."""
        # Create bullets with different importance
        low_importance = UnifiedBullet(
            id="low-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Low importance preference",
            section="preferences",
            severity=4,
            helpful_count=0,
            harmful_count=1
        )

        high_importance = UnifiedBullet(
            id="high-001",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="Critical strategy with high success",
            section="task_guidance",
            severity=10,
            helpful_count=10,
            harmful_count=0,
            reinforcement_count=5
        )

        formatted = format_unified_context([low_importance, high_importance])

        # High importance should be marked critical
        lines = formatted.split('\n')
        critical_lines = [l for l in lines if '[!]' in l]
        self.assertGreater(len(critical_lines), 0)


@unittest.skipIf(not QDRANT_AVAILABLE, "Qdrant not available")
class TestUnifiedIndexIntegration(unittest.TestCase):
    """Integration tests with real UnifiedMemoryIndex and Qdrant."""

    @classmethod
    def setUpClass(cls):
        """Set up test collection."""
        cls.test_collection = "ace_test_integration_unified"
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

    def test_index_and_retrieve_workflow(self):
        """Test full workflow: index bullet -> retrieve -> format."""
        # Create test bullet
        bullet = UnifiedBullet(
            id="test-workflow-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Test preference for TypeScript workflow",
            section="preferences",
            severity=8
        )

        # Index bullet
        result = self.index.index_bullet(bullet)
        self.assertTrue(result)

        # Retrieve
        results = self.index.retrieve(
            "TypeScript preference workflow",
            namespace=UnifiedNamespace.USER_PREFS,
            limit=10
        )

        # Verify
        self.assertGreater(len(results), 0)

        # Find our bullet in results
        found = any(r.id == "test-workflow-001" for r in results)
        self.assertTrue(found)

        # Format
        formatted = format_unified_context(results)
        self.assertIn("User Preferences:", formatted)
        self.assertIn("TypeScript", formatted)

    def test_namespace_filtering_in_retrieval(self):
        """Test namespace filter is correctly applied in retrieval."""
        # Create bullets in different namespaces
        pref_bullet = UnifiedBullet(
            id="test-ns-pref-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Namespace test preference unique content",
            section="preferences",
            severity=5
        )

        strat_bullet = UnifiedBullet(
            id="test-ns-strat-001",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="Namespace test strategy unique content",
            section="task_guidance",
            severity=5
        )

        # Index both
        self.index.index_bullet(pref_bullet)
        self.index.index_bullet(strat_bullet)

        # Search with namespace filter
        results = self.index.retrieve(
            "Namespace test unique content",
            namespace=UnifiedNamespace.USER_PREFS,
            limit=10
        )

        # Verify only USER_PREFS results
        for r in results:
            self.assertEqual(r.namespace, "user_prefs")


if __name__ == '__main__':
    unittest.main()
