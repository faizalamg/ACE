"""
Test suite for Phase 3: Unified Memory Retrieval Integration

These tests verify:
- SmartBulletIndex integration with UnifiedMemoryIndex
- Namespace filtering in retrieval
- Hybrid search across unified and playbook sources
- Backwards compatibility with existing retrieval API

TDD Protocol: These tests WILL FAIL until Phase 3 implementation is complete.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from typing import List, Optional

# Import existing ACE components
from ace.playbook import Bullet, EnrichedBullet, Playbook
from ace.retrieval import SmartBulletIndex, ScoredBullet

# Import unified memory components
try:
    from ace.unified_memory import (
        UnifiedBullet,
        UnifiedNamespace,
        UnifiedSource,
        UnifiedMemoryIndex,
    )
    UNIFIED_MODULE_EXISTS = True
except ImportError:
    UNIFIED_MODULE_EXISTS = False
    UnifiedBullet = None
    UnifiedNamespace = None
    UnifiedSource = None
    UnifiedMemoryIndex = None


class TestSmartBulletIndexUnifiedIntegration(unittest.TestCase):
    """Test SmartBulletIndex integration with UnifiedMemoryIndex"""

    def setUp(self):
        if not UNIFIED_MODULE_EXISTS:
            self.skipTest("ace.unified_memory not implemented yet - TDD RED phase")

        # Create playbook with traditional bullets
        self.playbook = Playbook()
        bullet = self.playbook.add_enriched_bullet(
            section="task_guidance",
            content="Use async/await for IO operations",
            task_types=["programming", "optimization"],
            trigger_patterns=["async", "io", "performance"]
        )
        # Set effectiveness manually (bullets don't accept helpful/harmful in constructor)
        bullet.helpful = 5
        bullet.harmful = 1

        # Create mock unified index
        self.mock_unified_index = Mock(spec=UnifiedMemoryIndex)
        self.mock_unified_index.retrieve.return_value = []

        # Create SmartBulletIndex with both backends
        self.index = SmartBulletIndex(
            playbook=self.playbook,
            unified_index=self.mock_unified_index
        )

    def test_smart_index_accepts_unified_index(self):
        """Test SmartBulletIndex constructor accepts unified_index parameter"""
        self.assertIsNotNone(self.index._unified_index)
        self.assertEqual(self.index._unified_index, self.mock_unified_index)

    def test_smart_index_retrieve_with_namespace_filter(self):
        """Test retrieve() supports namespace filtering"""
        # Should be able to filter by namespace when unified index is available
        results = self.index.retrieve(
            query="test query",
            namespace=UnifiedNamespace.USER_PREFS,
            limit=5
        )

        # Should call unified index with namespace filter
        self.mock_unified_index.retrieve.assert_called_once()
        call_kwargs = self.mock_unified_index.retrieve.call_args[1]
        self.assertEqual(call_kwargs["namespace"], UnifiedNamespace.USER_PREFS)

    def test_smart_index_retrieve_multiple_namespaces(self):
        """Test retrieve() supports list of namespaces"""
        results = self.index.retrieve(
            query="test query",
            namespace=[UnifiedNamespace.USER_PREFS, UnifiedNamespace.TASK_STRATEGIES],
            limit=5
        )

        # Should pass list to unified index
        self.mock_unified_index.retrieve.assert_called_once()
        call_kwargs = self.mock_unified_index.retrieve.call_args[1]
        self.assertEqual(
            call_kwargs["namespace"],
            [UnifiedNamespace.USER_PREFS, UnifiedNamespace.TASK_STRATEGIES]
        )

    def test_smart_index_hybrid_search_combines_sources(self):
        """Test retrieve() combines results from playbook and unified index"""
        # Mock unified index returns user preferences
        self.mock_unified_index.retrieve.return_value = [
            UnifiedBullet(
                id="unified-001",
                namespace=UnifiedNamespace.USER_PREFS,
                source=UnifiedSource.USER_FEEDBACK,
                content="User prefers TypeScript",
                section="preferences",
                severity=8
            )
        ]

        # Retrieve with no namespace filter (search all sources)
        results = self.index.retrieve(
            query="typescript async",
            namespace=None,
            limit=10
        )

        # Should combine playbook bullets + unified bullets
        # Playbook has 1 bullet, unified returns 1
        self.assertGreaterEqual(len(results), 2)

        # Check both sources represented
        contents = [r.content for r in results]
        self.assertIn("Use async/await for IO operations", contents)  # From playbook
        self.assertIn("User prefers TypeScript", contents)  # From unified

    def test_smart_index_backwards_compatibility(self):
        """Test retrieve() still works without namespace parameter (backwards compatible)"""
        # Old code doesn't pass namespace parameter
        results = self.index.retrieve(
            query="async io",
            task_type="programming",
            limit=5
        )

        # Should still work - retrieves from playbook only
        self.assertEqual(len(results), 1)
        self.assertIn("async/await", results[0].content)

    def test_smart_index_namespace_only_retrieval(self):
        """Test retrieve() with namespace but no query retrieves from specific namespace"""
        self.mock_unified_index.retrieve.return_value = [
            UnifiedBullet(
                id="pref-001",
                namespace=UnifiedNamespace.USER_PREFS,
                source=UnifiedSource.USER_FEEDBACK,
                content="Preference 1",
                section="preferences",
                severity=7
            ),
            UnifiedBullet(
                id="pref-002",
                namespace=UnifiedNamespace.USER_PREFS,
                source=UnifiedSource.USER_FEEDBACK,
                content="Preference 2",
                section="preferences",
                severity=6
            )
        ]

        # Retrieve all user preferences (no query)
        results = self.index.retrieve(
            namespace=UnifiedNamespace.USER_PREFS,
            limit=10
        )

        # Should retrieve from unified index only
        self.assertEqual(len(results), 2)

    def test_smart_index_scoring_unified_bullets(self):
        """Test ScoredBullet wraps UnifiedBullet correctly"""
        self.mock_unified_index.retrieve.return_value = [
            UnifiedBullet(
                id="score-001",
                namespace=UnifiedNamespace.TASK_STRATEGIES,
                source=UnifiedSource.TASK_EXECUTION,
                content="Test strategy",
                section="task_guidance",
                helpful_count=5,
                harmful_count=1,
                severity=7
            )
        ]

        results = self.index.retrieve(
            query="test strategy",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            limit=5
        )

        # Should return ScoredBullet objects
        self.assertTrue(all(isinstance(r, ScoredBullet) for r in results))

        # Bullet inside should be UnifiedBullet
        self.assertIsInstance(results[0].bullet, UnifiedBullet)

        # Should have relevance score
        self.assertGreater(results[0].score, 0)

    def test_smart_index_effectiveness_from_unified_bullet(self):
        """Test effectiveness scoring works with UnifiedBullet"""
        self.mock_unified_index.retrieve.return_value = [
            UnifiedBullet(
                id="eff-001",
                namespace=UnifiedNamespace.TASK_STRATEGIES,
                source=UnifiedSource.TASK_EXECUTION,
                content="Effective strategy",
                section="task_guidance",
                helpful_count=8,
                harmful_count=2  # 80% effectiveness
            )
        ]

        results = self.index.retrieve(
            query="strategy",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            min_effectiveness=0.7,  # Should pass 80% threshold
            limit=5
        )

        # Should return the bullet (80% > 70%)
        self.assertEqual(len(results), 1)

    def test_smart_index_filters_by_min_effectiveness(self):
        """Test min_effectiveness filter works with unified bullets"""
        self.mock_unified_index.retrieve.return_value = [
            UnifiedBullet(
                id="low-eff",
                namespace=UnifiedNamespace.TASK_STRATEGIES,
                source=UnifiedSource.TASK_EXECUTION,
                content="Untested strategy",
                section="task_guidance",
                helpful_count=1,
                harmful_count=4  # 20% effectiveness
            )
        ]

        results = self.index.retrieve(
            query="strategy",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            min_effectiveness=0.5,  # Requires 50%
            limit=5
        )

        # Should filter out low effectiveness bullet (20% < 50%)
        self.assertEqual(len(results), 0)


class TestUnifiedRetrievalInterface(unittest.TestCase):
    """Test unified retrieval interface for seamless hybrid search"""

    def setUp(self):
        if not UNIFIED_MODULE_EXISTS:
            self.skipTest("ace.unified_memory not implemented yet - TDD RED phase")

    def test_retrieve_user_preferences(self):
        """Test dedicated method for retrieving user preferences"""
        playbook = Playbook()
        mock_unified = Mock(spec=UnifiedMemoryIndex)
        mock_unified.retrieve.return_value = [
            UnifiedBullet(
                id="pref-001",
                namespace=UnifiedNamespace.USER_PREFS,
                source=UnifiedSource.USER_FEEDBACK,
                content="User prefers concise responses",
                section="communication",
                severity=8
            )
        ]

        index = SmartBulletIndex(playbook=playbook, unified_index=mock_unified)

        # Should have convenience method for user prefs
        results = index.retrieve_user_preferences(query="communication style", limit=5)

        # Verify it filtered by USER_PREFS namespace
        mock_unified.retrieve.assert_called_once()
        call_kwargs = mock_unified.retrieve.call_args[1]
        self.assertEqual(call_kwargs["namespace"], UnifiedNamespace.USER_PREFS)

    def test_retrieve_task_strategies(self):
        """Test dedicated method for retrieving task strategies"""
        playbook = Playbook()
        mock_unified = Mock(spec=UnifiedMemoryIndex)
        mock_unified.retrieve.return_value = []

        index = SmartBulletIndex(playbook=playbook, unified_index=mock_unified)

        # Should have convenience method for task strategies
        results = index.retrieve_task_strategies(query="debugging", limit=5)

        # Verify namespace filter
        mock_unified.retrieve.assert_called_once()
        call_kwargs = mock_unified.retrieve.call_args[1]
        self.assertEqual(call_kwargs["namespace"], UnifiedNamespace.TASK_STRATEGIES)

    def test_retrieve_project_context(self):
        """Test dedicated method for retrieving project-specific context"""
        playbook = Playbook()
        mock_unified = Mock(spec=UnifiedMemoryIndex)
        mock_unified.retrieve.return_value = []

        index = SmartBulletIndex(playbook=playbook, unified_index=mock_unified)

        # Should have convenience method for project context
        results = index.retrieve_project_context(query="architecture", limit=5)

        # Verify namespace filter
        mock_unified.retrieve.assert_called_once()
        call_kwargs = mock_unified.retrieve.call_args[1]
        self.assertEqual(call_kwargs["namespace"], UnifiedNamespace.PROJECT_SPECIFIC)

    def test_retrieve_all_sources(self):
        """Test retrieving from all sources (playbook + all namespaces)"""
        playbook = Playbook()
        bullet = playbook.add_enriched_bullet(
            section="task_guidance",
            content="Playbook strategy",
            task_types=["testing"]
        )

        mock_unified = Mock(spec=UnifiedMemoryIndex)
        mock_unified.retrieve.return_value = [
            UnifiedBullet(
                id="all-001",
                namespace=UnifiedNamespace.USER_PREFS,
                source=UnifiedSource.USER_FEEDBACK,
                content="User preference",
                section="preferences",
                severity=7
            )
        ]

        index = SmartBulletIndex(playbook=playbook, unified_index=mock_unified)

        # Retrieve with no namespace filter - should search all
        results = index.retrieve(query="test", namespace=None, limit=10)

        # Should combine results from both sources
        self.assertGreater(len(results), 0)


class TestUnifiedBulletCompatibility(unittest.TestCase):
    """Test UnifiedBullet works as drop-in replacement for Bullet"""

    def setUp(self):
        if not UNIFIED_MODULE_EXISTS:
            self.skipTest("ace.unified_memory not implemented yet - TDD RED phase")

    def test_unified_bullet_has_effectiveness_property(self):
        """Test UnifiedBullet has effectiveness_score property like Bullet"""
        bullet = UnifiedBullet(
            id="compat-001",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="Test",
            section="test",
            helpful_count=5,
            harmful_count=1
        )

        # Should have effectiveness_score property
        self.assertTrue(hasattr(bullet, "effectiveness_score"))
        self.assertAlmostEqual(bullet.effectiveness_score, 5/6)

    def test_unified_bullet_has_content_property(self):
        """Test UnifiedBullet has content property like Bullet"""
        bullet = UnifiedBullet(
            id="compat-002",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Test content",
            section="test"
        )

        self.assertEqual(bullet.content, "Test content")

    def test_unified_bullet_works_in_scored_bullet(self):
        """Test UnifiedBullet can be wrapped in ScoredBullet"""
        bullet = UnifiedBullet(
            id="compat-003",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="Compatible bullet",
            section="test"
        )

        scored = ScoredBullet(bullet=bullet, score=0.8, match_reasons=["test"])

        # Should access content through ScoredBullet
        self.assertEqual(scored.content, "Compatible bullet")
        self.assertEqual(scored.bullet, bullet)


class TestNamespaceFilteringLogic(unittest.TestCase):
    """Test namespace filtering logic in retrieval"""

    def setUp(self):
        if not UNIFIED_MODULE_EXISTS:
            self.skipTest("ace.unified_memory not implemented yet - TDD RED phase")

    def test_namespace_string_values_accepted(self):
        """Test namespace parameter accepts string values"""
        playbook = Playbook()
        mock_unified = Mock(spec=UnifiedMemoryIndex)
        mock_unified.retrieve.return_value = []

        index = SmartBulletIndex(playbook=playbook, unified_index=mock_unified)

        # Should accept string namespace
        results = index.retrieve(
            query="test",
            namespace="user_prefs",  # String instead of enum
            limit=5
        )

        # Should normalize to enum internally
        mock_unified.retrieve.assert_called_once()

    def test_namespace_enum_values_accepted(self):
        """Test namespace parameter accepts enum values"""
        playbook = Playbook()
        mock_unified = Mock(spec=UnifiedMemoryIndex)
        mock_unified.retrieve.return_value = []

        index = SmartBulletIndex(playbook=playbook, unified_index=mock_unified)

        # Should accept enum namespace
        results = index.retrieve(
            query="test",
            namespace=UnifiedNamespace.USER_PREFS,
            limit=5
        )

        mock_unified.retrieve.assert_called_once()

    def test_namespace_list_filtering(self):
        """Test filtering by list of namespaces"""
        playbook = Playbook()
        mock_unified = Mock(spec=UnifiedMemoryIndex)
        mock_unified.retrieve.return_value = [
            UnifiedBullet(
                id="list-001",
                namespace=UnifiedNamespace.USER_PREFS,
                source=UnifiedSource.USER_FEEDBACK,
                content="Preference",
                section="preferences"
            ),
            UnifiedBullet(
                id="list-002",
                namespace=UnifiedNamespace.TASK_STRATEGIES,
                source=UnifiedSource.TASK_EXECUTION,
                content="Strategy",
                section="task_guidance"
            )
        ]

        index = SmartBulletIndex(playbook=playbook, unified_index=mock_unified)

        # Retrieve from multiple namespaces
        results = index.retrieve(
            query="test",
            namespace=[UnifiedNamespace.USER_PREFS, UnifiedNamespace.TASK_STRATEGIES],
            limit=10
        )

        # Should return bullets from both namespaces
        self.assertGreaterEqual(len(results), 2)


class TestFallbackBehavior(unittest.TestCase):
    """Test graceful fallback when unified index is unavailable"""

    def setUp(self):
        if not UNIFIED_MODULE_EXISTS:
            self.skipTest("ace.unified_memory not implemented yet - TDD RED phase")

    def test_smart_index_works_without_unified_index(self):
        """Test SmartBulletIndex works without unified_index (backwards compatible)"""
        playbook = Playbook()
        bullet = playbook.add_enriched_bullet(
            section="task_guidance",
            content="Fallback bullet",
            task_types=["testing"]
        )

        # Create index without unified backend
        index = SmartBulletIndex(playbook=playbook)

        # Should still work - retrieve from playbook only
        results = index.retrieve(query="fallback", limit=5)

        self.assertEqual(len(results), 1)
        self.assertIn("Fallback bullet", results[0].content)

    def test_namespace_parameter_ignored_without_unified_index(self):
        """Test namespace parameter gracefully ignored when no unified index"""
        playbook = Playbook()
        playbook.add_enriched_bullet(
            section="task_guidance",
            content="Test bullet",
            task_types=["testing"]
        )

        index = SmartBulletIndex(playbook=playbook)

        # Passing namespace should not crash
        results = index.retrieve(
            query="test",
            namespace=UnifiedNamespace.USER_PREFS,  # Ignored - no unified backend
            limit=5
        )

        # Should retrieve from playbook (namespace filter ignored)
        self.assertGreaterEqual(len(results), 0)


if __name__ == '__main__':
    unittest.main()
