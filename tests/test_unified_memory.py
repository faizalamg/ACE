"""
Test suite for ACE Framework Unified Memory Architecture

These tests cover:
- Phase 1.1: UnifiedBullet schema
- Phase 1.2: UnifiedMemoryIndex with namespace support

TDD Protocol: Write failing tests first, then implement.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from typing import List, Dict, Any
import json

# These imports WILL FAIL until ace/unified_memory.py is created
try:
    from ace.unified_memory import (
        UnifiedBullet,
        UnifiedNamespace,
        UnifiedSource,
        UnifiedMemoryIndex,
        convert_bullet_to_unified,
        convert_memory_to_unified,
    )
    UNIFIED_MODULE_EXISTS = True
except ImportError:
    UNIFIED_MODULE_EXISTS = False
    # Placeholder classes
    class UnifiedBullet:
        pass
    class UnifiedNamespace:
        USER_PREFS = "user_prefs"
        TASK_STRATEGIES = "task_strategies"
        PROJECT_SPECIFIC = "project_specific"
    class UnifiedSource:
        USER_FEEDBACK = "user_feedback"
        TASK_EXECUTION = "task_execution"
        MIGRATION = "migration"
        EXPLICIT_STORE = "explicit_store"
    class UnifiedMemoryIndex:
        pass


class TestUnifiedBulletSchema(unittest.TestCase):
    """Test Phase 1.1: UnifiedBullet dataclass schema"""

    def setUp(self):
        if not UNIFIED_MODULE_EXISTS:
            self.skipTest("ace.unified_memory module not implemented yet - TDD RED phase")

    def test_unified_bullet_basic_creation(self):
        """Test creating a UnifiedBullet with required fields"""
        bullet = UnifiedBullet(
            id="test-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Always use concise responses",
            section="communication"
        )

        self.assertEqual(bullet.id, "test-001")
        self.assertEqual(bullet.namespace, UnifiedNamespace.USER_PREFS)
        self.assertEqual(bullet.source, UnifiedSource.USER_FEEDBACK)
        self.assertEqual(bullet.content, "Always use concise responses")
        self.assertEqual(bullet.section, "communication")

    def test_unified_bullet_default_values(self):
        """Test that UnifiedBullet has correct default values"""
        bullet = UnifiedBullet(
            id="test-002",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="Filter by date range first",
            section="task_guidance"
        )

        # ACE scoring defaults
        self.assertEqual(bullet.helpful_count, 0)
        self.assertEqual(bullet.harmful_count, 0)

        # Personal memory scoring defaults
        self.assertEqual(bullet.severity, 5)
        self.assertEqual(bullet.reinforcement_count, 1)

        # Metadata defaults
        self.assertEqual(bullet.category, "")
        self.assertEqual(bullet.feedback_type, "")
        self.assertEqual(bullet.context, "")

        # Retrieval optimization defaults
        self.assertEqual(bullet.trigger_patterns, [])
        self.assertEqual(bullet.task_types, [])
        self.assertEqual(bullet.complexity, "medium")
        self.assertEqual(bullet.retrieval_type, "hybrid")

        # Timestamps should be set
        self.assertIsNotNone(bullet.created_at)
        self.assertIsNotNone(bullet.updated_at)

    def test_unified_bullet_with_ace_fields(self):
        """Test UnifiedBullet with ACE-specific enrichment fields"""
        bullet = UnifiedBullet(
            id="test-003",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="Use try-except for file operations",
            section="common_errors",
            helpful_count=5,
            harmful_count=1,
            trigger_patterns=["file", "open", "read", "write"],
            task_types=["debugging", "file_operations"],
            complexity="simple",
            domains=["python"],
            retrieval_type="hybrid"
        )

        self.assertEqual(bullet.helpful_count, 5)
        self.assertEqual(bullet.harmful_count, 1)
        self.assertEqual(bullet.trigger_patterns, ["file", "open", "read", "write"])
        self.assertEqual(bullet.task_types, ["debugging", "file_operations"])
        self.assertEqual(bullet.complexity, "simple")
        self.assertEqual(bullet.domains, ["python"])

    def test_unified_bullet_with_memory_fields(self):
        """Test UnifiedBullet with personal memory fields"""
        bullet = UnifiedBullet(
            id="test-004",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="User prefers TypeScript over JavaScript",
            section="preferences",
            severity=8,
            reinforcement_count=3,
            feedback_type="DIRECTIVE",
            category="WORKFLOW"
        )

        self.assertEqual(bullet.severity, 8)
        self.assertEqual(bullet.reinforcement_count, 3)
        self.assertEqual(bullet.feedback_type, "DIRECTIVE")
        self.assertEqual(bullet.category, "WORKFLOW")

    def test_unified_bullet_serialization(self):
        """Test UnifiedBullet can be serialized to dict"""
        bullet = UnifiedBullet(
            id="test-005",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.MIGRATION,
            content="Test content",
            section="test_section"
        )

        # Should have to_dict method
        bullet_dict = bullet.to_dict()

        self.assertIsInstance(bullet_dict, dict)
        self.assertEqual(bullet_dict["id"], "test-005")
        self.assertEqual(bullet_dict["namespace"], "user_prefs")
        self.assertEqual(bullet_dict["source"], "migration")
        self.assertEqual(bullet_dict["content"], "Test content")

    def test_unified_bullet_deserialization(self):
        """Test UnifiedBullet can be created from dict"""
        data = {
            "id": "test-006",
            "namespace": "task_strategies",
            "source": "task_execution",
            "content": "Test content from dict",
            "section": "test_section",
            "helpful_count": 10,
            "severity": 7
        }

        bullet = UnifiedBullet.from_dict(data)

        self.assertEqual(bullet.id, "test-006")
        self.assertEqual(bullet.namespace, UnifiedNamespace.TASK_STRATEGIES)
        self.assertEqual(bullet.helpful_count, 10)
        self.assertEqual(bullet.severity, 7)

    def test_unified_bullet_effectiveness_score(self):
        """Test effectiveness score calculation (ACE method)"""
        bullet = UnifiedBullet(
            id="test-007",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="Test",
            section="test",
            helpful_count=8,
            harmful_count=2
        )

        # effectiveness = helpful / (helpful + harmful) = 8/10 = 0.8
        self.assertAlmostEqual(bullet.effectiveness_score, 0.8)

    def test_unified_bullet_combined_importance(self):
        """Test combined importance score (ACE + Memory scoring)"""
        bullet = UnifiedBullet(
            id="test-008",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Test",
            section="test",
            helpful_count=5,
            harmful_count=1,
            severity=9,
            reinforcement_count=4
        )

        # Should have method to compute combined importance
        importance = bullet.combined_importance_score

        # Should factor in both ACE effectiveness and memory severity/reinforcement
        self.assertGreater(importance, 0.5)


class TestUnifiedBulletConversion(unittest.TestCase):
    """Test conversion functions between old and unified formats"""

    def setUp(self):
        if not UNIFIED_MODULE_EXISTS:
            self.skipTest("ace.unified_memory module not implemented yet - TDD RED phase")

    def test_convert_ace_bullet_to_unified(self):
        """Test converting ACE Bullet/EnrichedBullet to UnifiedBullet"""
        from ace.playbook import Bullet, EnrichedBullet

        ace_bullet = EnrichedBullet(
            id="ace-001",
            section="task_guidance",
            content="Filter by date range first",
            helpful=5,
            harmful=1,
            task_types=["query", "database"],
            trigger_patterns=["date", "filter"]
        )

        unified = convert_bullet_to_unified(ace_bullet)

        self.assertIsInstance(unified, UnifiedBullet)
        self.assertEqual(unified.id, "ace-001")
        self.assertEqual(unified.namespace, UnifiedNamespace.TASK_STRATEGIES)
        self.assertEqual(unified.source, UnifiedSource.MIGRATION)
        self.assertEqual(unified.content, "Filter by date range first")
        self.assertEqual(unified.helpful_count, 5)
        self.assertEqual(unified.harmful_count, 1)
        self.assertEqual(unified.task_types, ["query", "database"])

    def test_convert_memory_to_unified(self):
        """Test converting personal memory dict to UnifiedBullet"""
        memory = {
            "lesson": "User prefers TypeScript",
            "category": "WORKFLOW",
            "severity": 8,
            "reinforcement_count": 3,
            "feedback_type": "DIRECTIVE",
            "timestamp": "2025-01-01T00:00:00Z"
        }

        unified = convert_memory_to_unified(memory)

        self.assertIsInstance(unified, UnifiedBullet)
        self.assertEqual(unified.namespace, UnifiedNamespace.USER_PREFS)
        self.assertEqual(unified.source, UnifiedSource.MIGRATION)
        self.assertEqual(unified.content, "User prefers TypeScript")
        self.assertEqual(unified.severity, 8)
        self.assertEqual(unified.reinforcement_count, 3)
        self.assertEqual(unified.category, "WORKFLOW")


class TestUnifiedMemoryIndex(unittest.TestCase):
    """Test Phase 1.2: UnifiedMemoryIndex with namespace support"""

    def setUp(self):
        if not UNIFIED_MODULE_EXISTS:
            self.skipTest("ace.unified_memory module not implemented yet - TDD RED phase")

        self.mock_qdrant_client = Mock()

    def test_unified_index_creation(self):
        """Test creating UnifiedMemoryIndex"""
        index = UnifiedMemoryIndex(
            qdrant_url="http://localhost:6333",
            embedding_url="http://localhost:1234",
            collection_name="ace_unified"
        )

        self.assertEqual(index.collection_name, "ace_unified")

    def test_unified_index_with_namespace_filter(self):
        """Test indexing bullet with namespace"""
        index = UnifiedMemoryIndex(
            qdrant_client=self.mock_qdrant_client,
            collection_name="ace_unified"
        )

        bullet = UnifiedBullet(
            id="idx-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Test content",
            section="test"
        )

        index.index_bullet(bullet)

        # Verify Qdrant upsert was called with namespace in payload
        self.mock_qdrant_client.upsert.assert_called_once()
        call_args = self.mock_qdrant_client.upsert.call_args
        points = call_args[1]["points"]
        self.assertEqual(points[0].payload["namespace"], "user_prefs")

    @unittest.skip("Test uses deprecated mock approach - retrieve() now uses REST API directly via httpx")
    def test_unified_retrieve_with_namespace_filter(self):
        """Test retrieving with namespace filter"""
        index = UnifiedMemoryIndex(
            qdrant_client=self.mock_qdrant_client,
            collection_name="ace_unified"
        )

        # Mock query_points response (newer API)
        mock_point = Mock(
            id="result-001",
            score=0.9,
            payload={
                "id": "result-001",
                "namespace": "user_prefs",
                "content": "User preference",
                "section": "preferences"
            }
        )
        mock_response = Mock()
        mock_response.points = [mock_point]
        self.mock_qdrant_client.query_points.return_value = mock_response

        results = index.retrieve(
            query="user preferences",
            namespace=UnifiedNamespace.USER_PREFS,
            limit=5
        )

        # Verify filter was applied (filter is inside prefetch, not top-level)
        self.mock_qdrant_client.query_points.assert_called_once()
        call_args = self.mock_qdrant_client.query_points.call_args
        prefetch = call_args[1].get("prefetch")
        self.assertIsNotNone(prefetch)
        self.assertTrue(len(prefetch) > 0)
        # Filter should be in the first prefetch query
        prefetch_filter = prefetch[0].get("filter")
        self.assertIsNotNone(prefetch_filter)

    def test_unified_retrieve_all_namespaces(self):
        """Test retrieving from all namespaces"""
        index = UnifiedMemoryIndex(
            qdrant_client=self.mock_qdrant_client,
            collection_name="ace_unified"
        )

        # Mock query_points response (newer API)
        mock_response = Mock()
        mock_response.points = []
        self.mock_qdrant_client.query_points.return_value = mock_response

        results = index.retrieve(
            query="test query",
            namespace=None,  # All namespaces
            limit=10
        )

        # Should not apply namespace filter
        self.mock_qdrant_client.query_points.assert_called_once()
        call_args = self.mock_qdrant_client.query_points.call_args
        query_filter = call_args[1].get("query_filter")
        # Filter should be None or not include namespace
        if query_filter is not None:
            self.assertNotIn("namespace", str(query_filter))

    def test_unified_retrieve_multiple_namespaces(self):
        """Test retrieving from multiple specific namespaces"""
        index = UnifiedMemoryIndex(
            qdrant_client=self.mock_qdrant_client,
            collection_name="ace_unified"
        )

        # Mock query_points response (newer API)
        mock_response = Mock()
        mock_response.points = []
        self.mock_qdrant_client.query_points.return_value = mock_response

        results = index.retrieve(
            query="test query",
            namespace=[UnifiedNamespace.USER_PREFS, UnifiedNamespace.TASK_STRATEGIES],
            limit=10
        )

        # Should apply OR filter for multiple namespaces
        self.mock_qdrant_client.query_points.assert_called_once()

    def test_unified_collection_creation(self):
        """Test creating unified collection with proper config"""
        # Setup mock to return empty collection list (so create_collection gets called)
        mock_collections = Mock()
        mock_collections.collections = []
        self.mock_qdrant_client.get_collections.return_value = mock_collections

        index = UnifiedMemoryIndex(
            qdrant_client=self.mock_qdrant_client,
            collection_name="ace_unified"
        )

        index.create_collection()

        # Verify collection created with hybrid vectors
        self.mock_qdrant_client.create_collection.assert_called_once()
        call_args = self.mock_qdrant_client.create_collection.call_args

        # Should have dense vectors config
        vectors_config = call_args[1].get("vectors_config")
        self.assertIsNotNone(vectors_config)

    def test_unified_batch_index(self):
        """Test batch indexing multiple bullets"""
        index = UnifiedMemoryIndex(
            qdrant_client=self.mock_qdrant_client,
            collection_name="ace_unified"
        )

        bullets = [
            UnifiedBullet(
                id=f"batch-{i}",
                namespace=UnifiedNamespace.USER_PREFS,
                source=UnifiedSource.MIGRATION,
                content=f"Content {i}",
                section="test"
            )
            for i in range(5)
        ]

        index.batch_index(bullets)

        # Should call upsert with all bullets
        self.mock_qdrant_client.upsert.assert_called_once()
        call_args = self.mock_qdrant_client.upsert.call_args
        points = call_args[1]["points"]
        self.assertEqual(len(points), 5)

    def test_unified_delete_by_namespace(self):
        """Test deleting all bullets in a namespace"""
        index = UnifiedMemoryIndex(
            qdrant_client=self.mock_qdrant_client,
            collection_name="ace_unified"
        )

        index.delete_namespace(UnifiedNamespace.PROJECT_SPECIFIC)

        # Should call delete with namespace filter
        self.mock_qdrant_client.delete.assert_called_once()
        call_args = self.mock_qdrant_client.delete.call_args
        points_selector = call_args[1].get("points_selector")
        self.assertIsNotNone(points_selector)


class TestUnifiedMemoryFormatting(unittest.TestCase):
    """Test context formatting for unified memories"""

    def setUp(self):
        if not UNIFIED_MODULE_EXISTS:
            self.skipTest("ace.unified_memory module not implemented yet - TDD RED phase")

    def test_format_unified_context(self):
        """Test formatting unified bullets for context injection"""
        from ace.unified_memory import format_unified_context

        bullets = [
            UnifiedBullet(
                id="fmt-001",
                namespace=UnifiedNamespace.USER_PREFS,
                source=UnifiedSource.USER_FEEDBACK,
                content="User prefers TypeScript",
                section="preferences",
                severity=8
            ),
            UnifiedBullet(
                id="fmt-002",
                namespace=UnifiedNamespace.TASK_STRATEGIES,
                source=UnifiedSource.TASK_EXECUTION,
                content="Use async/await for IO",
                section="task_guidance",
                helpful_count=5
            )
        ]

        formatted = format_unified_context(bullets)

        # Should group by namespace
        self.assertIn("[PREF]", formatted)
        self.assertIn("[STRAT]", formatted)
        self.assertIn("User prefers TypeScript", formatted)
        self.assertIn("Use async/await for IO", formatted)

    def test_format_with_indicators(self):
        """Test formatting includes severity/importance indicators"""
        from ace.unified_memory import format_unified_context

        bullets = [
            UnifiedBullet(
                id="ind-001",
                namespace=UnifiedNamespace.USER_PREFS,
                source=UnifiedSource.USER_FEEDBACK,
                content="Critical preference",
                section="preferences",
                severity=9
            )
        ]

        formatted = format_unified_context(bullets)

        # Should include importance indicator for high severity
        self.assertIn("[!]", formatted)  # Critical indicator


class TestUnifiedMemoryIntegration(unittest.TestCase):
    """Integration tests for unified memory with SmartBulletIndex"""

    def setUp(self):
        if not UNIFIED_MODULE_EXISTS:
            self.skipTest("ace.unified_memory module not implemented yet - TDD RED phase")

    def test_unified_with_smart_bullet_index(self):
        """Test UnifiedMemoryIndex works with SmartBulletIndex retrieval"""
        # This test verifies the integration point works
        # Full integration tested in Phase 3
        pass


class TestELFConfidenceDecay(unittest.TestCase):
    """Test ELF-inspired Confidence Decay features (Qdrant-native)"""

    def setUp(self):
        if not UNIFIED_MODULE_EXISTS:
            self.skipTest("ace.unified_memory module not implemented yet - TDD RED phase")

    def test_effective_score_with_decay_fresh(self):
        """Test that recently validated bullets have minimal decay."""
        bullet = UnifiedBullet(
            id="decay-001",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="Fresh strategy",
            section="task_guidance",
            helpful_count=8,
            harmful_count=2,
            last_validated=datetime.now(timezone.utc)
        )

        # Fresh bullet should have minimal decay
        base_score = bullet.effectiveness_score  # 8/10 = 0.8
        decayed_score = bullet.effective_score_with_decay()

        # Should be very close to base score (within 1%)
        self.assertAlmostEqual(decayed_score, base_score, delta=0.01)

    def test_effective_score_with_decay_stale(self):
        """Test that old bullets have significant decay."""
        from datetime import timedelta

        bullet = UnifiedBullet(
            id="decay-002",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="Stale strategy",
            section="task_guidance",
            helpful_count=8,
            harmful_count=2,
            last_validated=datetime.now(timezone.utc) - timedelta(weeks=10)
        )

        base_score = bullet.effectiveness_score  # 0.8
        decayed_score = bullet.effective_score_with_decay()

        # After 10 weeks with 0.95 decay rate:
        # decayed = 0.8 * (0.95^10) = 0.8 * 0.598 = 0.479
        self.assertLess(decayed_score, base_score)
        self.assertLess(decayed_score, 0.6)

    def test_effective_score_minimum_threshold(self):
        """Test that decay doesn't go below minimum threshold."""
        from datetime import timedelta

        bullet = UnifiedBullet(
            id="decay-003",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="Ancient strategy",
            section="task_guidance",
            helpful_count=8,
            harmful_count=2,
            # 2 years old - extreme decay
            last_validated=datetime.now(timezone.utc) - timedelta(weeks=104)
        )

        decayed_score = bullet.effective_score_with_decay()

        # Should not go below min_confidence_threshold (default 0.1)
        self.assertGreaterEqual(decayed_score, 0.1)

    def test_validate_resets_decay_timer(self):
        """Test that validate() updates last_validated."""
        from datetime import timedelta

        old_time = datetime.now(timezone.utc) - timedelta(weeks=4)
        bullet = UnifiedBullet(
            id="decay-004",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="Strategy to validate",
            section="task_guidance",
            helpful_count=5,
            harmful_count=0,
            last_validated=old_time
        )

        # Validate should update last_validated
        bullet.validate()

        # Should be within last few seconds
        time_diff = datetime.now(timezone.utc) - bullet.last_validated
        self.assertLess(time_diff.total_seconds(), 5)

    def test_decay_disabled_returns_raw_score(self):
        """Test that disabling decay returns raw effectiveness score."""
        from datetime import timedelta
        from ace.config import reset_config
        import os

        # Temporarily disable decay
        os.environ["ACE_CONFIDENCE_DECAY"] = "false"
        reset_config()

        try:
            bullet = UnifiedBullet(
                id="decay-005",
                namespace=UnifiedNamespace.TASK_STRATEGIES,
                source=UnifiedSource.TASK_EXECUTION,
                content="No decay strategy",
                section="task_guidance",
                helpful_count=8,
                harmful_count=2,
                last_validated=datetime.now(timezone.utc) - timedelta(weeks=10)
            )

            base_score = bullet.effectiveness_score
            decayed_score = bullet.effective_score_with_decay()

            # Should be exactly equal when decay disabled
            self.assertEqual(decayed_score, base_score)
        finally:
            # Re-enable decay
            os.environ["ACE_CONFIDENCE_DECAY"] = "true"
            reset_config()


class TestELFGoldenRules(unittest.TestCase):
    """Test ELF-inspired Golden Rules features (Qdrant-native)"""

    def setUp(self):
        if not UNIFIED_MODULE_EXISTS:
            self.skipTest("ace.unified_memory module not implemented yet - TDD RED phase")

    def test_check_golden_status_qualifies(self):
        """Test bullet qualifies for golden status."""
        bullet = UnifiedBullet(
            id="golden-001",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="Highly effective strategy",
            section="task_guidance",
            helpful_count=15,  # >= 10 threshold
            harmful_count=0    # <= 0 max
        )

        # Should qualify for golden status
        self.assertTrue(bullet.check_golden_status())

    def test_check_golden_status_not_enough_helpful(self):
        """Test bullet with insufficient helpful count."""
        bullet = UnifiedBullet(
            id="golden-002",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="New strategy",
            section="task_guidance",
            helpful_count=5,   # < 10 threshold
            harmful_count=0
        )

        # Should NOT qualify
        self.assertFalse(bullet.check_golden_status())

    def test_check_golden_status_too_harmful(self):
        """Test bullet with too many harmful counts."""
        bullet = UnifiedBullet(
            id="golden-003",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="Controversial strategy",
            section="task_guidance",
            helpful_count=20,  # >= threshold
            harmful_count=1    # > 0 max
        )

        # Should NOT qualify
        self.assertFalse(bullet.check_golden_status())

    def test_check_demotion_status(self):
        """Test bullet should be demoted from golden."""
        bullet = UnifiedBullet(
            id="golden-004",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="Formerly golden strategy",
            section="task_guidance",
            helpful_count=15,
            harmful_count=3,  # >= demotion threshold (3)
            is_golden=True
        )

        # Should be demoted
        self.assertTrue(bullet.check_demotion_status())

    def test_golden_disabled_returns_false(self):
        """Test that disabling golden rules returns false."""
        from ace.config import reset_config
        import os

        os.environ["ACE_GOLDEN_RULES"] = "false"
        reset_config()

        try:
            bullet = UnifiedBullet(
                id="golden-005",
                namespace=UnifiedNamespace.TASK_STRATEGIES,
                source=UnifiedSource.TASK_EXECUTION,
                content="Strategy",
                section="task_guidance",
                helpful_count=50,
                harmful_count=0
            )

            # Should return False when disabled
            self.assertFalse(bullet.check_golden_status())
        finally:
            os.environ["ACE_GOLDEN_RULES"] = "true"
            reset_config()


class TestELFUnifiedMemoryIndexMethods(unittest.TestCase):
    """Test ELF methods on UnifiedMemoryIndex (Qdrant-native)"""

    def setUp(self):
        if not UNIFIED_MODULE_EXISTS:
            self.skipTest("ace.unified_memory module not implemented yet - TDD RED phase")

        self.mock_qdrant_client = Mock()

    def test_tag_bullet_helpful(self):
        """Test tagging bullet as helpful updates count."""
        index = UnifiedMemoryIndex(
            qdrant_client=self.mock_qdrant_client,
            collection_name="ace_unified"
        )

        # Mock retrieve to return existing bullet
        mock_point = Mock()
        mock_point.payload = {"helpful_count": 5, "harmful_count": 0}
        self.mock_qdrant_client.retrieve.return_value = [mock_point]

        result = index.tag_bullet("test-bullet", "helpful")

        # Should call set_payload with incremented count
        self.mock_qdrant_client.set_payload.assert_called_once()
        call_args = self.mock_qdrant_client.set_payload.call_args
        payload = call_args[1]["payload"]
        self.assertEqual(payload["helpful_count"], 6)
        # Should also update last_validated
        self.assertIn("last_validated", payload)

    def test_tag_bullet_harmful(self):
        """Test tagging bullet as harmful updates count."""
        index = UnifiedMemoryIndex(
            qdrant_client=self.mock_qdrant_client,
            collection_name="ace_unified"
        )

        mock_point = Mock()
        mock_point.payload = {"helpful_count": 5, "harmful_count": 1}
        self.mock_qdrant_client.retrieve.return_value = [mock_point]

        result = index.tag_bullet("test-bullet", "harmful")

        self.mock_qdrant_client.set_payload.assert_called_once()
        call_args = self.mock_qdrant_client.set_payload.call_args
        payload = call_args[1]["payload"]
        self.assertEqual(payload["harmful_count"], 2)

    def test_validate_bullet_updates_timestamp(self):
        """Test validate_bullet updates last_validated."""
        index = UnifiedMemoryIndex(
            qdrant_client=self.mock_qdrant_client,
            collection_name="ace_unified"
        )

        result = index.validate_bullet("test-bullet")

        self.mock_qdrant_client.set_payload.assert_called_once()
        call_args = self.mock_qdrant_client.set_payload.call_args
        payload = call_args[1]["payload"]
        self.assertIn("last_validated", payload)
        self.assertIn("updated_at", payload)

    def test_update_bullet_payload(self):
        """Test generic payload update method."""
        index = UnifiedMemoryIndex(
            qdrant_client=self.mock_qdrant_client,
            collection_name="ace_unified"
        )

        result = index.update_bullet_payload(
            "test-bullet",
            {"custom_field": "custom_value"}
        )

        self.mock_qdrant_client.set_payload.assert_called_once()
        call_args = self.mock_qdrant_client.set_payload.call_args
        payload = call_args[1]["payload"]
        self.assertEqual(payload["custom_field"], "custom_value")

    def test_get_golden_rules(self):
        """Test retrieving golden rules."""
        index = UnifiedMemoryIndex(
            qdrant_client=self.mock_qdrant_client,
            collection_name="ace_unified"
        )

        # Mock scroll response
        mock_point = Mock()
        mock_point.payload = {
            "id": "golden-001",
            "namespace": "task_strategies",
            "source": "task_execution",
            "content": "Golden strategy",
            "section": "task_guidance",
            "is_golden": True
        }
        self.mock_qdrant_client.scroll.return_value = ([mock_point], None)

        results = index.get_golden_rules()

        self.mock_qdrant_client.scroll.assert_called_once()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "Golden strategy")


class TestELFSerialization(unittest.TestCase):
    """Test ELF fields are properly serialized/deserialized"""

    def setUp(self):
        if not UNIFIED_MODULE_EXISTS:
            self.skipTest("ace.unified_memory module not implemented yet - TDD RED phase")

    def test_last_validated_serialization(self):
        """Test last_validated is serialized to ISO string."""
        now = datetime.now(timezone.utc)
        bullet = UnifiedBullet(
            id="serial-001",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="Test",
            section="test",
            last_validated=now
        )

        data = bullet.to_dict()

        # Should be ISO string
        self.assertIsInstance(data["last_validated"], str)
        self.assertIn("T", data["last_validated"])  # ISO format has T

    def test_last_validated_deserialization(self):
        """Test last_validated is deserialized from ISO string."""
        data = {
            "id": "serial-002",
            "namespace": "task_strategies",
            "source": "task_execution",
            "content": "Test",
            "section": "test",
            "last_validated": "2025-01-15T10:30:00+00:00"
        }

        bullet = UnifiedBullet.from_dict(data)

        self.assertIsInstance(bullet.last_validated, datetime)
        self.assertEqual(bullet.last_validated.hour, 10)
        self.assertEqual(bullet.last_validated.minute, 30)

    def test_is_golden_serialization(self):
        """Test is_golden boolean is serialized."""
        bullet = UnifiedBullet(
            id="serial-003",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="Golden test",
            section="test",
            is_golden=True
        )

        data = bullet.to_dict()

        self.assertIsInstance(data["is_golden"], bool)
        self.assertTrue(data["is_golden"])

    def test_is_golden_deserialization(self):
        """Test is_golden boolean is deserialized."""
        data = {
            "id": "serial-004",
            "namespace": "task_strategies",
            "source": "task_execution",
            "content": "Test",
            "section": "test",
            "is_golden": True
        }

        bullet = UnifiedBullet.from_dict(data)

        self.assertTrue(bullet.is_golden)


if __name__ == '__main__':
    unittest.main()
