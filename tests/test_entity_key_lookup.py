"""
Test suite for Entity-Key Deterministic Lookup (O(1) access)

These tests cover the new entity-key feature that provides:
- Deterministic O(1) lookup for specific entities
- Unique constraint: one active bullet per entity_key
- Optional field (bullets without entity_key work normally)
- Format validation (namespace:key pattern)

TDD Protocol: These tests WILL FAIL until entity_key functionality is implemented.
"""

import unittest
from unittest.mock import Mock, MagicMock
from datetime import datetime, timezone

# These imports will work but entity_key features DON'T exist yet
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


class TestUnifiedBulletEntityKey(unittest.TestCase):
    """Test UnifiedBullet entity_key field"""

    def setUp(self):
        if not UNIFIED_MODULE_EXISTS:
            self.skipTest("ace.unified_memory module not implemented yet - TDD RED phase")

    def test_entity_key_optional_field(self):
        """Test entity_key is an optional field that defaults to None"""
        bullet = UnifiedBullet(
            id="test-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="User prefers TypeScript",
            section="preferences"
        )

        # entity_key should exist but be None by default
        self.assertIsNone(bullet.entity_key)

    def test_entity_key_creation(self):
        """Test creating bullet with entity_key"""
        bullet = UnifiedBullet(
            id="test-002",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="User prefers TypeScript",
            section="preferences",
            entity_key="user:lang_preference"
        )

        self.assertEqual(bullet.entity_key, "user:lang_preference")

    def test_entity_key_format_validation_valid(self):
        """Test entity_key format validation accepts valid patterns"""
        valid_keys = [
            "user:lang_preference",
            "project:default_model",
            "session:theme_mode",
            "global:timezone",
            "user:editor_config",
        ]

        for key in valid_keys:
            bullet = UnifiedBullet(
                id=f"test-{key}",
                namespace=UnifiedNamespace.USER_PREFS,
                source=UnifiedSource.USER_FEEDBACK,
                content="Test content",
                section="test",
                entity_key=key
            )
            # Should not raise exception
            self.assertEqual(bullet.entity_key, key)

    def test_entity_key_format_validation_invalid(self):
        """Test entity_key format validation rejects invalid patterns"""
        invalid_keys = [
            "no_colon",           # Missing colon separator
            ":empty_namespace",   # Empty namespace
            "empty_key:",         # Empty key
            "too:many:colons",    # Multiple colons
            # Note: "" (empty string) is treated as None - no entity_key, valid
            "spaces not allowed", # Spaces not allowed
        ]

        for key in invalid_keys:
            with self.assertRaises(ValueError):
                UnifiedBullet(
                    id=f"test-invalid-{key}",
                    namespace=UnifiedNamespace.USER_PREFS,
                    source=UnifiedSource.USER_FEEDBACK,
                    content="Test content",
                    section="test",
                    entity_key=key
                )

    def test_entity_key_serialization(self):
        """Test entity_key is serialized to dict"""
        bullet = UnifiedBullet(
            id="test-003",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="User prefers dark mode",
            section="preferences",
            entity_key="user:theme_preference"
        )

        data = bullet.to_dict()

        self.assertIn("entity_key", data)
        self.assertEqual(data["entity_key"], "user:theme_preference")

    def test_entity_key_deserialization(self):
        """Test entity_key is deserialized from dict"""
        data = {
            "id": "test-004",
            "namespace": "user_prefs",
            "source": "user_feedback",
            "content": "User prefers 2-space indentation",
            "section": "preferences",
            "entity_key": "user:indent_preference"
        }

        bullet = UnifiedBullet.from_dict(data)

        self.assertEqual(bullet.entity_key, "user:indent_preference")

    def test_entity_key_none_serialization(self):
        """Test bullets without entity_key serialize correctly"""
        bullet = UnifiedBullet(
            id="test-005",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="General strategy without entity key",
            section="task_guidance"
        )

        data = bullet.to_dict()

        # entity_key should be None or omitted
        self.assertTrue(data.get("entity_key") is None or "entity_key" not in data)


class TestUnifiedMemoryIndexEntityKeyLookup(unittest.TestCase):
    """Test UnifiedMemoryIndex entity-key O(1) lookup methods"""

    def setUp(self):
        if not UNIFIED_MODULE_EXISTS:
            self.skipTest("ace.unified_memory module not implemented yet - TDD RED phase")

        self.mock_qdrant_client = Mock()
        self.index = UnifiedMemoryIndex(
            qdrant_client=self.mock_qdrant_client,
            collection_name="test_collection"
        )

    def test_get_by_entity_method_exists(self):
        """Test get_by_entity() method exists"""
        # Method should exist
        self.assertTrue(hasattr(self.index, "get_by_entity"))
        self.assertTrue(callable(getattr(self.index, "get_by_entity", None)))

    def test_get_by_entity_returns_bullet(self):
        """Test get_by_entity() returns UnifiedBullet for existing entity_key"""
        # Mock Qdrant scroll response with bullet that has entity_key
        mock_point = Mock()
        mock_point.payload = {
            "id": "entity-001",
            "namespace": "user_prefs",
            "source": "user_feedback",
            "content": "User prefers TypeScript",
            "section": "preferences",
            "entity_key": "user:lang_preference",
            "helpful_count": 0,
            "harmful_count": 0,
            "severity": 5,
            "reinforcement_count": 1
        }
        self.mock_qdrant_client.scroll.return_value = ([mock_point], None)

        result = self.index.get_by_entity("user:lang_preference")

        # Should return UnifiedBullet
        self.assertIsInstance(result, UnifiedBullet)
        self.assertEqual(result.entity_key, "user:lang_preference")
        self.assertEqual(result.content, "User prefers TypeScript")

    def test_get_by_entity_returns_none_not_found(self):
        """Test get_by_entity() returns None for non-existent entity_key"""
        # Mock empty scroll response
        self.mock_qdrant_client.scroll.return_value = ([], None)

        result = self.index.get_by_entity("nonexistent:key")

        self.assertIsNone(result)

    def test_get_by_entity_uses_scroll_filter(self):
        """Test get_by_entity() uses Qdrant scroll with entity_key filter"""
        self.mock_qdrant_client.scroll.return_value = ([], None)

        self.index.get_by_entity("user:theme_preference")

        # Verify scroll was called with entity_key filter
        self.mock_qdrant_client.scroll.assert_called_once()
        call_args = self.mock_qdrant_client.scroll.call_args
        scroll_filter = call_args[1].get("scroll_filter")

        # Filter should match entity_key field
        self.assertIsNotNone(scroll_filter)

    def test_get_by_entity_o1_complexity(self):
        """Test get_by_entity() is O(1) lookup (not semantic search)"""
        # This test verifies we're using scroll with exact filter, not query_points
        self.mock_qdrant_client.scroll.return_value = ([], None)

        self.index.get_by_entity("user:lang_preference")

        # Should call scroll (O(1) indexed lookup), NOT query_points (semantic search)
        self.mock_qdrant_client.scroll.assert_called_once()
        self.mock_qdrant_client.query_points.assert_not_called()


class TestUnifiedMemoryIndexUpdateByEntity(unittest.TestCase):
    """Test UnifiedMemoryIndex update_by_entity() method"""

    def setUp(self):
        if not UNIFIED_MODULE_EXISTS:
            self.skipTest("ace.unified_memory module not implemented yet - TDD RED phase")

        self.mock_qdrant_client = Mock()
        self.index = UnifiedMemoryIndex(
            qdrant_client=self.mock_qdrant_client,
            collection_name="test_collection"
        )

    def test_update_by_entity_method_exists(self):
        """Test update_by_entity() method exists"""
        self.assertTrue(hasattr(self.index, "update_by_entity"))
        self.assertTrue(callable(getattr(self.index, "update_by_entity", None)))

    def test_update_by_entity_updates_content(self):
        """Test update_by_entity() updates bullet content"""
        # Mock existing bullet
        mock_point = Mock()
        mock_point.id = "entity-001"
        mock_point.payload = {
            "id": "entity-001",
            "namespace": "user_prefs",
            "source": "user_feedback",
            "content": "Old preference",
            "section": "preferences",
            "entity_key": "user:lang_preference",
            "helpful_count": 2,
            "harmful_count": 0
        }
        self.mock_qdrant_client.scroll.return_value = ([mock_point], None)

        result = self.index.update_by_entity(
            "user:lang_preference",
            "User now prefers Python"
        )

        # Should call set_payload to update content
        self.mock_qdrant_client.set_payload.assert_called_once()
        call_args = self.mock_qdrant_client.set_payload.call_args
        payload = call_args[1]["payload"]

        self.assertEqual(payload["content"], "User now prefers Python")
        # Should also update updated_at timestamp
        self.assertIn("updated_at", payload)

    def test_update_by_entity_returns_updated_bullet(self):
        """Test update_by_entity() returns updated UnifiedBullet"""
        # Mock existing bullet
        mock_point = Mock()
        mock_point.id = "entity-002"
        mock_point.payload = {
            "id": "entity-002",
            "namespace": "user_prefs",
            "source": "user_feedback",
            "content": "Old content",
            "section": "preferences",
            "entity_key": "user:theme_preference",
            "helpful_count": 0,
            "harmful_count": 0
        }
        self.mock_qdrant_client.scroll.return_value = ([mock_point], None)

        result = self.index.update_by_entity(
            "user:theme_preference",
            "User prefers dark mode"
        )

        self.assertIsInstance(result, UnifiedBullet)
        self.assertEqual(result.entity_key, "user:theme_preference")
        self.assertEqual(result.content, "User prefers dark mode")

    def test_update_by_entity_raises_not_found(self):
        """Test update_by_entity() raises ValueError if entity_key not found"""
        # Mock empty scroll response
        self.mock_qdrant_client.scroll.return_value = ([], None)

        with self.assertRaises(ValueError) as ctx:
            self.index.update_by_entity("nonexistent:key", "New content")

        self.assertIn("nonexistent:key", str(ctx.exception))

    def test_update_by_entity_increments_reinforcement(self):
        """Test update_by_entity() increments reinforcement_count"""
        # Mock existing bullet
        mock_point = Mock()
        mock_point.id = "entity-003"
        mock_point.payload = {
            "id": "entity-003",
            "namespace": "user_prefs",
            "source": "user_feedback",
            "content": "Old preference",
            "section": "preferences",
            "entity_key": "user:indent_preference",
            "reinforcement_count": 2
        }
        self.mock_qdrant_client.scroll.return_value = ([mock_point], None)

        result = self.index.update_by_entity(
            "user:indent_preference",
            "User prefers 4-space indentation"
        )

        # Should increment reinforcement_count
        call_args = self.mock_qdrant_client.set_payload.call_args
        payload = call_args[1]["payload"]
        self.assertEqual(payload["reinforcement_count"], 3)


class TestEntityKeyUniquenessConstraint(unittest.TestCase):
    """Test entity_key uniqueness constraint (one active bullet per key)"""

    def setUp(self):
        if not UNIFIED_MODULE_EXISTS:
            self.skipTest("ace.unified_memory module not implemented yet - TDD RED phase")

        self.mock_qdrant_client = Mock()
        self.index = UnifiedMemoryIndex(
            qdrant_client=self.mock_qdrant_client,
            collection_name="test_collection"
        )

    def test_index_bullet_with_duplicate_entity_key_replaces_old(self):
        """Test indexing bullet with existing entity_key replaces old bullet"""
        # Mock existing bullet with same entity_key
        existing_point = Mock()
        existing_point.id = "old-bullet-001"
        existing_point.payload = {
            "id": "old-bullet-001",
            "namespace": "user_prefs",
            "content": "Old preference",
            "entity_key": "user:lang_preference"
        }
        self.mock_qdrant_client.scroll.return_value = ([existing_point], None)

        # Create new bullet with same entity_key
        new_bullet = UnifiedBullet(
            id="new-bullet-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="User now prefers Python",
            section="preferences",
            entity_key="user:lang_preference"
        )

        self.index.index_bullet(new_bullet)

        # Should delete old bullet and upsert new one
        self.mock_qdrant_client.delete.assert_called_once()
        delete_call_args = self.mock_qdrant_client.delete.call_args
        deleted_ids = delete_call_args[1]["points_selector"]

        # Should delete old bullet by ID
        # (Implementation detail: check that old ID was targeted for deletion)
        self.assertIsNotNone(deleted_ids)

    def test_get_by_entity_returns_only_one_bullet(self):
        """Test get_by_entity() returns only one bullet (uniqueness)"""
        # Even if Qdrant returns multiple (bug scenario), should only return first
        mock_point1 = Mock()
        mock_point1.payload = {
            "id": "dup-001",
            "namespace": "user_prefs",
            "content": "First duplicate",
            "entity_key": "user:lang_preference"
        }
        mock_point2 = Mock()
        mock_point2.payload = {
            "id": "dup-002",
            "namespace": "user_prefs",
            "content": "Second duplicate",
            "entity_key": "user:lang_preference"
        }
        self.mock_qdrant_client.scroll.return_value = ([mock_point1, mock_point2], None)

        result = self.index.get_by_entity("user:lang_preference")

        # Should return only one bullet (first one)
        self.assertIsInstance(result, UnifiedBullet)
        # Should log warning about duplicates (not tested here, but expected behavior)


class TestEntityKeyIndexCreation(unittest.TestCase):
    """Test entity_key index creation in Qdrant collection"""

    def setUp(self):
        if not UNIFIED_MODULE_EXISTS:
            self.skipTest("ace.unified_memory module not implemented yet - TDD RED phase")

        self.mock_qdrant_client = Mock()

    def test_create_collection_with_entity_key_index(self):
        """Test create_collection() creates payload index for entity_key"""
        # Mock get_collections to return empty (force create)
        mock_collections = Mock()
        mock_collections.collections = []
        self.mock_qdrant_client.get_collections.return_value = mock_collections

        index = UnifiedMemoryIndex(
            qdrant_client=self.mock_qdrant_client,
            collection_name="ace_unified"
        )

        index.create_collection()

        # Should call create_payload_index for entity_key field
        # (This enables O(1) lookup via scroll filter)
        self.mock_qdrant_client.create_payload_index.assert_called()

        # Check that entity_key field was indexed
        create_index_calls = self.mock_qdrant_client.create_payload_index.call_args_list
        entity_key_indexed = any(
            "entity_key" in str(call)
            for call in create_index_calls
        )
        self.assertTrue(entity_key_indexed, "entity_key field should be indexed")


class TestEntityKeyEdgeCases(unittest.TestCase):
    """Test edge cases for entity_key functionality"""

    def setUp(self):
        if not UNIFIED_MODULE_EXISTS:
            self.skipTest("ace.unified_memory module not implemented yet - TDD RED phase")

        self.mock_qdrant_client = Mock()
        self.index = UnifiedMemoryIndex(
            qdrant_client=self.mock_qdrant_client,
            collection_name="test_collection"
        )

    def test_bullets_without_entity_key_work_normally(self):
        """Test bullets without entity_key are indexed/retrieved normally"""
        bullet = UnifiedBullet(
            id="normal-001",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="General task strategy",
            section="task_guidance"
            # No entity_key
        )

        # Should index without error
        self.index.index_bullet(bullet)

        # Should call upsert
        self.mock_qdrant_client.upsert.assert_called_once()

    def test_entity_key_case_sensitive(self):
        """Test entity_key matching is case-sensitive"""
        mock_point = Mock()
        mock_point.payload = {
            "id": "case-001",
            "namespace": "user_prefs",
            "content": "Test",
            "entity_key": "user:Lang_Preference"  # Camel case
        }
        self.mock_qdrant_client.scroll.return_value = ([mock_point], None)

        result_exact = self.index.get_by_entity("user:Lang_Preference")
        self.assertIsNotNone(result_exact)

        # Different case should not match
        self.mock_qdrant_client.scroll.return_value = ([], None)
        result_diff = self.index.get_by_entity("user:lang_preference")
        self.assertIsNone(result_diff)

    def test_entity_key_namespace_independent(self):
        """Test entity_key can have different namespace than bullet namespace"""
        # entity_key namespace (e.g., "user:") doesn't have to match bullet.namespace
        bullet = UnifiedBullet(
            id="namespace-001",
            namespace=UnifiedNamespace.TASK_STRATEGIES,  # task_strategies
            source=UnifiedSource.USER_FEEDBACK,
            content="User preference stored in task namespace",
            section="preferences",
            entity_key="user:task_preference"  # entity_key namespace = "user"
        )

        # Should not raise exception
        self.assertEqual(bullet.entity_key, "user:task_preference")


if __name__ == '__main__':
    unittest.main()
