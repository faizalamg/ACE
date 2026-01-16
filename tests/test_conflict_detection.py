"""
Test suite for conflict detection in UnifiedMemoryIndex.

Tests the ability to detect semantically contradictory bullets and resolve conflicts.
All tests should FAIL initially - this is the RED phase of TDD.
"""

import unittest
from unittest.mock import Mock, MagicMock
from datetime import datetime
from typing import List

from ace.unified_memory import (
    UnifiedBullet,
    UnifiedNamespace,
    UnifiedSource,
    UnifiedMemoryIndex
)


class TestConflictDetection(unittest.TestCase):
    """Test conflict detection and resolution in UnifiedMemoryIndex."""

    def setUp(self):
        """Setup test fixtures with mocked Qdrant client."""
        self.mock_qdrant_client = Mock()
        self.index = UnifiedMemoryIndex(
            qdrant_client=self.mock_qdrant_client,
            collection_name="test_conflicts"
        )

        # Common test bullet
        self.test_bullet = UnifiedBullet(
            id="test-bullet-1",
            content="Always use Python for scripting tasks",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            section="task_guidance",
            helpful_count=5,
            harmful_count=0,
            category="language_preference"
        )

        # Contradictory bullet
        self.contradictory_bullet = UnifiedBullet(
            id="test-bullet-2",
            content="Never use Python, prefer JavaScript for scripting",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            section="task_guidance",
            helpful_count=3,
            harmful_count=0,
            category="language_preference"
        )

        # Similar but non-conflicting bullet
        self.similar_bullet = UnifiedBullet(
            id="test-bullet-3",
            content="Use Python 3.11+ for optimal performance",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            section="task_guidance",
            helpful_count=4,
            harmful_count=0,
            category="language_preference"
        )

        # Unrelated bullet
        self.unrelated_bullet = UnifiedBullet(
            id="test-bullet-4",
            content="Always commit with descriptive messages",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            section="best_practices",
            helpful_count=10,
            harmful_count=0,
            category="git_practices"
        )

    def test_detect_conflicts_method_exists(self):
        """Test that detect_conflicts method exists on UnifiedMemoryIndex."""
        self.assertTrue(
            hasattr(self.index, 'detect_conflicts'),
            "UnifiedMemoryIndex should have detect_conflicts method"
        )

    def test_detect_conflicts_with_contradictory_bullet(self):
        """Test detecting semantically contradictory bullets."""
        # Mock existing bullet in index
        self.mock_qdrant_client.search.return_value = [
            Mock(id=self.test_bullet.id, score=0.90, payload={
                'content': self.test_bullet.content,
                'namespace': self.test_bullet.namespace,
                'source': self.test_bullet.source,
                'section': self.test_bullet.section,
                'helpful_count': self.test_bullet.helpful_count,
                'harmful_count': self.test_bullet.harmful_count,
                'category': self.test_bullet.category
            })
        ]

        # Detect conflicts with contradictory bullet
        conflicts = self.index.detect_conflicts(self.contradictory_bullet)

        # Should detect the contradictory bullet
        self.assertIsInstance(conflicts, list, "detect_conflicts should return a list")
        self.assertGreater(len(conflicts), 0, "Should detect at least one conflict")
        self.assertEqual(
            conflicts[0].id,
            self.test_bullet.id,
            "Should detect test_bullet as conflicting"
        )

    def test_detect_conflicts_high_similarity_threshold(self):
        """Test that conflict detection uses >0.85 similarity threshold."""
        # Mock bullets with varying similarity scores
        mock_results = [
            Mock(id="similar-high", score=0.90, payload={'content': "Use Python always"}),
            Mock(id="similar-low", score=0.70, payload={'content': "Use Python sometimes"}),
        ]
        self.mock_qdrant_client.search.return_value = mock_results

        conflicts = self.index.detect_conflicts(self.test_bullet)

        # Should only flag high similarity (>0.85) bullets
        conflict_ids = [c.id for c in conflicts]
        self.assertIn("similar-high", conflict_ids, "High similarity should be flagged")
        self.assertNotIn("similar-low", conflict_ids, "Low similarity should not be flagged")

    def test_detect_conflicts_non_conflicting_similar_bullets(self):
        """Test that similar but non-conflicting bullets are NOT flagged."""
        # Mock similar but complementary bullet
        self.mock_qdrant_client.search.return_value = [
            Mock(id=self.similar_bullet.id, score=0.88, payload={
                'content': self.similar_bullet.content,
                'namespace': UnifiedNamespace.TASK_STRATEGIES.value,
                'source': UnifiedSource.TASK_EXECUTION.value,
                'section': self.similar_bullet.section,
                'helpful_count': self.similar_bullet.helpful_count,
                'harmful_count': self.similar_bullet.harmful_count,
                'category': self.similar_bullet.category
            })
        ]

        conflicts = self.index.detect_conflicts(self.test_bullet)

        # Similar bullet should NOT be flagged as conflict
        # (both say "use Python" - they agree)
        self.assertEqual(
            len(conflicts),
            0,
            "Complementary bullets should not be flagged as conflicts"
        )

    def test_detect_conflicts_returns_empty_for_unrelated(self):
        """Test that unrelated bullets return empty conflicts list."""
        # Mock completely unrelated bullet
        self.mock_qdrant_client.search.return_value = [
            Mock(id=self.unrelated_bullet.id, score=0.20, payload={
                'content': self.unrelated_bullet.content,
                'namespace': UnifiedNamespace.TASK_STRATEGIES.value,
                'source': UnifiedSource.TASK_EXECUTION.value,
                'section': self.unrelated_bullet.section,
                'helpful_count': self.unrelated_bullet.helpful_count,
                'harmful_count': self.unrelated_bullet.harmful_count,
                'category': self.unrelated_bullet.category
            })
        ]

        conflicts = self.index.detect_conflicts(self.test_bullet)

        self.assertEqual(len(conflicts), 0, "Unrelated bullets should not conflict")

    def test_index_bullet_returns_conflicts_on_insert(self):
        """Test that index_bullet returns detected conflicts when inserting."""
        # Mock conflict detection on insert
        self.mock_qdrant_client.search.return_value = [
            Mock(id=self.test_bullet.id, score=0.92, payload={
                'content': self.test_bullet.content,
                'namespace': UnifiedNamespace.TASK_STRATEGIES.value,
                'source': UnifiedSource.TASK_EXECUTION.value,
                'section': self.test_bullet.section,
                'helpful_count': self.test_bullet.helpful_count,
                'harmful_count': self.test_bullet.harmful_count,
                'category': self.test_bullet.category
            })
        ]

        # Insert contradictory bullet - should return conflicts
        result = self.index.index_bullet(self.contradictory_bullet)

        # Result should contain conflict information
        self.assertIsNotNone(result, "index_bullet should return conflict data")
        self.assertTrue(
            hasattr(result, 'conflicts') or isinstance(result, dict),
            "Result should contain conflicts information"
        )

        # Extract conflicts from result (dict or object)
        conflicts = result.get('conflicts', []) if isinstance(result, dict) else result.conflicts

        self.assertGreater(
            len(conflicts),
            0,
            "Should return detected conflicts on insert"
        )

    def test_resolve_conflict_method_exists(self):
        """Test that resolve_conflict method exists on UnifiedMemoryIndex."""
        self.assertTrue(
            hasattr(self.index, 'resolve_conflict'),
            "UnifiedMemoryIndex should have resolve_conflict method"
        )

    def test_resolve_conflict_keeps_winner_removes_losers(self):
        """Test that resolve_conflict keeps winning bullet and removes losing bullets."""
        keep_id = self.test_bullet.id
        remove_ids = [self.contradictory_bullet.id, "another-conflict-id"]

        # Mock delete operation
        self.mock_qdrant_client.delete.return_value = Mock(status="ok")

        # Resolve conflict - keep test_bullet, remove contradictory_bullet
        self.index.resolve_conflict(keep_id=keep_id, remove_ids=remove_ids)

        # Verify delete was called with correct IDs
        self.mock_qdrant_client.delete.assert_called_once()
        delete_call_args = self.mock_qdrant_client.delete.call_args

        # Check that remove_ids were passed to delete
        points_selector = delete_call_args.kwargs.get('points_selector')
        self.assertIsNotNone(points_selector, "Should call delete with points_selector")

    def test_resolve_conflict_updates_winning_bullet_metadata(self):
        """Test that resolve_conflict updates metadata of winning bullet."""
        keep_id = self.test_bullet.id
        remove_ids = [self.contradictory_bullet.id]

        # Mock update operation
        self.mock_qdrant_client.set_payload.return_value = Mock(status="ok")

        self.index.resolve_conflict(keep_id=keep_id, remove_ids=remove_ids)

        # Verify winning bullet metadata was updated with conflict resolution info
        self.mock_qdrant_client.set_payload.assert_called()
        update_call = self.mock_qdrant_client.set_payload.call_args

        # Check that metadata includes conflict resolution info
        payload = update_call.kwargs.get('payload', {})
        self.assertIn(
            'conflict_resolved',
            str(payload),
            "Winning bullet should have conflict_resolved metadata"
        )

    def test_detect_conflicts_uses_llm_for_semantic_analysis(self):
        """Test that conflict detection uses LLM for semantic contradiction analysis."""
        # Mock high similarity bullets
        self.mock_qdrant_client.search.return_value = [
            Mock(id="candidate-1", score=0.91, payload={
                'content': "Never use Python for scripting",
                'namespace': UnifiedNamespace.TASK_STRATEGIES.value,
                'source': UnifiedSource.TASK_EXECUTION.value,
                'section': "task_guidance",
                'helpful_count': 2,
                'harmful_count': 0,
                'category': "language_preference"
            })
        ]

        # Mock LLM client for semantic analysis
        mock_llm = Mock()
        mock_llm.complete.return_value = '{"is_contradictory": true, "reason": "Direct negation"}'
        self.index._llm_client = mock_llm

        conflicts = self.index.detect_conflicts(self.test_bullet)

        # Verify LLM was called for semantic analysis
        mock_llm.complete.assert_called()
        self.assertGreater(
            len(conflicts),
            0,
            "LLM should detect semantic contradiction"
        )

    def test_detect_conflicts_empty_index(self):
        """Test conflict detection when index is empty returns no conflicts."""
        # Mock empty search results
        self.mock_qdrant_client.search.return_value = []

        conflicts = self.index.detect_conflicts(self.test_bullet)

        self.assertEqual(len(conflicts), 0, "Empty index should have no conflicts")

    def test_resolve_conflict_validates_keep_id_exists(self):
        """Test that resolve_conflict validates winning bullet exists before deletion."""
        keep_id = "non-existent-id"
        remove_ids = [self.contradictory_bullet.id]

        # Mock retrieve returning None (bullet doesn't exist)
        self.mock_qdrant_client.retrieve.return_value = None

        # Should raise ValueError for non-existent winning bullet
        with self.assertRaises(ValueError) as context:
            self.index.resolve_conflict(keep_id=keep_id, remove_ids=remove_ids)

        self.assertIn(
            "keep_id",
            str(context.exception).lower(),
            "Error should mention keep_id validation"
        )


if __name__ == '__main__':
    unittest.main()
