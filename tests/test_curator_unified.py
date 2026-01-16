"""
Tests for Curator unified storage integration.

TDD: Writing failing tests first for Phase 5.1 - Curator unified storage.
These tests verify that the Curator can optionally store bullets to the unified memory system.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import json

from ace.roles import Curator, CuratorOutput
from ace.playbook import Playbook
from ace.delta import DeltaBatch, DeltaOperation


class TestCuratorUnifiedStorage(unittest.TestCase):
    """Test Curator with unified memory storage integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = Mock()
        self.mock_llm.complete.return_value = Mock(text=json.dumps({
            "reasoning": "Test reasoning",
            "deduplication_check": {"similar_bullets": []},
            "operations": [
                {
                    "type": "ADD",
                    "section": "task_guidance",
                    "content": "New strategy from test"
                }
            ],
            "quality_metrics": {
                "avg_atomicity": 0.9,
                "operations_count": 1,
                "estimated_impact": 0.5
            }
        }))

        self.playbook = Playbook()

        # Mock reflection output
        self.mock_reflection = Mock()
        self.mock_reflection.raw = {
            "reasoning": "Test reflection",
            "error_identification": "None",
            "root_cause_analysis": "N/A",
            "correct_approach": "Test approach",
            "key_insight": "Test insight"
        }
        self.mock_reflection.bullet_tags = []

    def test_curator_accepts_unified_index_parameter(self):
        """Test that Curator accepts unified_index parameter in constructor."""
        mock_unified_index = Mock()

        # This should not raise an error
        curator = Curator(
            self.mock_llm,
            unified_index=mock_unified_index
        )

        self.assertIsNotNone(curator)
        self.assertEqual(curator.unified_index, mock_unified_index)

    def test_curator_without_unified_index_works(self):
        """Test that Curator works without unified_index (backward compat)."""
        curator = Curator(self.mock_llm)

        # Should have None or not have the attribute
        unified_index = getattr(curator, 'unified_index', None)
        self.assertIsNone(unified_index)

    def test_curate_stores_to_unified_when_enabled(self):
        """Test that curate() stores ADD operations to unified index."""
        mock_unified_index = Mock()
        mock_unified_index.index_bullet.return_value = True

        curator = Curator(
            self.mock_llm,
            unified_index=mock_unified_index
        )

        output = curator.curate(
            reflection=self.mock_reflection,
            playbook=self.playbook,
            question_context="Test context",
            progress="1/1"
        )

        # Verify unified index was called for ADD operation
        mock_unified_index.index_bullet.assert_called()

        # Verify the bullet has correct namespace
        call_args = mock_unified_index.index_bullet.call_args
        bullet = call_args[0][0]  # First positional arg

        # Check it's a UnifiedBullet with task_strategies namespace
        # Handle both enum (UnifiedNamespace.TASK_STRATEGIES) and string ("task_strategies")
        namespace_val = bullet.namespace.value if hasattr(bullet.namespace, 'value') else bullet.namespace
        self.assertEqual(namespace_val, "task_strategies")
        self.assertEqual(bullet.section, "task_guidance")
        self.assertIn("New strategy from test", bullet.content)

    def test_curate_does_not_store_when_no_unified_index(self):
        """Test that curate() does not fail without unified index."""
        curator = Curator(self.mock_llm)

        # Should complete without error
        output = curator.curate(
            reflection=self.mock_reflection,
            playbook=self.playbook,
            question_context="Test context",
            progress="1/1"
        )

        self.assertIsInstance(output, CuratorOutput)
        self.assertIsNotNone(output.delta)

    def test_curate_stores_with_task_execution_source(self):
        """Test that stored bullets have source=TASK_EXECUTION."""
        mock_unified_index = Mock()
        mock_unified_index.index_bullet.return_value = True

        curator = Curator(
            self.mock_llm,
            unified_index=mock_unified_index
        )

        output = curator.curate(
            reflection=self.mock_reflection,
            playbook=self.playbook,
            question_context="Test context",
            progress="1/1"
        )

        call_args = mock_unified_index.index_bullet.call_args
        bullet = call_args[0][0]

        # Handle both enum and string values
        source_val = bullet.source.value if hasattr(bullet.source, 'value') else bullet.source
        self.assertEqual(source_val, "task_execution")

    def test_curate_handles_unified_storage_failure_gracefully(self):
        """Test that curate() continues when unified storage fails."""
        mock_unified_index = Mock()
        mock_unified_index.index_bullet.side_effect = Exception("Storage failed")

        curator = Curator(
            self.mock_llm,
            unified_index=mock_unified_index
        )

        # Should complete without raising exception
        output = curator.curate(
            reflection=self.mock_reflection,
            playbook=self.playbook,
            question_context="Test context",
            progress="1/1"
        )

        self.assertIsInstance(output, CuratorOutput)

    def test_curate_only_stores_add_operations(self):
        """Test that only ADD operations are stored to unified index."""
        # Configure LLM to return UPDATE operation
        self.mock_llm.complete.return_value = Mock(text=json.dumps({
            "reasoning": "Update test",
            "deduplication_check": {"similar_bullets": []},
            "operations": [
                {
                    "type": "UPDATE",
                    "bullet_id": "task_guidance-00001",
                    "section": "task_guidance",
                    "content": "Updated strategy"
                }
            ],
            "quality_metrics": {
                "avg_atomicity": 0.9,
                "operations_count": 1,
                "estimated_impact": 0.3
            }
        }))

        mock_unified_index = Mock()

        curator = Curator(
            self.mock_llm,
            unified_index=mock_unified_index
        )

        output = curator.curate(
            reflection=self.mock_reflection,
            playbook=self.playbook,
            question_context="Test context",
            progress="1/1"
        )

        # UPDATE operations should NOT be stored to unified
        mock_unified_index.index_bullet.assert_not_called()

    def test_curate_preserves_playbook_apply_delta(self):
        """Test that playbook.apply_delta still works after unified storage."""
        mock_unified_index = Mock()
        mock_unified_index.index_bullet.return_value = True

        curator = Curator(
            self.mock_llm,
            unified_index=mock_unified_index
        )

        output = curator.curate(
            reflection=self.mock_reflection,
            playbook=self.playbook,
            question_context="Test context",
            progress="1/1"
        )

        # Apply delta to playbook
        self.playbook.apply_delta(output.delta)

        # Verify playbook has the new bullet
        bullets = self.playbook.bullets()
        self.assertTrue(any("New strategy from test" in b.content for b in bullets))


class TestCuratorUnifiedStorageDisabled(unittest.TestCase):
    """Test Curator with unified storage explicitly disabled."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = Mock()
        self.mock_llm.complete.return_value = Mock(text=json.dumps({
            "reasoning": "Test reasoning",
            "deduplication_check": {"similar_bullets": []},
            "operations": [
                {"type": "ADD", "section": "task_guidance", "content": "Test bullet"}
            ],
            "quality_metrics": {
                "avg_atomicity": 0.9,
                "operations_count": 1,
                "estimated_impact": 0.5
            }
        }))

        self.mock_reflection = Mock()
        self.mock_reflection.raw = {
            "reasoning": "Test",
            "error_identification": "None",
            "root_cause_analysis": "N/A",
            "correct_approach": "Test",
            "key_insight": "Test"
        }
        self.mock_reflection.bullet_tags = []

    def test_store_to_unified_false_disables_storage(self):
        """Test that store_to_unified=False prevents unified storage."""
        mock_unified_index = Mock()

        curator = Curator(
            self.mock_llm,
            unified_index=mock_unified_index,
            store_to_unified=False
        )

        output = curator.curate(
            reflection=self.mock_reflection,
            playbook=Playbook(),
            question_context="Test",
            progress="1/1"
        )

        # Should NOT call index_bullet when disabled
        mock_unified_index.index_bullet.assert_not_called()


if __name__ == "__main__":
    unittest.main()
