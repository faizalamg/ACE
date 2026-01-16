"""
Tests for Adapter unified storage integration.

TDD: Writing failing tests first for Phase 5.2 - Adapter unified storage.
These tests verify that OfflineAdapter and OnlineAdapter can optionally use unified memory.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import json

from ace.roles import Generator, Reflector, Curator, GeneratorOutput, ReflectorOutput, CuratorOutput
from ace.playbook import Playbook
from ace.adaptation import OfflineAdapter, OnlineAdapter, Sample, SimpleEnvironment
from ace.delta import DeltaBatch


class TestOfflineAdapterUnifiedStorage(unittest.TestCase):
    """Test OfflineAdapter with unified memory storage integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock LLM responses
        self.mock_llm = Mock()

        # Generator response
        self.generator_response = {
            "reasoning": "Test reasoning with [task_guidance-00001]",
            "final_answer": "4",
            "bullet_ids": ["task_guidance-00001"]
        }

        # Reflector response
        self.reflector_response = {
            "reasoning": "Good approach",
            "error_identification": "None",
            "root_cause_analysis": "N/A",
            "correct_approach": "Direct calculation",
            "key_insight": "Simple arithmetic works",
            "bullet_tags": []
        }

        # Curator response
        self.curator_response = {
            "reasoning": "Learning from success",
            "deduplication_check": {"similar_bullets": []},
            "operations": [
                {"type": "ADD", "section": "task_guidance", "content": "Learned strategy"}
            ],
            "quality_metrics": {
                "avg_atomicity": 0.9,
                "operations_count": 1,
                "estimated_impact": 0.5
            }
        }

        # Use sequential responses: Generator -> Reflector -> Curator
        # Each role gets its own mock response in order
        call_index = [0]
        responses = [
            json.dumps(self.generator_response),
            json.dumps(self.reflector_response),
            json.dumps(self.curator_response),
        ]

        def mock_complete(prompt, **kwargs):
            mock_response = Mock()
            idx = call_index[0] % 3  # Cycle through responses
            mock_response.text = responses[idx]
            call_index[0] += 1
            return mock_response

        self.mock_llm.complete.side_effect = mock_complete

        self.generator = Generator(self.mock_llm)
        self.reflector = Reflector(self.mock_llm)
        self.curator = Curator(self.mock_llm)
        self.playbook = Playbook()
        self.environment = SimpleEnvironment()

    def test_offline_adapter_accepts_unified_index_parameter(self):
        """Test that OfflineAdapter accepts unified_index parameter."""
        mock_unified_index = Mock()

        adapter = OfflineAdapter(
            playbook=self.playbook,
            generator=self.generator,
            reflector=self.reflector,
            curator=self.curator,
            unified_index=mock_unified_index
        )

        self.assertEqual(adapter.unified_index, mock_unified_index)

    def test_offline_adapter_accepts_use_unified_storage_parameter(self):
        """Test that OfflineAdapter accepts use_unified_storage parameter."""
        adapter = OfflineAdapter(
            playbook=self.playbook,
            generator=self.generator,
            reflector=self.reflector,
            curator=self.curator,
            use_unified_storage=True
        )

        self.assertTrue(adapter.use_unified_storage)

    def test_offline_adapter_default_use_unified_storage_true(self):
        """Test that use_unified_storage defaults to True (unified-only architecture)."""
        adapter = OfflineAdapter(
            playbook=self.playbook,
            generator=self.generator,
            reflector=self.reflector,
            curator=self.curator
        )

        self.assertTrue(getattr(adapter, 'use_unified_storage', True))

    def test_offline_adapter_run_stores_to_unified(self):
        """Test that run() stores curator operations to unified index."""
        mock_unified_index = Mock()
        mock_unified_index.index_bullet.return_value = True

        adapter = OfflineAdapter(
            playbook=self.playbook,
            generator=self.generator,
            reflector=self.reflector,
            curator=self.curator,
            unified_index=mock_unified_index,
            use_unified_storage=True
        )

        samples = [Sample(question="What is 2+2?", ground_truth="4")]

        results = adapter.run(samples, self.environment, epochs=1)

        # Verify unified index was called
        self.assertTrue(mock_unified_index.index_bullet.called)

    def test_offline_adapter_run_without_unified_works(self):
        """Test that run() works without unified storage."""
        adapter = OfflineAdapter(
            playbook=self.playbook,
            generator=self.generator,
            reflector=self.reflector,
            curator=self.curator
        )

        samples = [Sample(question="What is 2+2?", ground_truth="4")]

        # Should not raise
        results = adapter.run(samples, self.environment, epochs=1)

        self.assertEqual(len(results), 1)


class TestOnlineAdapterUnifiedStorage(unittest.TestCase):
    """Test OnlineAdapter with unified memory storage integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = Mock()

        # Generator response
        self.generator_response = {
            "reasoning": "Test reasoning",
            "final_answer": "test answer",
            "bullet_ids": []
        }

        # Reflector response
        self.reflector_response = {
            "reasoning": "Analysis",
            "error_identification": "None",
            "root_cause_analysis": "N/A",
            "correct_approach": "Good",
            "key_insight": "Works",
            "bullet_tags": []
        }

        # Curator response
        self.curator_response = {
            "reasoning": "Learning",
            "deduplication_check": {"similar_bullets": []},
            "operations": [],
            "quality_metrics": {
                "avg_atomicity": 0.9,
                "operations_count": 0,
                "estimated_impact": 0
            }
        }

        # Use sequential responses: Generator -> Reflector -> Curator
        call_index = [0]
        responses = [
            json.dumps(self.generator_response),
            json.dumps(self.reflector_response),
            json.dumps(self.curator_response),
        ]

        def mock_complete(prompt, **kwargs):
            mock_response = Mock()
            idx = call_index[0] % 3
            mock_response.text = responses[idx]
            call_index[0] += 1
            return mock_response

        self.mock_llm.complete.side_effect = mock_complete

        self.generator = Generator(self.mock_llm)
        self.reflector = Reflector(self.mock_llm)
        self.curator = Curator(self.mock_llm)
        self.playbook = Playbook()
        self.environment = SimpleEnvironment()

    def test_online_adapter_accepts_unified_index_parameter(self):
        """Test that OnlineAdapter accepts unified_index parameter."""
        mock_unified_index = Mock()

        adapter = OnlineAdapter(
            playbook=self.playbook,
            generator=self.generator,
            reflector=self.reflector,
            curator=self.curator,
            unified_index=mock_unified_index
        )

        self.assertEqual(adapter.unified_index, mock_unified_index)

    def test_online_adapter_accepts_use_unified_storage_parameter(self):
        """Test that OnlineAdapter accepts use_unified_storage parameter."""
        adapter = OnlineAdapter(
            playbook=self.playbook,
            generator=self.generator,
            reflector=self.reflector,
            curator=self.curator,
            use_unified_storage=True
        )

        self.assertTrue(adapter.use_unified_storage)

    def test_online_adapter_run_stores_to_unified(self):
        """Test that online run() stores to unified index when enabled."""
        # Create fresh mock with ADD operation in curator response
        fresh_llm = Mock()
        curator_with_add = {
            "reasoning": "Learning",
            "deduplication_check": {"similar_bullets": []},
            "operations": [{"type": "ADD", "section": "common_errors", "content": "Online learned pattern"}],
            "quality_metrics": {"avg_atomicity": 0.9, "operations_count": 1, "estimated_impact": 0.5}
        }

        call_index = [0]
        responses = [
            json.dumps(self.generator_response),
            json.dumps(self.reflector_response),
            json.dumps(curator_with_add),
        ]

        def mock_complete(prompt, **kwargs):
            mock_response = Mock()
            idx = call_index[0] % 3
            mock_response.text = responses[idx]
            call_index[0] += 1
            return mock_response

        fresh_llm.complete.side_effect = mock_complete

        mock_unified_index = Mock()
        mock_unified_index.index_bullet.return_value = True

        adapter = OnlineAdapter(
            playbook=Playbook(),
            generator=Generator(fresh_llm),
            reflector=Reflector(fresh_llm),
            curator=Curator(fresh_llm),
            unified_index=mock_unified_index,
            use_unified_storage=True
        )

        samples = [Sample(question="Test?", ground_truth="test")]

        results = adapter.run(samples, self.environment)

        # Verify unified storage was called
        self.assertTrue(mock_unified_index.index_bullet.called)


class TestAdapterUnifiedStorageIntegration(unittest.TestCase):
    """Integration tests for Adapter unified storage."""

    def test_curator_unified_index_propagates_from_adapter(self):
        """Test that adapter passes unified_index to curator."""
        mock_llm = Mock()
        mock_llm.complete.return_value = Mock(text=json.dumps({
            "reasoning": "Test",
            "final_answer": "4",
            "bullet_ids": []
        }))

        mock_unified_index = Mock()

        # Create curator without unified_index
        curator = Curator(mock_llm)

        adapter = OfflineAdapter(
            playbook=Playbook(),
            generator=Generator(mock_llm),
            reflector=Reflector(mock_llm),
            curator=curator,
            unified_index=mock_unified_index,
            use_unified_storage=True
        )

        # Verify adapter sets curator's unified_index
        self.assertEqual(curator.unified_index, mock_unified_index)

    def test_adapter_unified_storage_respects_flag(self):
        """Test that unified storage only happens when use_unified_storage=True."""
        mock_llm = Mock()
        mock_llm.complete.return_value = Mock(text=json.dumps({
            "reasoning": "Test",
            "deduplication_check": {},
            "operations": [{"type": "ADD", "section": "test", "content": "bullet"}],
            "quality_metrics": {"avg_atomicity": 0.9, "operations_count": 1, "estimated_impact": 0.5}
        }))

        mock_unified_index = Mock()

        adapter = OfflineAdapter(
            playbook=Playbook(),
            generator=Generator(mock_llm),
            reflector=Reflector(mock_llm),
            curator=Curator(mock_llm),
            unified_index=mock_unified_index,
            use_unified_storage=False  # Explicitly disabled
        )

        samples = [Sample(question="Test?", ground_truth="answer")]

        adapter.run(samples, SimpleEnvironment(), epochs=1)

        # Should NOT store to unified
        mock_unified_index.index_bullet.assert_not_called()


if __name__ == "__main__":
    unittest.main()
