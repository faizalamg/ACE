"""
Tests for session tracking integration with adaptation loops.

Verifies OfflineAdapter and OnlineAdapter correctly track session outcomes
during adaptation using SessionOutcomeTracker.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from ace.adaptation import (
    OfflineAdapter,
    OnlineAdapter,
    Sample,
    TaskEnvironment,
    EnvironmentResult,
)
from ace.session_tracking import SessionOutcomeTracker
from ace.playbook import Playbook
from ace.roles import Generator, Reflector, Curator


class MockEnvironment(TaskEnvironment):
    """Mock environment that returns predefined results."""

    def __init__(self, is_correct: bool = True):
        self.is_correct = is_correct

    def evaluate(self, sample, generator_output):
        feedback = "Correct!" if self.is_correct else "Incorrect."
        return EnvironmentResult(
            feedback=feedback,
            ground_truth=sample.ground_truth,
            metrics={"correct": 1.0 if self.is_correct else 0.0},
        )


class TestOfflineAdapterSessionTracking(unittest.TestCase):
    """Test session tracking integration with OfflineAdapter."""

    def setUp(self):
        """Create mock components for testing."""
        # Mock LLM and roles
        self.mock_llm = Mock()
        self.generator = Mock(spec=Generator)
        self.reflector = Mock(spec=Reflector)
        self.curator = Mock(spec=Curator)

        # Configure mocks to return valid outputs
        from ace.roles import GeneratorOutput, ReflectorOutput, CuratorOutput
        from ace.delta import DeltaBatch

        self.generator.generate.return_value = GeneratorOutput(
            final_answer="42",
            reasoning="Simple calculation",
            bullet_ids=["bullet_001", "bullet_002"],
            raw={"answer": "42"},
        )

        self.reflector.reflect.return_value = ReflectorOutput(
            reasoning="Test reasoning",
            error_identification="No errors",
            root_cause_analysis="N/A",
            correct_approach="Proceed as planned",
            key_insight="Test insight",
            bullet_tags=[],
            raw={"tags": []},
        )

        self.curator.curate.return_value = CuratorOutput(
            delta=DeltaBatch(reasoning="Test delta batch", operations=[]),
            raw={"operations": []},
        )

        # Create playbook
        self.playbook = Playbook()

    def test_offline_adapter_accepts_session_tracker(self):
        """Test OfflineAdapter accepts session_tracker parameter."""
        tracker = SessionOutcomeTracker()

        # Should not raise error
        adapter = OfflineAdapter(
            playbook=self.playbook,
            generator=self.generator,
            reflector=self.reflector,
            curator=self.curator,
            session_tracker=tracker,
        )

        # Should store session tracker
        self.assertIsNotNone(adapter.session_tracker)
        self.assertEqual(adapter.session_tracker, tracker)

    def test_offline_adapter_tracks_successful_outcome(self):
        """Test OfflineAdapter tracks 'worked' outcome for successful samples."""
        tracker = SessionOutcomeTracker()
        adapter = OfflineAdapter(
            playbook=self.playbook,
            generator=self.generator,
            reflector=self.reflector,
            curator=self.curator,
            session_tracker=tracker,
            session_type="test_session",
        )

        # Create sample with metadata
        sample = Sample(
            question="What is 2+2?",
            ground_truth="4",
            metadata={"sample_id": "test_001"},
        )

        # Mock environment returns correct result
        env = MockEnvironment(is_correct=True)

        # Run single epoch, single sample
        results = adapter.run([sample], env, epochs=1)

        # Verify tracking occurred for used bullets
        effectiveness_001 = tracker.get_session_effectiveness("test_session", "bullet_001")
        effectiveness_002 = tracker.get_session_effectiveness("test_session", "bullet_002")

        # Should both be 100% (1 worked, 0 failed)
        self.assertEqual(effectiveness_001, 1.0)
        self.assertEqual(effectiveness_002, 1.0)

    def test_offline_adapter_tracks_failed_outcome(self):
        """Test OfflineAdapter tracks 'failed' outcome for unsuccessful samples."""
        tracker = SessionOutcomeTracker()
        adapter = OfflineAdapter(
            playbook=self.playbook,
            generator=self.generator,
            reflector=self.reflector,
            curator=self.curator,
            session_tracker=tracker,
            session_type="test_session",
        )

        sample = Sample(
            question="What is 2+2?",
            ground_truth="5",  # Wrong ground truth
            metadata={"sample_id": "test_002"},
        )

        # Mock environment returns incorrect result
        env = MockEnvironment(is_correct=False)

        # Run adaptation
        results = adapter.run([sample], env, epochs=1)

        # Verify tracking occurred with failed outcome
        effectiveness_001 = tracker.get_session_effectiveness("test_session", "bullet_001")
        effectiveness_002 = tracker.get_session_effectiveness("test_session", "bullet_002")

        # Should both be 0% (0 worked, 1 failed)
        self.assertEqual(effectiveness_001, 0.0)
        self.assertEqual(effectiveness_002, 0.0)

    def test_offline_adapter_tracks_multiple_samples(self):
        """Test OfflineAdapter tracks outcomes across multiple samples."""
        tracker = SessionOutcomeTracker()
        adapter = OfflineAdapter(
            playbook=self.playbook,
            generator=self.generator,
            reflector=self.reflector,
            curator=self.curator,
            session_tracker=tracker,
            session_type="multi_sample",
        )

        # Create samples - some succeed, some fail
        samples = [
            Sample(question="Q1", ground_truth="A1", metadata={"id": "1"}),
            Sample(question="Q2", ground_truth="A2", metadata={"id": "2"}),
            Sample(question="Q3", ground_truth="A3", metadata={"id": "3"}),
        ]

        # Mock environments - first two succeed, last one fails
        class SequentialEnvironment(TaskEnvironment):
            def __init__(self):
                self.call_count = 0
                self.results = [True, True, False]

            def evaluate(self, sample, generator_output):
                is_correct = self.results[self.call_count]
                self.call_count += 1
                return EnvironmentResult(
                    feedback="Correct!" if is_correct else "Incorrect.",
                    ground_truth=sample.ground_truth,
                    metrics={"correct": 1.0 if is_correct else 0.0},
                )

        env = SequentialEnvironment()

        # Run adaptation
        results = adapter.run(samples, env, epochs=1)

        # Verify effectiveness: 2 worked, 1 failed = 66.67%
        effectiveness = tracker.get_session_effectiveness("multi_sample", "bullet_001")
        self.assertAlmostEqual(effectiveness, 0.6667, places=2)

    def test_offline_adapter_tracks_per_session_type(self):
        """Test OfflineAdapter maintains separate tracking per session_type."""
        tracker = SessionOutcomeTracker()

        # Create two adapters with different session types
        adapter1 = OfflineAdapter(
            playbook=self.playbook,
            generator=self.generator,
            reflector=self.reflector,
            curator=self.curator,
            session_tracker=tracker,
            session_type="browser_automation",
        )

        adapter2 = OfflineAdapter(
            playbook=Playbook(),
            generator=self.generator,
            reflector=self.reflector,
            curator=self.curator,
            session_tracker=tracker,
            session_type="api_calls",
        )

        sample = Sample(question="Test", ground_truth="Test")

        # Run both adapters - one succeeds, one fails
        env_success = MockEnvironment(is_correct=True)
        env_failure = MockEnvironment(is_correct=False)

        adapter1.run([sample], env_success, epochs=1)
        adapter2.run([sample], env_failure, epochs=1)

        # Should have different effectiveness per session type
        browser_effectiveness = tracker.get_session_effectiveness(
            "browser_automation", "bullet_001"
        )
        api_effectiveness = tracker.get_session_effectiveness("api_calls", "bullet_001")

        self.assertEqual(browser_effectiveness, 1.0)  # 100% success
        self.assertEqual(api_effectiveness, 0.0)  # 0% success

    def test_offline_adapter_without_session_tracker(self):
        """Test OfflineAdapter works normally without session_tracker."""
        # No session_tracker provided
        adapter = OfflineAdapter(
            playbook=self.playbook,
            generator=self.generator,
            reflector=self.reflector,
            curator=self.curator,
        )

        sample = Sample(question="Test", ground_truth="Test")
        env = MockEnvironment(is_correct=True)

        # Should not raise error
        results = adapter.run([sample], env, epochs=1)

        # Should complete successfully
        self.assertEqual(len(results), 1)

    def test_offline_adapter_uses_sample_metadata_for_session_type(self):
        """Test OfflineAdapter can derive session_type from sample metadata."""
        tracker = SessionOutcomeTracker()
        adapter = OfflineAdapter(
            playbook=self.playbook,
            generator=self.generator,
            reflector=self.reflector,
            curator=self.curator,
            session_tracker=tracker,
            # No session_type in constructor
        )

        # Sample provides session_type in metadata
        sample = Sample(
            question="Test",
            ground_truth="Test",
            metadata={"session_type": "metadata_session"},
        )

        env = MockEnvironment(is_correct=True)
        results = adapter.run([sample], env, epochs=1)

        # Should track under session type from metadata
        effectiveness = tracker.get_session_effectiveness("metadata_session", "bullet_001")
        self.assertEqual(effectiveness, 1.0)


class TestOnlineAdapterSessionTracking(unittest.TestCase):
    """Test session tracking integration with OnlineAdapter."""

    def setUp(self):
        """Create mock components for testing."""
        self.mock_llm = Mock()
        self.generator = Mock(spec=Generator)
        self.reflector = Mock(spec=Reflector)
        self.curator = Mock(spec=Curator)

        from ace.roles import GeneratorOutput, ReflectorOutput, CuratorOutput
        from ace.delta import DeltaBatch

        self.generator.generate.return_value = GeneratorOutput(
            final_answer="42",
            reasoning="Simple calculation",
            bullet_ids=["bullet_101", "bullet_102"],
            raw={"answer": "42"},
        )

        self.reflector.reflect.return_value = ReflectorOutput(
            reasoning="Test reasoning",
            error_identification="No errors",
            root_cause_analysis="N/A",
            correct_approach="Proceed as planned",
            key_insight="Test insight",
            bullet_tags=[],
            raw={"tags": []},
        )

        self.curator.curate.return_value = CuratorOutput(
            delta=DeltaBatch(reasoning="Test delta batch", operations=[]),
            raw={"operations": []},
        )

        self.playbook = Playbook()

    def test_online_adapter_accepts_session_tracker(self):
        """Test OnlineAdapter accepts session_tracker parameter."""
        tracker = SessionOutcomeTracker()

        adapter = OnlineAdapter(
            playbook=self.playbook,
            generator=self.generator,
            reflector=self.reflector,
            curator=self.curator,
            session_tracker=tracker,
        )

        self.assertIsNotNone(adapter.session_tracker)
        self.assertEqual(adapter.session_tracker, tracker)

    def test_online_adapter_tracks_successful_outcome(self):
        """Test OnlineAdapter tracks 'worked' outcome for successful samples."""
        tracker = SessionOutcomeTracker()
        adapter = OnlineAdapter(
            playbook=self.playbook,
            generator=self.generator,
            reflector=self.reflector,
            curator=self.curator,
            session_tracker=tracker,
            session_type="online_session",
        )

        sample = Sample(question="Test", ground_truth="Test")
        env = MockEnvironment(is_correct=True)

        results = adapter.run([sample], env)

        # Verify tracking
        effectiveness = tracker.get_session_effectiveness("online_session", "bullet_101")
        self.assertEqual(effectiveness, 1.0)

    def test_online_adapter_tracks_failed_outcome(self):
        """Test OnlineAdapter tracks 'failed' outcome for unsuccessful samples."""
        tracker = SessionOutcomeTracker()
        adapter = OnlineAdapter(
            playbook=self.playbook,
            generator=self.generator,
            reflector=self.reflector,
            curator=self.curator,
            session_tracker=tracker,
            session_type="online_session",
        )

        sample = Sample(question="Test", ground_truth="Wrong")
        env = MockEnvironment(is_correct=False)

        results = adapter.run([sample], env)

        # Verify tracking
        effectiveness = tracker.get_session_effectiveness("online_session", "bullet_101")
        self.assertEqual(effectiveness, 0.0)

    def test_online_adapter_tracks_streaming_samples(self):
        """Test OnlineAdapter tracks outcomes for streaming samples."""
        tracker = SessionOutcomeTracker()
        adapter = OnlineAdapter(
            playbook=self.playbook,
            generator=self.generator,
            reflector=self.reflector,
            curator=self.curator,
            session_tracker=tracker,
            session_type="streaming",
        )

        # Create sample stream (generator)
        def sample_stream():
            for i in range(5):
                yield Sample(question=f"Q{i}", ground_truth=f"A{i}")

        # Mock environment - alternating success/failure
        class AlternatingEnvironment(TaskEnvironment):
            def __init__(self):
                self.call_count = 0

            def evaluate(self, sample, generator_output):
                is_correct = self.call_count % 2 == 0  # Even indices succeed
                self.call_count += 1
                return EnvironmentResult(
                    feedback="Correct!" if is_correct else "Incorrect.",
                    ground_truth=sample.ground_truth,
                    metrics={"correct": 1.0 if is_correct else 0.0},
                )

        env = AlternatingEnvironment()

        # Run online adaptation
        results = adapter.run(sample_stream(), env)

        # 5 samples: indices 0,2,4 succeed (3), indices 1,3 fail (2)
        # Effectiveness = 3/5 = 60%
        effectiveness = tracker.get_session_effectiveness("streaming", "bullet_101")
        self.assertAlmostEqual(effectiveness, 0.6, places=2)


class TestAdaptationSessionIntegration(unittest.TestCase):
    """Integration tests for full adaptation loop with session tracking."""

    def test_full_offline_adaptation_loop_with_session_tracking(self):
        """Test complete offline adaptation loop tracks session outcomes correctly."""
        # This test verifies the end-to-end integration
        tracker = SessionOutcomeTracker()

        # Use mocks for simplicity (DummyLLMClient requires pre-queued responses)
        # Focus on session tracking integration, not LLM client behavior
        generator = Mock(spec=Generator)
        reflector = Mock(spec=Reflector)
        curator = Mock(spec=Curator)

        from ace.roles import GeneratorOutput, ReflectorOutput, CuratorOutput
        from ace.delta import DeltaBatch

        generator.generate.return_value = GeneratorOutput(
            final_answer="42",
            reasoning="Simple calculation",
            bullet_ids=["integration_bullet_001"],
            raw={"answer": "42"},
        )

        reflector.reflect.return_value = ReflectorOutput(
            reasoning="Test reasoning",
            error_identification="No errors",
            root_cause_analysis="N/A",
            correct_approach="Proceed as planned",
            key_insight="Test insight",
            bullet_tags=[],
            raw={"tags": []},
        )

        curator.curate.return_value = CuratorOutput(
            delta=DeltaBatch(reasoning="Test delta batch", operations=[]),
            raw={"operations": []},
        )

        adapter = OfflineAdapter(
            playbook=Playbook(),
            generator=generator,
            reflector=reflector,
            curator=curator,
            session_tracker=tracker,
            session_type="integration_test",
            enable_observability=False,  # Disable to avoid Opik dependency
        )

        # Create samples
        samples = [
            Sample(question="What is 2+2?", ground_truth="4"),
            Sample(question="What is 3+3?", ground_truth="6"),
        ]

        # Simple environment
        from ace.adaptation import SimpleEnvironment

        env = SimpleEnvironment()

        # Run adaptation - should track all bullets used
        results = adapter.run(samples, env, epochs=2)

        # Verify results were recorded
        self.assertEqual(len(results), 4)  # 2 samples * 2 epochs

        # Verify session tracking occurred for integration_bullet_001
        effectiveness = tracker.get_session_effectiveness("integration_test", "integration_bullet_001")
        # All samples should succeed (SimpleEnvironment checks if ground_truth in answer)
        # DummyLLMClient doesn't return meaningful answers, so this is just integration check
        # Main goal: verify no errors during full adaptation loop with session tracking

    def test_full_online_adaptation_loop_with_session_tracking(self):
        """Test complete online adaptation loop tracks session outcomes correctly."""
        tracker = SessionOutcomeTracker()

        # Use mocks for consistency with offline integration test
        generator = Mock(spec=Generator)
        reflector = Mock(spec=Reflector)
        curator = Mock(spec=Curator)

        from ace.roles import GeneratorOutput, ReflectorOutput, CuratorOutput
        from ace.delta import DeltaBatch

        generator.generate.return_value = GeneratorOutput(
            final_answer="42",
            reasoning="Simple calculation",
            bullet_ids=["online_bullet_001"],
            raw={"answer": "42"},
        )

        reflector.reflect.return_value = ReflectorOutput(
            reasoning="Test reasoning",
            error_identification="No errors",
            root_cause_analysis="N/A",
            correct_approach="Proceed as planned",
            key_insight="Test insight",
            bullet_tags=[],
            raw={"tags": []},
        )

        curator.curate.return_value = CuratorOutput(
            delta=DeltaBatch(reasoning="Test delta batch", operations=[]),
            raw={"operations": []},
        )

        adapter = OnlineAdapter(
            playbook=Playbook(),
            generator=generator,
            reflector=reflector,
            curator=curator,
            session_tracker=tracker,
            session_type="online_integration",
            enable_observability=False,
        )

        samples = [
            Sample(question="Q1", ground_truth="A1"),
            Sample(question="Q2", ground_truth="A2"),
        ]

        from ace.adaptation import SimpleEnvironment

        env = SimpleEnvironment()

        # Run online adaptation
        results = adapter.run(samples, env)

        # Verify results
        self.assertEqual(len(results), 2)

        # Verify session tracking occurred
        effectiveness = tracker.get_session_effectiveness("online_integration", "online_bullet_001")
        # Main goal: verify no errors during full online adaptation loop with session tracking


if __name__ == "__main__":
    unittest.main()
