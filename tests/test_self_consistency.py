"""Tests for self-consistency sampling in Generator.

TDD: These tests define the expected behavior BEFORE implementation.

Self-consistency sampling generates multiple responses and selects
the most consistent answer via majority voting, improving accuracy
for tasks where reasoning can vary but the answer should converge.
"""

import unittest
from typing import List
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestSelfConsistencyBasic(unittest.TestCase):
    """Test basic self-consistency functionality."""

    def test_self_consistency_generator_exists(self):
        """Test that SelfConsistencyGenerator class exists."""
        from ace.self_consistency import SelfConsistencyGenerator

        self.assertIsNotNone(SelfConsistencyGenerator)

    def test_self_consistency_inherits_generator_interface(self):
        """Test that SelfConsistencyGenerator has generate method."""
        from ace.self_consistency import SelfConsistencyGenerator

        llm = MagicMock()
        generator = SelfConsistencyGenerator(llm, num_samples=3)

        self.assertTrue(hasattr(generator, "generate"))
        self.assertTrue(callable(generator.generate))

    def test_self_consistency_configurable_samples(self):
        """Test that number of samples is configurable."""
        from ace.self_consistency import SelfConsistencyGenerator

        llm = MagicMock()
        generator = SelfConsistencyGenerator(llm, num_samples=5)

        self.assertEqual(generator.num_samples, 5)

    def test_default_num_samples(self):
        """Test default number of samples."""
        from ace.self_consistency import SelfConsistencyGenerator

        llm = MagicMock()
        generator = SelfConsistencyGenerator(llm)

        self.assertEqual(generator.num_samples, 3)


@pytest.mark.unit
class TestSelfConsistencyGeneration(unittest.TestCase):
    """Test self-consistency generation behavior."""

    def test_generates_multiple_responses(self):
        """Test that generator produces multiple responses internally."""
        from ace import Playbook
        from ace.self_consistency import SelfConsistencyGenerator

        llm = MagicMock()
        # Return valid JSON for each call
        llm.complete.return_value = MagicMock(
            text='{"reasoning": "test", "final_answer": "Paris", "bullet_ids": []}'
        )

        generator = SelfConsistencyGenerator(llm, num_samples=3)
        playbook = Playbook()

        generator.generate(
            question="What is the capital of France?",
            context=None,
            playbook=playbook,
        )

        # Should call LLM 3 times
        self.assertEqual(llm.complete.call_count, 3)

    def test_majority_voting_selects_most_common(self):
        """Test that majority voting selects the most common answer."""
        from ace import Playbook
        from ace.self_consistency import SelfConsistencyGenerator

        llm = MagicMock()
        # Simulate 3 responses: 2 say "Paris", 1 says "London"
        responses = [
            MagicMock(text='{"reasoning": "r1", "final_answer": "Paris", "bullet_ids": []}'),
            MagicMock(text='{"reasoning": "r2", "final_answer": "London", "bullet_ids": []}'),
            MagicMock(text='{"reasoning": "r3", "final_answer": "Paris", "bullet_ids": []}'),
        ]
        llm.complete.side_effect = responses

        generator = SelfConsistencyGenerator(llm, num_samples=3)
        playbook = Playbook()

        result = generator.generate(
            question="What is the capital of France?",
            context=None,
            playbook=playbook,
        )

        # Should select "Paris" (2 votes vs 1)
        self.assertEqual(result.final_answer, "Paris")

    def test_tie_breaking_uses_first_occurrence(self):
        """Test tie-breaking behavior when answers have equal votes."""
        from ace import Playbook
        from ace.self_consistency import SelfConsistencyGenerator

        llm = MagicMock()
        # Simulate 4 responses: 2 say "A", 2 say "B"
        responses = [
            MagicMock(text='{"reasoning": "r1", "final_answer": "A", "bullet_ids": []}'),
            MagicMock(text='{"reasoning": "r2", "final_answer": "B", "bullet_ids": []}'),
            MagicMock(text='{"reasoning": "r3", "final_answer": "A", "bullet_ids": []}'),
            MagicMock(text='{"reasoning": "r4", "final_answer": "B", "bullet_ids": []}'),
        ]
        llm.complete.side_effect = responses

        generator = SelfConsistencyGenerator(llm, num_samples=4)
        playbook = Playbook()

        result = generator.generate(
            question="Test question",
            context=None,
            playbook=playbook,
        )

        # Should select "A" (first to reach max count)
        self.assertEqual(result.final_answer, "A")

    def test_returns_generator_output(self):
        """Test that result is a GeneratorOutput instance."""
        from ace import Playbook
        from ace.roles import GeneratorOutput
        from ace.self_consistency import SelfConsistencyGenerator

        llm = MagicMock()
        llm.complete.return_value = MagicMock(
            text='{"reasoning": "test", "final_answer": "42", "bullet_ids": []}'
        )

        generator = SelfConsistencyGenerator(llm, num_samples=3)
        playbook = Playbook()

        result = generator.generate(
            question="What is 6*7?",
            context=None,
            playbook=playbook,
        )

        self.assertIsInstance(result, GeneratorOutput)


@pytest.mark.unit
class TestSelfConsistencyMetrics(unittest.TestCase):
    """Test self-consistency metrics and confidence."""

    def test_tracks_vote_distribution(self):
        """Test that vote distribution is tracked."""
        from ace import Playbook
        from ace.self_consistency import SelfConsistencyGenerator

        llm = MagicMock()
        responses = [
            MagicMock(text='{"reasoning": "r1", "final_answer": "Paris", "bullet_ids": []}'),
            MagicMock(text='{"reasoning": "r2", "final_answer": "London", "bullet_ids": []}'),
            MagicMock(text='{"reasoning": "r3", "final_answer": "Paris", "bullet_ids": []}'),
        ]
        llm.complete.side_effect = responses

        generator = SelfConsistencyGenerator(llm, num_samples=3)
        playbook = Playbook()

        result = generator.generate(
            question="What is the capital of France?",
            context=None,
            playbook=playbook,
        )

        # Check raw output contains vote info
        self.assertIn("vote_distribution", result.raw)
        self.assertEqual(result.raw["vote_distribution"]["Paris"], 2)
        self.assertEqual(result.raw["vote_distribution"]["London"], 1)

    def test_confidence_score_calculation(self):
        """Test confidence score based on vote unanimity."""
        from ace import Playbook
        from ace.self_consistency import SelfConsistencyGenerator

        llm = MagicMock()
        # All 3 agree
        llm.complete.return_value = MagicMock(
            text='{"reasoning": "test", "final_answer": "Paris", "bullet_ids": []}'
        )

        generator = SelfConsistencyGenerator(llm, num_samples=3)
        playbook = Playbook()

        result = generator.generate(
            question="What is the capital of France?",
            context=None,
            playbook=playbook,
        )

        # Unanimous vote = 100% confidence
        self.assertEqual(result.raw["consistency_confidence"], 1.0)

    def test_lower_confidence_for_split_votes(self):
        """Test lower confidence when votes are split."""
        from ace import Playbook
        from ace.self_consistency import SelfConsistencyGenerator

        llm = MagicMock()
        responses = [
            MagicMock(text='{"reasoning": "r1", "final_answer": "Paris", "bullet_ids": []}'),
            MagicMock(text='{"reasoning": "r2", "final_answer": "London", "bullet_ids": []}'),
            MagicMock(text='{"reasoning": "r3", "final_answer": "Paris", "bullet_ids": []}'),
        ]
        llm.complete.side_effect = responses

        generator = SelfConsistencyGenerator(llm, num_samples=3)
        playbook = Playbook()

        result = generator.generate(
            question="What is the capital of France?",
            context=None,
            playbook=playbook,
        )

        # 2/3 votes = ~0.67 confidence
        self.assertAlmostEqual(result.raw["consistency_confidence"], 2 / 3, places=2)


@pytest.mark.unit
class TestSelfConsistencyTemperature(unittest.TestCase):
    """Test temperature control for diverse sampling."""

    def test_configurable_temperature(self):
        """Test that sampling temperature is configurable."""
        from ace.self_consistency import SelfConsistencyGenerator

        llm = MagicMock()
        generator = SelfConsistencyGenerator(llm, num_samples=3, temperature=0.7)

        self.assertEqual(generator.temperature, 0.7)

    def test_default_temperature(self):
        """Test default temperature for self-consistency."""
        from ace.self_consistency import SelfConsistencyGenerator

        llm = MagicMock()
        generator = SelfConsistencyGenerator(llm, num_samples=3)

        # Self-consistency typically uses higher temperature for diversity
        self.assertEqual(generator.temperature, 0.7)

    def test_temperature_passed_to_llm(self):
        """Test that temperature is passed to LLM calls."""
        from ace import Playbook
        from ace.self_consistency import SelfConsistencyGenerator

        llm = MagicMock()
        llm.complete.return_value = MagicMock(
            text='{"reasoning": "test", "final_answer": "Paris", "bullet_ids": []}'
        )

        generator = SelfConsistencyGenerator(llm, num_samples=3, temperature=0.9)
        playbook = Playbook()

        generator.generate(
            question="Test",
            context=None,
            playbook=playbook,
        )

        # Check temperature was passed
        for call in llm.complete.call_args_list:
            self.assertEqual(call.kwargs.get("temperature"), 0.9)


@pytest.mark.unit
class TestSelfConsistencyNormalization(unittest.TestCase):
    """Test answer normalization for better matching."""

    def test_normalizes_whitespace(self):
        """Test that answers are normalized for whitespace."""
        from ace import Playbook
        from ace.self_consistency import SelfConsistencyGenerator

        llm = MagicMock()
        responses = [
            MagicMock(text='{"reasoning": "r1", "final_answer": "Paris", "bullet_ids": []}'),
            MagicMock(text='{"reasoning": "r2", "final_answer": "  Paris  ", "bullet_ids": []}'),
            MagicMock(text='{"reasoning": "r3", "final_answer": "paris", "bullet_ids": []}'),
        ]
        llm.complete.side_effect = responses

        generator = SelfConsistencyGenerator(llm, num_samples=3, normalize_answers=True)
        playbook = Playbook()

        result = generator.generate(
            question="What is the capital of France?",
            context=None,
            playbook=playbook,
        )

        # All should be counted as same answer after normalization
        self.assertEqual(result.raw["consistency_confidence"], 1.0)

    def test_normalization_disabled_by_default(self):
        """Test that normalization is disabled by default for exact matching."""
        from ace.self_consistency import SelfConsistencyGenerator

        llm = MagicMock()
        generator = SelfConsistencyGenerator(llm, num_samples=3)

        self.assertFalse(generator.normalize_answers)


@pytest.mark.unit
class TestSelfConsistencyErrorHandling(unittest.TestCase):
    """Test error handling in self-consistency sampling."""

    def test_handles_partial_failures(self):
        """Test that partial LLM failures don't break voting."""
        from ace import Playbook
        from ace.self_consistency import SelfConsistencyGenerator

        llm = MagicMock()
        # 2 valid responses, 1 invalid
        responses = [
            MagicMock(text='{"reasoning": "r1", "final_answer": "Paris", "bullet_ids": []}'),
            MagicMock(text="invalid json"),
            MagicMock(text='{"reasoning": "r3", "final_answer": "Paris", "bullet_ids": []}'),
        ]
        llm.complete.side_effect = responses

        generator = SelfConsistencyGenerator(llm, num_samples=3)
        playbook = Playbook()

        result = generator.generate(
            question="What is the capital of France?",
            context=None,
            playbook=playbook,
        )

        # Should still work with 2 valid responses
        self.assertEqual(result.final_answer, "Paris")
        self.assertEqual(result.raw["valid_samples"], 2)

    def test_fails_if_all_samples_fail(self):
        """Test that generation fails if all samples fail."""
        from ace import Playbook
        from ace.self_consistency import SelfConsistencyGenerator

        llm = MagicMock()
        llm.complete.return_value = MagicMock(text="invalid json")

        generator = SelfConsistencyGenerator(llm, num_samples=3)
        playbook = Playbook()

        with self.assertRaises(RuntimeError):
            generator.generate(
                question="Test",
                context=None,
                playbook=playbook,
            )

    def test_minimum_valid_samples_configurable(self):
        """Test that minimum valid samples is configurable."""
        from ace.self_consistency import SelfConsistencyGenerator

        llm = MagicMock()
        generator = SelfConsistencyGenerator(llm, num_samples=5, min_valid_samples=3)

        self.assertEqual(generator.min_valid_samples, 3)


if __name__ == "__main__":
    unittest.main()
