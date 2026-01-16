"""Tests for Chain-of-Verification (CoVe) in Reflector.

TDD: These tests define the expected behavior BEFORE implementation.

Chain-of-Verification generates verification questions about the initial
response, answers them independently, and uses the answers to refine
the final output, improving accuracy through self-verification.

Reference: Dhuliawala et al., "Chain-of-Verification Reduces Hallucination"
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestCoVeReflectorBasic(unittest.TestCase):
    """Test basic CoVe Reflector functionality."""

    def test_cove_reflector_exists(self):
        """Test that CoVeReflector class exists."""
        from ace.chain_of_verification import CoVeReflector

        self.assertIsNotNone(CoVeReflector)

    def test_cove_reflector_has_reflect_method(self):
        """Test that CoVeReflector has reflect method."""
        from ace.chain_of_verification import CoVeReflector

        llm = MagicMock()
        reflector = CoVeReflector(llm)

        self.assertTrue(hasattr(reflector, "reflect"))
        self.assertTrue(callable(reflector.reflect))

    def test_cove_configurable_questions(self):
        """Test that number of verification questions is configurable."""
        from ace.chain_of_verification import CoVeReflector

        llm = MagicMock()
        reflector = CoVeReflector(llm, num_questions=5)

        self.assertEqual(reflector.num_questions, 5)

    def test_default_num_questions(self):
        """Test default number of verification questions."""
        from ace.chain_of_verification import CoVeReflector

        llm = MagicMock()
        reflector = CoVeReflector(llm)

        self.assertEqual(reflector.num_questions, 3)


@pytest.mark.unit
class TestCoVeVerificationProcess(unittest.TestCase):
    """Test the verification question generation and answering."""

    def test_generates_verification_questions(self):
        """Test that verification questions are generated."""
        from ace import Playbook
        from ace.chain_of_verification import CoVeReflector
        from ace.roles import GeneratorOutput

        llm = MagicMock()
        # First call: initial reflection
        # Second call: generate questions
        # Third call: answer questions and refine
        llm.complete.side_effect = [
            # Initial reflection
            MagicMock(text='''{
                "reasoning": "Initial analysis",
                "error_identification": "None",
                "root_cause_analysis": "N/A",
                "correct_approach": "N/A",
                "key_insight": "Answer is correct",
                "bullet_tags": []
            }'''),
            # Verification questions
            MagicMock(text='''{
                "verification_questions": [
                    "Is the reasoning logically sound?",
                    "Are all facts correct?",
                    "Is the conclusion supported?"
                ]
            }'''),
            # Verified answers
            MagicMock(text='''{
                "verified_answers": [
                    {"question": "Is the reasoning logically sound?", "answer": "Yes", "confidence": 0.9},
                    {"question": "Are all facts correct?", "answer": "Yes", "confidence": 0.85},
                    {"question": "Is the conclusion supported?", "answer": "Yes", "confidence": 0.95}
                ]
            }'''),
            # Final refined reflection
            MagicMock(text='''{
                "reasoning": "Refined analysis after verification",
                "error_identification": "None",
                "root_cause_analysis": "N/A",
                "correct_approach": "N/A",
                "key_insight": "Answer is correct and verified",
                "bullet_tags": []
            }'''),
        ]

        reflector = CoVeReflector(llm, num_questions=3)
        playbook = Playbook()
        gen_output = GeneratorOutput(
            reasoning="2+2=4",
            final_answer="4",
            bullet_ids=[],
            raw={},
        )

        result = reflector.reflect(
            question="What is 2+2?",
            generator_output=gen_output,
            playbook=playbook,
            ground_truth="4",
            feedback="Correct",
        )

        # Should have verification metadata
        self.assertIn("verification_questions", result.raw)
        self.assertEqual(len(result.raw["verification_questions"]), 3)

    def test_verification_improves_reflection(self):
        """Test that verification leads to improved reflection."""
        from ace import Playbook
        from ace.chain_of_verification import CoVeReflector
        from ace.roles import GeneratorOutput

        llm = MagicMock()
        llm.complete.side_effect = [
            # Initial reflection (with potential issue)
            MagicMock(text='''{
                "reasoning": "Initial analysis - seems correct",
                "error_identification": "None",
                "root_cause_analysis": "N/A",
                "correct_approach": "N/A",
                "key_insight": "Calculation correct",
                "bullet_tags": []
            }'''),
            # Verification questions
            MagicMock(text='''{
                "verification_questions": [
                    "Was the multiplication performed correctly?"
                ]
            }'''),
            # Verified answers reveal issue
            MagicMock(text='''{
                "verified_answers": [
                    {"question": "Was the multiplication performed correctly?", "answer": "No, 6*7=42 not 43", "confidence": 0.99}
                ]
            }'''),
            # Refined reflection with correction
            MagicMock(text='''{
                "reasoning": "Refined: Calculation error detected",
                "error_identification": "Arithmetic error: 6*7 was computed as 43 instead of 42",
                "root_cause_analysis": "Off-by-one error in multiplication",
                "correct_approach": "6*7=42",
                "key_insight": "Always verify arithmetic",
                "bullet_tags": []
            }'''),
        ]

        reflector = CoVeReflector(llm, num_questions=1)
        playbook = Playbook()
        gen_output = GeneratorOutput(
            reasoning="6*7=43",
            final_answer="43",
            bullet_ids=[],
            raw={},
        )

        result = reflector.reflect(
            question="What is 6*7?",
            generator_output=gen_output,
            playbook=playbook,
            ground_truth="42",
            feedback="Incorrect",
        )

        # Should have error identification after verification
        self.assertIn("error", result.error_identification.lower())


@pytest.mark.unit
class TestCoVeReflectorOutput(unittest.TestCase):
    """Test CoVe Reflector output structure."""

    def test_returns_reflector_output(self):
        """Test that result is a ReflectorOutput instance."""
        from ace import Playbook
        from ace.chain_of_verification import CoVeReflector
        from ace.roles import GeneratorOutput, ReflectorOutput

        llm = MagicMock()
        llm.complete.side_effect = [
            MagicMock(text='''{
                "reasoning": "Analysis",
                "error_identification": "None",
                "root_cause_analysis": "N/A",
                "correct_approach": "N/A",
                "key_insight": "Correct",
                "bullet_tags": []
            }'''),
            MagicMock(text='{"verification_questions": ["Q1"]}'),
            MagicMock(text='{"verified_answers": [{"question": "Q1", "answer": "Yes", "confidence": 0.9}]}'),
            MagicMock(text='''{
                "reasoning": "Refined",
                "error_identification": "None",
                "root_cause_analysis": "N/A",
                "correct_approach": "N/A",
                "key_insight": "Verified correct",
                "bullet_tags": []
            }'''),
        ]

        reflector = CoVeReflector(llm, num_questions=1)
        playbook = Playbook()
        gen_output = GeneratorOutput(
            reasoning="test",
            final_answer="test",
            bullet_ids=[],
            raw={},
        )

        result = reflector.reflect(
            question="Test",
            generator_output=gen_output,
            playbook=playbook,
        )

        self.assertIsInstance(result, ReflectorOutput)

    def test_raw_contains_verification_metadata(self):
        """Test that raw output contains verification metadata."""
        from ace import Playbook
        from ace.chain_of_verification import CoVeReflector
        from ace.roles import GeneratorOutput

        llm = MagicMock()
        llm.complete.side_effect = [
            MagicMock(text='''{
                "reasoning": "Analysis",
                "error_identification": "None",
                "root_cause_analysis": "N/A",
                "correct_approach": "N/A",
                "key_insight": "Correct",
                "bullet_tags": []
            }'''),
            MagicMock(text='{"verification_questions": ["Is this correct?"]}'),
            MagicMock(text='{"verified_answers": [{"question": "Is this correct?", "answer": "Yes", "confidence": 0.95}]}'),
            MagicMock(text='''{
                "reasoning": "Refined",
                "error_identification": "None",
                "root_cause_analysis": "N/A",
                "correct_approach": "N/A",
                "key_insight": "Verified",
                "bullet_tags": []
            }'''),
        ]

        reflector = CoVeReflector(llm, num_questions=1)
        playbook = Playbook()
        gen_output = GeneratorOutput(
            reasoning="test",
            final_answer="test",
            bullet_ids=[],
            raw={},
        )

        result = reflector.reflect(
            question="Test",
            generator_output=gen_output,
            playbook=playbook,
        )

        self.assertIn("chain_of_verification", result.raw)
        self.assertTrue(result.raw["chain_of_verification"])
        self.assertIn("verification_questions", result.raw)
        self.assertIn("verified_answers", result.raw)


@pytest.mark.unit
class TestCoVeConfidence(unittest.TestCase):
    """Test verification confidence tracking."""

    def test_tracks_verification_confidence(self):
        """Test that verification confidence is tracked."""
        from ace import Playbook
        from ace.chain_of_verification import CoVeReflector
        from ace.roles import GeneratorOutput

        llm = MagicMock()
        llm.complete.side_effect = [
            MagicMock(text='''{
                "reasoning": "Analysis",
                "error_identification": "None",
                "root_cause_analysis": "N/A",
                "correct_approach": "N/A",
                "key_insight": "Correct",
                "bullet_tags": []
            }'''),
            MagicMock(text='{"verification_questions": ["Q1", "Q2"]}'),
            MagicMock(text='''{
                "verified_answers": [
                    {"question": "Q1", "answer": "Yes", "confidence": 0.9},
                    {"question": "Q2", "answer": "Yes", "confidence": 0.8}
                ]
            }'''),
            MagicMock(text='''{
                "reasoning": "Refined",
                "error_identification": "None",
                "root_cause_analysis": "N/A",
                "correct_approach": "N/A",
                "key_insight": "Verified",
                "bullet_tags": []
            }'''),
        ]

        reflector = CoVeReflector(llm, num_questions=2)
        playbook = Playbook()
        gen_output = GeneratorOutput(
            reasoning="test",
            final_answer="test",
            bullet_ids=[],
            raw={},
        )

        result = reflector.reflect(
            question="Test",
            generator_output=gen_output,
            playbook=playbook,
        )

        # Should have average confidence
        self.assertIn("verification_confidence", result.raw)
        self.assertAlmostEqual(result.raw["verification_confidence"], 0.85, places=2)


@pytest.mark.unit
class TestCoVeSkipOption(unittest.TestCase):
    """Test option to skip verification for simple cases."""

    def test_can_skip_verification(self):
        """Test that verification can be skipped."""
        from ace.chain_of_verification import CoVeReflector

        llm = MagicMock()
        reflector = CoVeReflector(llm, skip_verification=True)

        self.assertTrue(reflector.skip_verification)

    def test_skip_when_high_confidence(self):
        """Test that verification is skipped when initial confidence is high."""
        from ace import Playbook
        from ace.chain_of_verification import CoVeReflector
        from ace.roles import GeneratorOutput

        llm = MagicMock()
        # Only one call when skipping
        llm.complete.return_value = MagicMock(text='''{
            "reasoning": "High confidence analysis",
            "error_identification": "None",
            "root_cause_analysis": "N/A",
            "correct_approach": "N/A",
            "key_insight": "Clearly correct",
            "bullet_tags": [],
            "initial_confidence": 0.99
        }''')

        reflector = CoVeReflector(llm, skip_verification=True, confidence_threshold=0.95)
        playbook = Playbook()
        gen_output = GeneratorOutput(
            reasoning="2+2=4",
            final_answer="4",
            bullet_ids=[],
            raw={},
        )

        result = reflector.reflect(
            question="What is 2+2?",
            generator_output=gen_output,
            playbook=playbook,
            ground_truth="4",
            feedback="Correct",
        )

        # Should only call LLM once (no verification)
        self.assertEqual(llm.complete.call_count, 1)
        self.assertIn("verification_skipped", result.raw)


@pytest.mark.unit
class TestCoVeErrorHandling(unittest.TestCase):
    """Test error handling in CoVe process."""

    def test_handles_verification_failure(self):
        """Test graceful handling when verification fails."""
        from ace import Playbook
        from ace.chain_of_verification import CoVeReflector
        from ace.roles import GeneratorOutput

        llm = MagicMock()
        llm.complete.side_effect = [
            # Initial reflection succeeds
            MagicMock(text='''{
                "reasoning": "Analysis",
                "error_identification": "None",
                "root_cause_analysis": "N/A",
                "correct_approach": "N/A",
                "key_insight": "Correct",
                "bullet_tags": []
            }'''),
            # Question generation fails
            MagicMock(text="invalid json"),
        ]

        reflector = CoVeReflector(llm, num_questions=1)
        playbook = Playbook()
        gen_output = GeneratorOutput(
            reasoning="test",
            final_answer="test",
            bullet_ids=[],
            raw={},
        )

        # Should fall back to initial reflection
        result = reflector.reflect(
            question="Test",
            generator_output=gen_output,
            playbook=playbook,
        )

        self.assertIsNotNone(result)
        self.assertIn("verification_error", result.raw)

    def test_configurable_fallback_behavior(self):
        """Test that fallback behavior is configurable."""
        from ace.chain_of_verification import CoVeReflector

        llm = MagicMock()
        reflector = CoVeReflector(llm, fallback_on_error=False)

        self.assertFalse(reflector.fallback_on_error)


if __name__ == "__main__":
    unittest.main()
