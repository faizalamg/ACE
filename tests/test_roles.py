"""Unit tests for Generator, Reflector, and Curator roles.

NOTE: Tests requiring LLM will be skipped if no API key is available.
All tests use REAL implementations - NO MOCKING/FAKING/STUBBING.
"""

import unittest
import os
from pathlib import Path
import tempfile

import pytest

from ace import Generator, Reflector, Curator, Playbook
from ace.roles import (
    _safe_json_loads,
    GeneratorOutput,
    ReflectorOutput,
    CuratorOutput,
    BulletTag,
)


def check_llm_available():
    """Check if an LLM API is available."""
    return bool(
        os.getenv("ZAI_API_KEY") or
        os.getenv("OPENAI_API_KEY") or
        os.getenv("ANTHROPIC_API_KEY")
    )


def get_test_llm_client():
    """Get a properly configured LLM client for tests."""
    from ace.llm_providers.litellm_client import LiteLLMClient
    from ace.config import get_llm_config
    config = get_llm_config()
    return LiteLLMClient(
        model=config.model,
        api_base=config.api_base,
        api_key=config.api_key,
        max_tokens=2048  # Ensure responses aren't truncated
    )


LLM_AVAILABLE = check_llm_available()


@pytest.mark.unit
class TestSafeJsonLoads(unittest.TestCase):
    """Test JSON parsing utility with edge cases - no LLM required."""

    def test_valid_json(self):
        """Test parsing valid JSON."""
        json_str = '{"key": "value", "number": 42}'
        result = _safe_json_loads(json_str)
        self.assertEqual(result["key"], "value")
        self.assertEqual(result["number"], 42)

    def test_json_with_markdown_fences_json_lang(self):
        """Test stripping ```json markdown fences."""
        json_str = '```json\n{"key": "value"}\n```'
        result = _safe_json_loads(json_str)
        self.assertEqual(result["key"], "value")

    def test_json_with_generic_markdown_fences(self):
        """Test stripping generic ``` markdown fences."""
        json_str = '```\n{"key": "value"}\n```'
        result = _safe_json_loads(json_str)
        self.assertEqual(result["key"], "value")

    def test_json_with_only_closing_fence(self):
        """Test JSON with only closing fence."""
        json_str = '{"key": "value"}\n```'
        result = _safe_json_loads(json_str)
        self.assertEqual(result["key"], "value")

    def test_json_with_whitespace(self):
        """Test JSON with leading/trailing whitespace."""
        json_str = '  \n  {"key": "value"}  \n  '
        result = _safe_json_loads(json_str)
        self.assertEqual(result["key"], "value")

    def test_invalid_json_raises_value_error(self):
        """Test that invalid JSON raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            _safe_json_loads("This is not JSON")
        self.assertIn("not valid JSON", str(ctx.exception))

    def test_non_dict_json_raises_value_error(self):
        """Test that non-dict JSON (array, string) raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            _safe_json_loads('["array", "not", "object"]')
        self.assertIn("Expected a JSON object", str(ctx.exception))

    def test_truncated_json_detection_unmatched_braces(self):
        """Test detection of truncated JSON with unmatched braces."""
        truncated = '{"key": "value", "incomplete": {'
        with self.assertRaises(ValueError) as ctx:
            _safe_json_loads(truncated)
        self.assertIn("truncated", str(ctx.exception).lower())

    def test_debug_logging_on_failure(self):
        """Test that invalid JSON is logged to debug file."""
        # Clean up any existing debug log
        debug_path = Path("logs/json_failures.log")
        if debug_path.exists():
            debug_path.unlink()

        with self.assertRaises(ValueError):
            _safe_json_loads("Invalid JSON!")

        # Verify debug log was created
        self.assertTrue(debug_path.exists())


@pytest.mark.unit
@pytest.mark.skipif(not LLM_AVAILABLE, reason="No LLM API key available")
class TestGenerator(unittest.TestCase):
    """Test Generator role with REAL LLM."""

    def setUp(self):
        """Set up test fixtures."""
        self.playbook = Playbook()
        self.llm = get_test_llm_client()

    def test_generate_basic(self):
        """Test basic generation with real LLM."""
        generator = Generator(self.llm)
        output = generator.generate(
            question="What is 2 + 2?",
            context="Simple arithmetic",
            playbook=self.playbook,
        )

        self.assertIsNotNone(output.final_answer)
        self.assertIsNotNone(output.reasoning)
        # The answer should contain "4" somewhere
        self.assertIn("4", output.final_answer)

    def test_generate_with_playbook_bullets(self):
        """Test generation uses bullets from playbook."""
        bullet = self.playbook.add_bullet(
            "math", "Show your work step by step", bullet_id="math-001"
        )

        generator = Generator(self.llm)
        output = generator.generate(
            question="What is 15 * 3?",
            context="Calculate step by step",
            playbook=self.playbook,
        )

        self.assertIsNotNone(output.final_answer)
        # Should get the correct answer
        self.assertIn("45", output.final_answer)

    def test_generate_with_reflection(self):
        """Test generation with reflection from previous attempt."""
        generator = Generator(self.llm)
        output = generator.generate(
            question="What is 10 / 2?",
            context="Division",
            playbook=self.playbook,
            reflection="Make sure to show the calculation step by step",
        )

        self.assertIsNotNone(output.final_answer)
        self.assertIn("5", output.final_answer)


@pytest.mark.unit
@pytest.mark.skipif(not LLM_AVAILABLE, reason="No LLM API key available")
class TestReflector(unittest.TestCase):
    """Test Reflector role with REAL LLM."""

    def setUp(self):
        """Set up test fixtures."""
        self.playbook = Playbook()
        self.llm = get_test_llm_client()

    def test_reflect_basic(self):
        """Test basic reflection with real LLM."""
        reflector = Reflector(self.llm)
        generator_output = GeneratorOutput(
            reasoning="2+2 equals 4 because addition combines quantities",
            final_answer="4",
            bullet_ids=[],
            raw={},
        )

        reflection = reflector.reflect(
            question="What is 2+2?",
            generator_output=generator_output,
            playbook=self.playbook,
            ground_truth="4",
            feedback="Correct!",
        )

        self.assertIsNotNone(reflection.reasoning)

    def test_reflect_with_incorrect_answer(self):
        """Test reflection when answer is wrong."""
        reflector = Reflector(self.llm)
        generator_output = GeneratorOutput(
            reasoning="2+2 equals 5",
            final_answer="5",
            bullet_ids=[],
            raw={},
        )

        reflection = reflector.reflect(
            question="What is 2+2?",
            generator_output=generator_output,
            playbook=self.playbook,
            ground_truth="4",
            feedback="Incorrect - the answer should be 4",
        )

        self.assertIsNotNone(reflection.reasoning)


@pytest.mark.unit
@pytest.mark.skipif(not LLM_AVAILABLE, reason="No LLM API key available")
class TestCurator(unittest.TestCase):
    """Test Curator role with REAL LLM."""

    def setUp(self):
        """Set up test fixtures."""
        self.playbook = Playbook()
        self.llm = get_test_llm_client()

    def test_curate_basic(self):
        """Test basic curation with real LLM."""
        curator = Curator(self.llm)
        reflection = ReflectorOutput(
            reasoning="Missing verification step led to error",
            error_identification="Did not verify calculation",
            root_cause_analysis="Skipped verification",
            correct_approach="Should verify",
            key_insight="Always verify calculations",
            bullet_tags=[],
            raw={},
        )

        curator_output = curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context="Math problem",
            progress="1/10",
        )

        self.assertIsNotNone(curator_output.delta)

    def test_curate_empty_playbook(self):
        """Test curation suggests additions for empty playbook."""
        curator = Curator(self.llm)
        reflection = ReflectorOutput(
            reasoning="Need a strategy for verification",
            error_identification="",
            root_cause_analysis="",
            correct_approach="Add verification bullet",
            key_insight="Verification is important",
            bullet_tags=[],
            raw={},
        )

        curator_output = curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context="Test",
            progress="1/1",
        )

        self.assertIsNotNone(curator_output.delta)


class TestExtractCitedBulletIds(unittest.TestCase):
    """Test bullet ID extraction utility - no LLM required."""

    def test_extract_single_id(self):
        """Extract single bullet ID."""
        from ace.roles import extract_cited_bullet_ids

        text = "Following [general-00042], I will proceed."
        result = extract_cited_bullet_ids(text)
        self.assertEqual(result, ["general-00042"])

    def test_extract_multiple_ids(self):
        """Extract multiple IDs in order."""
        from ace.roles import extract_cited_bullet_ids

        text = "Using [general-00042] and [geo-00003] strategies."
        result = extract_cited_bullet_ids(text)
        self.assertEqual(result, ["general-00042", "geo-00003"])

    def test_deduplicate_preserving_order(self):
        """Deduplicate while preserving first occurrence."""
        from ace.roles import extract_cited_bullet_ids

        text = "Start with [id-001], then [id-002], revisit [id-001]."
        result = extract_cited_bullet_ids(text)
        self.assertEqual(result, ["id-001", "id-002"])

    def test_no_ids_found(self):
        """Return empty list when no IDs."""
        from ace.roles import extract_cited_bullet_ids

        text = "This has no bullet citations at all."
        result = extract_cited_bullet_ids(text)
        self.assertEqual(result, [])

    def test_mixed_with_noise(self):
        """Extract IDs ignoring other bracketed content."""
        from ace.roles import extract_cited_bullet_ids

        text = "Use [strategy-123] but not [this is not an id] or [123]."
        result = extract_cited_bullet_ids(text)
        self.assertEqual(result, ["strategy-123"])

    def test_various_section_names(self):
        """Handle different section naming conventions."""
        from ace.roles import extract_cited_bullet_ids

        text = "[general-001] [content_extraction-042] [API_calls-999]"
        result = extract_cited_bullet_ids(text)
        self.assertEqual(
            result, ["general-001", "content_extraction-042", "API_calls-999"]
        )

    def test_empty_string(self):
        """Handle empty input."""
        from ace.roles import extract_cited_bullet_ids

        self.assertEqual(extract_cited_bullet_ids(""), [])

    def test_multiline_text(self):
        """Extract from multiline text."""
        from ace.roles import extract_cited_bullet_ids

        text = """
        Step 1: Following [setup-001], initialize.
        Step 2: Apply [process-042] for data.
        Step 3: Using [setup-001] again.
        """
        result = extract_cited_bullet_ids(text)
        self.assertEqual(result, ["setup-001", "process-042"])


if __name__ == "__main__":
    unittest.main()
