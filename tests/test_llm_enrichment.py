"""Tests for LLM-based bullet enrichment (production-grade).

TDD: These tests define the expected behavior BEFORE implementation.

Production enrichment uses LLM with CURATOR_ENRICHMENT_PROMPT instead of
heuristics for accurate semantic scaffolding.
"""

import json
import unittest
from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
class TestLLMEnrichmentBasic(unittest.TestCase):
    """Test basic LLM-based enrichment functionality."""

    def test_llm_enricher_exists(self):
        """Test that LLMBulletEnricher class exists."""
        from ace.enrichment import LLMBulletEnricher

        self.assertIsNotNone(LLMBulletEnricher)

    def test_llm_enricher_requires_llm_client(self):
        """Test that LLMBulletEnricher requires an LLM client."""
        from ace.enrichment import LLMBulletEnricher

        llm = MagicMock()
        enricher = LLMBulletEnricher(llm)

        self.assertEqual(enricher.llm, llm)

    def test_llm_enricher_has_enrich_method(self):
        """Test that LLMBulletEnricher has enrich method."""
        from ace.enrichment import LLMBulletEnricher

        llm = MagicMock()
        enricher = LLMBulletEnricher(llm)

        self.assertTrue(hasattr(enricher, "enrich"))
        self.assertTrue(callable(enricher.enrich))


@pytest.mark.unit
class TestLLMEnrichmentProcess(unittest.TestCase):
    """Test the LLM enrichment process."""

    def test_enrich_calls_llm_with_prompt(self):
        """Test that enrich calls LLM with enrichment prompt."""
        from ace.enrichment import LLMBulletEnricher
        from ace.playbook import Bullet

        llm = MagicMock()
        llm.complete.return_value = MagicMock(text='''{
            "task_types": ["debugging"],
            "domains": ["python"],
            "complexity_level": "medium",
            "preconditions": ["error present"],
            "trigger_patterns": ["exception"],
            "anti_patterns": ["simple typo"],
            "retrieval_type": "semantic",
            "embedding_text": "debug error exception handling"
        }''')

        enricher = LLMBulletEnricher(llm)
        bullet = Bullet(id="test-001", section="debugging", content="Check stack traces")

        enricher.enrich(bullet, context="Fixed exception in parser")

        # LLM should be called
        llm.complete.assert_called_once()

        # Prompt should contain bullet content and context
        call_args = llm.complete.call_args
        prompt = call_args[0][0]
        self.assertIn("Check stack traces", prompt)
        self.assertIn("Fixed exception in parser", prompt)

    def test_enrich_returns_enriched_bullet(self):
        """Test that enrich returns an EnrichedBullet."""
        from ace.enrichment import LLMBulletEnricher
        from ace.playbook import Bullet, EnrichedBullet

        llm = MagicMock()
        llm.complete.return_value = MagicMock(text='''{
            "task_types": ["debugging", "code_review"],
            "domains": ["python", "general"],
            "complexity_level": "medium",
            "preconditions": ["error message present"],
            "trigger_patterns": ["exception", "traceback"],
            "anti_patterns": ["performance issue"],
            "retrieval_type": "semantic",
            "embedding_text": "debug error exception traceback handling"
        }''')

        enricher = LLMBulletEnricher(llm)
        bullet = Bullet(id="test-001", section="debugging", content="Check stack traces")

        result = enricher.enrich(bullet, context="Fixed exception")

        self.assertIsInstance(result, EnrichedBullet)
        self.assertEqual(result.id, "test-001")
        self.assertEqual(result.content, "Check stack traces")
        self.assertEqual(result.task_types, ["debugging", "code_review"])
        self.assertEqual(result.domains, ["python", "general"])
        self.assertEqual(result.complexity_level, "medium")

    def test_enrich_preserves_bullet_stats(self):
        """Test that enrichment preserves helpful/harmful/neutral counts."""
        from ace.enrichment import LLMBulletEnricher
        from ace.playbook import Bullet

        llm = MagicMock()
        llm.complete.return_value = MagicMock(text='''{
            "task_types": ["debugging"],
            "domains": ["python"],
            "complexity_level": "simple",
            "preconditions": [],
            "trigger_patterns": [],
            "anti_patterns": [],
            "retrieval_type": "semantic",
            "embedding_text": "test"
        }''')

        enricher = LLMBulletEnricher(llm)
        bullet = Bullet(
            id="test-001",
            section="test",
            content="Test content",
            helpful=10,
            harmful=2,
            neutral=5,
        )

        result = enricher.enrich(bullet, context="Test context")

        self.assertEqual(result.helpful, 10)
        self.assertEqual(result.harmful, 2)
        self.assertEqual(result.neutral, 5)


@pytest.mark.unit
class TestLLMEnrichmentOutput(unittest.TestCase):
    """Test LLM enrichment output parsing."""

    def test_parses_all_enrichment_fields(self):
        """Test that all enrichment fields are parsed from LLM output."""
        from ace.enrichment import LLMBulletEnricher
        from ace.playbook import Bullet

        llm = MagicMock()
        llm.complete.return_value = MagicMock(text='''{
            "task_types": ["reasoning", "problem_solving"],
            "domains": ["math", "logic"],
            "complexity_level": "complex",
            "preconditions": ["multi-step problem", "numerical data"],
            "trigger_patterns": ["solve", "calculate", "derive"],
            "anti_patterns": ["simple arithmetic", "lookup only"],
            "retrieval_type": "hybrid",
            "embedding_text": "mathematical reasoning step-by-step problem solving calculation"
        }''')

        enricher = LLMBulletEnricher(llm)
        bullet = Bullet(id="math-001", section="math", content="Break problems into steps")

        result = enricher.enrich(bullet, context="Solved complex equation")

        self.assertEqual(result.task_types, ["reasoning", "problem_solving"])
        self.assertEqual(result.domains, ["math", "logic"])
        self.assertEqual(result.complexity_level, "complex")
        self.assertEqual(result.preconditions, ["multi-step problem", "numerical data"])
        self.assertEqual(result.trigger_patterns, ["solve", "calculate", "derive"])
        self.assertEqual(result.anti_patterns, ["simple arithmetic", "lookup only"])
        self.assertEqual(result.retrieval_type, "hybrid")
        self.assertEqual(result.embedding_text, "mathematical reasoning step-by-step problem solving calculation")

    def test_handles_markdown_code_blocks(self):
        """Test that LLM response with markdown code blocks is handled."""
        from ace.enrichment import LLMBulletEnricher
        from ace.playbook import Bullet

        llm = MagicMock()
        llm.complete.return_value = MagicMock(text='''```json
{
    "task_types": ["debugging"],
    "domains": ["python"],
    "complexity_level": "simple",
    "preconditions": [],
    "trigger_patterns": [],
    "anti_patterns": [],
    "retrieval_type": "semantic",
    "embedding_text": "test"
}
```''')

        enricher = LLMBulletEnricher(llm)
        bullet = Bullet(id="test-001", section="test", content="Test")

        result = enricher.enrich(bullet, context="Test")

        self.assertEqual(result.task_types, ["debugging"])


@pytest.mark.unit
class TestLLMEnrichmentErrorHandling(unittest.TestCase):
    """Test error handling in LLM enrichment."""

    def test_retries_on_json_error(self):
        """Test that enricher retries on JSON parse errors."""
        from ace.enrichment import LLMBulletEnricher
        from ace.playbook import Bullet

        llm = MagicMock()
        llm.complete.side_effect = [
            MagicMock(text="invalid json"),
            MagicMock(text='''{
                "task_types": ["debugging"],
                "domains": ["python"],
                "complexity_level": "simple",
                "preconditions": [],
                "trigger_patterns": [],
                "anti_patterns": [],
                "retrieval_type": "semantic",
                "embedding_text": "test"
            }'''),
        ]

        enricher = LLMBulletEnricher(llm, max_retries=3)
        bullet = Bullet(id="test-001", section="test", content="Test")

        result = enricher.enrich(bullet, context="Test")

        # Should succeed on second try
        self.assertEqual(result.task_types, ["debugging"])
        self.assertEqual(llm.complete.call_count, 2)

    def test_fallback_to_heuristic_on_failure(self):
        """Test fallback to heuristic enrichment when LLM fails."""
        from ace.enrichment import LLMBulletEnricher
        from ace.playbook import Bullet, EnrichedBullet

        llm = MagicMock()
        llm.complete.return_value = MagicMock(text="invalid json always")

        enricher = LLMBulletEnricher(llm, max_retries=2, fallback_to_heuristic=True)
        bullet = Bullet(id="test-001", section="debugging", content="Check errors")

        result = enricher.enrich(bullet, context="debug error fix")

        # Should return heuristic-enriched bullet
        self.assertIsInstance(result, EnrichedBullet)
        # Should have used context keywords
        self.assertIn("debugging", result.task_types)

    def test_raises_on_failure_without_fallback(self):
        """Test that error is raised when fallback is disabled."""
        from ace.enrichment import LLMBulletEnricher
        from ace.playbook import Bullet

        llm = MagicMock()
        llm.complete.return_value = MagicMock(text="invalid json")

        enricher = LLMBulletEnricher(llm, max_retries=2, fallback_to_heuristic=False)
        bullet = Bullet(id="test-001", section="test", content="Test")

        with self.assertRaises(RuntimeError):
            enricher.enrich(bullet, context="Test")


@pytest.mark.unit
class TestLLMEnrichmentBatch(unittest.TestCase):
    """Test batch enrichment functionality."""

    def test_batch_enrich_multiple_bullets(self):
        """Test enriching multiple bullets in batch."""
        from ace.enrichment import LLMBulletEnricher
        from ace.playbook import Bullet

        llm = MagicMock()
        llm.complete.return_value = MagicMock(text='''{
            "task_types": ["general"],
            "domains": ["general"],
            "complexity_level": "simple",
            "preconditions": [],
            "trigger_patterns": [],
            "anti_patterns": [],
            "retrieval_type": "semantic",
            "embedding_text": "general strategy"
        }''')

        enricher = LLMBulletEnricher(llm)
        bullets = [
            Bullet(id="test-001", section="test", content="Content 1"),
            Bullet(id="test-002", section="test", content="Content 2"),
            Bullet(id="test-003", section="test", content="Content 3"),
        ]

        results = enricher.enrich_batch(bullets, context="Batch context")

        self.assertEqual(len(results), 3)
        self.assertEqual(llm.complete.call_count, 3)


@pytest.mark.unit
class TestEnrichBulletFunction(unittest.TestCase):
    """Test the module-level enrich_bullet function with LLM support."""

    def test_enrich_bullet_with_llm(self):
        """Test enrich_bullet can use LLM enricher."""
        from ace.enrichment import enrich_bullet_llm
        from ace.playbook import Bullet, EnrichedBullet

        llm = MagicMock()
        llm.complete.return_value = MagicMock(text='''{
            "task_types": ["debugging"],
            "domains": ["python"],
            "complexity_level": "medium",
            "preconditions": [],
            "trigger_patterns": ["error"],
            "anti_patterns": [],
            "retrieval_type": "semantic",
            "embedding_text": "debug error"
        }''')

        bullet = Bullet(id="test-001", section="test", content="Check logs")
        result = enrich_bullet_llm(bullet, "Error in logs", llm=llm)

        self.assertIsInstance(result, EnrichedBullet)
        self.assertEqual(result.task_types, ["debugging"])


@pytest.mark.unit
class TestEnrichmentIntegration(unittest.TestCase):
    """Test enrichment integration with Playbook."""

    def test_playbook_add_enriched_with_llm(self):
        """Test adding bullet with LLM enrichment to playbook."""
        from ace import Playbook
        from ace.enrichment import LLMBulletEnricher
        from ace.playbook import EnrichedBullet

        llm = MagicMock()
        llm.complete.return_value = MagicMock(text='''{
            "task_types": ["validation"],
            "domains": ["testing"],
            "complexity_level": "simple",
            "preconditions": ["test file exists"],
            "trigger_patterns": ["run test", "verify"],
            "anti_patterns": ["production code"],
            "retrieval_type": "keyword",
            "embedding_text": "test validation verification"
        }''')

        enricher = LLMBulletEnricher(llm)
        playbook = Playbook()

        # Add bullet and enrich with LLM
        bullet = playbook.add_bullet("testing", "Always run tests before commit")
        enriched = enricher.enrich(bullet, context="Test failed before push")

        # Replace in playbook
        playbook._bullets[bullet.id] = enriched

        retrieved = playbook.get_bullet(bullet.id)
        self.assertIsInstance(retrieved, EnrichedBullet)
        self.assertEqual(retrieved.task_types, ["validation"])


if __name__ == "__main__":
    unittest.main()
