"""Tests for Curator enrichment pipeline that adds semantic scaffolding to bullets.

TDD: These tests define the expected behavior BEFORE implementation.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestCuratorEnrichmentPrompt(unittest.TestCase):
    """Test the enrichment prompt template for Curator."""

    def test_enrichment_prompt_exists(self):
        """Test that enrichment prompt template exists in prompts module."""
        from ace.prompts_v2_1 import CURATOR_ENRICHMENT_PROMPT

        self.assertIsInstance(CURATOR_ENRICHMENT_PROMPT, str)
        self.assertIn("{content}", CURATOR_ENRICHMENT_PROMPT)
        self.assertIn("{context}", CURATOR_ENRICHMENT_PROMPT)

    def test_enrichment_prompt_requests_all_scaffolding_fields(self):
        """Test that prompt requests all semantic scaffolding fields."""
        from ace.prompts_v2_1 import CURATOR_ENRICHMENT_PROMPT

        # Should request dimensional metadata
        self.assertIn("task_types", CURATOR_ENRICHMENT_PROMPT)
        self.assertIn("domains", CURATOR_ENRICHMENT_PROMPT)
        self.assertIn("complexity_level", CURATOR_ENRICHMENT_PROMPT)

        # Should request structural metadata
        self.assertIn("preconditions", CURATOR_ENRICHMENT_PROMPT)
        self.assertIn("trigger_patterns", CURATOR_ENRICHMENT_PROMPT)
        self.assertIn("anti_patterns", CURATOR_ENRICHMENT_PROMPT)

        # Should request retrieval hints
        self.assertIn("retrieval_type", CURATOR_ENRICHMENT_PROMPT)
        self.assertIn("embedding_text", CURATOR_ENRICHMENT_PROMPT)


@pytest.mark.unit
class TestDeltaOperationEnrichment(unittest.TestCase):
    """Test DeltaOperation with enrichment metadata."""

    def test_delta_operation_accepts_enrichment(self):
        """Test DeltaOperation can store enrichment metadata."""
        from ace.delta import DeltaOperation

        op = DeltaOperation(
            type="ADD",
            section="test",
            content="Test content",
            enrichment={
                "task_types": ["reasoning"],
                "domains": ["math"],
                "complexity_level": "medium",
            }
        )

        self.assertEqual(op.enrichment["task_types"], ["reasoning"])
        self.assertEqual(op.enrichment["domains"], ["math"])

    def test_delta_operation_enrichment_defaults_to_none(self):
        """Test DeltaOperation enrichment is None by default."""
        from ace.delta import DeltaOperation

        op = DeltaOperation(
            type="ADD",
            section="test",
            content="Test content",
        )

        self.assertIsNone(op.enrichment)

    def test_delta_operation_serialization_includes_enrichment(self):
        """Test DeltaOperation serializes enrichment correctly."""
        from ace.delta import DeltaOperation
        from dataclasses import asdict

        op = DeltaOperation(
            type="ADD",
            section="test",
            content="Test content",
            enrichment={
                "task_types": ["debugging"],
                "trigger_patterns": ["error", "exception"],
            }
        )

        data = asdict(op)
        self.assertIn("enrichment", data)
        self.assertEqual(data["enrichment"]["task_types"], ["debugging"])


@pytest.mark.unit
class TestPlaybookApplyEnrichedDelta(unittest.TestCase):
    """Test Playbook.apply_delta with enrichment metadata."""

    def test_apply_delta_with_enrichment_creates_enriched_bullet(self):
        """Test that delta with enrichment creates EnrichedBullet."""
        from ace import Playbook
        from ace.playbook import EnrichedBullet
        from ace.delta import DeltaOperation, DeltaBatch

        playbook = Playbook()

        delta = DeltaBatch(
            reasoning="Test",
            operations=[
                DeltaOperation(
                    type="ADD",
                    section="math",
                    content="Use step-by-step reasoning",
                    enrichment={
                        "task_types": ["reasoning", "problem_solving"],
                        "domains": ["math", "logic"],
                        "complexity_level": "complex",
                        "preconditions": ["multi-step problem"],
                        "trigger_patterns": ["solve", "calculate", "derive"],
                        "anti_patterns": ["simple arithmetic"],
                        "retrieval_type": "semantic",
                        "embedding_text": "mathematical reasoning step by step problem solving",
                    }
                )
            ]
        )

        playbook.apply_delta(delta)

        bullet = playbook.bullets()[0]
        self.assertIsInstance(bullet, EnrichedBullet)
        self.assertEqual(bullet.task_types, ["reasoning", "problem_solving"])
        self.assertEqual(bullet.domains, ["math", "logic"])
        self.assertEqual(bullet.complexity_level, "complex")
        self.assertEqual(bullet.preconditions, ["multi-step problem"])
        self.assertEqual(bullet.retrieval_type, "semantic")

    def test_apply_delta_without_enrichment_creates_basic_bullet(self):
        """Test that delta without enrichment creates basic Bullet."""
        from ace import Playbook
        from ace.playbook import Bullet, EnrichedBullet
        from ace.delta import DeltaOperation, DeltaBatch

        playbook = Playbook()

        delta = DeltaBatch(
            reasoning="Test",
            operations=[
                DeltaOperation(
                    type="ADD",
                    section="general",
                    content="Be concise",
                    # No enrichment
                )
            ]
        )

        playbook.apply_delta(delta)

        bullet = playbook.bullets()[0]
        # Should be basic Bullet, not EnrichedBullet
        self.assertIsInstance(bullet, Bullet)
        self.assertNotIsInstance(bullet, EnrichedBullet)

    def test_apply_delta_update_preserves_enrichment(self):
        """Test that UPDATE operation preserves existing enrichment."""
        from ace import Playbook
        from ace.playbook import EnrichedBullet
        from ace.delta import DeltaOperation, DeltaBatch

        playbook = Playbook()

        # Add enriched bullet
        playbook.add_enriched_bullet(
            section="test",
            content="Original content",
            task_types=["debugging"],
            domains=["python"],
        )
        bullet_id = playbook.bullets()[0].id

        # Update content only
        delta = DeltaBatch(
            reasoning="Update",
            operations=[
                DeltaOperation(
                    type="UPDATE",
                    section="test",
                    bullet_id=bullet_id,
                    content="Updated content",
                    # No enrichment in update - should preserve existing
                )
            ]
        )

        playbook.apply_delta(delta)

        bullet = playbook.get_bullet(bullet_id)
        self.assertIsInstance(bullet, EnrichedBullet)
        self.assertEqual(bullet.content, "Updated content")
        # Enrichment should be preserved
        self.assertEqual(bullet.task_types, ["debugging"])
        self.assertEqual(bullet.domains, ["python"])


if __name__ == "__main__":
    unittest.main()
