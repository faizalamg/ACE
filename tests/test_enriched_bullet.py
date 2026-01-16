"""Tests for EnrichedBullet with semantic scaffolding metadata.

TDD: These tests define the expected behavior BEFORE implementation.
"""

import json
import tempfile
import unittest
from pathlib import Path

import pytest


@pytest.mark.unit
class TestEnrichedBullet(unittest.TestCase):
    """Test EnrichedBullet class with semantic scaffolding."""

    def test_enriched_bullet_creation_with_defaults(self):
        """Test creating enriched bullet with default scaffolding values."""
        from ace.playbook import EnrichedBullet

        bullet = EnrichedBullet(
            id="math-00001",
            section="math",
            content="Break complex problems into smaller steps",
        )

        # Core fields
        self.assertEqual(bullet.id, "math-00001")
        self.assertEqual(bullet.section, "math")
        self.assertEqual(bullet.content, "Break complex problems into smaller steps")

        # Effectiveness metrics (defaults)
        self.assertEqual(bullet.helpful, 0)
        self.assertEqual(bullet.harmful, 0)
        self.assertEqual(bullet.neutral, 0)
        self.assertEqual(bullet.confidence, 0.0)

        # Semantic scaffolding (defaults)
        self.assertEqual(bullet.task_types, [])
        self.assertEqual(bullet.domains, [])
        self.assertEqual(bullet.complexity_level, "medium")
        self.assertEqual(bullet.preconditions, [])
        self.assertEqual(bullet.trigger_patterns, [])
        self.assertEqual(bullet.anti_patterns, [])

        # Relational metadata (defaults)
        self.assertEqual(bullet.related_bullets, [])
        self.assertIsNone(bullet.supersedes)
        self.assertIsNone(bullet.derived_from)

        # Usage context (defaults)
        self.assertEqual(bullet.successful_contexts, [])
        self.assertEqual(bullet.failure_contexts, [])

        # Retrieval hints (defaults)
        self.assertEqual(bullet.retrieval_type, "semantic")
        self.assertIsNone(bullet.embedding_text)

    def test_enriched_bullet_full_scaffolding(self):
        """Test creating enriched bullet with full semantic scaffolding."""
        from ace.playbook import EnrichedBullet

        bullet = EnrichedBullet(
            id="debug-00005",
            section="debugging",
            content="Check for null pointer exceptions before accessing object properties",
            helpful=15,
            harmful=2,
            neutral=3,
            confidence=0.85,
            # Dimensional metadata
            task_types=["debugging", "code_review"],
            domains=["python", "java", "typescript"],
            complexity_level="simple",
            # Structural metadata
            preconditions=["error message mentions null", "accessing object property"],
            trigger_patterns=["NullPointerException", "TypeError: Cannot read", "AttributeError"],
            anti_patterns=["primitive types", "already null-checked"],
            # Relational metadata
            related_bullets=["debug-00003", "validation-00012"],
            supersedes="debug-00001",
            derived_from=None,
            # Usage context
            successful_contexts=["Fixed NPE in user service", "Prevented crash in payment flow"],
            failure_contexts=["False positive on Optional types"],
            # Retrieval hints
            retrieval_type="keyword",
            embedding_text="null pointer exception check object property access validation",
        )

        # Verify all fields
        self.assertEqual(bullet.task_types, ["debugging", "code_review"])
        self.assertEqual(bullet.domains, ["python", "java", "typescript"])
        self.assertEqual(bullet.complexity_level, "simple")
        self.assertEqual(bullet.preconditions, ["error message mentions null", "accessing object property"])
        self.assertEqual(len(bullet.trigger_patterns), 3)
        self.assertEqual(bullet.anti_patterns, ["primitive types", "already null-checked"])
        self.assertEqual(bullet.related_bullets, ["debug-00003", "validation-00012"])
        self.assertEqual(bullet.supersedes, "debug-00001")
        self.assertEqual(bullet.successful_contexts, ["Fixed NPE in user service", "Prevented crash in payment flow"])
        self.assertEqual(bullet.failure_contexts, ["False positive on Optional types"])
        self.assertEqual(bullet.retrieval_type, "keyword")
        self.assertEqual(bullet.embedding_text, "null pointer exception check object property access validation")
        self.assertEqual(bullet.confidence, 0.85)

    def test_enriched_bullet_to_llm_dict(self):
        """Test that to_llm_dict includes scaffolding fields useful for LLM."""
        from ace.playbook import EnrichedBullet

        bullet = EnrichedBullet(
            id="test-001",
            section="test",
            content="Test strategy",
            helpful=5,
            harmful=1,
            task_types=["reasoning"],
            domains=["math"],
            trigger_patterns=["calculate", "compute"],
            anti_patterns=["creative writing"],
            confidence=0.75,
        )

        llm_dict = bullet.to_llm_dict()

        # Core fields (compressed TOON format)
        self.assertEqual(llm_dict["i"], "test-001")  # id -> i
        self.assertEqual(llm_dict["c"], "Test strategy")  # content -> c
        self.assertEqual(llm_dict["h"], 5)  # helpful -> h
        self.assertEqual(llm_dict["x"], 1)  # harmful -> x

        # Should include retrieval-relevant scaffolding (compressed)
        self.assertEqual(llm_dict["tt"], ["reasoning"])  # task_types -> tt
        self.assertEqual(llm_dict["dm"], ["math"])  # domains -> dm
        self.assertEqual(llm_dict["tp"], ["calculate", "compute"])  # trigger_patterns -> tp
        self.assertEqual(llm_dict["cf"], 0.75)  # confidence -> cf (only included if != 0.5)

        # Should NOT include internal metadata
        self.assertNotIn("created_at", llm_dict)
        self.assertNotIn("updated_at", llm_dict)
        self.assertNotIn("embedding_text", llm_dict)  # Internal retrieval optimization
        self.assertNotIn("successful_contexts", llm_dict)  # Too verbose for prompt
        self.assertNotIn("failure_contexts", llm_dict)

    def test_enriched_bullet_to_retrieval_dict(self):
        """Test retrieval-specific dictionary for indexing."""
        from ace.playbook import EnrichedBullet

        bullet = EnrichedBullet(
            id="test-001",
            section="test",
            content="Test strategy for complex problems",
            task_types=["reasoning"],
            domains=["math"],
            complexity_level="complex",
            trigger_patterns=["calculate", "compute"],
            embedding_text="custom embedding text for search",
        )

        retrieval_dict = bullet.to_retrieval_dict()

        # Should include all retrieval-relevant fields
        self.assertEqual(retrieval_dict["id"], "test-001")
        self.assertEqual(retrieval_dict["task_types"], ["reasoning"])
        self.assertEqual(retrieval_dict["domains"], ["math"])
        self.assertEqual(retrieval_dict["complexity_level"], "complex")
        self.assertEqual(retrieval_dict["trigger_patterns"], ["calculate", "compute"])

        # Should use embedding_text if provided, else content
        self.assertEqual(retrieval_dict["text_for_embedding"], "custom embedding text for search")

    def test_enriched_bullet_to_retrieval_dict_fallback(self):
        """Test retrieval dict uses content when embedding_text not set."""
        from ace.playbook import EnrichedBullet

        bullet = EnrichedBullet(
            id="test-001",
            section="test",
            content="Test strategy for complex problems",
        )

        retrieval_dict = bullet.to_retrieval_dict()

        # Should fall back to content
        self.assertEqual(retrieval_dict["text_for_embedding"], "Test strategy for complex problems")

    def test_enriched_bullet_serialization(self):
        """Test JSON serialization preserves all scaffolding fields."""
        from ace.playbook import EnrichedBullet
        from dataclasses import asdict

        bullet = EnrichedBullet(
            id="test-001",
            section="test",
            content="Test strategy",
            task_types=["reasoning", "analysis"],
            domains=["math"],
            preconditions=["has numbers"],
            trigger_patterns=["calculate"],
            related_bullets=["test-002"],
            confidence=0.8,
        )

        # Serialize to dict
        data = asdict(bullet)

        # Verify all scaffolding fields present
        self.assertEqual(data["task_types"], ["reasoning", "analysis"])
        self.assertEqual(data["domains"], ["math"])
        self.assertEqual(data["preconditions"], ["has numbers"])
        self.assertEqual(data["trigger_patterns"], ["calculate"])
        self.assertEqual(data["related_bullets"], ["test-002"])
        self.assertEqual(data["confidence"], 0.8)

        # Should be JSON serializable
        json_str = json.dumps(data)
        self.assertIsInstance(json_str, str)

        # Should deserialize correctly
        loaded = json.loads(json_str)
        self.assertEqual(loaded["task_types"], ["reasoning", "analysis"])

    def test_enriched_bullet_backward_compatible_with_bullet(self):
        """Test EnrichedBullet can be used where Bullet is expected."""
        from ace.playbook import Bullet, EnrichedBullet

        # EnrichedBullet should have all Bullet fields
        enriched = EnrichedBullet(
            id="test-001",
            section="test",
            content="Test content",
            helpful=5,
            harmful=1,
            neutral=2,
        )

        # Should have all base Bullet attributes
        self.assertEqual(enriched.id, "test-001")
        self.assertEqual(enriched.section, "test")
        self.assertEqual(enriched.content, "Test content")
        self.assertEqual(enriched.helpful, 5)
        self.assertEqual(enriched.harmful, 1)
        self.assertEqual(enriched.neutral, 2)
        self.assertIsNotNone(enriched.created_at)
        self.assertIsNotNone(enriched.updated_at)

        # Should support tag() method
        enriched.tag("helpful", 3)
        self.assertEqual(enriched.helpful, 8)

    def test_enriched_bullet_effectiveness_score(self):
        """Test computed effectiveness score from helpful/harmful."""
        from ace.playbook import EnrichedBullet

        # High effectiveness
        good_bullet = EnrichedBullet(
            id="good-001",
            section="test",
            content="Effective strategy",
            helpful=10,
            harmful=0,
        )
        self.assertGreater(good_bullet.effectiveness_score, 0.9)

        # Low effectiveness
        bad_bullet = EnrichedBullet(
            id="bad-001",
            section="test",
            content="Bad strategy",
            helpful=1,
            harmful=9,
        )
        self.assertLess(bad_bullet.effectiveness_score, 0.2)

        # Mixed effectiveness
        mixed_bullet = EnrichedBullet(
            id="mixed-001",
            section="test",
            content="Mixed strategy",
            helpful=5,
            harmful=5,
        )
        self.assertAlmostEqual(mixed_bullet.effectiveness_score, 0.5, places=1)

        # No data (cold start)
        new_bullet = EnrichedBullet(
            id="new-001",
            section="test",
            content="New strategy",
        )
        self.assertEqual(new_bullet.effectiveness_score, 0.5)  # Neutral default

    def test_enriched_bullet_matches_intent(self):
        """Test intent matching for smart retrieval."""
        from ace.playbook import EnrichedBullet

        bullet = EnrichedBullet(
            id="math-001",
            section="math",
            content="Use step-by-step reasoning",
            task_types=["reasoning", "problem_solving"],
            domains=["math", "logic"],
            complexity_level="complex",
        )

        # Should match analytical intent in math domain
        self.assertTrue(bullet.matches_intent(
            task_type="reasoning",
            domain="math",
            complexity="complex"
        ))

        # Should match partial (domain only)
        self.assertTrue(bullet.matches_intent(domain="math"))

        # Should not match different domain
        self.assertFalse(bullet.matches_intent(domain="creative_writing"))

        # Should not match simpler complexity requirement
        # (complex bullet shouldn't be used for simple tasks)
        self.assertFalse(bullet.matches_intent(
            domain="math",
            complexity="simple"
        ))


@pytest.mark.unit
class TestPlaybookWithEnrichedBullets(unittest.TestCase):
    """Test Playbook integration with EnrichedBullet."""

    def test_playbook_add_enriched_bullet(self):
        """Test adding EnrichedBullet to Playbook."""
        from ace import Playbook
        from ace.playbook import EnrichedBullet

        playbook = Playbook()

        # Add enriched bullet with scaffolding
        bullet = playbook.add_enriched_bullet(
            section="math",
            content="Break problems into steps",
            task_types=["reasoning"],
            domains=["math", "logic"],
            trigger_patterns=["solve", "calculate"],
        )

        self.assertIsInstance(bullet, EnrichedBullet)
        self.assertEqual(bullet.task_types, ["reasoning"])
        self.assertEqual(bullet.domains, ["math", "logic"])
        self.assertEqual(len(playbook.bullets()), 1)

    def test_playbook_retrieves_by_intent(self):
        """Test playbook can retrieve bullets matching intent."""
        from ace import Playbook

        playbook = Playbook()

        # Add bullets with different scaffolding
        playbook.add_enriched_bullet(
            section="math",
            content="Use algebra",
            task_types=["reasoning"],
            domains=["math"],
            complexity_level="complex",
        )
        playbook.add_enriched_bullet(
            section="writing",
            content="Be concise",
            task_types=["creative"],
            domains=["writing"],
            complexity_level="simple",
        )
        playbook.add_enriched_bullet(
            section="math",
            content="Check units",
            task_types=["validation"],
            domains=["math", "physics"],
            complexity_level="simple",
        )

        # Retrieve math bullets only
        math_bullets = playbook.get_bullets_by_intent(domain="math")
        self.assertEqual(len(math_bullets), 2)

        # Retrieve simple bullets only
        simple_bullets = playbook.get_bullets_by_intent(complexity="simple")
        self.assertEqual(len(simple_bullets), 2)

        # Retrieve reasoning in math domain
        reasoning_math = playbook.get_bullets_by_intent(
            task_type="reasoning",
            domain="math"
        )
        self.assertEqual(len(reasoning_math), 1)
        self.assertEqual(reasoning_math[0].content, "Use algebra")

    def test_playbook_backward_compatible_with_basic_bullets(self):
        """Test that Playbook still works with basic Bullet objects."""
        from ace import Playbook
        from ace.playbook import Bullet

        playbook = Playbook()

        # Old-style bullet addition should still work
        bullet = playbook.add_bullet(
            section="test",
            content="Basic bullet",
            metadata={"helpful": 5}
        )

        self.assertIsInstance(bullet, Bullet)
        self.assertEqual(bullet.helpful, 5)
        self.assertEqual(len(playbook.bullets()), 1)


@pytest.mark.unit
class TestBulletEnrichmentPipeline(unittest.TestCase):
    """Test the enrichment pipeline that adds scaffolding to bullets."""

    def test_enrich_bullet_from_context(self):
        """Test enriching a bullet based on usage context."""
        from ace.playbook import Bullet, enrich_bullet

        # Basic bullet
        basic = Bullet(
            id="test-001",
            section="debugging",
            content="Always check for null before accessing properties",
        )

        # Context where it was successful
        success_context = """
        Question: Why is my code throwing NullPointerException?
        Answer: The user object was null when accessing user.getName().
        Feedback: Correct! This fixed the crash.
        """

        # Enrich the bullet
        enriched = enrich_bullet(basic, success_context)

        # Should infer task types
        self.assertIn("debugging", enriched.task_types)

        # Should infer domains from context
        self.assertTrue(any(d in enriched.domains for d in ["java", "python", "code"]))

        # Should extract trigger patterns
        self.assertTrue(len(enriched.trigger_patterns) > 0)

        # Should record success context (compressed)
        self.assertTrue(len(enriched.successful_contexts) > 0)


if __name__ == "__main__":
    unittest.main()
