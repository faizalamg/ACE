"""Tests for SmartBulletIndex with purpose-aware retrieval.

TDD: These tests define the expected behavior BEFORE implementation.
"""

import unittest
from typing import List

import pytest


@pytest.mark.unit
class TestSmartBulletIndexCreation(unittest.TestCase):
    """Test SmartBulletIndex initialization and basic operations."""

    def test_smart_bullet_index_exists(self):
        """Test that SmartBulletIndex class exists."""
        from ace.retrieval import SmartBulletIndex

        index = SmartBulletIndex()
        self.assertIsNotNone(index)

    def test_index_accepts_playbook(self):
        """Test that index can be initialized with a playbook."""
        from ace import Playbook
        from ace.retrieval import SmartBulletIndex

        playbook = Playbook()
        playbook.add_bullet("math", "Use step-by-step reasoning")

        index = SmartBulletIndex(playbook=playbook)
        self.assertEqual(len(index), 1)

    def test_index_updates_from_playbook(self):
        """Test that index can be updated when playbook changes."""
        from ace import Playbook
        from ace.retrieval import SmartBulletIndex

        playbook = Playbook()
        index = SmartBulletIndex(playbook=playbook)

        playbook.add_bullet("math", "Use step-by-step reasoning")
        index.update()

        self.assertEqual(len(index), 1)


@pytest.mark.unit
class TestSmartBulletIndexRetrieval(unittest.TestCase):
    """Test purpose-aware retrieval capabilities."""

    def setUp(self):
        """Set up test fixtures with enriched bullets."""
        from ace import Playbook
        from ace.retrieval import SmartBulletIndex

        self.playbook = Playbook()

        # Add enriched bullets with different purposes
        self.playbook.add_enriched_bullet(
            section="math",
            content="Use step-by-step reasoning for complex calculations",
            task_types=["reasoning", "problem_solving"],
            domains=["math", "logic"],
            complexity_level="complex",
            trigger_patterns=["calculate", "solve", "derive"],
        )

        self.playbook.add_enriched_bullet(
            section="debugging",
            content="Check error logs first when debugging",
            task_types=["debugging", "troubleshooting"],
            domains=["software", "devops"],
            complexity_level="medium",
            trigger_patterns=["error", "bug", "exception", "crash"],
        )

        self.playbook.add_enriched_bullet(
            section="coding",
            content="Write tests before implementation",
            task_types=["development", "testing"],
            domains=["software"],
            complexity_level="simple",
            trigger_patterns=["implement", "code", "build", "create"],
        )

        self.index = SmartBulletIndex(playbook=self.playbook)

    def test_retrieve_by_task_type(self):
        """Test retrieval by task type."""
        results = self.index.retrieve(task_type="debugging")

        self.assertEqual(len(results), 1)
        self.assertIn("error logs", results[0].content)

    def test_retrieve_by_domain(self):
        """Test retrieval by domain."""
        results = self.index.retrieve(domain="math")

        self.assertEqual(len(results), 1)
        self.assertIn("step-by-step", results[0].content)

    def test_retrieve_by_trigger_pattern(self):
        """Test retrieval by trigger pattern matching."""
        results = self.index.retrieve(query="How do I calculate the total?")

        self.assertGreater(len(results), 0)
        # Math bullet should rank highest due to "calculate" trigger
        self.assertIn("step-by-step", results[0].content)

    def test_retrieve_by_complexity(self):
        """Test retrieval filtered by complexity level."""
        results = self.index.retrieve(complexity="simple")

        self.assertEqual(len(results), 1)
        self.assertIn("tests", results[0].content)

    def test_retrieve_combined_filters(self):
        """Test retrieval with multiple filter criteria."""
        results = self.index.retrieve(
            domain="software",
            task_type="debugging",
        )

        self.assertEqual(len(results), 1)
        self.assertIn("error logs", results[0].content)

    def test_retrieve_returns_scored_results(self):
        """Test that retrieval returns results with relevance scores."""
        results = self.index.retrieve(query="Fix the bug in the code")

        self.assertGreater(len(results), 0)
        # Results should have a score attribute
        for result in results:
            self.assertTrue(hasattr(result, 'score') or isinstance(result, tuple))

    def test_retrieve_with_limit(self):
        """Test retrieval with result limit."""
        # Add more bullets to test limiting
        for i in range(5):
            self.playbook.add_enriched_bullet(
                section="general",
                content=f"General tip number {i}",
                task_types=["general"],
                domains=["all"],
            )
        self.index.update()

        results = self.index.retrieve(domain="all", limit=3)

        self.assertEqual(len(results), 3)


@pytest.mark.unit
class TestSmartBulletIndexSemanticSearch(unittest.TestCase):
    """Test semantic search capabilities."""

    def setUp(self):
        """Set up test fixtures."""
        from ace import Playbook
        from ace.retrieval import SmartBulletIndex

        self.playbook = Playbook()

        # Add bullets with custom embedding text
        self.playbook.add_enriched_bullet(
            section="api",
            content="Use retry with exponential backoff for transient failures",
            task_types=["api_design", "error_handling"],
            domains=["backend", "distributed_systems"],
            embedding_text="API retry exponential backoff transient failure handling resilience",
        )

        self.playbook.add_enriched_bullet(
            section="database",
            content="Index frequently queried columns",
            task_types=["optimization", "database"],
            domains=["backend", "performance"],
            embedding_text="database index query performance optimization SQL columns",
        )

        self.index = SmartBulletIndex(playbook=self.playbook)

    def test_semantic_search_uses_embedding_text(self):
        """Test that semantic search uses custom embedding text."""
        results = self.index.semantic_search("resilience patterns for API calls")

        self.assertGreater(len(results), 0)
        # Should match the retry bullet due to embedding text containing "resilience"
        self.assertIn("retry", results[0].content.lower())

    def test_semantic_search_with_threshold(self):
        """Test semantic search with minimum similarity threshold."""
        results = self.index.semantic_search(
            "completely unrelated topic like cooking recipes",
            threshold=0.8,  # High threshold should filter out irrelevant results
        )

        # Should return empty or very few results due to high threshold
        self.assertLessEqual(len(results), 1)


@pytest.mark.unit
class TestSmartBulletIndexIntentRouting(unittest.TestCase):
    """Test intent-based query routing."""

    def setUp(self):
        """Set up test fixtures."""
        from ace import Playbook
        from ace.retrieval import SmartBulletIndex

        self.playbook = Playbook()

        # Analytical bullet
        self.playbook.add_enriched_bullet(
            section="analysis",
            content="Compare alternatives before making architectural decisions",
            task_types=["analysis", "architecture"],
            domains=["software"],
            retrieval_type="analytical",
        )

        # Factual bullet
        self.playbook.add_enriched_bullet(
            section="reference",
            content="Python 3.12 requires explicit type annotations",
            task_types=["reference", "lookup"],
            domains=["python"],
            retrieval_type="factual",
        )

        # Procedural bullet
        self.playbook.add_enriched_bullet(
            section="process",
            content="Run tests, then build, then deploy",
            task_types=["deployment", "ci_cd"],
            domains=["devops"],
            retrieval_type="procedural",
        )

        self.index = SmartBulletIndex(playbook=self.playbook)

    def test_route_analytical_query(self):
        """Test routing analytical queries."""
        results = self.index.retrieve(
            query="Which database should I choose?",
            intent="analytical",
        )

        self.assertGreater(len(results), 0)
        self.assertIn("compare", results[0].content.lower())

    def test_route_factual_query(self):
        """Test routing factual queries."""
        results = self.index.retrieve(
            query="What Python version requires type annotations?",
            intent="factual",
        )

        self.assertGreater(len(results), 0)
        self.assertIn("python 3.12", results[0].content.lower())

    def test_route_procedural_query(self):
        """Test routing procedural queries."""
        results = self.index.retrieve(
            query="How do I deploy the application?",
            intent="procedural",
        )

        self.assertGreater(len(results), 0)
        self.assertIn("deploy", results[0].content.lower())


@pytest.mark.unit
class TestSmartBulletIndexEffectiveness(unittest.TestCase):
    """Test effectiveness-based ranking."""

    def setUp(self):
        """Set up test fixtures with bullets having different effectiveness."""
        from ace import Playbook
        from ace.retrieval import SmartBulletIndex

        self.playbook = Playbook()

        # High effectiveness bullet
        self.playbook.add_enriched_bullet(
            section="general",
            content="High effectiveness tip",
            task_types=["general"],
            domains=["all"],
        )
        bullet1 = self.playbook.bullets()[-1]
        bullet1.helpful = 10
        bullet1.harmful = 1

        # Low effectiveness bullet
        self.playbook.add_enriched_bullet(
            section="general",
            content="Low effectiveness tip",
            task_types=["general"],
            domains=["all"],
        )
        bullet2 = self.playbook.bullets()[-1]
        bullet2.helpful = 1
        bullet2.harmful = 5

        self.index = SmartBulletIndex(playbook=self.playbook)

    def test_effectiveness_influences_ranking(self):
        """Test that effectiveness scores influence retrieval ranking."""
        results = self.index.retrieve(domain="all", rank_by_effectiveness=True)

        self.assertEqual(len(results), 2)
        # High effectiveness should rank first
        self.assertIn("High effectiveness", results[0].content)

    def test_filter_by_minimum_effectiveness(self):
        """Test filtering bullets by minimum effectiveness threshold."""
        results = self.index.retrieve(
            domain="all",
            min_effectiveness=0.5,  # Exclude low effectiveness bullets
        )

        self.assertEqual(len(results), 1)
        self.assertIn("High effectiveness", results[0].content)


if __name__ == "__main__":
    unittest.main()
