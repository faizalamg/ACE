"""Tests for metadata-enhanced retrieval scoring (Phase 1A).

TDD: These tests are written FIRST and will FAIL until production code is implemented.
"""

import unittest

import pytest


@pytest.mark.unit
class TestQueryTypeMetadataScoring(unittest.TestCase):
    """Test query_type parameter boosts score when matching task_types metadata."""

    def setUp(self):
        """Set up test fixtures with bullets having task_types metadata."""
        from ace import Playbook
        from ace.retrieval import SmartBulletIndex

        self.playbook = Playbook()

        # Bullet with debugging task_type
        self.playbook.add_enriched_bullet(
            section="debugging",
            content="Check error logs systematically",
            task_types=["debugging", "troubleshooting"],
            domains=["software"],
        )

        # Bullet with reasoning task_type
        self.playbook.add_enriched_bullet(
            section="reasoning",
            content="Break down complex problems step-by-step",
            task_types=["reasoning", "analysis"],
            domains=["general"],
        )

        # Bullet with coding task_type
        self.playbook.add_enriched_bullet(
            section="coding",
            content="Write unit tests before implementation",
            task_types=["coding", "testing"],
            domains=["software"],
        )

        self.index = SmartBulletIndex(playbook=self.playbook)

    def test_query_type_matching_boosts_score(self):
        """Test that query_type parameter adds +0.25 boost when matching task_types.

        This test WILL FAIL until retrieve() is updated to handle query_type parameter.
        """
        # Retrieve with query_type that matches first bullet's task_types
        results = self.index.retrieve(query_type="debugging")

        # Should find the debugging bullet
        self.assertGreater(len(results), 0, "Should find at least one bullet")

        # Find the debugging bullet in results
        debug_result = None
        for result in results:
            if "error logs" in result.content:
                debug_result = result
                break

        self.assertIsNotNone(debug_result, "Should find debugging bullet")

        # Score calculation with dynamic weighting:
        # - query_type boost (raw) = 0.25
        # - effectiveness (cold start) = 0.5
        # - Dynamic weights (cold start): similarity=0.8, outcome=0.2
        # - Final score = 0.8 * 0.25 + 0.2 * 0.5 = 0.2 + 0.1 = 0.3
        self.assertGreaterEqual(
            debug_result.score,
            0.25,
            f"Score {debug_result.score} should be >= 0.25 (query_type boost with dynamic weighting)"
        )

        # Check match_reasons includes query_type match
        match_reasons_str = " ".join(debug_result.match_reasons)
        self.assertIn(
            "task_type_match:debugging",
            match_reasons_str,
            f"Match reasons should include 'task_type_match:debugging', got: {debug_result.match_reasons}"
        )

    def test_query_type_non_matching_no_boost(self):
        """Test that query_type doesn't boost score when not matching task_types."""
        results = self.index.retrieve(query_type="debugging")

        # Find the reasoning bullet (should NOT get query_type boost)
        reasoning_result = None
        for result in results:
            if "Break down complex" in result.content:
                reasoning_result = result
                break

        # If reasoning bullet appears, it should NOT have the query_type boost
        if reasoning_result:
            match_reasons_str = " ".join(reasoning_result.match_reasons)
            self.assertNotIn(
                "task_type_match:debugging",
                match_reasons_str,
                "Reasoning bullet should not get debugging query_type boost"
            )

    def test_query_type_with_multiple_task_types(self):
        """Test that query_type matches any of the bullet's task_types."""
        # Debugging bullet has task_types=["debugging", "troubleshooting"]
        # Both should match
        results_debugging = self.index.retrieve(query_type="debugging")
        results_troubleshooting = self.index.retrieve(query_type="troubleshooting")

        # Both should find the debugging bullet with boost
        self.assertGreater(len(results_debugging), 0)
        self.assertGreater(len(results_troubleshooting), 0)

        # Both should have the task_type_match in reasons
        for results, query_type in [
            (results_debugging, "debugging"),
            (results_troubleshooting, "troubleshooting")
        ]:
            debug_result = [r for r in results if "error logs" in r.content]
            if debug_result:
                match_reasons_str = " ".join(debug_result[0].match_reasons)
                self.assertIn(
                    f"task_type_match:{query_type}",
                    match_reasons_str,
                    f"Should match {query_type} in task_types"
                )


@pytest.mark.unit
class TestDomainMetadataScoring(unittest.TestCase):
    """Test domain parameter enhancement with scoring boost."""

    def setUp(self):
        """Set up test fixtures with bullets having domain metadata."""
        from ace import Playbook
        from ace.retrieval import SmartBulletIndex

        self.playbook = Playbook()

        # Software domain bullet
        self.playbook.add_enriched_bullet(
            section="software",
            content="Use design patterns for maintainability",
            task_types=["coding"],
            domains=["software", "architecture"],
        )

        # Math domain bullet
        self.playbook.add_enriched_bullet(
            section="math",
            content="Apply mathematical induction for proofs",
            task_types=["reasoning"],
            domains=["math", "logic"],
        )

        # Multi-domain bullet
        self.playbook.add_enriched_bullet(
            section="general",
            content="Document assumptions clearly",
            task_types=["documentation"],
            domains=["software", "math", "general"],
        )

        self.index = SmartBulletIndex(playbook=self.playbook)

    def test_domain_matching_boosts_score(self):
        """Test that domain parameter adds scoring boost when matching domains metadata.

        This test WILL FAIL until retrieve() is updated to add domain scoring boost.
        """
        results = self.index.retrieve(domain="math")

        # Find the math bullet
        math_result = None
        for result in results:
            if "mathematical induction" in result.content:
                math_result = result
                break

        self.assertIsNotNone(math_result, "Should find math bullet")

        # Existing code: domain match = 0.3
        # Phase 1A enhancement: should add additional boost (similar to query_type)
        # For now, check that we get the base 0.3 score
        # (The enhancement might add more, but existing code should pass)
        self.assertGreaterEqual(
            math_result.score,
            0.3,
            f"Score should be >= 0.3 for domain match"
        )

        # Check match_reasons includes domain
        match_reasons_str = " ".join(math_result.match_reasons)
        self.assertIn(
            "domain:",
            match_reasons_str,
            "Match reasons should include domain match"
        )

    def test_domain_with_multiple_domains(self):
        """Test that domain matches any of the bullet's domains."""
        # Multi-domain bullet has domains=["software", "math", "general"]
        for domain in ["software", "math", "general"]:
            results = self.index.retrieve(domain=domain)

            # Should find the multi-domain bullet
            multi_result = [r for r in results if "assumptions" in r.content]
            self.assertGreater(
                len(multi_result),
                0,
                f"Should find multi-domain bullet when filtering by {domain}"
            )


@pytest.mark.unit
class TestCombinedMetadataScoring(unittest.TestCase):
    """Test combined query_type and domain scoring."""

    def setUp(self):
        """Set up test fixtures for combined metadata testing."""
        from ace import Playbook
        from ace.retrieval import SmartBulletIndex

        self.playbook = Playbook()

        # Bullet matching both query_type and domain
        self.playbook.add_enriched_bullet(
            section="software_debugging",
            content="Use debugger breakpoints for software issues",
            task_types=["debugging", "troubleshooting"],
            domains=["software", "devops"],
        )

        # Bullet matching only query_type
        self.playbook.add_enriched_bullet(
            section="math_debugging",
            content="Debug mathematical proofs by checking axioms",
            task_types=["debugging", "reasoning"],
            domains=["math", "logic"],
        )

        # Bullet matching only domain
        self.playbook.add_enriched_bullet(
            section="software_coding",
            content="Follow coding standards for software projects",
            task_types=["coding", "best_practices"],
            domains=["software", "engineering"],
        )

        self.index = SmartBulletIndex(playbook=self.playbook)

    def test_combined_query_type_and_domain_boost(self):
        """Test that combining query_type and domain parameters provides cumulative boost.

        This integration test ensures both enhancements work together.
        """
        results = self.index.retrieve(query_type="debugging", domain="software")

        # Should find all three bullets, but software_debugging should rank highest
        self.assertGreater(len(results), 0, "Should find bullets")

        # Find the software debugging bullet (should have highest score)
        top_result = results[0]

        self.assertIn(
            "debugger breakpoints",
            top_result.content,
            "Bullet matching both query_type and domain should rank highest"
        )

        # Score calculation with dynamic weighting:
        # - domain match (existing): 0.3
        # - query_type match (new): 0.25
        # - Raw similarity score = 0.55
        # - effectiveness (cold start) = 0.5
        # - Dynamic weights (cold start): similarity=0.8, outcome=0.2
        # - Final score = 0.8 * 0.55 + 0.2 * 0.5 = 0.44 + 0.1 = 0.54
        self.assertGreaterEqual(
            top_result.score,
            0.50,
            f"Combined score should be >= 0.50 with dynamic weighting, got {top_result.score}"
        )

    def test_partial_match_lower_score(self):
        """Test that partial matches (only query_type or only domain) score lower."""
        results = self.index.retrieve(query_type="debugging", domain="software")

        # Find bullets by content
        full_match = None  # debugger breakpoints (both match)
        partial_match_1 = None  # mathematical proofs (only query_type)
        partial_match_2 = None  # coding standards (only domain)

        for result in results:
            if "debugger breakpoints" in result.content:
                full_match = result
            elif "mathematical proofs" in result.content:
                partial_match_1 = result
            elif "coding standards" in result.content:
                partial_match_2 = result

        self.assertIsNotNone(full_match, "Should find full match bullet")

        # Full match should have highest score
        if partial_match_1:
            self.assertGreater(
                full_match.score,
                partial_match_1.score,
                "Full match should score higher than query_type-only match"
            )

        if partial_match_2:
            self.assertGreater(
                full_match.score,
                partial_match_2.score,
                "Full match should score higher than domain-only match"
            )


if __name__ == "__main__":
    unittest.main()
