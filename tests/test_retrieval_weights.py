"""Tests for dynamic weight shifting in SmartBulletIndex based on bullet maturity.

Phase 1D: Dynamic Weight Shifting
- New bullets (0 feedback): 80% similarity, 20% outcome
- Early bullets (1-4 feedback): 50/50
- Mature bullets (5+ feedback): 30% similarity, 70% outcome

TDD: These tests define the expected behavior BEFORE implementation.
"""

import unittest

import pytest


@pytest.mark.unit
class TestDynamicWeightShifting(unittest.TestCase):
    """Test weight progression based on bullet maturity."""

    def test_new_bullet_uses_similarity_weighting(self):
        """Test that new bullets with 0 feedback use 80% similarity, 20% outcome."""
        from ace import Playbook
        from ace.retrieval import SmartBulletIndex

        playbook = Playbook()
        playbook.add_enriched_bullet(
            section="test",
            content="New bullet with no feedback",
            task_types=["general"],
            domains=["all"],
        )
        bullet = playbook.bullets()[-1]
        # Verify no feedback (cold start)
        self.assertEqual(bullet.helpful, 0)
        self.assertEqual(bullet.harmful, 0)

        index = SmartBulletIndex(playbook=playbook)

        # Access the private method for testing weight calculation
        weights = index._get_dynamic_weights(bullet)
        similarity_weight, outcome_weight = weights

        # New bullets should trust similarity more (cold start)
        self.assertEqual(similarity_weight, 0.8)
        self.assertEqual(outcome_weight, 0.2)

    def test_mature_bullet_uses_outcome_weighting(self):
        """Test that mature bullets with 5+ feedback use 30% similarity, 70% outcome."""
        from ace import Playbook
        from ace.retrieval import SmartBulletIndex

        playbook = Playbook()
        playbook.add_enriched_bullet(
            section="test",
            content="Mature bullet with lots of feedback",
            task_types=["general"],
            domains=["all"],
        )
        bullet = playbook.bullets()[-1]
        # Simulate mature bullet with 10 total signals
        bullet.helpful = 8
        bullet.harmful = 2

        index = SmartBulletIndex(playbook=playbook)

        # Access the private method for testing
        weights = index._get_dynamic_weights(bullet)
        similarity_weight, outcome_weight = weights

        # Mature bullets should trust outcomes more
        self.assertEqual(similarity_weight, 0.3)
        self.assertEqual(outcome_weight, 0.7)

    def test_early_bullet_uses_balanced_weighting(self):
        """Test that early bullets with 1-4 feedback use 50/50 weighting."""
        from ace import Playbook
        from ace.retrieval import SmartBulletIndex

        playbook = Playbook()
        playbook.add_enriched_bullet(
            section="test",
            content="Early bullet with some feedback",
            task_types=["general"],
            domains=["all"],
        )
        bullet = playbook.bullets()[-1]
        # Simulate early bullet with 3 total signals
        bullet.helpful = 2
        bullet.harmful = 1

        index = SmartBulletIndex(playbook=playbook)

        weights = index._get_dynamic_weights(bullet)
        similarity_weight, outcome_weight = weights

        # Early bullets should use balanced weighting
        self.assertEqual(similarity_weight, 0.5)
        self.assertEqual(outcome_weight, 0.5)

    def test_weight_progression(self):
        """Test weight progression from new -> early -> mature."""
        from ace import Playbook
        from ace.retrieval import SmartBulletIndex

        playbook = Playbook()
        playbook.add_enriched_bullet(
            section="test",
            content="Test bullet",
            task_types=["general"],
            domains=["all"],
        )
        bullet = playbook.bullets()[-1]

        index = SmartBulletIndex(playbook=playbook)

        # Stage 1: New bullet (0 signals)
        bullet.helpful = 0
        bullet.harmful = 0
        sim_w, out_w = index._get_dynamic_weights(bullet)
        self.assertEqual((sim_w, out_w), (0.8, 0.2))

        # Stage 2: Early bullet (1 signal)
        bullet.helpful = 1
        sim_w, out_w = index._get_dynamic_weights(bullet)
        self.assertEqual((sim_w, out_w), (0.5, 0.5))

        # Stage 3: Early bullet (4 signals)
        bullet.helpful = 3
        bullet.harmful = 1
        sim_w, out_w = index._get_dynamic_weights(bullet)
        self.assertEqual((sim_w, out_w), (0.5, 0.5))

        # Stage 4: Mature bullet (5 signals)
        bullet.helpful = 4
        bullet.harmful = 1
        sim_w, out_w = index._get_dynamic_weights(bullet)
        self.assertEqual((sim_w, out_w), (0.3, 0.7))

        # Stage 5: Mature bullet (20 signals)
        bullet.helpful = 15
        bullet.harmful = 5
        sim_w, out_w = index._get_dynamic_weights(bullet)
        self.assertEqual((sim_w, out_w), (0.3, 0.7))

    def test_neutral_signals_count_toward_maturity(self):
        """Test that neutral signals count toward maturity threshold."""
        from ace import Playbook
        from ace.retrieval import SmartBulletIndex

        playbook = Playbook()
        playbook.add_enriched_bullet(
            section="test",
            content="Test bullet",
            task_types=["general"],
            domains=["all"],
        )
        bullet = playbook.bullets()[-1]

        # Set 5 total signals (mostly neutral)
        bullet.helpful = 1
        bullet.harmful = 1
        bullet.neutral = 3

        index = SmartBulletIndex(playbook=playbook)

        weights = index._get_dynamic_weights(bullet)
        similarity_weight, outcome_weight = weights

        # Should be mature (5 total signals)
        self.assertEqual(similarity_weight, 0.3)
        self.assertEqual(outcome_weight, 0.7)


@pytest.mark.unit
class TestDynamicWeightIntegration(unittest.TestCase):
    """Test that dynamic weights are integrated into retrieve() scoring."""

    def setUp(self):
        """Set up test fixtures with bullets at different maturity levels."""
        from ace import Playbook
        from ace.retrieval import SmartBulletIndex

        self.playbook = Playbook()

        # New bullet (0 feedback) - should rely on similarity
        self.playbook.add_enriched_bullet(
            section="new",
            content="New debugging strategy: check network first",
            task_types=["debugging"],
            domains=["networking"],
            trigger_patterns=["network", "connection"],
        )
        self.new_bullet = self.playbook.bullets()[-1]

        # Mature bullet (10 feedback) - should rely on outcomes
        self.playbook.add_enriched_bullet(
            section="mature",
            content="Mature debugging strategy: check logs first",
            task_types=["debugging"],
            domains=["all"],
            trigger_patterns=["error", "debug"],
        )
        self.mature_bullet = self.playbook.bullets()[-1]
        self.mature_bullet.helpful = 8
        self.mature_bullet.harmful = 2  # 80% effectiveness

        self.index = SmartBulletIndex(playbook=self.playbook)

    def test_retrieve_uses_dynamic_weights_in_scoring(self):
        """Test that retrieve() method uses dynamic weights in score calculation."""
        # Query that matches both bullets
        results = self.index.retrieve(
            query="debugging network issues",
            task_type="debugging",
        )

        self.assertGreater(len(results), 0)

        # Find our test bullets in results
        new_result = None
        mature_result = None
        for r in results:
            if "network first" in r.content:
                new_result = r
            elif "logs first" in r.content:
                mature_result = r

        # Both should be present
        self.assertIsNotNone(new_result)
        self.assertIsNotNone(mature_result)

        # Verify scores reflect dynamic weighting
        # New bullet uses 80% similarity -> high score if similar
        # Mature bullet uses 70% outcome -> high score if effective

        # Mature bullet has high effectiveness (0.8), should score well
        # despite possibly lower similarity
        self.assertGreater(mature_result.score, 0.0)

    def test_mature_bullet_benefits_from_high_effectiveness(self):
        """Test that mature bullets with high effectiveness rank higher."""
        from ace import Playbook
        from ace.retrieval import SmartBulletIndex

        playbook = Playbook()

        # Two bullets with same task type but different maturity
        playbook.add_enriched_bullet(
            section="new",
            content="New tip: try approach A",
            task_types=["general"],
            domains=["all"],
        )
        new_bullet = playbook.bullets()[-1]

        playbook.add_enriched_bullet(
            section="mature",
            content="Proven tip: try approach B",
            task_types=["general"],
            domains=["all"],
        )
        mature_bullet = playbook.bullets()[-1]
        mature_bullet.helpful = 20  # Very effective
        mature_bullet.harmful = 0

        index = SmartBulletIndex(playbook=playbook)

        results = index.retrieve(
            task_type="general",
            rank_by_effectiveness=True,
        )

        self.assertEqual(len(results), 2)
        # Mature bullet with high effectiveness should rank higher
        self.assertIn("approach B", results[0].content)


if __name__ == "__main__":
    unittest.main()
