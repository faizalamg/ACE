"""Tests for golden rules auto-promotion feature."""

import unittest
from ace.playbook import Playbook, Bullet
from ace.config import ELFConfig


class TestGoldenRulesPromotion(unittest.TestCase):
    """Test golden rules auto-promotion and demotion."""

    def setUp(self):
        """Set up test fixtures."""
        self.playbook = Playbook()
        self.config = ELFConfig(
            enable_golden_rules=True,
            golden_rule_helpful_threshold=10,
            golden_rule_max_harmful=0,
            golden_rule_demotion_harmful_threshold=3,
        )

    def test_promotion_basic(self):
        """Test basic promotion to golden_rules."""
        # Add bullet that qualifies (helpful >= 10, harmful <= 0)
        bullet = self.playbook.add_bullet("strategies", "Always validate input", metadata={"helpful": 10, "harmful": 0})

        # Check promotion
        promoted = self.playbook.check_and_promote_golden_rules(self.config)

        self.assertEqual(len(promoted), 1)
        self.assertIn(bullet.id, promoted)
        self.assertEqual(bullet.section, "golden_rules")
        self.assertIn("golden_rules", self.playbook._sections)
        self.assertIn(bullet.id, self.playbook._sections["golden_rules"])

    def test_promotion_threshold_exact(self):
        """Test promotion at exact threshold."""
        # Exactly at threshold (helpful=10, harmful=0)
        bullet = self.playbook.add_bullet("strategies", "Use proper error handling", metadata={"helpful": 10, "harmful": 0})

        promoted = self.playbook.check_and_promote_golden_rules(self.config)
        self.assertEqual(len(promoted), 1)
        self.assertIn(bullet.id, promoted)

    def test_no_promotion_below_threshold(self):
        """Test no promotion when below threshold."""
        # Below threshold (helpful=9, harmful=0)
        bullet = self.playbook.add_bullet("strategies", "Log errors", metadata={"helpful": 9, "harmful": 0})

        promoted = self.playbook.check_and_promote_golden_rules(self.config)
        self.assertEqual(len(promoted), 0)
        self.assertEqual(bullet.section, "strategies")

    def test_no_promotion_too_much_harmful(self):
        """Test no promotion when harmful count too high."""
        # High helpful but also harmful (helpful=15, harmful=1)
        bullet = self.playbook.add_bullet("strategies", "Complex strategy", metadata={"helpful": 15, "harmful": 1})

        promoted = self.playbook.check_and_promote_golden_rules(self.config)
        self.assertEqual(len(promoted), 0)
        self.assertEqual(bullet.section, "strategies")

    def test_multiple_promotions(self):
        """Test promoting multiple bullets at once."""
        bullet1 = self.playbook.add_bullet("strategies", "Strategy 1", metadata={"helpful": 10, "harmful": 0})
        bullet2 = self.playbook.add_bullet("tactics", "Tactic 1", metadata={"helpful": 12, "harmful": 0})
        bullet3 = self.playbook.add_bullet("strategies", "Strategy 2", metadata={"helpful": 8, "harmful": 0})  # Won't promote

        promoted = self.playbook.check_and_promote_golden_rules(self.config)

        self.assertEqual(len(promoted), 2)
        self.assertIn(bullet1.id, promoted)
        self.assertIn(bullet2.id, promoted)
        self.assertNotIn(bullet3.id, promoted)

    def test_no_re_promotion(self):
        """Test that golden_rules bullets are not re-promoted."""
        # Manually add to golden_rules
        bullet = self.playbook.add_bullet("golden_rules", "Already golden", metadata={"helpful": 20, "harmful": 0})

        promoted = self.playbook.check_and_promote_golden_rules(self.config)
        self.assertEqual(len(promoted), 0)  # Should not re-promote

    def test_demotion_basic(self):
        """Test basic demotion from golden_rules."""
        # Add to golden_rules then accumulate harmful feedback
        bullet = self.playbook.add_bullet("golden_rules", "Risky strategy", metadata={"helpful": 10, "harmful": 3})

        demoted = self.playbook.demote_from_golden_rules(self.config)

        self.assertEqual(len(demoted), 1)
        self.assertIn(bullet.id, demoted)
        self.assertEqual(bullet.section, "deprecated")
        self.assertIn("deprecated", self.playbook._sections)
        self.assertIn(bullet.id, self.playbook._sections["deprecated"])

    def test_demotion_threshold_exact(self):
        """Test demotion at exact threshold."""
        bullet = self.playbook.add_bullet("golden_rules", "Borderline strategy", metadata={"helpful": 10, "harmful": 3})

        demoted = self.playbook.demote_from_golden_rules(self.config)
        self.assertEqual(len(demoted), 1)
        self.assertIn(bullet.id, demoted)

    def test_no_demotion_below_threshold(self):
        """Test no demotion when below harmful threshold."""
        bullet = self.playbook.add_bullet("golden_rules", "Solid strategy", metadata={"helpful": 15, "harmful": 2})

        demoted = self.playbook.demote_from_golden_rules(self.config)
        self.assertEqual(len(demoted), 0)
        self.assertEqual(bullet.section, "golden_rules")

    def test_no_demotion_no_golden_section(self):
        """Test demotion when no golden_rules section exists."""
        # Don't add any bullets to golden_rules
        demoted = self.playbook.demote_from_golden_rules(self.config)
        self.assertEqual(len(demoted), 0)

    def test_feature_disabled(self):
        """Test that promotion/demotion is disabled when config says so."""
        disabled_config = ELFConfig(enable_golden_rules=False)

        bullet = self.playbook.add_bullet("strategies", "Test", metadata={"helpful": 20, "harmful": 0})

        promoted = self.playbook.check_and_promote_golden_rules(disabled_config)
        self.assertEqual(len(promoted), 0)
        self.assertEqual(bullet.section, "strategies")

    def test_section_cleanup_on_promotion(self):
        """Test that empty sections are removed after promotion."""
        bullet = self.playbook.add_bullet("temporary", "Only bullet", metadata={"helpful": 10, "harmful": 0})

        self.assertIn("temporary", self.playbook._sections)

        self.playbook.check_and_promote_golden_rules(self.config)

        # Section should be removed since it's now empty
        self.assertNotIn("temporary", self.playbook._sections)

    def test_section_cleanup_on_demotion(self):
        """Test that empty golden_rules section is removed after demotion."""
        bullet = self.playbook.add_bullet("golden_rules", "Only golden", metadata={"helpful": 10, "harmful": 5})

        self.assertIn("golden_rules", self.playbook._sections)

        self.playbook.demote_from_golden_rules(self.config)

        # golden_rules section should be removed since it's now empty
        self.assertNotIn("golden_rules", self.playbook._sections)

    def test_updated_at_on_promotion(self):
        """Test that updated_at timestamp is updated on promotion."""
        import time
        bullet = self.playbook.add_bullet("strategies", "Test", metadata={"helpful": 10, "harmful": 0})
        original_updated = bullet.updated_at

        # Small delay to ensure different timestamp
        time.sleep(0.001)
        self.playbook.check_and_promote_golden_rules(self.config)

        # Timestamp should be updated
        self.assertNotEqual(bullet.updated_at, original_updated)

    def test_updated_at_on_demotion(self):
        """Test that updated_at timestamp is updated on demotion."""
        import time
        bullet = self.playbook.add_bullet("golden_rules", "Test", metadata={"helpful": 10, "harmful": 5})
        original_updated = bullet.updated_at

        # Small delay to ensure different timestamp
        time.sleep(0.001)
        self.playbook.demote_from_golden_rules(self.config)

        # Timestamp should be updated
        self.assertNotEqual(bullet.updated_at, original_updated)

    def test_full_lifecycle(self):
        """Test full lifecycle: add -> promote -> demote."""
        # Start in regular section
        bullet = self.playbook.add_bullet("strategies", "Lifecycle test", metadata={"helpful": 5, "harmful": 0})
        self.assertEqual(bullet.section, "strategies")

        # Not promoted yet (below threshold)
        promoted = self.playbook.check_and_promote_golden_rules(self.config)
        self.assertEqual(len(promoted), 0)

        # Accumulate helpful feedback
        bullet.helpful = 10
        promoted = self.playbook.check_and_promote_golden_rules(self.config)
        self.assertEqual(len(promoted), 1)
        self.assertEqual(bullet.section, "golden_rules")

        # Accumulate harmful feedback
        bullet.harmful = 3
        demoted = self.playbook.demote_from_golden_rules(self.config)
        self.assertEqual(len(demoted), 1)
        self.assertEqual(bullet.section, "deprecated")

    def test_default_config(self):
        """Test that default config loads correctly."""
        from ace.config import get_elf_config

        # This should not raise an error
        config = get_elf_config()
        self.assertIsInstance(config, ELFConfig)
        self.assertIsInstance(config.enable_golden_rules, bool)
        self.assertIsInstance(config.golden_rule_helpful_threshold, int)
        self.assertIsInstance(config.golden_rule_max_harmful, int)
        self.assertIsInstance(config.golden_rule_demotion_harmful_threshold, int)


if __name__ == "__main__":
    unittest.main()
