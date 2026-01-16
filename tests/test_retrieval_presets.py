"""
Test suite for P7.1 Multi-Preset Retrieval System.

TDD RED PHASE - All tests should FAIL initially.
Implementation targets:
- 4 presets: fast, balanced, deep, diverse
- PresetConfig dataclass in ace/config.py
- get_preset() function for retrieval
- apply_preset_to_retrieval_config() modifier
- <1ms switching latency
- Zero regression on default behavior
"""

import unittest
import time
from typing import Optional

# These imports WILL FAIL - that's the RED phase
from ace.config import PresetConfig, get_preset, apply_preset_to_retrieval_config, RetrievalConfig


class TestPresetConfigDataclass(unittest.TestCase):
    """Test PresetConfig dataclass structure and defaults."""

    def test_preset_config_dataclass_exists(self):
        """P7.1.1: PresetConfig dataclass must exist with required fields."""
        config = PresetConfig(
            final_k=64,
            use_hyde=True,
            enable_reranking=True,
            num_expanded_queries=4
        )

        self.assertEqual(config.final_k, 64)
        self.assertIsInstance(config.use_hyde, bool)
        self.assertIsInstance(config.enable_reranking, bool)
        self.assertEqual(config.num_expanded_queries, 4)


class TestPresetConfigurations(unittest.TestCase):
    """Test individual preset configurations match specifications."""

    def test_preset_fast_config_values(self):
        """P7.1.2: 'fast' preset configuration (speed-optimized)."""
        preset = get_preset("fast")

        self.assertIsInstance(preset, PresetConfig)
        self.assertEqual(preset.final_k, 40, "fast preset should limit results to 40")
        self.assertFalse(preset.use_hyde, "fast preset should disable HyDE")
        self.assertFalse(preset.enable_reranking, "fast preset should disable reranking")
        self.assertEqual(preset.num_expanded_queries, 1, "fast preset should use 1 query (no expansion)")

    def test_preset_balanced_config_values(self):
        """P7.1.3: 'balanced' preset configuration (default recommended)."""
        preset = get_preset("balanced")

        self.assertIsInstance(preset, PresetConfig)
        self.assertEqual(preset.final_k, 64, "balanced preset should use 64 results")
        # HyDE can be bool or "auto" string
        self.assertIn(preset.use_hyde, [True, False, "auto"], "balanced preset should use auto HyDE")
        self.assertTrue(preset.enable_reranking, "balanced preset should enable reranking")
        self.assertEqual(preset.num_expanded_queries, 4, "balanced preset should expand to 4 queries")

    def test_preset_deep_config_values(self):
        """P7.1.4: 'deep' preset configuration (maximum recall)."""
        preset = get_preset("deep")

        self.assertIsInstance(preset, PresetConfig)
        self.assertEqual(preset.final_k, 96, "deep preset should use 96 results")
        self.assertTrue(preset.use_hyde, "deep preset should force HyDE on")
        self.assertTrue(preset.enable_reranking, "deep preset should enable reranking")
        self.assertEqual(preset.num_expanded_queries, 6, "deep preset should expand to 6 queries")

    def test_preset_diverse_config_values(self):
        """P7.1.5: 'diverse' preset configuration (exploration-focused)."""
        preset = get_preset("diverse")

        self.assertIsInstance(preset, PresetConfig)
        self.assertEqual(preset.final_k, 80, "diverse preset should use 80 results")
        self.assertFalse(preset.use_hyde, "diverse preset should disable HyDE to avoid query bias")
        self.assertTrue(preset.enable_reranking, "diverse preset should enable reranking")
        self.assertEqual(preset.num_expanded_queries, 4, "diverse preset should expand to 4 queries")


class TestPresetRetrieval(unittest.TestCase):
    """Test preset retrieval and error handling."""

    def test_get_preset_returns_correct_config(self):
        """P7.1.6: get_preset() should return PresetConfig for valid names."""
        valid_presets = ["fast", "balanced", "deep", "diverse"]

        for preset_name in valid_presets:
            with self.subTest(preset=preset_name):
                config = get_preset(preset_name)
                self.assertIsInstance(config, PresetConfig,
                                     f"get_preset('{preset_name}') should return PresetConfig")

    def test_get_preset_invalid_name_raises_error(self):
        """P7.1.7: get_preset() should raise ValueError for invalid preset names."""
        with self.assertRaises(ValueError, msg="Invalid preset name should raise ValueError"):
            get_preset("turbo_ultra_mega")

    def test_get_preset_case_insensitive(self):
        """P7.1.8: get_preset() should handle case-insensitive names."""
        preset_lower = get_preset("fast")
        preset_upper = get_preset("FAST")
        preset_mixed = get_preset("FaSt")

        self.assertEqual(preset_lower.final_k, preset_upper.final_k)
        self.assertEqual(preset_lower.final_k, preset_mixed.final_k)


class TestPresetApplication(unittest.TestCase):
    """Test applying presets to RetrievalConfig."""

    def test_apply_preset_to_retrieval_config(self):
        """P7.1.9: apply_preset_to_retrieval_config() modifies config correctly."""
        base_config = RetrievalConfig()

        # Apply 'fast' preset
        modified_config = apply_preset_to_retrieval_config(base_config, "fast")

        self.assertEqual(modified_config.final_k, 40)
        self.assertFalse(modified_config.use_hyde)
        self.assertFalse(modified_config.enable_reranking)
        self.assertEqual(modified_config.num_expanded_queries, 1)

        # Original config should be unchanged (immutability check)
        self.assertNotEqual(base_config.final_k, 40,
                           "Original config should not be modified")

    def test_apply_preset_preserves_other_config_fields(self):
        """P7.1.10: Preset application should only modify preset-controlled fields."""
        base_config = RetrievalConfig(
            hybrid_alpha=0.7,  # Custom value
            initial_k=200,     # Custom value
        )

        modified_config = apply_preset_to_retrieval_config(base_config, "balanced")

        # Preset-controlled fields should change
        self.assertEqual(modified_config.final_k, 64)

        # Non-preset fields should be preserved
        self.assertEqual(modified_config.hybrid_alpha, 0.7,
                        "Non-preset field should be preserved")
        self.assertEqual(modified_config.initial_k, 200,
                        "Non-preset field should be preserved")


class TestDefaultBehavior(unittest.TestCase):
    """Test that default behavior is unchanged without presets."""

    def test_default_behavior_unchanged_without_preset(self):
        """P7.1.11: RetrievalConfig without preset should use original defaults."""
        config = RetrievalConfig()

        # Default values should match original implementation
        # (These values are from current ace/config.py)
        self.assertEqual(config.final_k, 64, "Default final_k should be 64")
        self.assertEqual(config.use_hyde, "auto", "Default use_hyde should be 'auto'")
        self.assertTrue(config.enable_reranking, "Default reranking should be True")
        self.assertEqual(config.num_expanded_queries, 4, "Default expansion should be 4")

    def test_retrieval_config_without_preset_parameter(self):
        """P7.1.12: RetrievalConfig should work without preset parameter."""
        # This ensures backward compatibility
        config = RetrievalConfig(
            final_k=80,
            use_hyde=False
        )

        self.assertEqual(config.final_k, 80)
        self.assertFalse(config.use_hyde)


class TestPresetSwitchingPerformance(unittest.TestCase):
    """Test preset switching latency requirements."""

    def test_preset_switching_latency_under_1ms(self):
        """P7.1.13: Preset switching must complete in <1ms (target: <1ms)."""
        base_config = RetrievalConfig()

        iterations = 100
        total_time = 0.0

        for _ in range(iterations):
            start = time.perf_counter()
            _ = apply_preset_to_retrieval_config(base_config, "deep")
            end = time.perf_counter()
            total_time += (end - start)

        avg_latency_ms = (total_time / iterations) * 1000

        self.assertLess(avg_latency_ms, 1.0,
                       f"Average preset switching latency {avg_latency_ms:.3f}ms exceeds 1ms target")

    def test_get_preset_latency_under_1ms(self):
        """P7.1.14: get_preset() must complete in <1ms."""
        iterations = 100
        total_time = 0.0

        for _ in range(iterations):
            start = time.perf_counter()
            _ = get_preset("balanced")
            end = time.perf_counter()
            total_time += (end - start)

        avg_latency_ms = (total_time / iterations) * 1000

        self.assertLess(avg_latency_ms, 1.0,
                       f"Average get_preset() latency {avg_latency_ms:.3f}ms exceeds 1ms target")


class TestPresetImmutability(unittest.TestCase):
    """Test that preset configs are immutable."""

    def test_preset_config_immutable(self):
        """P7.1.15: PresetConfig instances should be immutable (frozen dataclass)."""
        preset = get_preset("fast")

        # Attempt to modify should raise FrozenInstanceError or AttributeError
        with self.assertRaises((AttributeError, TypeError),
                              msg="PresetConfig should be immutable"):
            preset.final_k = 100

    def test_get_preset_returns_copy_not_reference(self):
        """P7.1.16: get_preset() should return new instance, not shared reference."""
        preset1 = get_preset("balanced")
        preset2 = get_preset("balanced")

        # Same values but different objects
        self.assertEqual(preset1.final_k, preset2.final_k)
        self.assertIsNot(preset1, preset2,
                        "get_preset() should return new instance each time")


if __name__ == "__main__":
    unittest.main()
