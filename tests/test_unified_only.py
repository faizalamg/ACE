"""
Tests for unified-only memory architecture (no legacy fallbacks).

This test suite validates that:
1. All components use ONLY the unified memory system
2. No legacy fallback code paths exist
3. No dual-write mechanisms are active
4. Default parameters enable unified storage
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add ace module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ace.unified_memory import (
    UnifiedBullet,
    UnifiedMemoryIndex,
    UnifiedNamespace,
    UnifiedSource,
)


class TestAdapterUnifiedByDefault(unittest.TestCase):
    """Test that adapters use unified storage by default."""

    def test_adapter_base_defaults_to_unified_storage_true(self):
        """AdapterBase should default to use_unified_storage=True."""
        from ace.adaptation import AdapterBase
        import inspect

        sig = inspect.signature(AdapterBase.__init__)
        use_unified_param = sig.parameters.get("use_unified_storage")

        self.assertIsNotNone(use_unified_param, "use_unified_storage parameter should exist")
        self.assertTrue(
            use_unified_param.default,
            f"use_unified_storage should default to True, got {use_unified_param.default}"
        )

    def test_offline_adapter_uses_unified_by_default(self):
        """OfflineAdapter should use unified storage by default."""
        from ace.adaptation import OfflineAdapter
        from ace.playbook import Playbook

        # Create mock components
        mock_llm = Mock()
        mock_llm.complete = Mock(return_value=Mock(text='{"reasoning":"","final_answer":"test","bullet_ids":[]}'))

        generator = Mock()
        reflector = Mock()
        curator = Mock()

        # Create adapter without specifying use_unified_storage
        adapter = OfflineAdapter(
            playbook=Playbook(),
            generator=generator,
            reflector=reflector,
            curator=curator
        )

        # Should default to True
        self.assertTrue(
            adapter.use_unified_storage,
            "OfflineAdapter should default to use_unified_storage=True"
        )

    def test_online_adapter_uses_unified_by_default(self):
        """OnlineAdapter should use unified storage by default."""
        from ace.adaptation import OnlineAdapter
        from ace.playbook import Playbook

        generator = Mock()
        reflector = Mock()
        curator = Mock()

        adapter = OnlineAdapter(
            playbook=Playbook(),
            generator=generator,
            reflector=reflector,
            curator=curator
        )

        self.assertTrue(
            adapter.use_unified_storage,
            "OnlineAdapter should default to use_unified_storage=True"
        )


class TestCuratorUnifiedByDefault(unittest.TestCase):
    """Test that Curator uses unified storage by default."""

    def test_curator_defaults_to_store_unified_true(self):
        """Curator should default to store_to_unified=True."""
        from ace.roles import Curator
        import inspect

        sig = inspect.signature(Curator.__init__)
        store_param = sig.parameters.get("store_to_unified")

        self.assertIsNotNone(store_param, "store_to_unified parameter should exist")
        self.assertTrue(
            store_param.default,
            f"store_to_unified should default to True, got {store_param.default}"
        )


class TestNoLegacyFallbackInHooks(unittest.TestCase):
    """Test that hooks don't contain legacy fallback code."""

    def test_inject_context_has_no_legacy_fallback(self):
        """ace_inject_context.py should not have legacy fallback."""
        hook_path = Path.home() / ".claude" / "hooks" / "ace_inject_context.py"
        if not hook_path.exists():
            self.skipTest("Hook file not found - may be running in CI")

        content = hook_path.read_text(encoding="utf-8")

        # Should NOT contain legacy fallback patterns
        self.assertNotIn(
            "ace_qdrant_memory",
            content,
            "ace_inject_context.py should not import ace_qdrant_memory (legacy)"
        )
        self.assertNotIn(
            "fallback to legacy",
            content.lower(),
            "ace_inject_context.py should not contain 'fallback to legacy'"
        )
        self.assertNotIn(
            "falling back",
            content.lower(),
            "ace_inject_context.py should not contain 'falling back'"
        )

    def test_learn_feedback_has_no_legacy_fallback(self):
        """ace_learn_from_feedback.py should not have legacy fallback."""
        hook_path = Path.home() / ".claude" / "hooks" / "ace_learn_from_feedback.py"
        if not hook_path.exists():
            self.skipTest("Hook file not found - may be running in CI")

        content = hook_path.read_text(encoding="utf-8")

        # Should NOT contain legacy fallback patterns
        self.assertNotIn(
            "from ace_qdrant_memory import store_memory",
            content,
            "ace_learn_from_feedback.py should not import store_memory from ace_qdrant_memory"
        )
        self.assertNotIn(
            "fallback to old system",
            content.lower(),
            "ace_learn_from_feedback.py should not contain 'fallback to old system'"
        )

    def test_session_start_has_no_legacy_fallback(self):
        """ace_session_start.py should not have legacy fallback."""
        hook_path = Path.home() / ".claude" / "hooks" / "ace_session_start.py"
        if not hook_path.exists():
            self.skipTest("Hook file not found - may be running in CI")

        content = hook_path.read_text(encoding="utf-8")

        # Should NOT contain [LEGACY] indicator
        self.assertNotIn(
            "[LEGACY]",
            content,
            "ace_session_start.py should not output [LEGACY] indicator"
        )
        # Should NOT fallback to ace_qdrant_memory
        self.assertNotIn(
            "from ace_qdrant_memory import search_memories",
            content,
            "ace_session_start.py should not import search_memories from legacy module"
        )


class TestNoDualWriteInQdrantMemory(unittest.TestCase):
    """Test that ace_qdrant_memory.py doesn't dual-write."""

    def test_store_memory_does_not_dual_write(self):
        """store_memory should NOT write to both legacy and unified."""
        hook_path = Path.home() / ".claude" / "hooks" / "ace_qdrant_memory.py"
        if not hook_path.exists():
            self.skipTest("Hook file not found - may be running in CI")

        content = hook_path.read_text(encoding="utf-8")

        # Should NOT contain dual-write pattern
        # The old code had: "# UNIFIED STORAGE: Also store in unified memory index"
        self.assertNotIn(
            "Also store in unified",
            content,
            "store_memory should not dual-write to unified memory"
        )
        self.assertNotIn(
            "unified_index.index_bullet",
            content,
            "store_memory should not call unified_index.index_bullet (dual-write)"
        )


class TestUnifiedOnlyRetrieval(unittest.TestCase):
    """Test that retrieval uses only unified memory."""

    def test_search_memories_uses_unified_only(self):
        """search_memories should only use unified memory, not legacy collection."""
        hook_path = Path.home() / ".claude" / "hooks" / "ace_qdrant_memory.py"
        if not hook_path.exists():
            self.skipTest("Hook file not found - may be running in CI")

        content = hook_path.read_text(encoding="utf-8")

        # The function should be a simple wrapper around UnifiedMemoryIndex
        # Not contain the old hybrid collection search
        self.assertNotIn(
            "ace_memories_hybrid",
            content,
            "search_memories should not reference legacy collection ace_memories_hybrid"
        )


class TestWrapperFunctionsUseUnifiedOnly(unittest.TestCase):
    """Test that wrapper functions only use unified memory."""

    def test_store_preference_uses_unified_only(self):
        """store_preference should only use unified memory."""
        hook_path = Path.home() / ".claude" / "hooks" / "ace_qdrant_memory.py"
        if not hook_path.exists():
            self.skipTest("Hook file not found - may be running in CI")

        content = hook_path.read_text(encoding="utf-8")

        # Find store_preference function
        self.assertIn("def store_preference", content)

        # Extract function body and check it doesn't have legacy fallback
        import re
        func_match = re.search(r'def store_preference\([^)]*\):[^}]+?(?=\ndef |\Z)', content, re.DOTALL)
        if func_match:
            func_body = func_match.group(0)
            self.assertNotIn(
                "Fallback to legacy",
                func_body,
                "store_preference should not have legacy fallback"
            )

    def test_search_by_namespace_uses_unified_only(self):
        """search_by_namespace should only use unified memory."""
        hook_path = Path.home() / ".claude" / "hooks" / "ace_qdrant_memory.py"
        if not hook_path.exists():
            self.skipTest("Hook file not found - may be running in CI")

        content = hook_path.read_text(encoding="utf-8")

        # Find function
        self.assertIn("def search_by_namespace", content)

        # Should not have legacy fallback
        import re
        func_match = re.search(r'def search_by_namespace\([^)]*\):[^}]+?(?=\ndef |\Z)', content, re.DOTALL)
        if func_match:
            func_body = func_match.group(0)
            self.assertNotIn(
                "Fallback to legacy",
                func_body,
                "search_by_namespace should not have legacy fallback"
            )


class TestNoBackwardCompatibilityCode(unittest.TestCase):
    """Test that no backward compatibility code remains."""

    def test_no_unified_available_check(self):
        """Should not check if UNIFIED_AVAILABLE (always available)."""
        hook_path = Path.home() / ".claude" / "hooks" / "ace_qdrant_memory.py"
        if not hook_path.exists():
            self.skipTest("Hook file not found - may be running in CI")

        content = hook_path.read_text(encoding="utf-8")

        # Should not conditionally check for unified availability
        self.assertNotIn(
            "if not UNIFIED_AVAILABLE:",
            content,
            "Should not check UNIFIED_AVAILABLE - unified is always required"
        )
        self.assertNotIn(
            "UNIFIED_AVAILABLE = False",
            content,
            "Should not set UNIFIED_AVAILABLE to False"
        )


if __name__ == "__main__":
    unittest.main()
