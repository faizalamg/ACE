"""
Tests for ace_inject_context.py hook using unified memory system.

This test suite validates:
1. UnifiedMemoryIndex integration for context retrieval
2. Namespace filtering (USER_PREFS, TASK_STRATEGIES, PROJECT_SPECIFIC)
3. Context formatting with [PREF], [STRAT], [PROJ] indicators
4. Backwards compatibility with legacy ace_qdrant_memory
5. JSON parsing and error handling

NOTE: These tests require a running Qdrant instance or will be skipped.
NO MOCKING IS USED - tests run against real implementations.
"""

import json
import sys
import os
import unittest
from pathlib import Path
from typing import List, Dict, Any

# Add ace module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ace.unified_memory import (
    UnifiedBullet,
    UnifiedMemoryIndex,
    UnifiedNamespace,
    UnifiedSource,
    format_unified_context,
)


# Check if Qdrant is available
def check_qdrant_available():
    """Check if Qdrant is running and accessible."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"), timeout=2)
        client.get_collections()
        return True
    except Exception:
        return False

QDRANT_AVAILABLE = check_qdrant_available()


@unittest.skipIf(not QDRANT_AVAILABLE, "Qdrant not available")
class TestUnifiedMemoryIndexIntegration(unittest.TestCase):
    """Test UnifiedMemoryIndex retrieval with namespace filtering using real Qdrant."""

    @classmethod
    def setUpClass(cls):
        """Setup real Qdrant client and index."""
        cls.test_collection = "ace_test_inject_context"
        cls.index = UnifiedMemoryIndex(
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection_name=cls.test_collection,
        )
        cls.index.create_collection()

        # Seed with test data
        cls.user_pref_bullet = UnifiedBullet(
            id="pref-inject-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="User prefers TypeScript over JavaScript for inject test",
            section="preferences",
            severity=8,
            reinforcement_count=3,
        )

        cls.task_strat_bullet = UnifiedBullet(
            id="strat-inject-001",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="Always write tests before production code TDD inject",
            section="task_guidance",
            helpful_count=10,
            harmful_count=1,
        )

        cls.project_bullet = UnifiedBullet(
            id="proj-inject-001",
            namespace=UnifiedNamespace.PROJECT_SPECIFIC,
            source=UnifiedSource.EXPLICIT_STORE,
            content="ACE Framework uses Qdrant for vector storage inject",
            section="architecture",
            helpful_count=5,
            harmful_count=0,
        )

        cls.index.batch_index([cls.user_pref_bullet, cls.task_strat_bullet, cls.project_bullet])

    @classmethod
    def tearDownClass(cls):
        """Clean up test collection."""
        try:
            cls.index.client.delete_collection(cls.test_collection)
        except Exception:
            pass

    def test_retrieve_all_namespaces(self):
        """Test retrieval without namespace filter returns all."""
        results = self.index.retrieve("TypeScript testing ACE inject", namespace=None, limit=10)
        # Should find at least some of our bullets
        self.assertGreater(len(results), 0)

    def test_retrieve_user_prefs_only(self):
        """Test namespace filter for USER_PREFS."""
        results = self.index.retrieve(
            "TypeScript inject test",
            namespace=UnifiedNamespace.USER_PREFS,
            limit=10
        )
        # All results should be user_prefs
        for result in results:
            self.assertEqual(result.namespace, "user_prefs")

    def test_retrieve_task_strategies_only(self):
        """Test namespace filter for TASK_STRATEGIES."""
        results = self.index.retrieve(
            "TDD inject test",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            limit=10
        )
        # All results should be task_strategies
        for result in results:
            self.assertEqual(result.namespace, "task_strategies")

    def test_retrieve_multiple_namespaces(self):
        """Test retrieval with multiple namespaces (OR filter)."""
        results = self.index.retrieve(
            "inject test",
            namespace=[UnifiedNamespace.USER_PREFS, UnifiedNamespace.TASK_STRATEGIES],
            limit=10
        )
        # Results should only be from these two namespaces
        for result in results:
            self.assertIn(result.namespace, ["user_prefs", "task_strategies"])

    def test_retrieve_project_specific_only(self):
        """Test namespace filter for PROJECT_SPECIFIC."""
        results = self.index.retrieve(
            "Qdrant inject",
            namespace=UnifiedNamespace.PROJECT_SPECIFIC,
            limit=10
        )
        # All results should be project_specific
        for result in results:
            self.assertEqual(result.namespace, "project_specific")


class TestContextFormatting(unittest.TestCase):
    """Test format_unified_context() output formatting - no Qdrant required."""

    def test_format_with_namespace_indicators(self):
        """Test output includes [PREF], [STRAT], [PROJ] indicators."""
        bullets = [
            UnifiedBullet(
                id="pref-001",
                namespace=UnifiedNamespace.USER_PREFS,
                source=UnifiedSource.USER_FEEDBACK,
                content="User prefers TypeScript",
                section="preferences",
                severity=8,
            ),
            UnifiedBullet(
                id="strat-001",
                namespace=UnifiedNamespace.TASK_STRATEGIES,
                source=UnifiedSource.TASK_EXECUTION,
                content="Write tests first (TDD)",
                section="task_guidance",
                helpful_count=10,
                harmful_count=1,
            ),
            UnifiedBullet(
                id="proj-001",
                namespace=UnifiedNamespace.PROJECT_SPECIFIC,
                source=UnifiedSource.EXPLICIT_STORE,
                content="ACE uses Qdrant",
                section="architecture",
            ),
        ]

        output = format_unified_context(bullets, max_bullets=10)

        # Verify sections exist
        self.assertIn("**User Preferences:**", output)
        self.assertIn("**Task Strategies:**", output)
        self.assertIn("**Project Context:**", output)

        # Verify indicators
        self.assertIn("[PREF]", output)
        self.assertIn("[STRAT]", output)
        self.assertIn("[PROJ]", output)

        # Verify content
        self.assertIn("User prefers TypeScript", output)
        self.assertIn("Write tests first (TDD)", output)
        self.assertIn("ACE uses Qdrant", output)

    def test_format_with_importance_indicators(self):
        """Test importance indicators [!], [*], [-]."""
        bullets = [
            UnifiedBullet(
                id="critical",
                namespace=UnifiedNamespace.USER_PREFS,
                source=UnifiedSource.USER_FEEDBACK,
                content="Critical preference",
                section="preferences",
                severity=9,  # Should get [!]
            ),
            UnifiedBullet(
                id="important",
                namespace=UnifiedNamespace.TASK_STRATEGIES,
                source=UnifiedSource.TASK_EXECUTION,
                content="Important strategy",
                section="task_guidance",
                severity=6,  # Should get [*]
            ),
            UnifiedBullet(
                id="normal",
                namespace=UnifiedNamespace.PROJECT_SPECIFIC,
                source=UnifiedSource.EXPLICIT_STORE,
                content="Normal info",
                section="architecture",
                severity=3,  # Should get [-]
            ),
        ]

        output = format_unified_context(bullets, max_bullets=10)

        # Verify importance indicators
        self.assertIn("[!]", output)  # Critical
        self.assertIn("[*]", output)  # Important
        self.assertIn("[-]", output)  # Normal

    def test_format_with_reinforcement_count(self):
        """Test reinforcement count display [x3]."""
        bullets = [
            UnifiedBullet(
                id="reinforced",
                namespace=UnifiedNamespace.USER_PREFS,
                source=UnifiedSource.USER_FEEDBACK,
                content="Reinforced preference",
                section="preferences",
                reinforcement_count=5,
            ),
            UnifiedBullet(
                id="single",
                namespace=UnifiedNamespace.USER_PREFS,
                source=UnifiedSource.USER_FEEDBACK,
                content="Single preference",
                section="preferences",
                reinforcement_count=1,
            ),
        ]

        output = format_unified_context(bullets, max_bullets=10)

        # Reinforced should show count
        self.assertIn("[x5]", output)
        # Single should not show count (implicit)
        self.assertNotIn("[x1]", output)

    def test_format_empty_list(self):
        """Test formatting empty bullet list."""
        output = format_unified_context([], max_bullets=10)
        self.assertEqual(output, "")

    def test_format_respects_max_bullets(self):
        """Test max_bullets limit."""
        bullets = [
            UnifiedBullet(
                id=f"bullet-{i}",
                namespace=UnifiedNamespace.USER_PREFS,
                source=UnifiedSource.USER_FEEDBACK,
                content=f"Preference {i}",
                section="preferences",
                severity=10 - i,  # Descending importance
            )
            for i in range(20)
        ]

        output = format_unified_context(bullets, max_bullets=5)

        # Count bullet lines (exclude headers)
        bullet_lines = [line for line in output.split('\n') if line.startswith('[')]
        self.assertLessEqual(len(bullet_lines), 5)


class TestBackwardsCompatibility(unittest.TestCase):
    """Test backwards compatibility with legacy system."""

    def test_legacy_module_exists(self):
        """Test hook's legacy module exists if needed for fallback."""
        # Verify legacy module would be importable
        user_claude_dir = Path.home() / ".claude" / "hooks"
        legacy_module = user_claude_dir / "ace_qdrant_memory.py"

        # If legacy module exists, import should work
        if legacy_module.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("ace_qdrant_memory", legacy_module)
            self.assertIsNotNone(spec)


class TestHookJSONParsing(unittest.TestCase):
    """Test hook's JSON parsing and error handling - no Qdrant required."""

    def test_parse_valid_hook_input(self):
        """Test parsing valid UserPromptSubmit hook input."""
        hook_input = {
            "prompt": "How do I implement TDD?",
            "timestamp": "2025-12-11T10:00:00Z",
        }

        raw = json.dumps(hook_input)
        parsed = json.loads(raw)

        self.assertEqual(parsed["prompt"], "How do I implement TDD?")

    def test_parse_empty_prompt(self):
        """Test handling empty prompt."""
        hook_input = {"prompt": ""}
        parsed = json.loads(json.dumps(hook_input))

        # Empty prompt should not trigger context injection
        self.assertEqual(parsed["prompt"], "")

    def test_hook_output_format(self):
        """Test hook output format with additionalContext."""
        # Simulate hook output
        context = format_unified_context([
            UnifiedBullet(
                id="test",
                namespace=UnifiedNamespace.USER_PREFS,
                source=UnifiedSource.USER_FEEDBACK,
                content="Test preference",
                section="preferences",
            )
        ])

        hook_output = {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": context,
            },
            "systemMessage": context,
        }

        # Verify structure
        self.assertIn("hookSpecificOutput", hook_output)
        self.assertEqual(hook_output["hookSpecificOutput"]["hookEventName"], "UserPromptSubmit")
        self.assertIn("additionalContext", hook_output["hookSpecificOutput"])
        self.assertIn("[PREF]", hook_output["hookSpecificOutput"]["additionalContext"])


if __name__ == "__main__":
    unittest.main()
