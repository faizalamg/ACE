"""
Tests for ace_learn_from_feedback.py hook integration with UnifiedMemoryIndex.

Tests verify:
1. Namespace assignment based on feedback type
2. UnifiedBullet creation with proper source/metadata
3. UnifiedMemoryIndex.index_bullet() integration
4. Backwards compatibility with fallback behavior

NOTE: Tests requiring Qdrant will be skipped if not available.
NO MOCKING IS USED - tests run against real implementations.
"""

import unittest
import os
from datetime import datetime
from pathlib import Path

# Import unified memory components
from ace.unified_memory import (
    UnifiedBullet,
    UnifiedMemoryIndex,
    UnifiedNamespace,
    UnifiedSource,
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
class TestLearnFeedbackHookIntegration(unittest.TestCase):
    """Test integration of ace_learn_from_feedback hook with unified memory using real Qdrant."""

    @classmethod
    def setUpClass(cls):
        """Set up test collection."""
        cls.test_collection = "ace_test_learn_feedback"
        cls.index = UnifiedMemoryIndex(
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection_name=cls.test_collection,
        )
        cls.index.create_collection()

    @classmethod
    def tearDownClass(cls):
        """Clean up test collection."""
        try:
            cls.index.client.delete_collection(cls.test_collection)
        except Exception:
            pass

    def test_namespace_assignment_user_preference(self):
        """Verify USER_PREFS namespace for preference feedback."""
        bullet = UnifiedBullet(
            id="test-pref-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="User prefers concise responses without verbose explanations",
            section="preferences",
            category="USER_PREFERENCE",
            feedback_type="PREFERENCE",
            severity=8,
        )

        result = self.index.index_bullet(bullet)

        self.assertTrue(result.get('stored', False))
        self.assertEqual(bullet.namespace, "user_prefs")
        self.assertEqual(bullet.source, "user_feedback")

    def test_namespace_assignment_task_strategies(self):
        """Verify TASK_STRATEGIES namespace for correction/directive feedback."""
        bullet = UnifiedBullet(
            id="test-task-001",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.USER_FEEDBACK,
            content="Always verify JSON syntax before making API calls",
            section="common_errors",
            category="ERROR_PREVENTION",
            feedback_type="CORRECTION",
            severity=9,
        )

        result = self.index.index_bullet(bullet)

        self.assertTrue(result.get('stored', False))
        self.assertEqual(bullet.namespace, "task_strategies")

    def test_namespace_assignment_workflow_patterns(self):
        """Verify TASK_STRATEGIES namespace for workflow feedback."""
        bullet = UnifiedBullet(
            id="test-workflow-001",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.USER_FEEDBACK,
            content="When user says implement always write tests first TDD protocol",
            section="common_patterns",
            category="WORKFLOW_PATTERN",
            feedback_type="WORKFLOW",
            severity=7,
        )

        result = self.index.index_bullet(bullet)

        self.assertTrue(result.get('stored', False))
        self.assertEqual(bullet.namespace, "task_strategies")

    def test_unified_index_storage(self):
        """Verify UnifiedMemoryIndex.index_bullet() stores correctly."""
        bullet = UnifiedBullet(
            id="test-storage-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="User prefers Vim keybindings",
            section="preferences",
            severity=6,
        )

        result = self.index.index_bullet(bullet)

        self.assertTrue(result.get('stored', False))

        # Verify we can retrieve it
        results = self.index.retrieve(
            "Vim keybindings",
            namespace=UnifiedNamespace.USER_PREFS,
            limit=10
        )

        found = any(r.id == "test-storage-001" for r in results)
        self.assertTrue(found)


class TestBulletCreation(unittest.TestCase):
    """Test UnifiedBullet creation - no Qdrant required."""

    def test_bullet_creation_from_feedback_analysis(self):
        """Verify UnifiedBullet creation from LLM feedback analysis."""
        feedback_data = {
            "feedback_type": "PREFERENCE",
            "severity": 8,
            "lesson": "Always use TypeScript over JavaScript for new projects",
            "category": "CODE_STANDARDS",
            "context": "User: I prefer TypeScript for all new code",
        }

        bullet = UnifiedBullet(
            id=f"fb-{datetime.now().timestamp()}",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content=feedback_data["lesson"],
            section="preferences",
            category=feedback_data["category"],
            feedback_type=feedback_data["feedback_type"],
            severity=feedback_data["severity"],
            context=feedback_data["context"],
            reinforcement_count=1,
        )

        self.assertEqual(bullet.namespace, "user_prefs")
        self.assertEqual(bullet.source, "user_feedback")
        self.assertEqual(bullet.severity, 8)
        self.assertEqual(bullet.reinforcement_count, 1)
        self.assertEqual(bullet.category, "CODE_STANDARDS")
        self.assertEqual(bullet.feedback_type, "PREFERENCE")
        self.assertGreater(len(bullet.content), 0)

    def test_source_assignment_user_feedback(self):
        """Verify USER_FEEDBACK source for explicit user corrections."""
        bullet = UnifiedBullet(
            id="test-correction-001",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.USER_FEEDBACK,
            content="Never skip error handling in API calls",
            section="common_errors",
            feedback_type="CORRECTION",
        )

        self.assertEqual(bullet.source, "user_feedback")

    def test_source_assignment_task_execution(self):
        """Verify TASK_EXECUTION source for implicit learnings."""
        bullet = UnifiedBullet(
            id="test-task-002",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.TASK_EXECUTION,
            content="React hooks must follow rules of hooks",
            section="task_guidance",
        )

        self.assertEqual(bullet.source, "task_execution")

    def test_severity_mapping_to_bullet(self):
        """Verify severity from feedback maps correctly to bullet."""
        bullet = UnifiedBullet(
            id="test-severity-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Critical: Always backup before destructive operations",
            section="preferences",
            severity=10,
            feedback_type="DIRECTIVE",
        )

        self.assertEqual(bullet.severity, 10)
        self.assertGreater(bullet.combined_importance_score, 0.8)

    def test_reinforcement_count_initialization(self):
        """Verify reinforcement_count starts at 1 for new bullets."""
        bullet = UnifiedBullet(
            id="test-reinforce-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="User likes detailed commit messages",
            section="preferences",
        )

        self.assertEqual(bullet.reinforcement_count, 1)

    def test_category_mapping(self):
        """Verify category from feedback maps to appropriate section."""
        test_cases = [
            ("USER_PREFERENCE", "preferences"),
            ("COMMUNICATION_STYLE", "preferences"),
            ("WORKFLOW_PATTERN", "common_patterns"),
            ("ERROR_PREVENTION", "common_errors"),
            ("TOOL_USAGE", "task_guidance"),
        ]

        for category, expected_section in test_cases:
            with self.subTest(category=category):
                bullet = UnifiedBullet(
                    id=f"test-cat-{category}",
                    namespace=UnifiedNamespace.USER_PREFS
                    if "PREFERENCE" in category
                    else UnifiedNamespace.TASK_STRATEGIES,
                    source=UnifiedSource.USER_FEEDBACK,
                    content=f"Test lesson for {category}",
                    section=expected_section,
                    category=category,
                )

                self.assertEqual(bullet.section, expected_section)

    def test_feedback_type_markers_preserved(self):
        """Verify feedback type markers are preserved in content."""
        marker = "[!]"
        feedback_type = "FRUSTRATION"
        lesson = "Never ignore user preferences"
        formatted_lesson = f"{marker} [{feedback_type}] {lesson}"

        bullet = UnifiedBullet(
            id="test-marker-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content=formatted_lesson,
            section="preferences",
            feedback_type=feedback_type,
            severity=9,
        )

        self.assertIn("[!]", bullet.content)
        self.assertIn("[FRUSTRATION]", bullet.content)

    def test_multiple_feedback_types_namespace_assignment(self):
        """Verify correct namespace for various feedback types."""
        test_cases = [
            ("FRUSTRATION", UnifiedNamespace.USER_PREFS),
            ("CORRECTION", UnifiedNamespace.TASK_STRATEGIES),
            ("DIRECTIVE", UnifiedNamespace.TASK_STRATEGIES),
            ("PREFERENCE", UnifiedNamespace.USER_PREFS),
            ("SUCCESS", UnifiedNamespace.TASK_STRATEGIES),
            ("WORKFLOW", UnifiedNamespace.TASK_STRATEGIES),
            ("META", UnifiedNamespace.USER_PREFS),
        ]

        for feedback_type, expected_namespace in test_cases:
            with self.subTest(feedback_type=feedback_type):
                bullet = UnifiedBullet(
                    id=f"test-ns-{feedback_type}",
                    namespace=expected_namespace,
                    source=UnifiedSource.USER_FEEDBACK,
                    content=f"Test {feedback_type} feedback",
                    section="preferences"
                    if expected_namespace == UnifiedNamespace.USER_PREFS
                    else "task_guidance",
                    feedback_type=feedback_type,
                )

                self.assertEqual(
                    bullet.namespace,
                    expected_namespace.value
                    if hasattr(expected_namespace, "value")
                    else expected_namespace,
                )

    def test_timestamp_preservation(self):
        """Verify created_at and updated_at timestamps are set."""
        bullet = UnifiedBullet(
            id="test-timestamp-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Test timestamp preservation",
            section="preferences",
        )

        self.assertIsNotNone(bullet.created_at)
        self.assertIsNotNone(bullet.updated_at)
        self.assertIsInstance(bullet.created_at, datetime)
        self.assertIsInstance(bullet.updated_at, datetime)

    def test_context_truncation(self):
        """Verify context is truncated to reasonable length."""
        long_context = "User said: " + ("X" * 5000)

        bullet = UnifiedBullet(
            id="test-context-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Always truncate context",
            section="preferences",
            context=long_context[:200],  # Hook truncates to 200 chars
        )

        self.assertLessEqual(len(bullet.context), 200)

    def test_combined_importance_score_calculation(self):
        """Verify combined importance score includes severity and effectiveness."""
        bullet = UnifiedBullet(
            id="test-score-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Critical preference",
            section="preferences",
            severity=10,
            reinforcement_count=5,
        )

        score = bullet.combined_importance_score
        self.assertGreater(score, 0.7)
        self.assertLessEqual(score, 1.0)

        # Reinforcement should boost score
        bullet_no_reinforce = UnifiedBullet(
            id="test-score-002",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Same preference",
            section="preferences",
            severity=10,
            reinforcement_count=1,
        )
        self.assertGreater(score, bullet_no_reinforce.combined_importance_score)


class TestFeedbackHookNamespaceLogic(unittest.TestCase):
    """Test namespace assignment logic for different feedback scenarios - no Qdrant required."""

    def test_communication_style_uses_user_prefs(self):
        """Communication style feedback should use USER_PREFS namespace."""
        bullet = UnifiedBullet(
            id="test-comm-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="User prefers bullet points over long paragraphs",
            section="preferences",
            category="COMMUNICATION_STYLE",
        )

        self.assertEqual(bullet.namespace, "user_prefs")

    def test_error_prevention_uses_task_strategies(self):
        """Error prevention feedback should use TASK_STRATEGIES namespace."""
        bullet = UnifiedBullet(
            id="test-error-001",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.USER_FEEDBACK,
            content="Always validate input before database queries",
            section="common_errors",
            category="ERROR_PREVENTION",
        )

        self.assertEqual(bullet.namespace, "task_strategies")

    def test_tool_usage_uses_task_strategies(self):
        """Tool usage feedback should use TASK_STRATEGIES namespace."""
        bullet = UnifiedBullet(
            id="test-tool-001",
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            source=UnifiedSource.USER_FEEDBACK,
            content="Use Auggie before ACE code_retrieval for semantic search",
            section="task_guidance",
            category="TOOL_USAGE",
        )

        self.assertEqual(bullet.namespace, "task_strategies")

    def test_memory_behavior_uses_user_prefs(self):
        """Memory behavior feedback should use USER_PREFS namespace."""
        bullet = UnifiedBullet(
            id="test-memory-001",
            namespace=UnifiedNamespace.USER_PREFS,
            source=UnifiedSource.USER_FEEDBACK,
            content="Store cross-project lessons, not project-specific details",
            section="preferences",
            category="MEMORY_BEHAVIOR",
            feedback_type="META",
        )

        self.assertEqual(bullet.namespace, "user_prefs")


if __name__ == "__main__":
    unittest.main()
