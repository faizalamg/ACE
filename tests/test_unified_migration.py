"""
Test suite for ACE Framework Unified Memory Migration Tools

Phase 2: Migration Tools
- 2.1: Memory Migration Tool
- 2.2: Playbook Migration Tool

TDD Protocol: Write failing tests first, then implement.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from typing import List, Dict, Any
import json
import tempfile
import os

# Try to import migration tools
try:
    from scripts.migrate_memories_to_unified import (
        load_from_ace_memories_hybrid,
        migrate_memories_to_unified,
        MemoryMigrationResult,
    )
    MEMORY_MIGRATION_EXISTS = True
except ImportError:
    MEMORY_MIGRATION_EXISTS = False

try:
    from scripts.migrate_playbook_to_unified import (
        load_from_json_playbook,
        migrate_playbook_to_unified,
        PlaybookMigrationResult,
    )
    PLAYBOOK_MIGRATION_EXISTS = True
except ImportError:
    PLAYBOOK_MIGRATION_EXISTS = False

from ace.unified_memory import (
    UnifiedBullet,
    UnifiedNamespace,
    UnifiedSource,
    convert_bullet_to_unified,
    convert_memory_to_unified,
)


class TestMemoryMigration(unittest.TestCase):
    """Test Phase 2.1: Memory Migration Tool"""

    def setUp(self):
        if not MEMORY_MIGRATION_EXISTS:
            self.skipTest("scripts/migrate_memories_to_unified.py not implemented yet - TDD RED phase")

    def test_load_from_ace_memories_hybrid(self):
        """Test loading memories from existing Qdrant collection"""
        with patch('httpx.Client') as mock_client:
            # Mock scroll response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "result": {
                    "points": [
                        {
                            "id": 123,
                            "payload": {
                                "lesson": "Test memory 1",
                                "category": "WORKFLOW",
                                "severity": 8,
                                "reinforcement_count": 2,
                                "feedback_type": "DIRECTIVE"
                            }
                        },
                        {
                            "id": 456,
                            "payload": {
                                "lesson": "Test memory 2",
                                "category": "ARCHITECTURE",
                                "severity": 6,
                                "reinforcement_count": 1,
                                "feedback_type": "GENERAL"
                            }
                        }
                    ],
                    "next_page_offset": None
                }
            }
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response

            memories = load_from_ace_memories_hybrid()

            self.assertEqual(len(memories), 2)
            self.assertEqual(memories[0]["lesson"], "Test memory 1")
            self.assertEqual(memories[1]["category"], "ARCHITECTURE")

    def test_migrate_memories_dry_run(self):
        """Test migration with --dry-run flag (no actual writes)"""
        test_memories = [
            {
                "lesson": "Test memory",
                "category": "WORKFLOW",
                "severity": 7,
                "reinforcement_count": 1,
                "feedback_type": "GENERAL"
            }
        ]

        with patch('scripts.migrate_memories_to_unified.load_from_ace_memories_hybrid') as mock_load:
            mock_load.return_value = test_memories

            result = migrate_memories_to_unified(dry_run=True)

            self.assertIsInstance(result, MemoryMigrationResult)
            self.assertEqual(result.total_source, 1)
            self.assertEqual(result.migrated, 0)  # Dry run - no actual migration
            self.assertTrue(result.dry_run)

    def test_migrate_memories_full(self):
        """Test full migration of memories"""
        test_memories = [
            {
                "lesson": "Memory 1",
                "category": "WORKFLOW",
                "severity": 8,
                "reinforcement_count": 2,
                "feedback_type": "DIRECTIVE"
            },
            {
                "lesson": "Memory 2",
                "category": "DEBUGGING",
                "severity": 6,
                "reinforcement_count": 1,
                "feedback_type": "GENERAL"
            }
        ]

        with patch('scripts.migrate_memories_to_unified.load_from_ace_memories_hybrid') as mock_load:
            with patch('scripts.migrate_memories_to_unified.UnifiedMemoryIndex') as mock_index:
                mock_load.return_value = test_memories
                mock_index_instance = Mock()
                mock_index_instance.batch_index.return_value = 2
                mock_index.return_value = mock_index_instance

                result = migrate_memories_to_unified(dry_run=False)

                self.assertEqual(result.total_source, 2)
                self.assertEqual(result.migrated, 2)
                self.assertFalse(result.dry_run)

    def test_migrate_memories_verify(self):
        """Test migration verification (count before/after)"""
        with patch('scripts.migrate_memories_to_unified.load_from_ace_memories_hybrid') as mock_load:
            with patch('scripts.migrate_memories_to_unified.UnifiedMemoryIndex') as mock_index:
                mock_load.return_value = [{"lesson": "Test", "category": "TEST", "severity": 5}]
                mock_index_instance = Mock()
                mock_index_instance.batch_index.return_value = 1  # Must return int
                mock_index_instance.count.return_value = 1
                mock_index_instance.create_collection.return_value = True
                mock_index.return_value = mock_index_instance

                result = migrate_memories_to_unified(dry_run=False, verify=True)

                self.assertTrue(result.verified)
                self.assertEqual(result.target_count, 1)

    def test_convert_memory_preserves_data(self):
        """Test that memory conversion preserves all relevant data"""
        memory = {
            "lesson": "Important lesson learned",
            "category": "DEBUGGING",
            "severity": 9,
            "reinforcement_count": 5,
            "feedback_type": "FRUSTRATION",
            "timestamp": "2025-01-01T00:00:00Z"
        }

        unified = convert_memory_to_unified(memory)

        self.assertEqual(unified.content, "Important lesson learned")
        self.assertEqual(unified.namespace, "user_prefs")
        self.assertEqual(unified.source, "migration")
        self.assertEqual(unified.severity, 9)
        self.assertEqual(unified.reinforcement_count, 5)
        self.assertEqual(unified.category, "DEBUGGING")
        self.assertEqual(unified.feedback_type, "FRUSTRATION")

    def test_severity_mapping_to_section(self):
        """Test that severity/category maps to appropriate section"""
        high_severity = convert_memory_to_unified({
            "lesson": "Critical", "category": "PROTOCOL", "severity": 9
        })
        medium_severity = convert_memory_to_unified({
            "lesson": "Normal", "category": "WORKFLOW", "severity": 5
        })

        # Should map to task_guidance for protocol
        self.assertEqual(high_severity.section, "task_guidance")
        # Should map to common_patterns for workflow
        self.assertEqual(medium_severity.section, "common_patterns")


class TestPlaybookMigration(unittest.TestCase):
    """Test Phase 2.2: Playbook Migration Tool"""

    def setUp(self):
        if not PLAYBOOK_MIGRATION_EXISTS:
            self.skipTest("scripts/migrate_playbook_to_unified.py not implemented yet - TDD RED phase")

    def test_load_from_json_playbook(self):
        """Test loading playbook from JSON file"""
        playbook_data = {
            "bullets": {
                "task-001": {
                    "id": "task-001",
                    "section": "task_guidance",
                    "content": "Filter by date range first",
                    "helpful": 5,
                    "harmful": 1
                },
                "error-001": {
                    "id": "error-001",
                    "section": "common_errors",
                    "content": "Check null before access",
                    "helpful": 3,
                    "harmful": 0
                }
            },
            "sections": {
                "task_guidance": ["task-001"],
                "common_errors": ["error-001"]
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(playbook_data, f)
            temp_path = f.name

        try:
            bullets = load_from_json_playbook(temp_path)

            self.assertEqual(len(bullets), 2)
            self.assertIn("task-001", [b.id for b in bullets])
        finally:
            os.unlink(temp_path)

    def test_migrate_playbook_dry_run(self):
        """Test playbook migration with --dry-run flag"""
        from ace.playbook import Bullet

        test_bullets = [
            Bullet(id="b-001", section="task_guidance", content="Test bullet", helpful=1, harmful=0)
        ]

        with patch('scripts.migrate_playbook_to_unified.load_from_json_playbook') as mock_load:
            mock_load.return_value = test_bullets

            result = migrate_playbook_to_unified("fake_path.json", dry_run=True)

            self.assertIsInstance(result, PlaybookMigrationResult)
            self.assertEqual(result.total_source, 1)
            self.assertEqual(result.migrated, 0)  # Dry run
            self.assertTrue(result.dry_run)

    def test_migrate_playbook_full(self):
        """Test full playbook migration"""
        from ace.playbook import Bullet, EnrichedBullet

        test_bullets = [
            Bullet(id="b-001", section="task_guidance", content="Bullet 1"),
            EnrichedBullet(
                id="b-002",
                section="common_errors",
                content="Enriched bullet",
                task_types=["debugging"],
                trigger_patterns=["error", "exception"]
            )
        ]

        with patch('scripts.migrate_playbook_to_unified.load_from_json_playbook') as mock_load:
            with patch('scripts.migrate_playbook_to_unified.UnifiedMemoryIndex') as mock_index:
                mock_load.return_value = test_bullets
                mock_index_instance = Mock()
                mock_index_instance.batch_index.return_value = 2
                mock_index.return_value = mock_index_instance

                result = migrate_playbook_to_unified("fake_path.json", dry_run=False)

                self.assertEqual(result.total_source, 2)
                self.assertEqual(result.migrated, 2)

    def test_convert_enriched_bullet_preserves_scaffolding(self):
        """Test that EnrichedBullet scaffolding is preserved"""
        from ace.playbook import EnrichedBullet

        enriched = EnrichedBullet(
            id="enr-001",
            section="task_guidance",
            content="Use try-except",
            helpful=5,
            harmful=1,
            task_types=["debugging", "error_handling"],
            domains=["python"],
            trigger_patterns=["try", "except", "error"],
            complexity_level="simple"
        )

        unified = convert_bullet_to_unified(enriched)

        self.assertEqual(unified.task_types, ["debugging", "error_handling"])
        self.assertEqual(unified.domains, ["python"])
        self.assertEqual(unified.trigger_patterns, ["try", "except", "error"])
        self.assertEqual(unified.complexity, "simple")
        self.assertEqual(unified.helpful_count, 5)
        self.assertEqual(unified.harmful_count, 1)

    def test_migrate_playbook_verify(self):
        """Test playbook migration verification"""
        from ace.playbook import Bullet

        test_bullets = [Bullet(id="b-001", section="test", content="Test")]

        with patch('scripts.migrate_playbook_to_unified.load_from_json_playbook') as mock_load:
            with patch('scripts.migrate_playbook_to_unified.UnifiedMemoryIndex') as mock_index:
                mock_load.return_value = test_bullets
                mock_index_instance = Mock()
                mock_index_instance.batch_index.return_value = 1
                mock_index_instance.count.return_value = 1
                mock_index.return_value = mock_index_instance

                result = migrate_playbook_to_unified("fake.json", dry_run=False, verify=True)

                self.assertTrue(result.verified)


class TestMigrationCLI(unittest.TestCase):
    """Test CLI interface for migration scripts"""

    def setUp(self):
        if not (MEMORY_MIGRATION_EXISTS and PLAYBOOK_MIGRATION_EXISTS):
            self.skipTest("Migration scripts not implemented yet - TDD RED phase")

    def test_memory_migration_cli_dry_run(self):
        """Test memory migration CLI with --dry-run"""
        import subprocess
        result = subprocess.run(
            ['python', '-m', 'scripts.migrate_memories_to_unified', '--dry-run'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        # Should not fail
        self.assertEqual(result.returncode, 0)

    def test_playbook_migration_cli_dry_run(self):
        """Test playbook migration CLI with --dry-run"""
        # Create temp playbook file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"bullets": {}, "sections": {}}, f)
            temp_path = f.name

        try:
            import subprocess
            result = subprocess.run(
                ['python', '-m', 'scripts.migrate_playbook_to_unified', temp_path, '--dry-run'],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(__file__))
            )
            self.assertEqual(result.returncode, 0)
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()
