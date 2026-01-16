"""Tests for migrating existing bullets to enriched schema.

TDD: These tests define the expected behavior BEFORE implementation.
"""

import json
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest


@pytest.mark.unit
class TestBulletMigration(unittest.TestCase):
    """Test bullet migration from basic to enriched schema."""

    def test_migrate_basic_bullet_to_enriched(self):
        """Test migration of a basic bullet to enriched bullet."""
        from ace import Playbook
        from ace.playbook import Bullet, EnrichedBullet, migrate_bullet

        basic_bullet = Bullet(
            id="test-1",
            section="math",
            content="Use step-by-step reasoning",
            helpful=5,
            harmful=1,
            neutral=2,
        )

        enriched = migrate_bullet(basic_bullet)

        # Should be EnrichedBullet now
        self.assertIsInstance(enriched, EnrichedBullet)

        # Should preserve original fields
        self.assertEqual(enriched.id, "test-1")
        self.assertEqual(enriched.section, "math")
        self.assertEqual(enriched.content, "Use step-by-step reasoning")
        self.assertEqual(enriched.helpful, 5)
        self.assertEqual(enriched.harmful, 1)
        self.assertEqual(enriched.neutral, 2)

        # Should have enriched fields (from heuristic enrichment)
        self.assertIsInstance(enriched.task_types, list)
        self.assertIsInstance(enriched.domains, list)

    def test_migrate_already_enriched_bullet(self):
        """Test that already enriched bullets are not modified."""
        from ace.playbook import EnrichedBullet, migrate_bullet

        enriched_bullet = EnrichedBullet(
            id="test-2",
            section="debug",
            content="Check logs first",
            task_types=["debugging"],
            domains=["software"],
        )

        result = migrate_bullet(enriched_bullet)

        # Should be the same instance (or equal)
        self.assertEqual(result.id, enriched_bullet.id)
        self.assertEqual(result.task_types, ["debugging"])
        self.assertEqual(result.domains, ["software"])


@pytest.mark.unit
class TestPlaybookMigration(unittest.TestCase):
    """Test playbook-level migration."""

    def test_playbook_migrate_all_bullets(self):
        """Test migrating all bullets in a playbook."""
        from ace import Playbook
        from ace.playbook import EnrichedBullet

        playbook = Playbook()

        # Add basic bullets
        playbook.add_bullet("math", "Solve step by step")
        playbook.add_bullet("coding", "Write tests first")
        playbook.add_bullet("debug", "Check error logs")

        # Migrate all bullets
        migrated_count = playbook.migrate_to_enriched()

        # Should have migrated all bullets
        self.assertEqual(migrated_count, 3)

        # All bullets should now be EnrichedBullet
        for bullet in playbook.bullets():
            self.assertIsInstance(bullet, EnrichedBullet)

    def test_playbook_migration_preserves_sections(self):
        """Test that migration preserves section organization."""
        from ace import Playbook

        playbook = Playbook()
        playbook.add_bullet("math", "Math tip 1")
        playbook.add_bullet("math", "Math tip 2")
        playbook.add_bullet("coding", "Coding tip 1")

        playbook.migrate_to_enriched()

        # Sections should be preserved
        sections = playbook.list_sections()
        self.assertIn("math", sections)
        self.assertIn("coding", sections)

    def test_playbook_migration_idempotent(self):
        """Test that migration is idempotent (running twice doesn't change anything)."""
        from ace import Playbook

        playbook = Playbook()
        playbook.add_bullet("test", "Test content")

        # Migrate twice
        count1 = playbook.migrate_to_enriched()
        count2 = playbook.migrate_to_enriched()

        # First migration should migrate 1 bullet
        self.assertEqual(count1, 1)

        # Second migration should migrate 0 (already enriched)
        self.assertEqual(count2, 0)


@pytest.mark.unit
class TestPlaybookPersistenceMigration(unittest.TestCase):
    """Test migration during playbook persistence operations."""

    def test_load_legacy_playbook_auto_migrates(self):
        """Test that loading a legacy playbook auto-migrates bullets."""
        from ace import Playbook
        from ace.playbook import EnrichedBullet

        # Create legacy JSON format (without enrichment fields)
        legacy_data = {
            "sections": {
                "math": [
                    {
                        "id": "legacy-1",
                        "section": "math",
                        "content": "Old math tip",
                        "helpful": 3,
                        "harmful": 0,
                        "neutral": 1,
                    }
                ]
            }
        }

        with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(legacy_data, f)
            temp_path = f.name

        try:
            # Load with auto_migrate=True
            playbook = Playbook.from_json_file(temp_path, auto_migrate=True)

            # Should have 1 bullet
            bullets = list(playbook.bullets())
            self.assertEqual(len(bullets), 1)

            # Should be EnrichedBullet
            self.assertIsInstance(bullets[0], EnrichedBullet)

            # Original data preserved
            self.assertEqual(bullets[0].id, "legacy-1")
            self.assertEqual(bullets[0].helpful, 3)
        finally:
            Path(temp_path).unlink()

    def test_save_enriched_playbook(self):
        """Test that enriched playbook saves correctly."""
        from ace import Playbook
        from ace.playbook import EnrichedBullet

        playbook = Playbook()
        playbook.add_enriched_bullet(
            section="test",
            content="Enriched content",
            task_types=["testing"],
            domains=["software"],
            complexity_level="medium",
        )

        with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            playbook.to_json_file(temp_path)

            # Load and verify
            loaded = Playbook.from_json_file(temp_path)
            bullets = list(loaded.bullets())

            self.assertEqual(len(bullets), 1)
            self.assertIsInstance(bullets[0], EnrichedBullet)
            self.assertEqual(bullets[0].task_types, ["testing"])
            self.assertEqual(bullets[0].domains, ["software"])
        finally:
            Path(temp_path).unlink()


@pytest.mark.unit
class TestMigrationWithCustomEnricher(unittest.TestCase):
    """Test migration with custom enrichment functions."""

    def test_migrate_with_custom_enricher(self):
        """Test using a custom enrichment function during migration."""
        from ace import Playbook
        from ace.playbook import Bullet, EnrichedBullet

        playbook = Playbook()
        playbook.add_bullet("test", "Custom content")

        # Custom enricher that adds specific metadata
        def custom_enricher(bullet: Bullet) -> EnrichedBullet:
            return EnrichedBullet(
                id=bullet.id,
                section=bullet.section,
                content=bullet.content,
                helpful=bullet.helpful,
                harmful=bullet.harmful,
                neutral=bullet.neutral,
                task_types=["custom"],
                domains=["custom_domain"],
                complexity_level="high",
            )

        playbook.migrate_to_enriched(enricher=custom_enricher)

        bullet = list(playbook.bullets())[0]
        self.assertEqual(bullet.task_types, ["custom"])
        self.assertEqual(bullet.domains, ["custom_domain"])
        self.assertEqual(bullet.complexity_level, "high")


if __name__ == "__main__":
    unittest.main()
