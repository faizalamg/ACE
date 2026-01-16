#!/usr/bin/env python3
"""
Migrate ACE playbook from JSON file to unified collection.

This script migrates Bullet/EnrichedBullet entries from JSON playbook files
to the new unified architecture (ace_unified).

Usage:
    python scripts/migrate_playbook_to_unified.py playbook.json --dry-run
    python scripts/migrate_playbook_to_unified.py playbook.json
    python scripts/migrate_playbook_to_unified.py playbook.json --verify
"""

import argparse
import sys
import os
from dataclasses import dataclass
from typing import List, Union
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ace.playbook import Playbook, Bullet, EnrichedBullet
from ace.unified_memory import (
    UnifiedBullet,
    UnifiedNamespace,
    UnifiedSource,
    UnifiedMemoryIndex,
    convert_bullet_to_unified,
)


# Configuration from centralized config
from ace.config import get_qdrant_config, get_embedding_config

_qdrant_config = get_qdrant_config()
_embedding_config = get_embedding_config()

QDRANT_URL = _qdrant_config.url
EMBEDDING_URL = _embedding_config.url
TARGET_COLLECTION = _qdrant_config.unified_collection  # ace_memories_hybrid
BATCH_SIZE = 50


@dataclass
class PlaybookMigrationResult:
    """Result of playbook migration operation."""
    total_source: int
    migrated: int
    skipped: int
    errors: int
    dry_run: bool
    verified: bool = False
    target_count: int = 0


def load_from_json_playbook(playbook_path: str) -> List[Union[Bullet, EnrichedBullet]]:
    """
    Load all bullets from a JSON playbook file.

    Args:
        playbook_path: Path to the JSON playbook file

    Returns:
        List of Bullet/EnrichedBullet instances
    """
    path = Path(playbook_path)
    if not path.exists():
        raise FileNotFoundError(f"Playbook file not found: {playbook_path}")

    playbook = Playbook.load_from_file(str(path))
    return playbook.bullets()


def migrate_playbook_to_unified(
    playbook_path: str,
    dry_run: bool = False,
    verify: bool = False,
    qdrant_url: str = QDRANT_URL,
    embedding_url: str = EMBEDDING_URL,
) -> PlaybookMigrationResult:
    """
    Migrate playbook from JSON file to ace_unified collection.

    Args:
        playbook_path: Path to the JSON playbook file
        dry_run: If True, only preview migration without writing
        verify: If True, verify counts after migration
        qdrant_url: Qdrant server URL
        embedding_url: Embedding server URL

    Returns:
        PlaybookMigrationResult with migration stats
    """
    result = PlaybookMigrationResult(
        total_source=0,
        migrated=0,
        skipped=0,
        errors=0,
        dry_run=dry_run,
        verified=False,
        target_count=0
    )

    # Load source playbook
    print(f"Loading playbook from {playbook_path}...")
    try:
        bullets = load_from_json_playbook(playbook_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return result
    except Exception as e:
        print(f"Error loading playbook: {e}")
        result.errors += 1
        return result

    result.total_source = len(bullets)
    print(f"Found {result.total_source} bullets to migrate")

    if result.total_source == 0:
        print("No bullets to migrate")
        return result

    # Count enriched vs basic
    enriched_count = sum(1 for b in bullets if isinstance(b, EnrichedBullet))
    print(f"  - Basic Bullets: {result.total_source - enriched_count}")
    print(f"  - EnrichedBullets: {enriched_count}")

    if dry_run:
        print("\n[DRY RUN] Would migrate the following:")
        for i, bullet in enumerate(bullets[:10]):  # Show first 10
            content = bullet.content[:50]
            section = bullet.section
            bullet_type = "Enriched" if isinstance(bullet, EnrichedBullet) else "Basic"
            helpful = bullet.helpful if hasattr(bullet, 'helpful') else 0
            harmful = bullet.harmful if hasattr(bullet, 'harmful') else 0
            print(f"  {i+1}. [{bullet_type}] ({section}) +{helpful}/-{harmful} {content}...")
        if len(bullets) > 10:
            print(f"  ... and {len(bullets) - 10} more")
        print("\nTo execute migration, run without --dry-run flag")
        return result

    # Create unified index
    print(f"\nCreating unified index at {TARGET_COLLECTION}...")
    index = UnifiedMemoryIndex(
        qdrant_url=qdrant_url,
        embedding_url=embedding_url,
        collection_name=TARGET_COLLECTION
    )
    index.create_collection()

    # Convert and batch index
    print("Converting and indexing bullets...")
    unified_bullets = []

    for bullet in bullets:
        try:
            unified = convert_bullet_to_unified(bullet, source=UnifiedSource.MIGRATION)
            unified_bullets.append(unified)
        except Exception as e:
            print(f"  Error converting bullet {bullet.id}: {e}")
            result.errors += 1
            continue

    # Batch index in chunks
    for i in range(0, len(unified_bullets), BATCH_SIZE):
        batch = unified_bullets[i:i + BATCH_SIZE]
        indexed = index.batch_index(batch)
        result.migrated += indexed
        print(f"  Indexed batch {i//BATCH_SIZE + 1}: {indexed}/{len(batch)} bullets")

    result.skipped = result.total_source - result.migrated - result.errors

    # Verify if requested
    if verify:
        print("\nVerifying migration...")
        target_count = index.count(namespace=UnifiedNamespace.TASK_STRATEGIES)
        result.target_count = target_count
        result.verified = target_count >= result.migrated
        print(f"  Target collection count (task_strategies): {target_count}")
        print(f"  Verification: {'PASSED' if result.verified else 'FAILED'}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Migrate ACE playbook from JSON to ace_unified collection"
    )
    parser.add_argument(
        "playbook_path",
        help="Path to the JSON playbook file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration without writing"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify counts after migration"
    )
    parser.add_argument(
        "--qdrant-url",
        default=QDRANT_URL,
        help=f"Qdrant server URL (default: {QDRANT_URL})"
    )
    parser.add_argument(
        "--embedding-url",
        default=EMBEDDING_URL,
        help=f"Embedding server URL (default: {EMBEDDING_URL})"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ACE Playbook Migration Tool")
    print("=" * 60)
    print(f"Source: {args.playbook_path}")
    print(f"Target: {TARGET_COLLECTION}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("=" * 60)
    print()

    result = migrate_playbook_to_unified(
        playbook_path=args.playbook_path,
        dry_run=args.dry_run,
        verify=args.verify,
        qdrant_url=args.qdrant_url,
        embedding_url=args.embedding_url
    )

    print()
    print("=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"Total source bullets: {result.total_source}")
    print(f"Migrated: {result.migrated}")
    print(f"Skipped: {result.skipped}")
    print(f"Errors: {result.errors}")
    if result.verified:
        print(f"Target count: {result.target_count}")
        print(f"Verification: {'PASSED' if result.target_count >= result.migrated else 'FAILED'}")
    print("=" * 60)

    return 0 if result.errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
