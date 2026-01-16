#!/usr/bin/env python3
"""
Migrate memories from ace_memories_hybrid collection to unified collection.

This script migrates personal memories stored in the old Qdrant collection
(ace_memories_hybrid) to the new unified architecture (ace_unified).

Usage:
    python scripts/migrate_memories_to_unified.py --dry-run   # Preview migration
    python scripts/migrate_memories_to_unified.py             # Run migration
    python scripts/migrate_memories_to_unified.py --verify    # Verify after migration
"""

import argparse
import sys
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

# Fix Windows console encoding for UTF-8 output (memories may contain emojis)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("Warning: httpx not available, cannot connect to Qdrant")

from ace.unified_memory import (
    UnifiedBullet,
    UnifiedNamespace,
    UnifiedSource,
    UnifiedMemoryIndex,
    convert_memory_to_unified,
)


# Configuration from centralized config
from ace.config import get_qdrant_config, get_embedding_config

_qdrant_config = get_qdrant_config()
_embedding_config = get_embedding_config()

QDRANT_URL = _qdrant_config.url
EMBEDDING_URL = _embedding_config.url
SOURCE_COLLECTION = _qdrant_config.memories_collection  # ace_memories_hybrid
TARGET_COLLECTION = _qdrant_config.unified_collection   # ace_memories_hybrid (unified)
BATCH_SIZE = 50


@dataclass
class MemoryMigrationResult:
    """Result of memory migration operation."""
    total_source: int
    migrated: int
    skipped: int
    errors: int
    dry_run: bool
    verified: bool = False
    target_count: int = 0


def load_from_ace_memories_hybrid(
    qdrant_url: str = QDRANT_URL,
    collection_name: str = SOURCE_COLLECTION,
    batch_size: int = BATCH_SIZE
) -> List[Dict[str, Any]]:
    """
    Load all memories from the ace_memories_hybrid Qdrant collection.

    Args:
        qdrant_url: Qdrant server URL
        collection_name: Source collection name
        batch_size: Number of points to fetch per request

    Returns:
        List of memory dicts with payload data
    """
    if not HTTPX_AVAILABLE:
        return []

    memories = []
    offset = None

    try:
        with httpx.Client(timeout=60.0) as client:
            while True:
                scroll_params = {
                    "limit": batch_size,
                    "with_payload": True,
                    "with_vector": False  # Don't need vectors for migration
                }
                if offset is not None:
                    scroll_params["offset"] = offset

                resp = client.post(
                    f"{qdrant_url}/collections/{collection_name}/points/scroll",
                    json=scroll_params
                )

                if resp.status_code != 200:
                    print(f"Error scrolling collection: {resp.status_code}")
                    break

                data = resp.json().get("result", {})
                points = data.get("points", [])
                next_offset = data.get("next_page_offset")

                for point in points:
                    payload = point.get("payload", {})
                    if payload.get("lesson"):  # Only include memories with content
                        memories.append(payload)

                if not next_offset:
                    break
                offset = next_offset

    except Exception as e:
        print(f"Error loading memories: {e}")

    return memories


def migrate_memories_to_unified(
    dry_run: bool = False,
    verify: bool = False,
    qdrant_url: str = QDRANT_URL,
    embedding_url: str = EMBEDDING_URL,
) -> MemoryMigrationResult:
    """
    Migrate memories from ace_memories_hybrid to ace_unified.

    Args:
        dry_run: If True, only preview migration without writing
        verify: If True, verify counts after migration
        qdrant_url: Qdrant server URL
        embedding_url: Embedding server URL

    Returns:
        MemoryMigrationResult with migration stats
    """
    result = MemoryMigrationResult(
        total_source=0,
        migrated=0,
        skipped=0,
        errors=0,
        dry_run=dry_run,
        verified=False,
        target_count=0
    )

    # Load source memories
    print(f"Loading memories from {SOURCE_COLLECTION}...")
    memories = load_from_ace_memories_hybrid(qdrant_url)
    result.total_source = len(memories)
    print(f"Found {result.total_source} memories to migrate")

    if result.total_source == 0:
        print("No memories to migrate")
        return result

    if dry_run:
        print("\n[DRY RUN] Would migrate the following:")
        for i, mem in enumerate(memories[:10]):  # Show first 10
            lesson = mem.get("lesson", "")[:50]
            category = mem.get("category", "?")
            severity = mem.get("severity", 5)
            print(f"  {i+1}. [{category}] (sev={severity}) {lesson}...")
        if len(memories) > 10:
            print(f"  ... and {len(memories) - 10} more")
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
    print("Converting and indexing memories...")
    unified_bullets = []

    for mem in memories:
        try:
            unified = convert_memory_to_unified(mem, source=UnifiedSource.MIGRATION)
            unified_bullets.append(unified)
        except Exception as e:
            print(f"  Error converting memory: {e}")
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
        target_count = index.count(namespace=UnifiedNamespace.USER_PREFS)
        result.target_count = target_count
        result.verified = target_count >= result.migrated
        print(f"  Target collection count: {target_count}")
        print(f"  Verification: {'PASSED' if result.verified else 'FAILED'}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Migrate memories from ace_memories_hybrid to ace_unified"
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
    print("ACE Memory Migration Tool")
    print("=" * 60)
    print(f"Source: {SOURCE_COLLECTION}")
    print(f"Target: {TARGET_COLLECTION}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("=" * 60)
    print()

    result = migrate_memories_to_unified(
        dry_run=args.dry_run,
        verify=args.verify,
        qdrant_url=args.qdrant_url,
        embedding_url=args.embedding_url
    )

    print()
    print("=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"Total source memories: {result.total_source}")
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
