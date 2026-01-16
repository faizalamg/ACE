#!/usr/bin/env python3
"""
Deduplicate existing memories in ace_unified collection.

This script identifies and consolidates semantically similar memories:
1. Scans all memories in the collection
2. Groups memories by semantic similarity (>0.92 threshold)
3. For each group: keeps the best one, merges reinforcement counts, deletes duplicates

Usage:
    python scripts/deduplicate_memories.py --dry-run   # Preview without changes
    python scripts/deduplicate_memories.py             # Execute deduplication
    python scripts/deduplicate_memories.py --threshold 0.90  # Custom threshold
"""

import argparse
import sys

# Fix Windows console encoding for UTF-8/emoji output
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

import httpx
from qdrant_client import QdrantClient

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
from ace.config import get_qdrant_config, get_embedding_config

# Configuration from centralized config
_qdrant_config = get_qdrant_config()
_embedding_config = get_embedding_config()

QDRANT_URL = _qdrant_config.url
EMBEDDING_URL = _embedding_config.url
EMBEDDING_MODEL = _embedding_config.model
COLLECTION_NAME = _qdrant_config.unified_collection  # Uses ace_memories_hybrid
DEFAULT_THRESHOLD = 0.92


@dataclass
class DedupeResult:
    """Result of deduplication operation."""
    total_scanned: int
    duplicate_groups: int
    memories_merged: int
    memories_deleted: int
    dry_run: bool


def get_embedding(text: str, embedding_url: str = EMBEDDING_URL) -> Optional[List[float]]:
    """Get embedding for text from LM Studio."""
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{embedding_url}/v1/embeddings",
                json={"model": EMBEDDING_MODEL, "input": text[:8000]}
            )
            resp.raise_for_status()
            data = resp.json()
            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]["embedding"]
    except Exception as e:
        print(f"  Embedding error: {e}")
    return None


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def find_duplicate_groups(
    client: QdrantClient,
    threshold: float = DEFAULT_THRESHOLD
) -> List[List[Dict]]:
    """
    Find groups of semantically similar memories.

    Returns list of groups, where each group contains similar memories.
    """
    print("Loading all memories...")

    # Load all points with vectors
    all_points = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=offset,
            with_vectors=True,
            with_payload=True
        )
        all_points.extend(points)
        if offset is None:
            break

    print(f"Loaded {len(all_points)} memories")

    # Find duplicate groups using Union-Find
    processed: Set[int] = set()
    groups: List[List[Dict]] = []

    print(f"Analyzing similarity (threshold: {threshold})...")

    for i, p1 in enumerate(all_points):
        if p1.id in processed:
            continue

        vec1 = p1.vector.get("dense", []) if isinstance(p1.vector, dict) else p1.vector

        # Find all similar memories to this one
        group = [{
            "id": p1.id,
            "content": p1.payload.get("content", ""),
            "severity": p1.payload.get("severity", 5),
            "reinforcement_count": p1.payload.get("reinforcement_count", 1),
            "helpful_count": p1.payload.get("helpful_count", 0),
            "harmful_count": p1.payload.get("harmful_count", 0),
            "created_at": p1.payload.get("created_at", ""),
            "payload": p1.payload
        }]

        for j, p2 in enumerate(all_points):
            if i >= j or p2.id in processed:
                continue

            vec2 = p2.vector.get("dense", []) if isinstance(p2.vector, dict) else p2.vector
            sim = cosine_similarity(vec1, vec2)

            if sim >= threshold:
                group.append({
                    "id": p2.id,
                    "content": p2.payload.get("content", ""),
                    "severity": p2.payload.get("severity", 5),
                    "reinforcement_count": p2.payload.get("reinforcement_count", 1),
                    "helpful_count": p2.payload.get("helpful_count", 0),
                    "harmful_count": p2.payload.get("harmful_count", 0),
                    "created_at": p2.payload.get("created_at", ""),
                    "payload": p2.payload,
                    "similarity": sim
                })
                processed.add(p2.id)

        if len(group) > 1:
            groups.append(group)
            processed.add(p1.id)

        # Progress
        if (i + 1) % 200 == 0:
            print(f"  Progress: {i + 1}/{len(all_points)} ({100*(i+1)//len(all_points)}%)")

    return groups


def select_best_memory(group: List[Dict]) -> Tuple[Dict, List[Dict]]:
    """
    Select the best memory from a group to keep.

    Selection criteria (in order):
    1. Highest combined score (severity + reinforcement + helpful)
    2. Longest content
    3. Oldest (first created)

    Returns:
        Tuple of (best memory, list of duplicates to delete)
    """
    def score(mem: Dict) -> float:
        return (
            mem.get("severity", 5) * 0.3 +
            mem.get("reinforcement_count", 1) * 0.3 +
            mem.get("helpful_count", 0) * 0.2 +
            len(mem.get("content", "")) * 0.001 +
            (-mem.get("harmful_count", 0)) * 0.2
        )

    sorted_group = sorted(group, key=score, reverse=True)
    best = sorted_group[0]
    duplicates = sorted_group[1:]

    return best, duplicates


def merge_and_dedupe(
    client: QdrantClient,
    groups: List[List[Dict]],
    dry_run: bool = False,
    embedding_url: str = EMBEDDING_URL
) -> DedupeResult:
    """
    Merge duplicate groups and delete redundant memories.

    For each group:
    1. Select best memory
    2. Sum reinforcement_count, helpful_count, harmful_count from all
    3. Update best memory with merged counts
    4. Delete duplicates
    """
    result = DedupeResult(
        total_scanned=0,
        duplicate_groups=len(groups),
        memories_merged=0,
        memories_deleted=0,
        dry_run=dry_run
    )

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing {len(groups)} duplicate groups...")

    for i, group in enumerate(groups):
        best, duplicates = select_best_memory(group)

        print(f"\nGroup {i + 1}: {len(group)} memories")
        print(f"  Keep: [{best['id']}] \"{best['content'][:60]}...\"")

        # Merge counts
        total_reinforcement = sum(m.get("reinforcement_count", 1) for m in group)
        total_helpful = sum(m.get("helpful_count", 0) for m in group)
        total_harmful = sum(m.get("harmful_count", 0) for m in group)
        max_severity = max(m.get("severity", 5) for m in group)

        print(f"  Merged: severity={max_severity}, reinforcement={total_reinforcement}, helpful={total_helpful}, harmful={total_harmful}")

        if not dry_run:
            # Update best memory with merged counts
            updated_payload = {
                **best["payload"],
                "reinforcement_count": total_reinforcement,
                "helpful_count": total_helpful,
                "harmful_count": total_harmful,
                "severity": max_severity,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "merged_from": [d["id"] for d in duplicates]
            }

            # Get new embedding for the best content
            embedding = get_embedding(best["content"], embedding_url)
            if embedding:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[{
                        "id": best["id"],
                        "vector": {"dense": embedding},
                        "payload": updated_payload
                    }]
                )
                result.memories_merged += 1

            # Delete duplicates
            duplicate_ids = [d["id"] for d in duplicates]
            if duplicate_ids:
                from qdrant_client.models import PointIdsList
                client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=PointIdsList(points=duplicate_ids)
                )
                result.memories_deleted += len(duplicate_ids)
                print(f"  Deleted: {duplicate_ids}")
        else:
            result.memories_merged += 1
            result.memories_deleted += len(duplicates)
            for d in duplicates:
                sim = d.get("similarity", 0)
                print(f"  Would delete: [{d['id']}] (sim={sim:.3f}) \"{d['content'][:50]}...\"")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate memories in ace_unified collection"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without making changes"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Similarity threshold for deduplication (default: {DEFAULT_THRESHOLD})"
    )
    parser.add_argument(
        "--qdrant-url",
        default=QDRANT_URL,
        help=f"Qdrant URL (default: {QDRANT_URL})"
    )
    parser.add_argument(
        "--embedding-url",
        default=EMBEDDING_URL,
        help=f"Embedding service URL (default: {EMBEDDING_URL})"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MEMORY DEDUPLICATION")
    print("=" * 60)
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Threshold: {args.threshold}")
    print()

    client = QdrantClient(url=args.qdrant_url)

    # Get initial count
    collection_info = client.get_collection(COLLECTION_NAME)
    initial_count = collection_info.points_count
    print(f"Initial memory count: {initial_count}")

    # Find duplicate groups
    groups = find_duplicate_groups(client, args.threshold)
    print(f"\nFound {len(groups)} duplicate groups")

    if not groups:
        print("No duplicates found!")
        return 0

    # Show preview
    total_duplicates = sum(len(g) - 1 for g in groups)
    print(f"Total memories to merge: {sum(len(g) for g in groups)}")
    print(f"Memories to delete: {total_duplicates}")
    print(f"Expected final count: {initial_count - total_duplicates}")

    # Merge and dedupe
    result = merge_and_dedupe(
        client, groups,
        dry_run=args.dry_run,
        embedding_url=args.embedding_url
    )

    # Summary
    print("\n" + "=" * 60)
    print(f"{'[DRY RUN] ' if result.dry_run else ''}DEDUPLICATION COMPLETE")
    print("=" * 60)
    print(f"  Duplicate groups processed: {result.duplicate_groups}")
    print(f"  Memories merged: {result.memories_merged}")
    print(f"  Memories deleted: {result.memories_deleted}")

    if not args.dry_run:
        final_info = client.get_collection(COLLECTION_NAME)
        print(f"  Final memory count: {final_info.points_count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
