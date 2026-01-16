#!/usr/bin/env python3
"""
Re-embed all memories in ace_unified collection.

Fixes data quality issue where migrated records have duplicate/placeholder embeddings.
This script iterates through all points and generates fresh embeddings for each.

Usage:
    python scripts/reembed_unified_memories.py --dry-run   # Preview without changes
    python scripts/reembed_unified_memories.py             # Execute re-embedding
    python scripts/reembed_unified_memories.py --batch 100 # Custom batch size
"""

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import httpx
import os
from qdrant_client import QdrantClient

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ace.config import get_qdrant_config, get_embedding_config

# Configuration from centralized config
_qdrant_config = get_qdrant_config()
_embedding_config = get_embedding_config()

QDRANT_URL = _qdrant_config.url
EMBEDDING_URL = _embedding_config.url
EMBEDDING_MODEL = _embedding_config.model
COLLECTION_NAME = _qdrant_config.unified_collection  # Uses ace_memories_hybrid
DEFAULT_BATCH_SIZE = 50


@dataclass
class ReembedResult:
    """Result of re-embedding operation."""
    total: int
    processed: int
    updated: int
    skipped: int
    errors: int
    dry_run: bool


def get_embedding(text: str, embedding_url: str = EMBEDDING_URL) -> Optional[List[float]]:
    """Get embedding for text from LM Studio."""
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{embedding_url}/v1/embeddings",
                json={
                    "model": EMBEDDING_MODEL,
                    "input": text[:8000]  # Truncate to avoid token limits
                }
            )
            resp.raise_for_status()
            data = resp.json()
            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]["embedding"]
    except Exception as e:
        print(f"  Embedding error: {e}")
    return None


def vectors_are_similar(vec1: List[float], vec2: List[float], threshold: float = 0.999) -> bool:
    """Check if two vectors are nearly identical (indicating duplicate embedding)."""
    if len(vec1) != len(vec2):
        return False

    # Compute cosine similarity
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return True  # Zero vectors are "similar" (both bad)

    similarity = dot_product / (norm1 * norm2)
    return similarity > threshold


def reembed_memories(
    dry_run: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    qdrant_url: str = QDRANT_URL,
    embedding_url: str = EMBEDDING_URL,
) -> ReembedResult:
    """
    Re-embed all memories in the unified collection.

    Args:
        dry_run: If True, only analyze without updating
        batch_size: Number of points to process per batch
        qdrant_url: Qdrant server URL
        embedding_url: Embedding server URL

    Returns:
        ReembedResult with statistics
    """
    result = ReembedResult(
        total=0,
        processed=0,
        updated=0,
        skipped=0,
        errors=0,
        dry_run=dry_run
    )

    client = QdrantClient(url=qdrant_url)

    # Get total count
    collection_info = client.get_collection(COLLECTION_NAME)
    result.total = collection_info.points_count

    print(f"{'[DRY RUN] ' if dry_run else ''}Re-embedding {result.total} memories")
    print(f"Batch size: {batch_size}")
    print()

    # Track reference vector to detect duplicates
    reference_vector = None
    duplicate_count = 0

    # Scroll through all points
    offset = None
    batch_num = 0

    while True:
        batch_num += 1
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=batch_size,
            offset=offset,
            with_vectors=True,
            with_payload=True
        )

        if not points:
            break

        print(f"Batch {batch_num}: Processing {len(points)} points...")

        for point in points:
            result.processed += 1

            # Get content from payload
            content = point.payload.get("content", "")
            if not content:
                result.skipped += 1
                continue

            # Get current vector
            current_vector = point.vector.get("dense", []) if isinstance(point.vector, dict) else point.vector

            # Check if this is a duplicate vector
            if reference_vector is None:
                reference_vector = current_vector
            elif vectors_are_similar(current_vector, reference_vector):
                duplicate_count += 1

            # Generate new embedding
            new_embedding = get_embedding(content, embedding_url)

            if new_embedding is None:
                result.errors += 1
                print(f"  Error: Failed to embed point {point.id}")
                continue

            # Update point with new embedding
            if not dry_run:
                try:
                    client.update_vectors(
                        collection_name=COLLECTION_NAME,
                        points=[
                            {
                                "id": point.id,
                                "vector": {"dense": new_embedding}
                            }
                        ]
                    )
                    result.updated += 1
                except Exception as e:
                    result.errors += 1
                    print(f"  Error updating point {point.id}: {e}")
            else:
                result.updated += 1  # Would have updated

            # Progress indicator
            if result.processed % 100 == 0:
                print(f"  Progress: {result.processed}/{result.total} ({100*result.processed//result.total}%)")

        # Small delay between batches to not overwhelm embedding service
        if not dry_run:
            time.sleep(0.5)

        if offset is None:
            break

    print()
    print("=" * 50)
    print(f"{'[DRY RUN] ' if dry_run else ''}Re-embedding Complete")
    print(f"  Total points: {result.total}")
    print(f"  Processed: {result.processed}")
    print(f"  Updated: {result.updated}")
    print(f"  Skipped: {result.skipped}")
    print(f"  Errors: {result.errors}")
    print(f"  Duplicate vectors detected: {duplicate_count}")
    print("=" * 50)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Re-embed all memories in ace_unified collection"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without making changes"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})"
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

    result = reembed_memories(
        dry_run=args.dry_run,
        batch_size=args.batch,
        qdrant_url=args.qdrant_url,
        embedding_url=args.embedding_url
    )

    if result.errors > 0:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
