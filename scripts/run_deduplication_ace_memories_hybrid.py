#!/usr/bin/env python3
"""
Run advanced deduplication on ace_memories_hybrid collection.

This script:
1. Runs deduplication with --dry-run first to assess impact
2. If >10 duplicate groups found, proceeds with actual deduplication
3. Documents the number of memories merged/deleted

Usage:
    python scripts/run_deduplication_ace_memories_hybrid.py --dry-run
    python scripts/run_deduplication_ace_memories_hybrid.py  # Execute
    python scripts/run_deduplication_ace_memories_hybrid.py --method dbscan --eps 0.1
"""

import argparse
import logging
import sys
from pathlib import Path

# Add ace module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ace.deduplication import (
    DeduplicationEngine,
    ClusteringMethod,
    MergeStrategy,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Advanced deduplication for ace_memories_hybrid collection"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without making changes"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="hdbscan",
        choices=["hdbscan", "dbscan"],
        help="Clustering method (default: hdbscan)"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum cluster size for HDBSCAN (default: 2)"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.08,
        help="Epsilon for DBSCAN (default: 0.08)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="Min samples for DBSCAN (default: 2)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="keep_best",
        choices=["keep_best", "merge_content", "canonical_form"],
        help="Merge strategy (default: keep_best)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="ace_memories_hybrid",
        help="Qdrant collection name (default: ace_memories_hybrid)"
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default="http://localhost:6333",
        help="Qdrant URL (default: http://localhost:6333)"
    )
    parser.add_argument(
        "--embedding-url",
        type=str,
        default="http://localhost:1234",
        help="Embedding service URL (default: http://localhost:1234)"
    )

    args = parser.parse_args()

    # Initialize engine
    logger.info("Initializing DeduplicationEngine...")
    engine = DeduplicationEngine(
        collection_name=args.collection,
        qdrant_url=args.qdrant_url,
        embedding_url=args.embedding_url,
    )

    # Map string args to enums
    method = ClusteringMethod(args.method)
    strategy = MergeStrategy(args.strategy)

    # Run deduplication
    result = engine.run_deduplication(
        method=method,
        min_cluster_size=args.min_cluster_size,
        dbscan_eps=args.eps,
        dbscan_min_samples=args.min_samples,
        strategy=strategy,
        dry_run=args.dry_run,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total scanned: {result.total_scanned}")
    print(f"  Duplicate groups: {result.duplicate_groups}")
    print(f"  Memories merged: {result.memories_merged}")
    print(f"  Memories deleted: {result.memories_deleted}")

    if result.metrics:
        print(f"\n  Cluster Metrics:")
        print(f"    Clusters: {result.metrics.num_clusters}")
        print(f"    Noise points: {result.metrics.num_noise_points}")
        if result.metrics.silhouette_score is not None:
            print(f"    Silhouette score: {result.metrics.silhouette_score:.3f}")
        if result.metrics.davies_bouldin_score is not None:
            print(f"    Davies-Bouldin index: {result.metrics.davies_bouldin_score:.3f}")

    if args.dry_run and result.duplicate_groups > 10:
        print("\n" + "!" * 60)
        print(f"RECOMMENDATION: {result.duplicate_groups} duplicate groups found.")
        print("Run without --dry-run to execute deduplication.")
        print("!" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
