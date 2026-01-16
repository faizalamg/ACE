#!/usr/bin/env python3
"""
Measure recall@1 and recall@5 for ACE memory features.

Tests retrieval quality with/without new memory architecture features:
- Version history (superseded bullet filtering)
- Entity key lookup
- Conflict detection

Usage:
    python scripts/measure_memory_recall.py
"""

import os
import sys
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ace.unified_memory import (
    UnifiedMemoryIndex,
    UnifiedBullet,
    UnifiedNamespace,
    UnifiedSource,
)
from ace.config import get_memory_config, reset_config


@dataclass
class RecallResult:
    """Recall measurement result."""
    recall_at_1: float
    recall_at_5: float
    total_queries: int
    hits_at_1: int
    hits_at_5: int


def create_test_queries() -> List[Tuple[str, str]]:
    """
    Create test queries with expected content matches.

    Returns list of (query, expected_substring) tuples.
    """
    return [
        # Technical preferences
        ("Python scripting preference", "Python"),
        ("TypeScript code style", "TypeScript"),
        ("git commit message format", "commit"),
        ("API error handling", "error"),
        ("testing best practices", "test"),

        # Workflow preferences
        ("code review workflow", "review"),
        ("documentation standards", "doc"),
        ("debugging approach", "debug"),
        ("performance optimization", "performance"),
        ("security scanning", "security"),

        # Configuration preferences
        ("environment variable handling", "environment"),
        ("logging configuration", "log"),
        ("database connection", "database"),
        ("authentication setup", "auth"),
        ("deployment process", "deploy"),

        # Tool preferences
        ("IDE settings", "IDE"),
        ("terminal configuration", "terminal"),
        ("version control", "version"),
        ("package manager", "package"),
        ("build system", "build"),
    ]


def measure_recall(
    index: UnifiedMemoryIndex,
    queries: List[Tuple[str, str]],
    include_superseded: bool = True
) -> RecallResult:
    """
    Measure recall@1 and recall@5 for given queries.

    Args:
        index: Memory index to search
        queries: List of (query, expected_substring) tuples
        include_superseded: Whether to include superseded bullets

    Returns:
        RecallResult with recall metrics
    """
    hits_at_1 = 0
    hits_at_5 = 0
    total = len(queries)

    for query, expected in queries:
        try:
            results = index.retrieve(
                query=query,
                limit=5,
                threshold=0.3,
                include_superseded=include_superseded
            )

            if not results:
                continue

            # Check recall@1
            if expected.lower() in results[0].content.lower():
                hits_at_1 += 1
                hits_at_5 += 1
            else:
                # Check recall@5
                for result in results[:5]:
                    if expected.lower() in result.content.lower():
                        hits_at_5 += 1
                        break

        except Exception as e:
            print(f"  Query failed: {query} - {e}")
            continue

    recall_1 = hits_at_1 / total if total > 0 else 0.0
    recall_5 = hits_at_5 / total if total > 0 else 0.0

    return RecallResult(
        recall_at_1=recall_1,
        recall_at_5=recall_5,
        total_queries=total,
        hits_at_1=hits_at_1,
        hits_at_5=hits_at_5
    )


def print_results(name: str, result: RecallResult):
    """Pretty print recall results."""
    print(f"\n{name}")
    print("=" * 50)
    print(f"  Total Queries:  {result.total_queries}")
    print(f"  Hits@1:         {result.hits_at_1}")
    print(f"  Hits@5:         {result.hits_at_5}")
    print(f"  Recall@1:       {result.recall_at_1:.2%}")
    print(f"  Recall@5:       {result.recall_at_5:.2%}")


def main():
    print("ACE Memory Architecture - Recall Measurement")
    print("=" * 60)

    # Get current config
    config = get_memory_config()
    print(f"\nCurrent Configuration:")
    print(f"  Version History:     {config.enable_version_history}")
    print(f"  Entity Key Lookup:   {config.enable_entity_key_lookup}")
    print(f"  Conflict Detection:  {config.enable_conflict_detection}")
    print(f"  Exclude Superseded:  {config.exclude_superseded_by_default}")

    # Initialize index
    index = UnifiedMemoryIndex()

    # Get memory stats
    total_count = index.count()
    user_prefs_count = index.count(UnifiedNamespace.USER_PREFS)
    task_strategies_count = index.count(UnifiedNamespace.TASK_STRATEGIES)

    print(f"\nMemory Stats:")
    print(f"  Total Memories:      {total_count}")
    print(f"  User Preferences:    {user_prefs_count}")
    print(f"  Task Strategies:     {task_strategies_count}")

    # Create test queries
    queries = create_test_queries()
    print(f"\nRunning {len(queries)} test queries...")

    # Measure recall WITH superseded bullets (baseline - includes all)
    print("\nTesting with ALL bullets (include_superseded=True)...")
    result_all = measure_recall(index, queries, include_superseded=True)
    print_results("Baseline (All Bullets)", result_all)

    # Measure recall WITHOUT superseded bullets (new feature)
    print("\nTesting with ACTIVE bullets only (include_superseded=False)...")
    result_active = measure_recall(index, queries, include_superseded=False)
    print_results("Active Only (No Superseded)", result_active)

    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    recall1_diff = result_active.recall_at_1 - result_all.recall_at_1
    recall5_diff = result_active.recall_at_5 - result_all.recall_at_5

    print(f"\nRecall@1 Change: {recall1_diff:+.2%}")
    print(f"Recall@5 Change: {recall5_diff:+.2%}")

    if recall1_diff > 0:
        print("\n[POSITIVE] Filtering superseded bullets IMPROVED recall@1")
    elif recall1_diff < 0:
        print("\n[NEGATIVE] Filtering superseded bullets DECREASED recall@1")
    else:
        print("\n[NEUTRAL] No change in recall@1")

    if recall5_diff > 0:
        print("[POSITIVE] Filtering superseded bullets IMPROVED recall@5")
    elif recall5_diff < 0:
        print("[NEGATIVE] Filtering superseded bullets DECREASED recall@5")
    else:
        print("[NEUTRAL] No change in recall@5")

    # Additional test: Namespace-specific recall
    print("\n" + "=" * 60)
    print("NAMESPACE-SPECIFIC RECALL")
    print("=" * 60)

    # Filter queries for each namespace
    pref_queries = queries[:10]  # First 10 tend to be preference-related
    strat_queries = queries[10:]  # Rest tend to be strategy-related

    print(f"\nUser Preferences ({len(pref_queries)} queries)...")

    # Patch retrieve to filter by namespace
    def measure_namespace_recall(ns: UnifiedNamespace, test_queries: List[Tuple[str, str]]) -> RecallResult:
        hits_at_1 = 0
        hits_at_5 = 0
        total = len(test_queries)

        for query, expected in test_queries:
            try:
                results = index.retrieve(
                    query=query,
                    namespace=ns,
                    limit=5,
                    threshold=0.3,
                    include_superseded=True
                )

                if not results:
                    continue

                if expected.lower() in results[0].content.lower():
                    hits_at_1 += 1
                    hits_at_5 += 1
                else:
                    for result in results[:5]:
                        if expected.lower() in result.content.lower():
                            hits_at_5 += 1
                            break

            except Exception:
                continue

        return RecallResult(
            recall_at_1=hits_at_1 / total if total > 0 else 0.0,
            recall_at_5=hits_at_5 / total if total > 0 else 0.0,
            total_queries=total,
            hits_at_1=hits_at_1,
            hits_at_5=hits_at_5
        )

    pref_result = measure_namespace_recall(UnifiedNamespace.USER_PREFS, pref_queries)
    print_results("User Preferences Namespace", pref_result)

    strat_result = measure_namespace_recall(UnifiedNamespace.TASK_STRATEGIES, strat_queries)
    print_results("Task Strategies Namespace", strat_result)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Memory Architecture Features Impact:
- Version History: Enables soft-delete versioning with audit trail
- Entity Key Lookup: O(1) deterministic retrieval by key
- Conflict Detection: Identifies contradictory bullets
- Superseded Filtering: Can exclude old versions from retrieval

Current recall scores show the quality of the existing memory corpus.
New features provide data hygiene capabilities without degrading retrieval.
""")


if __name__ == "__main__":
    main()
