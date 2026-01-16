#!/usr/bin/env python3
"""
Post-Deduplication Evaluation Script

Runs the same test suite used for v2/v3 evaluation AFTER deduplication.
Compares Recall@1, Recall@5, MRR before and after deduplication.
Outputs results to optimization_results/v4_deduplication.json

Expected Impact: +10-15% from reducing duplicate confusion

Usage:
    python rag_training/evaluate_post_dedup.py
    python rag_training/evaluate_post_dedup.py --test-queries /path/to/test_queries.json
"""

import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import httpx

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# CONFIGURATION
# ============================================================================

QDRANT_URL = "http://localhost:6333"
EMBEDDING_URL = "http://localhost:1234"
COLLECTION_NAME = "ace_memories_hybrid"
EMBEDDING_MODEL = "text-embedding-qwen3-embedding-8b"
TOP_K = 10  # Retrieve top 10 results

# Load test queries
TEST_QUERIES_PATH = Path(__file__).parent / "test_queries_generated.json"
OUTPUT_PATH = Path(__file__).parent / "optimization_results" / "v4_deduplication.json"
BASELINE_PATH = Path(__file__).parent / "optimization_results" / "baseline_results.json"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class QueryResult:
    """Result of a single query evaluation."""
    query: str
    query_category: str
    difficulty: str
    expected_memory_id: int
    retrieved_ids: List[int]
    retrieved_scores: List[float]
    rank: Optional[int]  # Rank of expected memory (1-indexed), None if not found
    score: Optional[float]  # Score of expected memory
    success_at_1: bool
    success_at_3: bool
    success_at_5: bool
    success_at_10: bool
    reciprocal_rank: float
    latency_ms: float


@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    timestamp: str
    version: str
    configuration: Dict[str, str]

    # Overall metrics
    total_queries: int = 0
    overall_recall_at_1: float = 0.0
    overall_recall_at_3: float = 0.0
    overall_recall_at_5: float = 0.0
    overall_recall_at_10: float = 0.0
    overall_mrr: float = 0.0

    # Latency stats
    latency_avg_ms: float = 0.0

    # Per-query results
    query_results: List[Dict[str, Any]] = None


# ============================================================================
# EMBEDDING HELPER
# ============================================================================

def get_embedding(text: str) -> List[float]:
    """Get embedding from LM Studio."""
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{EMBEDDING_URL}/v1/embeddings",
            json={"model": EMBEDDING_MODEL, "input": text[:8000]}
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


# ============================================================================
# QUERY EXECUTION
# ============================================================================

def query_qdrant_hybrid(
    query_text: str,
    collection_name: str = COLLECTION_NAME,
    limit: int = TOP_K,
) -> Tuple[List[int], List[float], float]:
    """
    Query Qdrant using hybrid search (dense + sparse RRF).

    Returns:
        Tuple of (retrieved_ids, retrieved_scores, latency_ms)
    """
    # Get embedding
    start_time = time.time()
    query_embedding = get_embedding(query_text)

    # BM25 sparse vector (simple tokenization)
    tokens = query_text.lower().split()
    sparse_indices = [abs(hash(token)) % 100000 for token in tokens]
    sparse_values = [1.0] * len(tokens)

    # Build hybrid query
    prefetch_queries = [
        {
            "query": query_embedding,
            "using": "dense",
            "limit": limit * 3,
        }
    ]

    if sparse_indices:
        prefetch_queries.append({
            "query": {
                "indices": sparse_indices,
                "values": sparse_values,
            },
            "using": "sparse",
            "limit": limit * 3,
        })

    # Execute query
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{QDRANT_URL}/collections/{collection_name}/points/query",
            json={
                "prefetch": prefetch_queries,
                "query": {"fusion": "rrf"},
                "limit": limit,
                "with_payload": False,
            }
        )
        resp.raise_for_status()
        result = resp.json()

    latency_ms = (time.time() - start_time) * 1000

    # Extract IDs and scores
    retrieved_ids = [point["id"] for point in result["result"]["points"]]
    retrieved_scores = [point["score"] for point in result["result"]["points"]]

    return retrieved_ids, retrieved_scores, latency_ms


# ============================================================================
# EVALUATION LOGIC
# ============================================================================

def evaluate_query(
    query_text: str,
    expected_memory_id: int,
    query_category: str,
    difficulty: str,
) -> QueryResult:
    """Evaluate a single query."""
    # Execute query
    retrieved_ids, retrieved_scores, latency_ms = query_qdrant_hybrid(query_text)

    # Find rank of expected memory
    rank = None
    score = None
    if expected_memory_id in retrieved_ids:
        rank = retrieved_ids.index(expected_memory_id) + 1  # 1-indexed
        score = retrieved_scores[retrieved_ids.index(expected_memory_id)]

    # Calculate success metrics
    success_at_1 = rank == 1 if rank else False
    success_at_3 = rank <= 3 if rank else False
    success_at_5 = rank <= 5 if rank else False
    success_at_10 = rank <= 10 if rank else False

    # Reciprocal rank
    reciprocal_rank = 1.0 / rank if rank else 0.0

    return QueryResult(
        query=query_text,
        query_category=query_category,
        difficulty=difficulty,
        expected_memory_id=expected_memory_id,
        retrieved_ids=retrieved_ids,
        retrieved_scores=retrieved_scores,
        rank=rank,
        score=score,
        success_at_1=success_at_1,
        success_at_3=success_at_3,
        success_at_5=success_at_5,
        success_at_10=success_at_10,
        reciprocal_rank=reciprocal_rank,
        latency_ms=latency_ms,
    )


def run_evaluation(test_queries_path: Path) -> EvaluationResult:
    """Run full evaluation on test queries."""
    print("=" * 60)
    print("POST-DEDUPLICATION EVALUATION")
    print("=" * 60)
    print(f"Test queries: {test_queries_path}")
    print(f"Collection: {COLLECTION_NAME}")
    print("")

    # Load test queries
    with open(test_queries_path) as f:
        test_data = json.load(f)

    # Run evaluation
    all_results = []
    total_queries = 0

    for memory_data in test_data["memories"]:
        memory_id = memory_data["memory_id"]
        memory_content = memory_data["memory_content"]

        for query_data in memory_data["queries"]:
            query_text = query_data["query"]
            category = query_data["category"]
            difficulty = query_data["difficulty"]

            result = evaluate_query(query_text, memory_id, category, difficulty)
            all_results.append(result)
            total_queries += 1

            # Progress
            if total_queries % 10 == 0:
                print(f"  Evaluated {total_queries} queries...")

    print(f"\nEvaluated {total_queries} queries")

    # Calculate aggregate metrics
    recall_at_1 = sum(r.success_at_1 for r in all_results) / total_queries
    recall_at_3 = sum(r.success_at_3 for r in all_results) / total_queries
    recall_at_5 = sum(r.success_at_5 for r in all_results) / total_queries
    recall_at_10 = sum(r.success_at_10 for r in all_results) / total_queries
    mrr = sum(r.reciprocal_rank for r in all_results) / total_queries
    avg_latency = sum(r.latency_ms for r in all_results) / total_queries

    # Create result object
    result = EvaluationResult(
        timestamp=datetime.now().isoformat(),
        version="v4_deduplication",
        configuration={
            "collection_name": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "qdrant_url": QDRANT_URL,
            "embedding_url": EMBEDDING_URL,
            "top_k": TOP_K,
        },
        total_queries=total_queries,
        overall_recall_at_1=recall_at_1,
        overall_recall_at_3=recall_at_3,
        overall_recall_at_5=recall_at_5,
        overall_recall_at_10=recall_at_10,
        overall_mrr=mrr,
        latency_avg_ms=avg_latency,
        query_results=[asdict(r) for r in all_results],
    )

    return result


def compare_with_baseline(result: EvaluationResult, baseline_path: Path):
    """Compare results with baseline."""
    if not baseline_path.exists():
        print(f"\nBaseline not found at {baseline_path}")
        return

    with open(baseline_path) as f:
        baseline = json.load(f)

    print("\n" + "=" * 60)
    print("COMPARISON WITH BASELINE")
    print("=" * 60)

    # Extract baseline metrics
    baseline_r1 = baseline.get("overall_recall_at_1", 0)
    baseline_r5 = baseline.get("overall_recall_at_5", 0)
    baseline_mrr = baseline.get("overall_mrr", 0)

    # Calculate improvements
    r1_improvement = ((result.overall_recall_at_1 - baseline_r1) / baseline_r1 * 100) if baseline_r1 > 0 else 0
    r5_improvement = ((result.overall_recall_at_5 - baseline_r5) / baseline_r5 * 100) if baseline_r5 > 0 else 0
    mrr_improvement = ((result.overall_mrr - baseline_mrr) / baseline_mrr * 100) if baseline_mrr > 0 else 0

    print(f"Recall@1: {baseline_r1:.1%} -> {result.overall_recall_at_1:.1%} ({r1_improvement:+.1f}%)")
    print(f"Recall@5: {baseline_r5:.1%} -> {result.overall_recall_at_5:.1%} ({r5_improvement:+.1f}%)")
    print(f"MRR: {baseline_mrr:.3f} -> {result.overall_mrr:.3f} ({mrr_improvement:+.1f}%)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Post-deduplication evaluation")
    parser.add_argument(
        "--test-queries",
        type=Path,
        default=TEST_QUERIES_PATH,
        help=f"Path to test queries JSON (default: {TEST_QUERIES_PATH})"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output path for results (default: {OUTPUT_PATH})"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BASELINE_PATH,
        help=f"Baseline results for comparison (default: {BASELINE_PATH})"
    )

    args = parser.parse_args()

    # Run evaluation
    result = run_evaluation(args.test_queries)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total queries: {result.total_queries}")
    print(f"Recall@1: {result.overall_recall_at_1:.1%}")
    print(f"Recall@3: {result.overall_recall_at_3:.1%}")
    print(f"Recall@5: {result.overall_recall_at_5:.1%}")
    print(f"Recall@10: {result.overall_recall_at_10:.1%}")
    print(f"MRR: {result.overall_mrr:.3f}")
    print(f"Avg latency: {result.latency_avg_ms:.1f}ms")

    # Compare with baseline
    compare_with_baseline(result, args.baseline)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(asdict(result), f, indent=2)

    print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
