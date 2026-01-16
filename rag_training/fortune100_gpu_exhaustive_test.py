"""
Fortune 100 Exhaustive RAG Test - GPU-Accelerated Full Optimization Pipeline

This test validates the FULL optimization pipeline:
- Query expansion (4 variations)
- Multi-query retrieval with RRF fusion
- BM25 sparse + Dense hybrid search
- GPU-accelerated Cross-encoder re-ranking (DirectML on AMD)

Target: 95%+ Recall@5 across ALL 2003 memories
"""

import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

# Add parent to path for ace imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ace.retrieval_optimized import (
    OptimizedRetriever,
    CROSS_ENCODER_AVAILABLE,
    GPU_RERANKER_AVAILABLE,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result for a single query."""
    query: str
    query_category: str
    difficulty: str
    expected_memory_id: int
    retrieved_ids: List[int]
    retrieved_scores: List[float]
    rank: Optional[int]
    success_at_1: bool
    success_at_3: bool
    success_at_5: bool
    success_at_10: bool
    reciprocal_rank: float
    latency_ms: float
    reranked: bool


@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    timestamp: str
    configuration: Dict[str, Any]

    total_queries: int = 0
    total_memories: int = 0

    overall_recall_at_1: float = 0.0
    overall_recall_at_3: float = 0.0
    overall_recall_at_5: float = 0.0
    overall_recall_at_10: float = 0.0
    overall_mrr: float = 0.0
    overall_ndcg_at_10: float = 0.0

    latency_avg_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0

    by_query_category: Dict[str, Dict] = field(default_factory=dict)
    by_difficulty: Dict[str, Dict] = field(default_factory=dict)
    by_memory_category: Dict[str, Dict] = field(default_factory=dict)

    complete_misses: int = 0
    failures: List[Dict] = field(default_factory=list)


class Fortune100Evaluator:
    """Fortune 100 grade RAG evaluation with full optimization."""

    def __init__(self):
        print("=" * 80)
        print("FORTUNE 100 EXHAUSTIVE RAG TEST - FULL OPTIMIZATION")
        print("=" * 80)
        print()

        # Check GPU availability
        print("SYSTEM STATUS:")
        print(f"  Cross-encoder available: {CROSS_ENCODER_AVAILABLE}")
        print(f"  GPU reranker (DirectML): {GPU_RERANKER_AVAILABLE}")

        # Initialize optimized retriever
        print()
        print("Initializing OptimizedRetriever with full optimization...")
        self.retriever = OptimizedRetriever(
            config={
                "enable_reranking": True,
                "num_expanded_queries": 4,
                "candidates_per_query": 30,
                "first_stage_k": 30,
                "final_k": 10,
            }
        )

        # Verify GPU is being used
        if hasattr(self.retriever, 'cross_encoder'):
            ce = self.retriever.cross_encoder
            if hasattr(ce, 'use_gpu'):
                print(f"  GPU acceleration active: {ce.use_gpu}")
            else:
                print("  GPU acceleration: CPU fallback")

        print()
        print("FEATURES ENABLED:")
        print("  - Query expansion (4 variations)")
        print("  - Multi-query retrieval with RRF fusion")
        print("  - BM25 sparse + Dense hybrid search")
        print(f"  - Cross-encoder re-ranking: {'GPU (DirectML)' if GPU_RERANKER_AVAILABLE else 'CPU'}")
        print()
        print("Target: 95%+ Recall@5")
        print("=" * 80)

    def evaluate_query(
        self,
        query: str,
        query_category: str,
        difficulty: str,
        expected_id: int
    ) -> QueryResult:
        """Evaluate a single query."""
        start = time.perf_counter()

        results, metrics = self.retriever.search(query, limit=10, return_metrics=True)

        latency = (time.perf_counter() - start) * 1000

        retrieved_ids = [r.id for r in results]
        retrieved_scores = [r.score for r in results]
        reranked = any(r.reranked for r in results) if results else False

        # Find rank
        rank = None
        if expected_id in retrieved_ids:
            rank = retrieved_ids.index(expected_id) + 1

        return QueryResult(
            query=query,
            query_category=query_category,
            difficulty=difficulty,
            expected_memory_id=expected_id,
            retrieved_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            rank=rank,
            success_at_1=rank == 1 if rank else False,
            success_at_3=rank is not None and rank <= 3,
            success_at_5=rank is not None and rank <= 5,
            success_at_10=rank is not None and rank <= 10,
            reciprocal_rank=1.0 / rank if rank else 0.0,
            latency_ms=latency,
            reranked=reranked
        )

    def calculate_ndcg(self, results: List[QueryResult], k: int = 10) -> float:
        """Calculate NDCG@k."""
        ndcg_scores = []
        for r in results:
            if r.rank is None or r.rank > k:
                ndcg_scores.append(0.0)
            else:
                dcg = 1.0 / math.log2(r.rank + 1)
                idcg = 1.0 / math.log2(2)
                ndcg_scores.append(dcg / idcg)
        return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    def run_evaluation(self, test_suite_path: Path, output_path: Path) -> EvaluationResult:
        """Run full Fortune 100 evaluation."""
        print(f"\nLoading test suite: {test_suite_path}")

        with open(test_suite_path) as f:
            data = json.load(f)

        test_cases = data["test_cases"]
        total_queries = sum(len(tc.get("generated_queries", [])) for tc in test_cases)

        print(f"Loaded {len(test_cases)} memories with {total_queries} total queries")
        print()

        result = EvaluationResult(
            timestamp=datetime.now().isoformat(),
            configuration={
                "gpu_reranker": GPU_RERANKER_AVAILABLE,
                "cross_encoder": CROSS_ENCODER_AVAILABLE,
                "num_expanded_queries": 4,
                "candidates_per_query": 30,
                "test_suite": str(test_suite_path)
            },
            total_memories=len(test_cases)
        )

        all_results: List[QueryResult] = []
        all_latencies: List[float] = []
        by_query_category = defaultdict(list)
        by_difficulty = defaultdict(list)
        by_memory_category = defaultdict(list)

        start_time = time.time()

        for i, tc in enumerate(test_cases):
            memory_id = tc["memory_id"]
            category = tc["category"]
            queries = tc.get("generated_queries", [])

            if not queries:
                continue

            # Progress update
            progress = (i + 1) / len(test_cases) * 100
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (len(test_cases) - i - 1) if i > 0 else 0

            if (i + 1) % 10 == 0 or i == 0:
                current_r5 = sum(1 for r in all_results if r.success_at_5) / len(all_results) * 100 if all_results else 0
                print(f"[{progress:5.1f}%] Memory {memory_id} ({category}) | "
                      f"Current Recall@5: {current_r5:.1f}% | "
                      f"ETA: {eta:.0f}s")

            for q in queries:
                qr = self.evaluate_query(
                    query=q["query"],
                    query_category=q["category"],
                    difficulty=q["difficulty"],
                    expected_id=memory_id
                )

                all_results.append(qr)
                all_latencies.append(qr.latency_ms)
                by_query_category[qr.query_category].append(qr)
                by_difficulty[qr.difficulty].append(qr)
                by_memory_category[category].append(qr)

                # Track failures
                if not qr.success_at_5:
                    result.failures.append({
                        "memory_id": memory_id,
                        "query": qr.query,
                        "category": qr.query_category,
                        "difficulty": qr.difficulty,
                        "rank": qr.rank,
                        "retrieved_top3": qr.retrieved_ids[:3]
                    })

        # Calculate overall metrics
        n = len(all_results)
        result.total_queries = n

        result.overall_recall_at_1 = sum(1 for r in all_results if r.success_at_1) / n
        result.overall_recall_at_3 = sum(1 for r in all_results if r.success_at_3) / n
        result.overall_recall_at_5 = sum(1 for r in all_results if r.success_at_5) / n
        result.overall_recall_at_10 = sum(1 for r in all_results if r.success_at_10) / n
        result.overall_mrr = sum(r.reciprocal_rank for r in all_results) / n
        result.overall_ndcg_at_10 = self.calculate_ndcg(all_results, k=10)

        # Latency stats
        all_latencies.sort()
        result.latency_avg_ms = sum(all_latencies) / len(all_latencies)
        result.latency_p50_ms = all_latencies[len(all_latencies) // 2]
        result.latency_p95_ms = all_latencies[int(len(all_latencies) * 0.95)]
        result.latency_p99_ms = all_latencies[int(len(all_latencies) * 0.99)]

        # Breakdowns
        for cat, qrs in by_query_category.items():
            n_cat = len(qrs)
            result.by_query_category[cat] = {
                "total": n_cat,
                "recall_at_1": sum(1 for r in qrs if r.success_at_1) / n_cat,
                "recall_at_5": sum(1 for r in qrs if r.success_at_5) / n_cat,
                "mrr": sum(r.reciprocal_rank for r in qrs) / n_cat
            }

        for diff, qrs in by_difficulty.items():
            n_diff = len(qrs)
            result.by_difficulty[diff] = {
                "total": n_diff,
                "recall_at_1": sum(1 for r in qrs if r.success_at_1) / n_diff,
                "recall_at_5": sum(1 for r in qrs if r.success_at_5) / n_diff,
                "mrr": sum(r.reciprocal_rank for r in qrs) / n_diff
            }

        for cat, qrs in by_memory_category.items():
            n_cat = len(qrs)
            result.by_memory_category[cat] = {
                "total": n_cat,
                "recall_at_1": sum(1 for r in qrs if r.success_at_1) / n_cat,
                "recall_at_5": sum(1 for r in qrs if r.success_at_5) / n_cat,
                "mrr": sum(r.reciprocal_rank for r in qrs) / n_cat
            }

        result.complete_misses = sum(1 for r in all_results if r.rank is None)

        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)

        # Print summary
        total_time = time.time() - start_time
        print()
        print("=" * 80)
        print("FORTUNE 100 EXHAUSTIVE TEST COMPLETE")
        print("=" * 80)

        print(f"\nTOTAL EXECUTION TIME: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"Queries per second: {n/total_time:.1f}")

        print(f"\n{'='*40}")
        print("OVERALL METRICS")
        print(f"{'='*40}")
        print(f"  Total Queries: {result.total_queries}")
        print(f"  Total Memories: {result.total_memories}")
        print()

        r5_status = "PASS" if result.overall_recall_at_5 >= 0.95 else "FAIL"
        print(f"  Recall@1:  {result.overall_recall_at_1:.2%}")
        print(f"  Recall@3:  {result.overall_recall_at_3:.2%}")
        print(f"  Recall@5:  {result.overall_recall_at_5:.2%}  <-- TARGET 95% [{r5_status}]")
        print(f"  Recall@10: {result.overall_recall_at_10:.2%}")
        print(f"  MRR:       {result.overall_mrr:.4f}")
        print(f"  NDCG@10:   {result.overall_ndcg_at_10:.4f}")

        print(f"\n{'='*40}")
        print("LATENCY (GPU-accelerated reranking)")
        print(f"{'='*40}")
        print(f"  Avg:  {result.latency_avg_ms:.1f}ms")
        print(f"  P50:  {result.latency_p50_ms:.1f}ms")
        print(f"  P95:  {result.latency_p95_ms:.1f}ms")
        print(f"  P99:  {result.latency_p99_ms:.1f}ms")

        print(f"\n{'='*40}")
        print("BY QUERY CATEGORY")
        print(f"{'='*40}")
        for cat, data in sorted(result.by_query_category.items()):
            status = "PASS" if data['recall_at_5'] >= 0.95 else "FAIL"
            print(f"  {cat:20s}: R@5={data['recall_at_5']:.1%} [{status}] | MRR={data['mrr']:.3f} (n={data['total']})")

        print(f"\n{'='*40}")
        print("BY DIFFICULTY")
        print(f"{'='*40}")
        for diff, data in sorted(result.by_difficulty.items()):
            status = "PASS" if data['recall_at_5'] >= 0.95 else "FAIL"
            print(f"  {diff:20s}: R@5={data['recall_at_5']:.1%} [{status}] | MRR={data['mrr']:.3f} (n={data['total']})")

        print(f"\n{'='*40}")
        print("FAILURE ANALYSIS")
        print(f"{'='*40}")
        print(f"  Complete misses (not in top 10): {result.complete_misses} ({result.complete_misses/n:.1%})")
        print(f"  Failures at @5: {len(result.failures)} ({len(result.failures)/n:.1%})")

        if result.failures[:5]:
            print(f"\n  Sample failures:")
            for f in result.failures[:5]:
                print(f"    - Memory {f['memory_id']}: rank={f['rank']} | {f['query'][:50]}...")

        print(f"\n{'='*40}")
        print("VERDICT")
        print(f"{'='*40}")
        if result.overall_recall_at_5 >= 0.95:
            print("  >>> FORTUNE 100 QUALITY: ACHIEVED <<<")
            print(f"  >>> Recall@5 = {result.overall_recall_at_5:.2%} (>= 95% target)")
        else:
            print("  >>> FORTUNE 100 QUALITY: NOT ACHIEVED <<<")
            print(f"  >>> Recall@5 = {result.overall_recall_at_5:.2%} (< 95% target)")
            print(f"  >>> Gap: {0.95 - result.overall_recall_at_5:.2%}")

        print(f"\nResults saved to: {output_path}")
        print("=" * 80)

        return result


def main():
    """Run Fortune 100 exhaustive test."""
    test_suite = Path(__file__).parent / "test_suite" / "enhanced_test_suite.json"
    output = Path(__file__).parent / "optimization_results" / f"fortune100_gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    if not test_suite.exists():
        print(f"ERROR: Test suite not found: {test_suite}")
        print("Please generate the test suite first.")
        return None

    evaluator = Fortune100Evaluator()
    result = evaluator.run_evaluation(test_suite, output)

    return result


if __name__ == "__main__":
    main()
