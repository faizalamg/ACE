"""
Baseline Evaluation Script for RAG Optimization

Runs all generated test queries against Qdrant and measures:
- Recall@K (K=1, 3, 5, 10)
- Mean Reciprocal Rank (MRR)
- NDCG@10
- Precision@K
- Latency statistics
- Failure analysis by category and difficulty

Saves comprehensive results for later comparison with optimized versions.
"""

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import httpx


# ============================================================================
# CONFIGURATION
# ============================================================================

QDRANT_URL = "http://localhost:6333"
EMBEDDING_URL = "http://localhost:1234"
COLLECTION_NAME = "ace_memories_hybrid"
EMBEDDING_MODEL = "text-embedding-qwen3-embedding-8b"  # Qwen3-Embedding-8B
TOP_K = 10  # Retrieve top 10 results


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
    false_positives_above: int  # Number of wrong results ranked above correct


@dataclass
class MemoryTestResult:
    """Aggregated results for a single memory's test queries."""
    memory_id: int
    memory_content: str
    memory_category: str
    total_queries: int
    query_results: List[QueryResult] = field(default_factory=list)

    # Aggregated metrics
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    avg_latency_ms: float = 0.0


@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    timestamp: str
    configuration: Dict[str, str]
    test_suite_stats: Dict[str, Any]

    # Overall metrics
    total_queries: int = 0
    overall_recall_at_1: float = 0.0
    overall_recall_at_3: float = 0.0
    overall_recall_at_5: float = 0.0
    overall_recall_at_10: float = 0.0
    overall_mrr: float = 0.0
    overall_ndcg_at_10: float = 0.0

    # Latency stats
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_avg_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0

    # Breakdown by category
    results_by_query_category: Dict[str, Dict] = field(default_factory=dict)
    results_by_difficulty: Dict[str, Dict] = field(default_factory=dict)
    results_by_memory_category: Dict[str, Dict] = field(default_factory=dict)

    # Failure analysis
    complete_misses: int = 0  # Not in top 10
    low_rank: int = 0  # Found but rank > 5
    false_positive_dominant: int = 0  # Wrong result at rank 1

    # Memory-level results
    memory_results: List[Dict] = field(default_factory=list)


# ============================================================================
# EVALUATION ENGINE
# ============================================================================

class BaselineEvaluator:
    """Evaluator for RAG retrieval performance."""

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        embedding_url: str = EMBEDDING_URL,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL
    ):
        self.qdrant_url = qdrant_url
        self.embedding_url = embedding_url
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.client = httpx.Client(timeout=60.0)

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding vector for text."""
        try:
            resp = self.client.post(
                f"{self.embedding_url}/v1/embeddings",
                json={
                    "model": self.embedding_model,
                    "input": text
                }
            )
            if resp.status_code == 200:
                return resp.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"  Embedding error: {e}")
        return None

    def search(self, query: str, limit: int = TOP_K) -> Tuple[List[Dict], float]:
        """Execute hybrid search and return results with latency."""
        start = time.perf_counter()

        # Get query embedding
        embedding = self.get_embedding(query)
        if not embedding:
            return [], (time.perf_counter() - start) * 1000

        # Hybrid search with RRF
        search_body = {
            "prefetch": [
                {
                    "query": embedding,
                    "using": "dense",
                    "limit": limit * 3
                }
            ],
            "query": {"fusion": "rrf"},
            "limit": limit,
            "with_payload": True
        }

        try:
            resp = self.client.post(
                f"{self.qdrant_url}/collections/{self.collection_name}/points/query",
                json=search_body
            )
            latency = (time.perf_counter() - start) * 1000

            if resp.status_code == 200:
                points = resp.json().get("result", {}).get("points", [])
                return points, latency
        except Exception as e:
            print(f"  Search error: {e}")

        return [], (time.perf_counter() - start) * 1000

    def evaluate_query(
        self,
        query: str,
        query_category: str,
        difficulty: str,
        expected_memory_id: int
    ) -> QueryResult:
        """Evaluate a single query."""
        results, latency = self.search(query)

        retrieved_ids = [r["id"] for r in results]
        retrieved_scores = [r.get("score", 0) for r in results]

        # Find rank of expected memory
        rank = None
        score = None
        if expected_memory_id in retrieved_ids:
            rank = retrieved_ids.index(expected_memory_id) + 1
            score = retrieved_scores[retrieved_ids.index(expected_memory_id)]

        # Calculate metrics
        success_at_1 = rank == 1 if rank else False
        success_at_3 = rank is not None and rank <= 3
        success_at_5 = rank is not None and rank <= 5
        success_at_10 = rank is not None and rank <= 10
        reciprocal_rank = 1.0 / rank if rank else 0.0
        false_positives_above = rank - 1 if rank else len(retrieved_ids)

        return QueryResult(
            query=query,
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
            latency_ms=latency,
            false_positives_above=false_positives_above
        )

    def evaluate_memory(self, test_case: Dict) -> MemoryTestResult:
        """Evaluate all queries for a memory."""
        memory_id = test_case["memory_id"]
        content = test_case["content"]
        category = test_case["category"]
        queries = test_case.get("generated_queries", [])

        result = MemoryTestResult(
            memory_id=memory_id,
            memory_content=content,
            memory_category=category,
            total_queries=len(queries)
        )

        for q in queries:
            qr = self.evaluate_query(
                query=q["query"],
                query_category=q["category"],
                difficulty=q["difficulty"],
                expected_memory_id=memory_id
            )
            result.query_results.append(qr)

        # Aggregate metrics
        if result.query_results:
            n = len(result.query_results)
            result.recall_at_1 = sum(1 for r in result.query_results if r.success_at_1) / n
            result.recall_at_3 = sum(1 for r in result.query_results if r.success_at_3) / n
            result.recall_at_5 = sum(1 for r in result.query_results if r.success_at_5) / n
            result.recall_at_10 = sum(1 for r in result.query_results if r.success_at_10) / n
            result.mrr = sum(r.reciprocal_rank for r in result.query_results) / n
            result.avg_latency_ms = sum(r.latency_ms for r in result.query_results) / n

        return result

    def calculate_ndcg(self, query_results: List[QueryResult], k: int = 10) -> float:
        """Calculate NDCG@k for a set of query results."""
        ndcg_scores = []

        for qr in query_results:
            if qr.rank is None or qr.rank > k:
                ndcg_scores.append(0.0)
                continue

            # DCG: 1 / log2(rank + 1) for binary relevance
            dcg = 1.0 / math.log2(qr.rank + 1)

            # IDCG: Perfect ranking would have rank 1
            idcg = 1.0 / math.log2(2)

            ndcg_scores.append(dcg / idcg)

        return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    def run_evaluation(self, test_suite_path: Path, output_path: Path) -> EvaluationResult:
        """Run complete evaluation on test suite."""
        print(f"\n{'='*80}")
        print("BASELINE EVALUATION")
        print(f"{'='*80}")
        print(f"Test Suite: {test_suite_path}")
        print(f"Output: {output_path}")
        print(f"{'='*80}\n")

        # Load test suite
        with open(test_suite_path) as f:
            data = json.load(f)

        test_cases = data["test_cases"]
        metadata = data.get("metadata", {})

        print(f"Loaded {len(test_cases)} test cases")
        total_queries = sum(len(tc.get("generated_queries", [])) for tc in test_cases)
        print(f"Total queries to evaluate: {total_queries}")

        # Initialize result
        result = EvaluationResult(
            timestamp=datetime.now().isoformat(),
            configuration={
                "qdrant_url": self.qdrant_url,
                "embedding_url": self.embedding_url,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
                "top_k": TOP_K
            },
            test_suite_stats=metadata.get("generation_stats", {})
        )

        # Track all query results for aggregation
        all_query_results: List[QueryResult] = []
        all_latencies: List[float] = []

        # Aggregators by category
        by_query_category = defaultdict(list)
        by_difficulty = defaultdict(list)
        by_memory_category = defaultdict(list)

        # Evaluate each memory
        for i, tc in enumerate(test_cases):
            print(f"\n[{i+1}/{len(test_cases)}] Memory {tc['memory_id']} ({tc['category']})")
            print(f"  Content: {tc['content'][:60]}...")

            mem_result = self.evaluate_memory(tc)

            print(f"  Queries: {mem_result.total_queries}")
            print(f"  Recall@1: {mem_result.recall_at_1:.2%}")
            print(f"  Recall@5: {mem_result.recall_at_5:.2%}")
            print(f"  MRR: {mem_result.mrr:.3f}")

            # Collect all query results
            all_query_results.extend(mem_result.query_results)

            for qr in mem_result.query_results:
                all_latencies.append(qr.latency_ms)
                by_query_category[qr.query_category].append(qr)
                by_difficulty[qr.difficulty].append(qr)
                by_memory_category[tc["category"]].append(qr)

            # Store memory-level summary
            result.memory_results.append({
                "memory_id": mem_result.memory_id,
                "category": mem_result.memory_category,
                "total_queries": mem_result.total_queries,
                "recall_at_1": mem_result.recall_at_1,
                "recall_at_5": mem_result.recall_at_5,
                "recall_at_10": mem_result.recall_at_10,
                "mrr": mem_result.mrr,
                "avg_latency_ms": mem_result.avg_latency_ms
            })

        # Calculate overall metrics
        n = len(all_query_results)
        result.total_queries = n

        result.overall_recall_at_1 = sum(1 for r in all_query_results if r.success_at_1) / n
        result.overall_recall_at_3 = sum(1 for r in all_query_results if r.success_at_3) / n
        result.overall_recall_at_5 = sum(1 for r in all_query_results if r.success_at_5) / n
        result.overall_recall_at_10 = sum(1 for r in all_query_results if r.success_at_10) / n
        result.overall_mrr = sum(r.reciprocal_rank for r in all_query_results) / n
        result.overall_ndcg_at_10 = self.calculate_ndcg(all_query_results, k=10)

        # Latency statistics
        all_latencies.sort()
        result.latency_min_ms = min(all_latencies)
        result.latency_max_ms = max(all_latencies)
        result.latency_avg_ms = sum(all_latencies) / len(all_latencies)
        result.latency_p50_ms = all_latencies[len(all_latencies) // 2]
        result.latency_p95_ms = all_latencies[int(len(all_latencies) * 0.95)]
        result.latency_p99_ms = all_latencies[int(len(all_latencies) * 0.99)]

        # Breakdown by query category
        for cat, qrs in by_query_category.items():
            n_cat = len(qrs)
            result.results_by_query_category[cat] = {
                "total": n_cat,
                "recall_at_1": sum(1 for r in qrs if r.success_at_1) / n_cat,
                "recall_at_5": sum(1 for r in qrs if r.success_at_5) / n_cat,
                "mrr": sum(r.reciprocal_rank for r in qrs) / n_cat
            }

        # Breakdown by difficulty
        for diff, qrs in by_difficulty.items():
            n_diff = len(qrs)
            result.results_by_difficulty[diff] = {
                "total": n_diff,
                "recall_at_1": sum(1 for r in qrs if r.success_at_1) / n_diff,
                "recall_at_5": sum(1 for r in qrs if r.success_at_5) / n_diff,
                "mrr": sum(r.reciprocal_rank for r in qrs) / n_diff
            }

        # Breakdown by memory category
        for cat, qrs in by_memory_category.items():
            n_cat = len(qrs)
            result.results_by_memory_category[cat] = {
                "total": n_cat,
                "recall_at_1": sum(1 for r in qrs if r.success_at_1) / n_cat,
                "recall_at_5": sum(1 for r in qrs if r.success_at_5) / n_cat,
                "mrr": sum(r.reciprocal_rank for r in qrs) / n_cat
            }

        # Failure analysis
        result.complete_misses = sum(1 for r in all_query_results if r.rank is None)
        result.low_rank = sum(1 for r in all_query_results if r.rank and r.rank > 5)
        result.false_positive_dominant = sum(1 for r in all_query_results if r.rank and r.rank > 1)

        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)

        # Print summary
        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"\nOVERALL METRICS:")
        print(f"  Total Queries: {result.total_queries}")
        print(f"  Recall@1: {result.overall_recall_at_1:.2%}")
        print(f"  Recall@3: {result.overall_recall_at_3:.2%}")
        print(f"  Recall@5: {result.overall_recall_at_5:.2%}")
        print(f"  Recall@10: {result.overall_recall_at_10:.2%}")
        print(f"  MRR: {result.overall_mrr:.4f}")
        print(f"  NDCG@10: {result.overall_ndcg_at_10:.4f}")

        print(f"\nLATENCY:")
        print(f"  Min: {result.latency_min_ms:.1f}ms")
        print(f"  Avg: {result.latency_avg_ms:.1f}ms")
        print(f"  P50: {result.latency_p50_ms:.1f}ms")
        print(f"  P95: {result.latency_p95_ms:.1f}ms")
        print(f"  P99: {result.latency_p99_ms:.1f}ms")
        print(f"  Max: {result.latency_max_ms:.1f}ms")

        print(f"\nFAILURE ANALYSIS:")
        print(f"  Complete Misses (not in top 10): {result.complete_misses} ({result.complete_misses/n:.1%})")
        print(f"  Low Rank (rank > 5): {result.low_rank} ({result.low_rank/n:.1%})")
        print(f"  False Positive at #1: {result.false_positive_dominant} ({result.false_positive_dominant/n:.1%})")

        print(f"\nBY DIFFICULTY:")
        for diff, data in sorted(result.results_by_difficulty.items()):
            print(f"  {diff}: Recall@1={data['recall_at_1']:.2%}, MRR={data['mrr']:.3f} (n={data['total']})")

        print(f"\nBY QUERY CATEGORY:")
        for cat, data in sorted(result.results_by_query_category.items()):
            print(f"  {cat}: Recall@1={data['recall_at_1']:.2%}, MRR={data['mrr']:.3f} (n={data['total']})")

        print(f"\nResults saved to: {output_path}")

        return result

    def close(self):
        self.client.close()


def main():
    """Run baseline evaluation."""
    test_suite = Path(__file__).parent / "test_suite" / "enhanced_test_suite.json"
    output = Path(__file__).parent / "baseline_results" / "comprehensive_baseline.json"

    evaluator = BaselineEvaluator()
    try:
        result = evaluator.run_evaluation(test_suite, output)
    finally:
        evaluator.close()

    return result


if __name__ == "__main__":
    main()
