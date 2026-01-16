"""
Optimization V1: Cross-Encoder Re-ranking

This optimization applies cross-encoder re-ranking to the hybrid search results.
Cross-encoders jointly encode query+document pairs, providing more accurate
relevance scores than bi-encoder (embedding) approaches.

Expected improvement: +15-25% Recall@K improvement over baseline hybrid.

Strategy:
1. Retrieve top-30 candidates using hybrid search (dense + BM25 + RRF)
2. Re-rank top-30 using cross-encoder model
3. Return top-K from re-ranked results
"""

import json
import math
import re
import time
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import httpx

# Try to import sentence-transformers for cross-encoder
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    print("WARNING: sentence-transformers not installed. Install with: pip install sentence-transformers")


# ============================================================================
# CONFIGURATION
# ============================================================================

QDRANT_URL = "http://localhost:6333"
EMBEDDING_URL = "http://localhost:1234"
COLLECTION_NAME = "ace_memories_hybrid"
EMBEDDING_MODEL = "text-embedding-qwen3-embedding-8b"

# Re-ranking configuration
FIRST_STAGE_K = 30  # Retrieve more candidates for re-ranking
FINAL_K = 10  # Final number of results
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Fast, accurate model

# BM25 parameters
BM25_K1 = 1.5
BM25_B = 0.75
AVG_DOC_LENGTH = 50

# Technical programming stopwords
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
    'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have',
    'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
    'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'it', 'its',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'what',
    'which', 'who', 'whom', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
    'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now'
}


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
    rank: Optional[int]
    score: Optional[float]
    success_at_1: bool
    success_at_3: bool
    success_at_5: bool
    success_at_10: bool
    reciprocal_rank: float
    latency_ms: float
    false_positives_above: int
    rerank_latency_ms: float = 0.0


@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    timestamp: str
    configuration: Dict[str, str]
    test_suite_stats: Dict[str, Any]
    search_type: str
    optimization: str

    total_queries: int = 0
    overall_recall_at_1: float = 0.0
    overall_recall_at_3: float = 0.0
    overall_recall_at_5: float = 0.0
    overall_recall_at_10: float = 0.0
    overall_mrr: float = 0.0
    overall_ndcg_at_10: float = 0.0

    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_avg_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0

    rerank_latency_avg_ms: float = 0.0

    results_by_query_category: Dict[str, Dict] = field(default_factory=dict)
    results_by_difficulty: Dict[str, Dict] = field(default_factory=dict)
    results_by_memory_category: Dict[str, Dict] = field(default_factory=dict)

    complete_misses: int = 0
    low_rank: int = 0
    false_positive_dominant: int = 0

    improvement_over_baseline: Dict[str, float] = field(default_factory=dict)
    memory_results: List[Dict] = field(default_factory=list)


# ============================================================================
# BM25 TOKENIZATION
# ============================================================================

def tokenize_bm25(text: str) -> List[str]:
    """Tokenize text for BM25, preserving technical terms."""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = text.replace('_', ' ')
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def compute_bm25_sparse(text: str) -> Dict[str, Any]:
    """Compute BM25-style sparse vector for Qdrant."""
    tokens = tokenize_bm25(text)
    if not tokens:
        return {"indices": [], "values": []}

    tf = Counter(tokens)
    doc_length = len(tokens)

    indices = []
    values = []

    for term, freq in tf.items():
        term_hash = int(hashlib.md5(term.encode()).hexdigest()[:8], 16)
        indices.append(term_hash)

        tf_weight = (freq * (BM25_K1 + 1)) / (
            freq + BM25_K1 * (1 - BM25_B + BM25_B * doc_length / AVG_DOC_LENGTH)
        )
        values.append(float(tf_weight))

    return {"indices": indices, "values": values}


# ============================================================================
# CROSS-ENCODER RE-RANKING EVALUATOR
# ============================================================================

class CrossEncoderRerankerEvaluator:
    """Evaluator with cross-encoder re-ranking on top of hybrid search."""

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        embedding_url: str = EMBEDDING_URL,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        cross_encoder_model: str = CROSS_ENCODER_MODEL
    ):
        self.qdrant_url = qdrant_url
        self.embedding_url = embedding_url
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.client = httpx.Client(timeout=60.0)

        # Initialize cross-encoder
        if CROSS_ENCODER_AVAILABLE:
            print(f"Loading cross-encoder model: {cross_encoder_model}")
            self.cross_encoder = CrossEncoder(cross_encoder_model, max_length=512)
            print("Cross-encoder loaded successfully")
        else:
            self.cross_encoder = None
            print("WARNING: Cross-encoder not available, using score passthrough")

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get dense embedding vector."""
        try:
            resp = self.client.post(
                f"{self.embedding_url}/v1/embeddings",
                json={
                    "model": self.embedding_model,
                    "input": text[:8000]
                }
            )
            if resp.status_code == 200:
                return resp.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"  Embedding error: {e}")
        return None

    def hybrid_search_first_stage(self, query: str, limit: int = FIRST_STAGE_K) -> Tuple[List[Dict], float]:
        """
        First stage: Hybrid search to get candidate documents.
        Retrieves more candidates than needed for re-ranking.
        """
        start = time.perf_counter()

        dense_embedding = self.get_embedding(query)
        if not dense_embedding:
            return [], (time.perf_counter() - start) * 1000

        sparse_vector = compute_bm25_sparse(query)

        hybrid_query = {
            "prefetch": [
                {
                    "query": dense_embedding,
                    "using": "dense",
                    "limit": limit * 2  # Get even more for better recall
                }
            ],
            "query": {"fusion": "rrf"},
            "limit": limit,
            "with_payload": True
        }

        if sparse_vector.get("indices"):
            hybrid_query["prefetch"].append({
                "query": {
                    "indices": sparse_vector["indices"],
                    "values": sparse_vector["values"]
                },
                "using": "sparse",
                "limit": limit * 2
            })

        try:
            resp = self.client.post(
                f"{self.qdrant_url}/collections/{self.collection_name}/points/query",
                json=hybrid_query
            )
            latency = (time.perf_counter() - start) * 1000

            if resp.status_code == 200:
                points = resp.json().get("result", {}).get("points", [])
                return points, latency
        except Exception as e:
            print(f"  Search error: {e}")

        return [], (time.perf_counter() - start) * 1000

    def rerank_with_cross_encoder(
        self,
        query: str,
        candidates: List[Dict],
        final_k: int = FINAL_K
    ) -> Tuple[List[Dict], float]:
        """
        Re-rank candidates using cross-encoder.

        Cross-encoder jointly encodes (query, document) pairs and produces
        more accurate relevance scores than bi-encoder similarity.
        """
        if not self.cross_encoder or not candidates:
            return candidates[:final_k], 0.0

        start = time.perf_counter()

        # Prepare query-document pairs
        pairs = []
        for c in candidates:
            # Get document content from payload
            payload = c.get("payload", {})
            doc_text = payload.get("lesson", "") or payload.get("content", "") or str(payload)
            pairs.append([query, doc_text[:1000]])  # Truncate for speed

        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)

        # Sort by cross-encoder score
        scored_candidates = list(zip(scores, candidates))
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Update candidates with cross-encoder scores
        reranked = []
        for score, candidate in scored_candidates[:final_k]:
            candidate["original_score"] = candidate.get("score", 0)
            candidate["score"] = float(score)
            candidate["reranked"] = True
            reranked.append(candidate)

        latency = (time.perf_counter() - start) * 1000
        return reranked, latency

    def search_with_reranking(self, query: str, limit: int = FINAL_K) -> Tuple[List[Dict], float, float]:
        """
        Full search pipeline: hybrid retrieval + cross-encoder re-ranking.

        Returns: (results, retrieval_latency, rerank_latency)
        """
        # Stage 1: Get candidates
        candidates, retrieval_latency = self.hybrid_search_first_stage(query, FIRST_STAGE_K)

        # Stage 2: Re-rank with cross-encoder
        reranked, rerank_latency = self.rerank_with_cross_encoder(query, candidates, limit)

        return reranked, retrieval_latency, rerank_latency

    def evaluate_query(
        self,
        query: str,
        query_category: str,
        difficulty: str,
        expected_memory_id: int
    ) -> QueryResult:
        """Evaluate a single query with re-ranking."""
        results, retrieval_latency, rerank_latency = self.search_with_reranking(query)
        total_latency = retrieval_latency + rerank_latency

        retrieved_ids = [r["id"] for r in results]
        retrieved_scores = [r.get("score", 0) for r in results]

        rank = None
        score = None
        if expected_memory_id in retrieved_ids:
            rank = retrieved_ids.index(expected_memory_id) + 1
            score = retrieved_scores[retrieved_ids.index(expected_memory_id)]

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
            latency_ms=total_latency,
            false_positives_above=false_positives_above,
            rerank_latency_ms=rerank_latency
        )

    def evaluate_memory(self, test_case: Dict) -> Dict:
        """Evaluate all queries for a memory."""
        memory_id = test_case["memory_id"]
        content = test_case["content"]
        category = test_case["category"]
        queries = test_case.get("generated_queries", [])

        query_results = []
        for q in queries:
            qr = self.evaluate_query(
                query=q["query"],
                query_category=q["category"],
                difficulty=q["difficulty"],
                expected_memory_id=memory_id
            )
            query_results.append(qr)

        n = len(query_results)
        if n == 0:
            return {
                "memory_id": memory_id,
                "category": category,
                "total_queries": 0,
                "recall_at_1": 0,
                "recall_at_5": 0,
                "recall_at_10": 0,
                "mrr": 0,
                "avg_latency_ms": 0,
                "query_results": []
            }

        return {
            "memory_id": memory_id,
            "category": category,
            "content": content[:100],
            "total_queries": n,
            "recall_at_1": sum(1 for r in query_results if r.success_at_1) / n,
            "recall_at_5": sum(1 for r in query_results if r.success_at_5) / n,
            "recall_at_10": sum(1 for r in query_results if r.success_at_10) / n,
            "mrr": sum(r.reciprocal_rank for r in query_results) / n,
            "avg_latency_ms": sum(r.latency_ms for r in query_results) / n,
            "avg_rerank_latency_ms": sum(r.rerank_latency_ms for r in query_results) / n,
            "query_results": query_results
        }

    def calculate_ndcg(self, query_results: List[QueryResult], k: int = 10) -> float:
        """Calculate NDCG@k."""
        ndcg_scores = []
        for qr in query_results:
            if qr.rank is None or qr.rank > k:
                ndcg_scores.append(0.0)
                continue
            dcg = 1.0 / math.log2(qr.rank + 1)
            idcg = 1.0 / math.log2(2)
            ndcg_scores.append(dcg / idcg)
        return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    def load_baseline_results(self, baseline_path: Path) -> Optional[Dict]:
        """Load baseline results for comparison."""
        if baseline_path.exists():
            with open(baseline_path) as f:
                return json.load(f)
        return None

    def run_evaluation(
        self,
        test_suite_path: Path,
        output_path: Path,
        baseline_path: Optional[Path] = None
    ) -> EvaluationResult:
        """Run complete evaluation with cross-encoder re-ranking."""
        print(f"\n{'='*80}")
        print("OPTIMIZATION V1: CROSS-ENCODER RE-RANKING")
        print(f"{'='*80}")
        print(f"Test Suite: {test_suite_path}")
        print(f"Output: {output_path}")
        print(f"First Stage K: {FIRST_STAGE_K}")
        print(f"Final K: {FINAL_K}")
        print(f"Cross-Encoder: {CROSS_ENCODER_MODEL}")
        print(f"{'='*80}\n")

        # Load baseline for comparison
        baseline = None
        if baseline_path:
            baseline = self.load_baseline_results(baseline_path)
            if baseline:
                print(f"Loaded baseline from: {baseline_path}")
                print(f"  Baseline Recall@1: {baseline['overall_recall_at_1']:.2%}")
                print(f"  Baseline Recall@5: {baseline['overall_recall_at_5']:.2%}")
                print(f"  Baseline MRR: {baseline['overall_mrr']:.4f}")

        with open(test_suite_path) as f:
            data = json.load(f)

        test_cases = data["test_cases"]
        metadata = data.get("metadata", {})

        print(f"\nLoaded {len(test_cases)} test cases")
        total_queries = sum(len(tc.get("generated_queries", [])) for tc in test_cases)
        print(f"Total queries: {total_queries}")

        result = EvaluationResult(
            timestamp=datetime.now().isoformat(),
            configuration={
                "qdrant_url": self.qdrant_url,
                "embedding_url": self.embedding_url,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
                "first_stage_k": FIRST_STAGE_K,
                "final_k": FINAL_K,
                "cross_encoder_model": CROSS_ENCODER_MODEL,
                "bm25_k1": BM25_K1,
                "bm25_b": BM25_B
            },
            test_suite_stats=metadata.get("generation_stats", {}),
            search_type="hybrid_with_reranking",
            optimization="cross_encoder_v1"
        )

        all_query_results: List[QueryResult] = []
        all_latencies: List[float] = []
        all_rerank_latencies: List[float] = []
        by_query_category = defaultdict(list)
        by_difficulty = defaultdict(list)
        by_memory_category = defaultdict(list)

        for i, tc in enumerate(test_cases):
            print(f"\n[{i+1}/{len(test_cases)}] Memory {tc['memory_id']} ({tc['category']})")
            print(f"  Content: {tc['content'][:60]}...")

            mem_result = self.evaluate_memory(tc)

            print(f"  Queries: {mem_result['total_queries']}")
            print(f"  Recall@1: {mem_result['recall_at_1']:.2%}")
            print(f"  Recall@5: {mem_result['recall_at_5']:.2%}")
            print(f"  MRR: {mem_result['mrr']:.3f}")
            print(f"  Avg Rerank Latency: {mem_result['avg_rerank_latency_ms']:.1f}ms")

            for qr in mem_result["query_results"]:
                all_query_results.append(qr)
                all_latencies.append(qr.latency_ms)
                all_rerank_latencies.append(qr.rerank_latency_ms)
                by_query_category[qr.query_category].append(qr)
                by_difficulty[qr.difficulty].append(qr)
                by_memory_category[tc["category"]].append(qr)

            result.memory_results.append({
                "memory_id": mem_result["memory_id"],
                "category": mem_result["category"],
                "total_queries": mem_result["total_queries"],
                "recall_at_1": mem_result["recall_at_1"],
                "recall_at_5": mem_result["recall_at_5"],
                "recall_at_10": mem_result["recall_at_10"],
                "mrr": mem_result["mrr"],
                "avg_latency_ms": mem_result["avg_latency_ms"],
                "avg_rerank_latency_ms": mem_result["avg_rerank_latency_ms"]
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

        # Latency stats
        all_latencies.sort()
        result.latency_min_ms = min(all_latencies)
        result.latency_max_ms = max(all_latencies)
        result.latency_avg_ms = sum(all_latencies) / len(all_latencies)
        result.latency_p50_ms = all_latencies[len(all_latencies) // 2]
        result.latency_p95_ms = all_latencies[int(len(all_latencies) * 0.95)]
        result.latency_p99_ms = all_latencies[int(len(all_latencies) * 0.99)]
        result.rerank_latency_avg_ms = sum(all_rerank_latencies) / len(all_rerank_latencies)

        # Breakdowns
        for cat, qrs in by_query_category.items():
            n_cat = len(qrs)
            result.results_by_query_category[cat] = {
                "total": n_cat,
                "recall_at_1": sum(1 for r in qrs if r.success_at_1) / n_cat,
                "recall_at_5": sum(1 for r in qrs if r.success_at_5) / n_cat,
                "mrr": sum(r.reciprocal_rank for r in qrs) / n_cat
            }

        for diff, qrs in by_difficulty.items():
            n_diff = len(qrs)
            result.results_by_difficulty[diff] = {
                "total": n_diff,
                "recall_at_1": sum(1 for r in qrs if r.success_at_1) / n_diff,
                "recall_at_5": sum(1 for r in qrs if r.success_at_5) / n_diff,
                "mrr": sum(r.reciprocal_rank for r in qrs) / n_diff
            }

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

        # Calculate improvement over baseline
        if baseline:
            result.improvement_over_baseline = {
                "recall_at_1_delta": result.overall_recall_at_1 - baseline["overall_recall_at_1"],
                "recall_at_5_delta": result.overall_recall_at_5 - baseline["overall_recall_at_5"],
                "recall_at_10_delta": result.overall_recall_at_10 - baseline["overall_recall_at_10"],
                "mrr_delta": result.overall_mrr - baseline["overall_mrr"],
                "recall_at_1_pct_improvement": (
                    (result.overall_recall_at_1 - baseline["overall_recall_at_1"])
                    / baseline["overall_recall_at_1"] * 100
                ) if baseline["overall_recall_at_1"] > 0 else 0,
                "recall_at_5_pct_improvement": (
                    (result.overall_recall_at_5 - baseline["overall_recall_at_5"])
                    / baseline["overall_recall_at_5"] * 100
                ) if baseline["overall_recall_at_5"] > 0 else 0,
            }

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)

        # Print summary
        print(f"\n{'='*80}")
        print("CROSS-ENCODER RE-RANKING EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"\nOVERALL METRICS:")
        print(f"  Total Queries: {result.total_queries}")
        print(f"  Recall@1: {result.overall_recall_at_1:.2%}")
        print(f"  Recall@3: {result.overall_recall_at_3:.2%}")
        print(f"  Recall@5: {result.overall_recall_at_5:.2%}")
        print(f"  Recall@10: {result.overall_recall_at_10:.2%}")
        print(f"  MRR: {result.overall_mrr:.4f}")
        print(f"  NDCG@10: {result.overall_ndcg_at_10:.4f}")

        if baseline:
            print(f"\nIMPROVEMENT OVER BASELINE:")
            imp = result.improvement_over_baseline
            print(f"  Recall@1: {baseline['overall_recall_at_1']:.2%} -> {result.overall_recall_at_1:.2%} ({imp['recall_at_1_delta']:+.2%}, {imp['recall_at_1_pct_improvement']:+.1f}%)")
            print(f"  Recall@5: {baseline['overall_recall_at_5']:.2%} -> {result.overall_recall_at_5:.2%} ({imp['recall_at_5_delta']:+.2%}, {imp['recall_at_5_pct_improvement']:+.1f}%)")
            print(f"  MRR: {baseline['overall_mrr']:.4f} -> {result.overall_mrr:.4f} ({imp['mrr_delta']:+.4f})")

        print(f"\nLATENCY:")
        print(f"  Total Avg: {result.latency_avg_ms:.1f}ms")
        print(f"  Rerank Avg: {result.rerank_latency_avg_ms:.1f}ms")
        print(f"  P50: {result.latency_p50_ms:.1f}ms")
        print(f"  P95: {result.latency_p95_ms:.1f}ms")
        print(f"  P99: {result.latency_p99_ms:.1f}ms")

        print(f"\nFAILURE ANALYSIS:")
        print(f"  Complete Misses: {result.complete_misses} ({result.complete_misses/n:.1%})")
        print(f"  Low Rank (>5): {result.low_rank} ({result.low_rank/n:.1%})")
        print(f"  Not at #1: {result.false_positive_dominant} ({result.false_positive_dominant/n:.1%})")

        print(f"\nBY DIFFICULTY:")
        for diff, data in sorted(result.results_by_difficulty.items()):
            print(f"  {diff}: R@1={data['recall_at_1']:.2%}, MRR={data['mrr']:.3f} (n={data['total']})")

        print(f"\nResults saved to: {output_path}")

        return result

    def close(self):
        self.client.close()


def main():
    """Run cross-encoder re-ranking evaluation."""
    base_path = Path(__file__).parent.parent
    test_suite = base_path / "test_suite" / "enhanced_test_suite.json"
    baseline_path = base_path / "baseline_results" / "hybrid_baseline.json"
    output = base_path / "optimization_results" / "v1_cross_encoder_rerank.json"

    evaluator = CrossEncoderRerankerEvaluator()
    try:
        result = evaluator.run_evaluation(test_suite, output, baseline_path)
    finally:
        evaluator.close()

    return result


if __name__ == "__main__":
    main()
