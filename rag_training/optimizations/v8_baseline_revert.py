"""
V8: Baseline Revert - Verify 62%+ Recall@5 Restoration
========================================================

PURPOSE: Verify we can restore original 62.52% Recall@5 baseline performance
that existed BEFORE adding HyDE/reranker optimizations.

This script uses ONLY:
- Simple hybrid search (dense + BM25 sparse)
- LM Studio embeddings (nomic-embed-text-v1.5, 768 dims)
- No HyDE generation
- No reranker
- RRF fusion for combining dense + sparse results

Expected: ~62% Recall@5 (original baseline before optimization attempts)
Current: 29.87% Recall@5 (with HyDE/reranker - REGRESSION)

Target: Confirm we can restore 62%+ by removing problematic optimizations.
"""

import json
import math
import os
import re
import sys
import time
import hashlib
import logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx

# ============================================================================
# CONFIGURATION
# ============================================================================

QDRANT_URL = "http://localhost:6333"
EMBEDDING_URL = "http://localhost:1234"
COLLECTION_NAME = "ace_memories_hybrid"

# Retrieval Configuration - Simple baseline
FIRST_STAGE_K = 100  # Cast wide net
FINAL_K = 10  # Return top 10

# Parallel processing
MAX_PARALLEL_EVALS = 5

# BM25 parameters
BM25_K1 = 1.5
BM25_B = 0.75
AVG_DOC_LENGTH = 50
RRF_K = 60

STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
    'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have',
    'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
    'might', 'must', 'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you',
    'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'some', 'no',
    'not', 'only', 'so', 'than', 'too', 'very', 'just', 'also', 'now'
}

# Test suite paths
TEST_SUITE_PATH = Path(__file__).parent.parent / "test_suite" / "enhanced_test_suite.json"
OUTPUT_PATH = Path(__file__).parent.parent / "optimization_results" / "v8_baseline_revert.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# METRICS
# ============================================================================

def calculate_recall_at_k(retrieved_ids: List[str], ground_truth_id: str, k: int) -> float:
    return 1.0 if ground_truth_id in retrieved_ids[:k] else 0.0


def calculate_mrr(retrieved_ids: List[str], ground_truth_id: str) -> float:
    try:
        rank = retrieved_ids.index(ground_truth_id) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def calculate_ndcg_at_k(retrieved_ids: List[str], ground_truth_id: str, k: int) -> float:
    if ground_truth_id not in retrieved_ids[:k]:
        return 0.0
    rank = retrieved_ids[:k].index(ground_truth_id) + 1
    dcg = 1.0 / math.log2(rank + 1)
    idcg = 1.0 / math.log2(2)
    return dcg / idcg


# ============================================================================
# BM25 SPARSE VECTORS
# ============================================================================

def tokenize_bm25(text: str) -> List[str]:
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = text.replace('_', ' ')
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def compute_bm25_sparse(text: str) -> Dict[str, Any]:
    tokens = tokenize_bm25(text)
    if not tokens:
        return {"indices": [], "values": []}

    tf = Counter(tokens)
    doc_length = len(tokens)
    indices, values = [], []

    for term, freq in tf.items():
        term_hash = int(hashlib.md5(term.encode()).hexdigest()[:8], 16)
        indices.append(term_hash)
        tf_weight = (freq * (BM25_K1 + 1)) / (
            freq + BM25_K1 * (1 - BM25_B + BM25_B * doc_length / AVG_DOC_LENGTH)
        )
        values.append(float(tf_weight))

    return {"indices": indices, "values": values}


# ============================================================================
# BASELINE EVALUATOR (NO HYDE, NO RERANKER)
# ============================================================================

class BaselineEvaluator:
    """Simple hybrid search evaluator - dense + BM25 sparse only."""

    def __init__(self):
        self.http_client = httpx.Client(timeout=60.0)
        self.embedding_dim = 768
        logger.info("Baseline Evaluator initialized:")
        logger.info("  HyDE: FALSE")
        logger.info("  Reranker: FALSE")
        logger.info("  Embeddings: LM Studio nomic-embed-text-v1.5 (768 dims)")

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding from LM Studio (nomic-embed-text-v1.5)."""
        try:
            resp = self.http_client.post(
                f"{EMBEDDING_URL}/v1/embeddings",
                json={"model": "text-embedding-qwen3-embedding-8b", "input": text[:8000]}
            )
            if resp.status_code == 200:
                return resp.json()["data"][0]["embedding"]
            else:
                logger.error(f"Embedding request failed: {resp.status_code}")
        except Exception as e:
            logger.error(f"Embedding failed: {e}")

        return [0.0] * self.embedding_dim

    def search_qdrant(
        self,
        query_embedding: List[float],
        query_text: str,
        limit: int
    ) -> List[Dict]:
        """Execute hybrid search in Qdrant."""
        sparse_vector = compute_bm25_sparse(query_text)

        # Build hybrid query
        hybrid_query = {
            "prefetch": [
                {"query": query_embedding, "using": "dense", "limit": limit * 2}
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
            resp = self.http_client.post(
                f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/query",
                json=hybrid_query
            )
            if resp.status_code == 200:
                result = resp.json().get("result", [])
                if isinstance(result, dict):
                    return result.get("points", [])
                return result
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")

        return []

    def retrieve(self, query: str, limit: int = FINAL_K) -> Tuple[List[Dict], Dict[str, float]]:
        """Simple retrieval pipeline with latency tracking."""
        latencies = {}

        # Step 1: Generate embedding
        start = time.perf_counter()
        query_embedding = self.get_embedding(query)
        latencies["embedding_ms"] = (time.perf_counter() - start) * 1000

        # Step 2: Search Qdrant (hybrid with RRF)
        start = time.perf_counter()
        results = self.search_qdrant(query_embedding, query, limit)
        latencies["search_ms"] = (time.perf_counter() - start) * 1000

        latencies["total_ms"] = sum(latencies.values())

        return results, latencies

    def evaluate_single_query(
        self,
        query: str,
        ground_truth_id: str
    ) -> Dict[str, Any]:
        """Evaluate a single query."""
        try:
            results, latencies = self.retrieve(query)

            # Extract IDs - use Point ID as primary
            retrieved_ids = []
            for r in results:
                point_id = str(r.get("id", ""))
                payload = r.get("payload", {})
                bullet_id = str(payload.get("id", payload.get("bullet_id", point_id)))
                retrieved_ids.append(bullet_id)

            ground_truth_str = str(ground_truth_id)

            return {
                "query": query,
                "ground_truth_id": ground_truth_id,
                "retrieved_ids": retrieved_ids,
                "metrics": {
                    "recall@1": calculate_recall_at_k(retrieved_ids, ground_truth_str, 1),
                    "recall@3": calculate_recall_at_k(retrieved_ids, ground_truth_str, 3),
                    "recall@5": calculate_recall_at_k(retrieved_ids, ground_truth_str, 5),
                    "recall@10": calculate_recall_at_k(retrieved_ids, ground_truth_str, 10),
                    "mrr": calculate_mrr(retrieved_ids, ground_truth_str),
                    "ndcg@10": calculate_ndcg_at_k(retrieved_ids, ground_truth_str, 10),
                    **latencies
                },
                "success": True
            }
        except Exception as e:
            logger.error(f"Query failed: {query[:50]}... - {e}")
            return {
                "query": query,
                "ground_truth_id": ground_truth_id,
                "retrieved_ids": [],
                "metrics": {
                    "recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0, "recall@10": 0.0,
                    "mrr": 0.0, "ndcg@10": 0.0,
                    "embedding_ms": 0, "search_ms": 0, "total_ms": 0
                },
                "success": False,
                "error": str(e)
            }

    def run_evaluation(self, test_suite_path: Path) -> Dict[str, Any]:
        """Run full evaluation on test suite."""
        with open(test_suite_path, 'r', encoding='utf-8') as f:
            test_suite = json.load(f)

        logger.info(f"Loaded test suite: {len(test_suite['test_cases'])} memories")

        # Flatten queries
        test_queries = []
        for test_case in test_suite['test_cases']:
            memory_id = test_case['memory_id']
            for query_obj in test_case.get('generated_queries', []):
                test_queries.append({
                    'query': query_obj['query'],
                    'memory_id': memory_id,
                    'category': query_obj.get('category', 'general'),
                    'difficulty': query_obj.get('difficulty', 'medium')
                })

        logger.info(f"Total queries to evaluate: {len(test_queries)}")
        logger.info(f"Using {MAX_PARALLEL_EVALS} parallel workers")

        # Parallel evaluation
        def eval_single(args):
            idx, test_case = args
            if idx % 50 == 0:
                logger.info(f"[{idx}/{len(test_queries)}] Evaluating...")

            result = self.evaluate_single_query(
                query=test_case['query'],
                ground_truth_id=test_case['memory_id']
            )
            result['category'] = test_case['category']
            result['difficulty'] = test_case['difficulty']
            return result

        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_EVALS) as executor:
            results = list(executor.map(eval_single, enumerate(test_queries, 1)))

        # Aggregate metrics
        successful = [r for r in results if r['success']]

        if not successful:
            return {"error": "No successful queries"}

        # Overall metrics
        aggregate = {
            "recall@1": sum(r['metrics']['recall@1'] for r in successful) / len(successful),
            "recall@3": sum(r['metrics']['recall@3'] for r in successful) / len(successful),
            "recall@5": sum(r['metrics']['recall@5'] for r in successful) / len(successful),
            "recall@10": sum(r['metrics']['recall@10'] for r in successful) / len(successful),
            "mrr": sum(r['metrics']['mrr'] for r in successful) / len(successful),
            "ndcg@10": sum(r['metrics']['ndcg@10'] for r in successful) / len(successful),
            "avg_total_ms": sum(r['metrics']['total_ms'] for r in successful) / len(successful),
            "avg_embedding_ms": sum(r['metrics']['embedding_ms'] for r in successful) / len(successful),
            "avg_search_ms": sum(r['metrics']['search_ms'] for r in successful) / len(successful),
        }

        # By category
        by_category = defaultdict(list)
        for r in successful:
            by_category[r['category']].append(r)

        category_metrics = {}
        for cat, cat_results in by_category.items():
            category_metrics[cat] = {
                "count": len(cat_results),
                "recall@1": sum(r['metrics']['recall@1'] for r in cat_results) / len(cat_results),
                "recall@5": sum(r['metrics']['recall@5'] for r in cat_results) / len(cat_results),
                "mrr": sum(r['metrics']['mrr'] for r in cat_results) / len(cat_results),
            }

        # By difficulty
        by_difficulty = defaultdict(list)
        for r in successful:
            by_difficulty[r['difficulty']].append(r)

        difficulty_metrics = {}
        for diff, diff_results in by_difficulty.items():
            difficulty_metrics[diff] = {
                "count": len(diff_results),
                "recall@1": sum(r['metrics']['recall@1'] for r in diff_results) / len(diff_results),
                "recall@5": sum(r['metrics']['recall@5'] for r in diff_results) / len(diff_results),
                "mrr": sum(r['metrics']['mrr'] for r in diff_results) / len(diff_results),
            }

        return {
            "technique": "Baseline Revert (Simple Hybrid Search)",
            "version": "v8",
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "hyde_enabled": False,
                "reranker_enabled": False,
                "first_stage_k": FIRST_STAGE_K,
                "final_k": FINAL_K,
                "embedding_dim": self.embedding_dim,
                "embedding_model": "nomic-embed-text-v1.5",
                "search_method": "hybrid (dense + BM25 sparse with RRF fusion)"
            },
            "total_queries": len(test_queries),
            "successful_queries": len(successful),
            "aggregate_metrics": aggregate,
            "category_metrics": category_metrics,
            "difficulty_metrics": difficulty_metrics,
            "detailed_results": results[:100],  # First 100 for debugging
        }

    def close(self):
        self.http_client.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 80)
    logger.info("V8: BASELINE REVERT")
    logger.info("Target: Restore 62%+ Recall@5 (original baseline)")
    logger.info("Method: Simple hybrid search (NO HyDE, NO reranker)")
    logger.info("=" * 80)

    # Initialize baseline evaluator
    evaluator = BaselineEvaluator()

    try:
        # Run evaluation
        results = evaluator.run_evaluation(TEST_SUITE_PATH)

        # Save results
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"\nResults saved to: {OUTPUT_PATH}")

        # Print summary
        print("\n" + "=" * 80)
        print("BASELINE REVERT EVALUATION COMPLETE")
        print("=" * 80)

        agg = results['aggregate_metrics']
        print(f"\nOVERALL METRICS:")
        print(f"  Recall@1:  {agg['recall@1']:.2%}")
        print(f"  Recall@3:  {agg['recall@3']:.2%}")
        print(f"  Recall@5:  {agg['recall@5']:.2%}  {'PASS' if agg['recall@5'] >= 0.62 else 'NEEDS INVESTIGATION'}")
        print(f"  Recall@10: {agg['recall@10']:.2%}")
        print(f"  MRR:       {agg['mrr']:.4f}")
        print(f"  NDCG@10:   {agg['ndcg@10']:.4f}")

        print(f"\nLATENCY:")
        print(f"  Total:     {agg['avg_total_ms']:.1f}ms")
        print(f"  Embedding: {agg['avg_embedding_ms']:.1f}ms")
        print(f"  Search:    {agg['avg_search_ms']:.1f}ms")

        print(f"\nBY CATEGORY:")
        for cat, metrics in results['category_metrics'].items():
            print(f"  {cat}: R@1={metrics['recall@1']:.2%}, R@5={metrics['recall@5']:.2%}, MRR={metrics['mrr']:.3f} (n={metrics['count']})")

        print(f"\nBY DIFFICULTY:")
        for diff, metrics in results['difficulty_metrics'].items():
            print(f"  {diff}: R@1={metrics['recall@1']:.2%}, R@5={metrics['recall@5']:.2%}, MRR={metrics['mrr']:.3f} (n={metrics['count']})")

        # Target check
        print("\n" + "=" * 80)
        if agg['recall@5'] >= 0.62:
            print(f"BASELINE RESTORED: {agg['recall@5']:.2%} Recall@5")
            print("This confirms the original baseline performance can be restored.")
        else:
            gap = 0.62 - agg['recall@5']
            print(f"BASELINE NOT RESTORED: {agg['recall@5']:.2%} Recall@5 (need +{gap:.2%})")
            print("This suggests issues beyond HyDE/reranker. Investigate:")
            print("  1. Qdrant collection integrity")
            print("  2. Embedding model consistency")
            print("  3. Test suite changes")
        print("=" * 80)

    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
