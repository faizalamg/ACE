"""
V8: BM25 Hybrid Scoring with Tunable Weights
=============================================

Tests hybrid scoring combining dense semantic vectors with BM25 sparse vectors
to balance semantic understanding with exact phrase matching.

Formula: score = (weight * dense_score) + ((1-weight) * bm25_score)

Key Hypothesis:
- Dense vectors capture semantic meaning and context
- BM25 sparse vectors capture exact phrase matches and technical terms
- Optimal weight is expected in 0.4-0.6 range (expert consensus)

Pipeline:
  Query -> Dense embedding (nomic-embed-text-v1.5)
       -> Sparse BM25 vector
       -> Qdrant hybrid search with RRF fusion
       -> Test weights: [0.3, 0.4, 0.5, 0.6, 0.7]
       -> Find optimal weight for Recall@5

Target: Identify best weight for >90% Recall@5
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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
from dotenv import load_dotenv

load_dotenv()

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ============================================================================
# CONFIGURATION
# ============================================================================

QDRANT_URL = "http://localhost:6333"
EMBEDDING_URL = "http://localhost:1234"  # LM Studio
COLLECTION_NAME = "ace_memories_hybrid"

# Test weights for hybrid scoring
TEST_WEIGHTS = [0.3, 0.4, 0.5, 0.6, 0.7]

# Retrieval Configuration
RETRIEVAL_LIMIT = 100  # Retrieve more candidates for RRF fusion
FINAL_K = 10

# Parallel processing
MAX_PARALLEL_EVALS = 5

# BM25 parameters (from unified_memory.py)
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
OUTPUT_PATH = Path(__file__).parent.parent / "optimization_results" / "v8_bm25_hybrid.json"
WORK_LOG_PATH = Path(__file__).parent.parent / "optimization_results" / "work_log.md"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# METRICS
# ============================================================================

def calculate_recall_at_k(retrieved_ids: List[str], ground_truth_id: str, k: int) -> float:
    """Calculate Recall@K metric."""
    return 1.0 if ground_truth_id in retrieved_ids[:k] else 0.0


def calculate_mrr(retrieved_ids: List[str], ground_truth_id: str) -> float:
    """Calculate Mean Reciprocal Rank."""
    try:
        rank = retrieved_ids.index(ground_truth_id) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def calculate_ndcg_at_k(retrieved_ids: List[str], ground_truth_id: str, k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain@K."""
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
    """Tokenize text for BM25, preserving technical terms."""
    # Split CamelCase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Split snake_case
    text = text.replace('_', ' ')
    # Extract alphanumeric tokens
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    # Filter stopwords and short tokens
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def compute_bm25_sparse(text: str) -> Dict[str, Any]:
    """Compute BM25 sparse vector for Qdrant."""
    tokens = tokenize_bm25(text)
    if not tokens:
        return {"indices": [], "values": []}

    tf = Counter(tokens)
    doc_length = len(tokens)
    indices, values = [], []

    for term, freq in tf.items():
        # Consistent hash for term -> index
        term_hash = int(hashlib.md5(term.encode()).hexdigest()[:8], 16)
        indices.append(term_hash)

        # BM25 term weight
        tf_weight = (freq * (BM25_K1 + 1)) / (
            freq + BM25_K1 * (1 - BM25_B + BM25_B * doc_length / AVG_DOC_LENGTH)
        )
        values.append(float(tf_weight))

    return {"indices": indices, "values": values}


# ============================================================================
# HYBRID SCORING EVALUATOR
# ============================================================================

class HybridScoringEvaluator:
    """Evaluator for testing BM25 hybrid scoring with tunable weights."""

    def __init__(self, dense_weight: float = 0.5):
        """
        Initialize evaluator with specified dense/sparse weight.

        Args:
            dense_weight: Weight for dense semantic score (0.0-1.0)
                         sparse_weight = 1.0 - dense_weight
        """
        self.dense_weight = dense_weight
        self.sparse_weight = 1.0 - dense_weight
        self.http_client = httpx.Client(timeout=60.0)

        logger.info(f"Initialized with weights: dense={dense_weight:.2f}, sparse={self.sparse_weight:.2f}")

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

        # Return zero vector on failure
        return [0.0] * 768

    def search_qdrant_hybrid(
        self,
        query_embedding: List[float],
        query_text: str,
        limit: int
    ) -> List[Dict]:
        """
        Execute hybrid search in Qdrant using RRF fusion.

        Qdrant's built-in RRF fusion combines dense and sparse results,
        then we re-score with custom weights.
        """
        sparse_vector = compute_bm25_sparse(query_text)

        # Build hybrid query with RRF fusion
        hybrid_query = {
            "prefetch": [
                {"query": query_embedding, "using": "dense", "limit": limit * 2}
            ],
            "query": {"fusion": "rrf"},
            "limit": limit,
            "with_payload": True,
            "with_vector": False  # Don't need vectors back
        }

        # Add sparse query if we have tokens
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

    def search_qdrant_dense_only(
        self,
        query_embedding: List[float],
        limit: int
    ) -> List[Dict]:
        """Execute dense-only search for score extraction."""
        try:
            resp = self.http_client.post(
                f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/query",
                json={
                    "query": query_embedding,
                    "using": "dense",
                    "limit": limit,
                    "with_payload": True,
                    "with_vector": False
                }
            )
            if resp.status_code == 200:
                result = resp.json().get("result", [])
                if isinstance(result, dict):
                    return result.get("points", [])
                return result
        except Exception as e:
            logger.error(f"Dense search error: {e}")

        return []

    def search_qdrant_sparse_only(
        self,
        query_text: str,
        limit: int
    ) -> List[Dict]:
        """Execute sparse-only BM25 search for score extraction."""
        sparse_vector = compute_bm25_sparse(query_text)

        if not sparse_vector.get("indices"):
            return []

        try:
            resp = self.http_client.post(
                f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/query",
                json={
                    "query": {
                        "indices": sparse_vector["indices"],
                        "values": sparse_vector["values"]
                    },
                    "using": "sparse",
                    "limit": limit,
                    "with_payload": True,
                    "with_vector": False
                }
            )
            if resp.status_code == 200:
                result = resp.json().get("result", [])
                if isinstance(result, dict):
                    return result.get("points", [])
                return result
        except Exception as e:
            logger.error(f"Sparse search error: {e}")

        return []

    def apply_custom_weights(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict]
    ) -> List[Dict]:
        """
        Apply custom weights to dense and sparse scores.

        Qdrant normalizes scores to [0, 1] range. We combine them with custom weights.
        """
        # Build score maps
        dense_scores = {str(r.get("id", "")): r.get("score", 0.0) for r in dense_results}
        sparse_scores = {str(r.get("id", "")): r.get("score", 0.0) for r in sparse_results}

        # Get all unique IDs
        all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())

        # Compute weighted scores
        weighted_results = []
        for doc_id in all_ids:
            dense_score = dense_scores.get(doc_id, 0.0)
            sparse_score = sparse_scores.get(doc_id, 0.0)

            # Custom weighted combination
            final_score = (self.dense_weight * dense_score) + (self.sparse_weight * sparse_score)

            # Get document payload (prefer dense, fallback to sparse)
            doc = None
            for r in dense_results:
                if str(r.get("id", "")) == doc_id:
                    doc = r
                    break
            if not doc:
                for r in sparse_results:
                    if str(r.get("id", "")) == doc_id:
                        doc = r
                        break

            if doc:
                result = doc.copy()
                result["score"] = final_score
                result["dense_score"] = dense_score
                result["sparse_score"] = sparse_score
                weighted_results.append(result)

        # Sort by final weighted score
        weighted_results.sort(key=lambda x: x["score"], reverse=True)

        return weighted_results

    def retrieve(self, query: str, limit: int = FINAL_K) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Full retrieval pipeline with custom hybrid scoring.

        Returns:
            Tuple of (results, latencies)
        """
        latencies = {}

        # Step 1: Generate embedding
        start = time.perf_counter()
        query_embedding = self.get_embedding(query)
        latencies["embedding_ms"] = (time.perf_counter() - start) * 1000

        # Step 2: Execute dense and sparse searches separately
        start = time.perf_counter()
        dense_results = self.search_qdrant_dense_only(query_embedding, RETRIEVAL_LIMIT)
        sparse_results = self.search_qdrant_sparse_only(query, RETRIEVAL_LIMIT)
        latencies["search_ms"] = (time.perf_counter() - start) * 1000

        # Step 3: Apply custom weights
        start = time.perf_counter()
        results = self.apply_custom_weights(dense_results, sparse_results)
        latencies["rerank_ms"] = (time.perf_counter() - start) * 1000

        latencies["total_ms"] = sum(latencies.values())

        return results[:limit], latencies

    def evaluate_single_query(
        self,
        query: str,
        ground_truth_id: str
    ) -> Dict[str, Any]:
        """Evaluate a single query."""
        try:
            results, latencies = self.retrieve(query)

            # Extract IDs
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
                    "embedding_ms": 0, "search_ms": 0, "rerank_ms": 0, "total_ms": 0
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
            "avg_rerank_ms": sum(r['metrics']['rerank_ms'] for r in successful) / len(successful),
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
            "technique": f"BM25 Hybrid (dense_weight={self.dense_weight:.2f})",
            "version": "v8",
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(test_queries),
            "successful_queries": len(successful),
            "aggregate_metrics": aggregate,
            "category_metrics": category_metrics,
            "difficulty_metrics": difficulty_metrics,
            "detailed_results": results[:50],  # First 50 for debugging
        }

    def close(self):
        """Close HTTP client."""
        self.http_client.close()


# ============================================================================
# GRID SEARCH
# ============================================================================

def run_grid_search() -> Dict[str, Any]:
    """Run grid search over weight values."""
    logger.info("=" * 80)
    logger.info("V8: BM25 HYBRID SCORING - GRID SEARCH")
    logger.info(f"Testing weights: {TEST_WEIGHTS}")
    logger.info("=" * 80)

    all_results = []
    best_weight = None
    best_recall5 = 0.0

    for weight in TEST_WEIGHTS:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing dense_weight={weight:.2f}, sparse_weight={1-weight:.2f}")
        logger.info(f"{'='*80}")

        evaluator = HybridScoringEvaluator(dense_weight=weight)

        try:
            results = evaluator.run_evaluation(TEST_SUITE_PATH)
            all_results.append(results)

            # Check if this is the best weight
            recall5 = results['aggregate_metrics']['recall@5']
            if recall5 > best_recall5:
                best_recall5 = recall5
                best_weight = weight

            # Print summary
            agg = results['aggregate_metrics']
            logger.info(f"\nResults for weight={weight:.2f}:")
            logger.info(f"  Recall@1:  {agg['recall@1']:.2%}")
            logger.info(f"  Recall@5:  {agg['recall@5']:.2%}")
            logger.info(f"  MRR:       {agg['mrr']:.4f}")
            logger.info(f"  Latency:   {agg['avg_total_ms']:.1f}ms")

        finally:
            evaluator.close()

    # Compile final results
    final_results = {
        "technique": "BM25 Hybrid Scoring Grid Search",
        "version": "v8",
        "timestamp": datetime.now().isoformat(),
        "weights_tested": TEST_WEIGHTS,
        "best_weight": best_weight,
        "best_recall@5": best_recall5,
        "all_experiments": all_results
    }

    return final_results


# ============================================================================
# MAIN
# ============================================================================

def log_to_work_log(message: str):
    """Append status message to work log."""
    try:
        WORK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(WORK_LOG_PATH, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] v8_bm25_hybrid: {message}\n")
    except Exception as e:
        logger.error(f"Failed to write to work log: {e}")


def main():
    log_to_work_log("Starting BM25 hybrid scoring grid search")

    try:
        # Run grid search
        results = run_grid_search()

        # Save results
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"\nResults saved to: {OUTPUT_PATH}")

        # Print final summary
        print("\n" + "=" * 80)
        print("BM25 HYBRID SCORING - GRID SEARCH COMPLETE")
        print("=" * 80)

        print(f"\nWeights tested: {TEST_WEIGHTS}")
        print(f"Best weight: {results['best_weight']:.2f} (dense) / {1-results['best_weight']:.2f} (sparse)")
        print(f"Best Recall@5: {results['best_recall@5']:.2%}")

        print("\nAll Results:")
        print(f"{'Weight':<10} {'Recall@1':<12} {'Recall@5':<12} {'MRR':<10} {'Latency (ms)':<15}")
        print("-" * 60)
        for exp in results['all_experiments']:
            weight = exp['dense_weight']
            agg = exp['aggregate_metrics']
            print(f"{weight:<10.2f} {agg['recall@1']:<12.2%} {agg['recall@5']:<12.2%} "
                  f"{agg['mrr']:<10.4f} {agg['avg_total_ms']:<15.1f}")

        print("\n" + "=" * 80)

        log_message = (f"Grid search complete. Best weight: {results['best_weight']:.2f}, "
                      f"Recall@5: {results['best_recall@5']:.2%}")
        log_to_work_log(log_message)

    except Exception as e:
        logger.error(f"Grid search failed: {e}")
        log_to_work_log(f"ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
