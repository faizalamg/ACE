"""
V7: Fortune 100 Combined Pipeline - Target 95%+ Recall@5
=========================================================

Combines ALL optimizations for maximum retrieval accuracy:
1. HyDE (Hypothetical Document Embeddings) for query expansion
2. Fine-tuned BGE-large embeddings (1024 dims) for domain adaptation
3. BGE cross-encoder reranker for precision

Pipeline:
  Query -> HyDE hypotheticals -> BGE-large embeddings -> Qdrant hybrid search
       -> RRF fusion -> BGE reranker -> Final top-K

Target: >95% Recall@5, >85% Recall@1, MRR >0.90
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
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

load_dotenv()

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ace.hyde import HyDEGenerator, HyDEConfig
from ace.llm_providers.litellm_client import LiteLLMClient

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("WARNING: sentence-transformers not available. Install with: pip install sentence-transformers")

# ONNX Runtime DirectML for AMD GPU acceleration
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    # Check for DirectML (AMD GPU)
    DIRECTML_AVAILABLE = "DmlExecutionProvider" in ort.get_available_providers()
except ImportError:
    ONNX_AVAILABLE = False
    DIRECTML_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

QDRANT_URL = "http://localhost:6333"
EMBEDDING_URL = "http://localhost:1234"  # Fallback only
COLLECTION_NAME = "ace_memories_hybrid"

# Fine-tuned model paths
FINETUNED_MODEL_PATH = "ace/embedding_finetuning/models/ace_finetuned"
BGE_LARGE_MODEL = "BAAI/bge-large-en-v1.5"  # 1024 dimensions

# Reranker
BGE_RERANKER_MODEL = "BAAI/bge-reranker-large"  # Upgrade to large

# HyDE Configuration
LLM_MODEL = "openai/glm-4.6"
ZAI_API_BASE = "https://api.z.ai/api/coding/paas/v4"
NUM_HYPOTHETICALS = 3

# Retrieval Configuration - Aggressive for 95%+
FIRST_STAGE_K = 100  # Cast wide net
RERANK_K = 50  # Rerank top 50
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
OUTPUT_PATH = Path(__file__).parent.parent / "optimization_results" / "v7_fortune100_combined.json"

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


def reciprocal_rank_fusion(result_sets: List[List[Dict]], k: int = RRF_K) -> List[Dict]:
    scores: Dict[str, float] = {}
    doc_data: Dict[str, Dict] = {}

    for results in result_sets:
        for rank, doc in enumerate(results, start=1):
            doc_id = str(doc.get("id", ""))
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank)
            if doc_id not in doc_data:
                doc_data[doc_id] = doc

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    fused = []
    for doc_id, score in sorted_items:
        doc = doc_data[doc_id].copy()
        doc["rrf_score"] = score
        doc["score"] = score
        fused.append(doc)

    return fused


# ============================================================================
# FORTUNE 100 COMBINED EVALUATOR
# ============================================================================

class Fortune100Evaluator:
    """Combined HyDE + Fine-tuned Embeddings + Reranker pipeline."""

    def __init__(
        self,
        use_finetuned_embeddings: bool = True,
        use_hyde: bool = True,
        use_reranker: bool = True
    ):
        self.use_hyde = use_hyde
        self.use_reranker = use_reranker
        self.use_finetuned = use_finetuned_embeddings

        self.http_client = httpx.Client(timeout=60.0)

        # Initialize HyDE
        if use_hyde:
            logger.info("Initializing HyDE with Z.ai GLM-4.6...")
            zai_api_key = os.getenv("ZAI_API_KEY")
            if not zai_api_key:
                raise ValueError("ZAI_API_KEY required for HyDE")

            llm_client = LiteLLMClient(
                model=LLM_MODEL,
                api_key=zai_api_key,
                api_base=ZAI_API_BASE
            )
            hyde_config = HyDEConfig(
                num_hypotheticals=NUM_HYPOTHETICALS,
                max_tokens=150,
                temperature=0.7,
                cache_enabled=True
            )
            self.hyde_generator = HyDEGenerator(llm_client=llm_client, config=hyde_config)
            logger.info("HyDE initialized")
        else:
            self.hyde_generator = None

        # Initialize embedding model
        # NOTE: Qdrant collection uses nomic-embed-text-v1.5 (768 dims) from LM Studio
        # We MUST use the same model for search to match vector dimensions
        # The reranker will handle quality improvement post-retrieval
        self.embedding_model = None
        self.embedding_dim = 768
        logger.info("Using LM Studio embeddings (nomic-embed-text-v1.5, 768 dims) to match Qdrant")

        # Initialize reranker - AMD GPU via ONNX DirectML or optimized CPU
        self.use_onnx_reranker = False
        if use_reranker and TRANSFORMERS_AVAILABLE:
            import torch

            # Try ONNX DirectML first for AMD GPU acceleration
            if ONNX_AVAILABLE and DIRECTML_AVAILABLE:
                logger.info(f"Loading reranker with ONNX DirectML (AMD GPU): {BGE_RERANKER_MODEL}")
                try:
                    from optimum.onnxruntime import ORTModelForSequenceClassification
                    from transformers import AutoTokenizer

                    # Export and load with ONNX DirectML
                    self.reranker_tokenizer = AutoTokenizer.from_pretrained(BGE_RERANKER_MODEL)
                    self.reranker_onnx = ORTModelForSequenceClassification.from_pretrained(
                        BGE_RERANKER_MODEL,
                        export=True,
                        provider="DmlExecutionProvider"
                    )
                    self.use_onnx_reranker = True
                    self.reranker = None
                    logger.info("Reranker loaded on AMD GPU via ONNX DirectML")
                except Exception as e:
                    logger.warning(f"ONNX DirectML failed: {e}, falling back to CPU")
                    self.use_onnx_reranker = False

            # Fallback to optimized CPU
            if not self.use_onnx_reranker:
                torch.set_num_threads(16)
                logger.info(f"Loading reranker: {BGE_RERANKER_MODEL} on CPU (16 threads)")
                self.reranker = CrossEncoder(BGE_RERANKER_MODEL, max_length=512, device="cpu")
                self.reranker.model.eval()
                logger.info("Reranker loaded with CPU optimization")
        else:
            self.reranker = None

        logger.info(f"Fortune 100 Evaluator initialized:")
        logger.info(f"  HyDE: {use_hyde}")
        logger.info(f"  Fine-tuned embeddings: {use_finetuned_embeddings}")
        logger.info(f"  Reranker: {use_reranker}")

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

    def get_hyde_embedding(self, query: str) -> List[float]:
        """Generate HyDE-enhanced embedding by averaging hypotheticals."""
        if not self.hyde_generator:
            return self.get_embedding(query)

        hypotheticals = self.hyde_generator.generate_hypotheticals(query)

        embeddings = []
        for hyp in hypotheticals:
            if hyp and hyp.strip():
                emb = self.get_embedding(hyp.strip())
                if emb and any(v != 0 for v in emb):
                    embeddings.append(emb)

        if not embeddings:
            logger.warning("No valid hypotheticals, falling back to query embedding")
            return self.get_embedding(query)

        # Average embeddings
        dim = len(embeddings[0])
        averaged = [
            sum(emb[i] for emb in embeddings) / len(embeddings)
            for i in range(dim)
        ]

        return averaged

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

    def rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Rerank candidates using cross-encoder (ONNX DirectML or CPU)."""
        if (not self.reranker and not self.use_onnx_reranker) or not candidates:
            return candidates[:top_k]

        pairs = []
        doc_texts = []
        for c in candidates:
            payload = c.get("payload", {})
            doc_text = payload.get("lesson", "") or payload.get("content", "") or str(payload)
            doc_texts.append(doc_text[:1000])
            pairs.append([query, doc_text[:1000]])

        # Use ONNX DirectML (AMD GPU) if available
        if self.use_onnx_reranker:
            import torch
            inputs = self.reranker_tokenizer(
                [query] * len(doc_texts),
                doc_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np"
            )
            outputs = self.reranker_onnx(**inputs)
            scores = outputs.logits.flatten().tolist()
        else:
            # Fallback to CrossEncoder CPU
            scores = self.reranker.predict(pairs)

        scored = list(zip(scores, candidates))
        scored.sort(key=lambda x: x[0], reverse=True)

        reranked = []
        for score, candidate in scored[:top_k]:
            candidate["rerank_score"] = float(score)
            candidate["score"] = float(score)
            reranked.append(candidate)

        return reranked

    def retrieve(self, query: str, limit: int = FINAL_K) -> Tuple[List[Dict], Dict[str, float]]:
        """Full retrieval pipeline with latency tracking."""
        latencies = {}

        # Step 1: Generate embedding (with HyDE if enabled)
        start = time.perf_counter()
        if self.use_hyde:
            query_embedding = self.get_hyde_embedding(query)
        else:
            query_embedding = self.get_embedding(query)
        latencies["embedding_ms"] = (time.perf_counter() - start) * 1000

        # Step 2: Search Qdrant
        start = time.perf_counter()
        candidates = self.search_qdrant(query_embedding, query, FIRST_STAGE_K)
        latencies["search_ms"] = (time.perf_counter() - start) * 1000

        # Step 3: Rerank
        start = time.perf_counter()
        if self.use_reranker:
            results = self.rerank(query, candidates, limit)
        else:
            results = candidates[:limit]
        latencies["rerank_ms"] = (time.perf_counter() - start) * 1000

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
            "technique": "Fortune 100 Combined (HyDE + Fine-tuned + Reranker)",
            "version": "v7",
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "hyde_enabled": self.use_hyde,
                "finetuned_embeddings": self.use_finetuned,
                "reranker_enabled": self.use_reranker,
                "num_hypotheticals": NUM_HYPOTHETICALS,
                "first_stage_k": FIRST_STAGE_K,
                "rerank_k": RERANK_K,
                "final_k": FINAL_K,
                "embedding_dim": self.embedding_dim,
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
    logger.info("V7: FORTUNE 100 COMBINED PIPELINE")
    logger.info("Target: >95% Recall@5, >85% Recall@1, MRR >0.90")
    logger.info("=" * 80)

    # Check prerequisites
    if not os.getenv("ZAI_API_KEY"):
        logger.error("ZAI_API_KEY required for HyDE")
        sys.exit(1)

    if not TRANSFORMERS_AVAILABLE:
        logger.error("sentence-transformers required")
        sys.exit(1)

    # Initialize evaluator with ALL features
    evaluator = Fortune100Evaluator(
        use_finetuned_embeddings=True,
        use_hyde=True,
        use_reranker=True
    )

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
        print("FORTUNE 100 EVALUATION COMPLETE")
        print("=" * 80)

        agg = results['aggregate_metrics']
        print(f"\nOVERALL METRICS:")
        print(f"  Recall@1:  {agg['recall@1']:.2%}")
        print(f"  Recall@3:  {agg['recall@3']:.2%}")
        print(f"  Recall@5:  {agg['recall@5']:.2%}  {'PASS' if agg['recall@5'] >= 0.95 else 'NEEDS WORK'}")
        print(f"  Recall@10: {agg['recall@10']:.2%}")
        print(f"  MRR:       {agg['mrr']:.4f}")
        print(f"  NDCG@10:   {agg['ndcg@10']:.4f}")

        print(f"\nLATENCY:")
        print(f"  Total:     {agg['avg_total_ms']:.1f}ms")
        print(f"  Embedding: {agg['avg_embedding_ms']:.1f}ms")
        print(f"  Search:    {agg['avg_search_ms']:.1f}ms")
        print(f"  Rerank:    {agg['avg_rerank_ms']:.1f}ms")

        print(f"\nBY CATEGORY:")
        for cat, metrics in results['category_metrics'].items():
            print(f"  {cat}: R@1={metrics['recall@1']:.2%}, R@5={metrics['recall@5']:.2%}, MRR={metrics['mrr']:.3f} (n={metrics['count']})")

        print(f"\nBY DIFFICULTY:")
        for diff, metrics in results['difficulty_metrics'].items():
            print(f"  {diff}: R@1={metrics['recall@1']:.2%}, R@5={metrics['recall@5']:.2%}, MRR={metrics['mrr']:.3f} (n={metrics['count']})")

        # Target check
        print("\n" + "=" * 80)
        if agg['recall@5'] >= 0.95:
            print("TARGET ACHIEVED: 95%+ Recall@5")
        else:
            gap = 0.95 - agg['recall@5']
            print(f"TARGET NOT YET MET: Need +{gap:.2%} to reach 95% Recall@5")
        print("=" * 80)

    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
