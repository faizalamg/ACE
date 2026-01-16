"""
V6: HyDE (Hypothetical Document Embeddings) Optimization
=========================================================

**Technique**: HyDE - Generate hypothetical documents that would answer the query,
then use averaged embeddings for semantic search.

**Pipeline**:
1. Query -> LLM generates 3-5 hypothetical answer documents
2. Embed each hypothetical document
3. Average embeddings into single vector
4. Search Qdrant with averaged embedding + BM25 sparse (RRF fusion)

**Expected Improvement**: +5-10% for implicit/scenario/template queries

**Reference**: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
(Gao et al., 2022) - https://arxiv.org/abs/2212.10496

**Environment**:
- Qdrant: http://localhost:6333
- Embeddings: http://localhost:1234 (nomic-embed-text-v1.5)
- LLM: Z.ai GLM-4.6 (requires ZAI_API_KEY in .env)
"""

import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Any, Optional

import httpx
from dotenv import load_dotenv

# Parallel execution config
MAX_PARALLEL_EVALS = 5  # Max concurrent query evaluations

# Load environment variables from .env file
load_dotenv()

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ace.hyde import HyDEGenerator, HyDEConfig
from ace.hyde_retrieval import HyDEEnhancedRetriever
from ace.llm_providers.litellm_client import LiteLLMClient

# ============================================================================
# CONFIGURATION
# ============================================================================

QDRANT_URL = "http://localhost:6333"
EMBEDDING_URL = "http://localhost:1234"
COLLECTION_NAME = "ace_memories_hybrid"
EMBEDDING_MODEL = "text-embedding-qwen3-embedding-8b"

# LLM Configuration for HyDE
LLM_MODEL = "openai/glm-4.6"  # Z.ai GLM-4.6
ZAI_API_BASE = "https://api.z.ai/api/coding/paas/v4"  # Z.ai GLM endpoint (NOT root!)
NUM_HYPOTHETICALS = 3  # Number of hypothetical documents per query

# Test suite paths
TEST_SUITE_PATH = Path(__file__).parent.parent / "test_suite" / "enhanced_test_suite.json"
OUTPUT_PATH = Path(__file__).parent.parent / "optimization_results" / "v6_hyde.json"

# Top-K retrieval
TOP_K = 10

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_recall_at_k(retrieved_ids: List[str], ground_truth_id: str, k: int) -> float:
    """Calculate Recall@K metric."""
    top_k_ids = retrieved_ids[:k]
    return 1.0 if ground_truth_id in top_k_ids else 0.0


def calculate_mrr(retrieved_ids: List[str], ground_truth_id: str) -> float:
    """Calculate Mean Reciprocal Rank."""
    try:
        rank = retrieved_ids.index(ground_truth_id) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def calculate_ndcg_at_k(retrieved_ids: List[str], ground_truth_id: str, k: int) -> float:
    """Calculate NDCG@K (binary relevance)."""
    if ground_truth_id not in retrieved_ids[:k]:
        return 0.0

    rank = retrieved_ids[:k].index(ground_truth_id) + 1
    dcg = 1.0 / (1 + (rank - 1))  # log2(1 + rank) simplified for binary relevance
    idcg = 1.0  # Ideal case: relevant doc at rank 1
    return dcg / idcg


# ============================================================================
# EVALUATION HARNESS
# ============================================================================

class HyDEEvaluator:
    """Evaluate HyDE-enhanced retrieval performance."""

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        embedding_url: str = EMBEDDING_URL,
        collection_name: str = COLLECTION_NAME,
        num_hypotheticals: int = NUM_HYPOTHETICALS
    ):
        """Initialize HyDE evaluator.

        Args:
            qdrant_url: Qdrant server URL
            embedding_url: Embedding server URL
            collection_name: Qdrant collection name
            num_hypotheticals: Number of hypothetical documents to generate
        """
        # Initialize LLM client for HyDE (Z.ai GLM-4.6)
        logger.info(f"Initializing LLM client: {LLM_MODEL}")
        zai_api_key = os.getenv("ZAI_API_KEY")
        if not zai_api_key:
            raise ValueError("ZAI_API_KEY environment variable must be set for HyDE")
        self.llm_client = LiteLLMClient(
            model=LLM_MODEL,
            api_key=zai_api_key,
            api_base=ZAI_API_BASE
        )

        # Initialize HyDE generator
        hyde_config = HyDEConfig(
            num_hypotheticals=num_hypotheticals,
            max_tokens=150,
            temperature=0.7,
            cache_enabled=True
        )
        self.hyde_generator = HyDEGenerator(
            llm_client=self.llm_client,
            config=hyde_config
        )

        # Initialize HyDE-enhanced retriever
        self.retriever = HyDEEnhancedRetriever(
            hyde_generator=self.hyde_generator,
            qdrant_url=qdrant_url,
            embedding_url=embedding_url,
            collection_name=collection_name
        )

        logger.info(f"HyDE evaluator initialized with {num_hypotheticals} hypotheticals")

    def evaluate_single_query(
        self,
        query: str,
        ground_truth_id: str,
        use_hyde: bool,
        limit: int = TOP_K
    ) -> Dict[str, Any]:
        """Evaluate a single query.

        Args:
            query: Query text
            ground_truth_id: Expected bullet ID
            use_hyde: Enable HyDE expansion
            limit: Number of results to retrieve

        Returns:
            Dictionary with metrics and results
        """
        start_time = time.perf_counter()

        try:
            # Retrieve with or without HyDE
            results = self.retriever.retrieve(
                query=query,
                limit=limit,
                use_hyde=use_hyde
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Extract bullet IDs (as strings for consistent comparison)
            retrieved_ids = [str(r.bullet_id) for r in results]
            ground_truth_str = str(ground_truth_id)

            # Calculate metrics
            metrics = {
                "recall@1": calculate_recall_at_k(retrieved_ids, ground_truth_str, 1),
                "recall@5": calculate_recall_at_k(retrieved_ids, ground_truth_str, 5),
                "recall@10": calculate_recall_at_k(retrieved_ids, ground_truth_str, 10),
                "mrr": calculate_mrr(retrieved_ids, ground_truth_str),
                "ndcg@10": calculate_ndcg_at_k(retrieved_ids, ground_truth_str, 10),
                "latency_ms": latency_ms,
                "num_results": len(results)
            }

            return {
                "query": query,
                "ground_truth_id": ground_truth_id,
                "retrieved_ids": retrieved_ids,
                "metrics": metrics,
                "hyde_enabled": use_hyde,
                "success": True
            }

        except Exception as e:
            logger.error(f"Query failed: {query[:50]}... - {e}")
            return {
                "query": query,
                "ground_truth_id": ground_truth_id,
                "retrieved_ids": [],
                "metrics": {
                    "recall@1": 0.0,
                    "recall@5": 0.0,
                    "recall@10": 0.0,
                    "mrr": 0.0,
                    "ndcg@10": 0.0,
                    "latency_ms": 0.0,
                    "num_results": 0
                },
                "hyde_enabled": use_hyde,
                "success": False,
                "error": str(e)
            }

    def evaluate_test_suite(
        self,
        test_suite_path: Path,
        use_hyde: bool = True
    ) -> Dict[str, Any]:
        """Evaluate full test suite.

        Args:
            test_suite_path: Path to test_suite.json
            use_hyde: Enable HyDE expansion

        Returns:
            Complete evaluation results with per-category metrics
        """
        # Load test suite
        with open(test_suite_path, 'r', encoding='utf-8') as f:
            test_suite = json.load(f)

        logger.info(f"Loaded test suite: {len(test_suite['test_cases'])} memories")
        logger.info(f"HyDE enabled: {use_hyde}")

        # Evaluate each test case
        results_by_category = defaultdict(list)
        all_results = []

        # Flatten test cases - each memory has multiple generated queries
        test_queries = []
        for test_case in test_suite['test_cases']:
            memory_id = test_case['memory_id']
            memory_category = test_case['category']
            for query_obj in test_case.get('generated_queries', []):
                test_queries.append({
                    'query': query_obj['query'],
                    'memory_id': memory_id,
                    'category': query_obj.get('category', 'general'),
                    'memory_category': memory_category,
                    'difficulty': query_obj.get('difficulty', 'medium')
                })

        logger.info(f"Total queries to evaluate: {len(test_queries)}")
        logger.info(f"Using {MAX_PARALLEL_EVALS} parallel workers")

        def eval_single(args):
            """Worker function for parallel evaluation."""
            idx, test_case = args
            query = test_case['query']
            ground_truth_id = test_case['memory_id']
            category = test_case['category']

            if idx % 50 == 0:
                logger.info(f"[{idx}/{len(test_queries)}] Evaluating: {query[:60]}...")

            result = self.evaluate_single_query(
                query=query,
                ground_truth_id=ground_truth_id,
                use_hyde=use_hyde
            )
            result['category'] = category
            return result

        # Parallel evaluation with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_EVALS) as executor:
            indexed_queries = list(enumerate(test_queries, 1))
            results_list = list(executor.map(eval_single, indexed_queries))

        # Organize results by category
        for result in results_list:
            category = result['category']
            results_by_category[category].append(result)
            all_results.append(result)

        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(all_results)
        category_metrics = {
            category: self._calculate_aggregate_metrics(results)
            for category, results in results_by_category.items()
        }

        # Cache statistics
        cache_stats = self.hyde_generator.get_cache_stats()

        return {
            "technique": "HyDE (Hypothetical Document Embeddings)",
            "version": "v6",
            "hyde_enabled": use_hyde,
            "num_hypotheticals": self.retriever.config.num_hypotheticals,
            "llm_model": LLM_MODEL,
            "total_queries": len(all_results),
            "successful_queries": sum(1 for r in all_results if r['success']),
            "aggregate_metrics": aggregate_metrics,
            "category_metrics": category_metrics,
            "cache_stats": cache_stats,
            "detailed_results": all_results
        }

    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate aggregate metrics across results."""
        if not results:
            return {}

        successful_results = [r for r in results if r['success']]
        if not successful_results:
            return {"error": "No successful queries"}

        metrics = defaultdict(list)
        for result in successful_results:
            for metric_name, metric_value in result['metrics'].items():
                metrics[metric_name].append(metric_value)

        return {
            metric_name: {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }
            for metric_name, values in metrics.items()
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run HyDE evaluation."""
    logger.info("=" * 80)
    logger.info("V6: HyDE (Hypothetical Document Embeddings) Evaluation")
    logger.info("=" * 80)

    # Check environment
    if not os.getenv("ZAI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        logger.error("ERROR: Neither ZAI_API_KEY nor OPENAI_API_KEY found in environment")
        logger.error("Set ZAI_API_KEY for Z.ai GLM-4.6 or OPENAI_API_KEY for OpenAI")
        sys.exit(1)

    # Initialize evaluator
    evaluator = HyDEEvaluator(
        qdrant_url=QDRANT_URL,
        embedding_url=EMBEDDING_URL,
        collection_name=COLLECTION_NAME,
        num_hypotheticals=NUM_HYPOTHETICALS
    )

    # Run evaluation with HyDE enabled
    logger.info("\nRunning evaluation with HyDE enabled...")
    hyde_results = evaluator.evaluate_test_suite(
        test_suite_path=TEST_SUITE_PATH,
        use_hyde=True
    )

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(hyde_results, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to: {OUTPUT_PATH}")

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Technique: {hyde_results['technique']}")
    print(f"HyDE Enabled: {hyde_results['hyde_enabled']}")
    print(f"Hypothetical Documents: {hyde_results['num_hypotheticals']}")
    print(f"LLM Model: {hyde_results['llm_model']}")
    print(f"Total Queries: {hyde_results['total_queries']}")
    print(f"Successful: {hyde_results['successful_queries']}")
    print(f"\nCache Statistics:")
    for key, value in hyde_results['cache_stats'].items():
        print(f"  {key}: {value}")

    print(f"\nAGGREGATE METRICS:")
    for metric_name, metric_data in hyde_results['aggregate_metrics'].items():
        if isinstance(metric_data, dict):
            print(f"  {metric_name:12s}: mean={metric_data['mean']:.4f}, "
                  f"min={metric_data['min']:.4f}, max={metric_data['max']:.4f}")

    print(f"\nCATEGORY BREAKDOWN:")
    for category, metrics in hyde_results['category_metrics'].items():
        print(f"\n  {category}:")
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict) and 'mean' in metric_data:
                print(f"    {metric_name:12s}: {metric_data['mean']:.4f}")

    print("\n" + "=" * 80)
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
