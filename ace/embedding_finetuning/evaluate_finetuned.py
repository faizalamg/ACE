"""Evaluate fine-tuned embedding model performance.

Compares baseline (nomic-embed-text-v1.5) vs fine-tuned (all-MiniLM-L6-v2-finetuned)
on the test suite queries.

Metrics:
- Recall@1: Correct memory in top-1 result
- Recall@5: Correct memory in top-5 results
- MRR (Mean Reciprocal Rank): Average 1/rank of correct memory
- Average Similarity: Mean cosine similarity for correct pairs
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results for a single query evaluation."""

    query: str
    correct_memory_id: int
    correct_rank: Optional[int]  # None if not found in top-K
    top_5_ids: List[int]
    found_in_top_1: bool
    found_in_top_5: bool
    reciprocal_rank: float
    similarity_score: Optional[float] = None
    difficulty: str = "unknown"
    category: str = "unknown"


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all queries."""

    recall_at_1: float
    recall_at_5: float
    mrr: float
    avg_similarity: float
    total_queries: int
    by_difficulty: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)


class BaselineEmbeddingClient:
    """Client for baseline nomic embeddings via LM Studio."""

    def __init__(
        self,
        embedding_url: str = "http://localhost:1234",
        model: str = "text-embedding-qwen3-embedding-8b",
    ):
        self.embedding_url = embedding_url
        self.model = model
        self.client = httpx.Client(timeout=30.0)

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for batch of texts."""
        embeddings = []
        for text in texts:
            try:
                resp = self.client.post(
                    f"{self.embedding_url}/v1/embeddings",
                    json={"model": self.model, "input": text[:8000]},
                )
                if resp.status_code == 200:
                    embeddings.append(resp.json()["data"][0]["embedding"])
                else:
                    embeddings.append(None)
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                embeddings.append(None)

        return embeddings

    def close(self):
        self.client.close()


class EmbeddingEvaluator:
    """Evaluate embedding models on RAG retrieval task."""

    def __init__(
        self,
        test_suite_path: str,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "ace_memories_hybrid",
    ):
        """Initialize evaluator.

        Args:
            test_suite_path: Path to enhanced_test_suite.json
            qdrant_url: Qdrant server URL
            collection_name: Qdrant collection name
        """
        self.test_suite_path = Path(test_suite_path)
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.test_suite: Optional[Dict] = None

    def load_test_suite(self) -> None:
        """Load test suite."""
        with open(self.test_suite_path, "r", encoding="utf-8") as f:
            self.test_suite = json.load(f)
        logger.info(
            f"Loaded {len(self.test_suite['test_cases'])} test cases with "
            f"{self.test_suite['metadata']['generation_stats']['total_queries_generated']} queries"
        )

    def evaluate_model(
        self,
        model,
        model_name: str,
        max_queries: Optional[int] = None,
        use_dense_only: bool = True,
    ) -> Tuple[List[EvaluationResult], AggregateMetrics]:
        """Evaluate a single embedding model.

        Args:
            model: Embedding model (SentenceTransformer or BaselineEmbeddingClient)
            model_name: Name for logging
            max_queries: Maximum queries to evaluate (None = all)
            use_dense_only: If True, use dense-only search (faster, fair comparison)

        Returns:
            Tuple of (individual_results, aggregate_metrics)
        """
        if not self.test_suite:
            raise RuntimeError("Test suite not loaded. Call load_test_suite() first.")

        logger.info(f"Evaluating {model_name}...")
        start_time = time.time()

        results: List[EvaluationResult] = []
        queries_evaluated = 0

        for test_case in self.test_suite["test_cases"]:
            correct_memory_id = test_case["memory_id"]
            memory_category = test_case["category"]

            for query_obj in test_case["generated_queries"]:
                query = query_obj["query"]
                difficulty = query_obj["difficulty"]
                query_category = query_obj["category"]

                # Get query embedding
                if isinstance(model, SentenceTransformer):
                    query_embedding = model.encode(query).tolist()
                else:
                    query_embedding = model.encode([query])[0]

                if query_embedding is None:
                    continue

                # Search Qdrant (dense-only for fair comparison)
                try:
                    search_results = self.qdrant_client.search(
                        collection_name=self.collection_name,
                        query_vector=("dense", query_embedding),
                        limit=10,
                        with_payload=True,
                    )

                    # Find rank of correct memory
                    correct_rank = None
                    top_5_ids = []
                    similarity_score = None

                    for idx, result in enumerate(search_results, start=1):
                        result_id = result.payload.get(
                            "memory_id"
                        ) or result.payload.get("bullet_id")
                        top_5_ids.append(result_id)

                        if result_id == correct_memory_id:
                            correct_rank = idx
                            similarity_score = result.score

                    # Calculate metrics
                    found_in_top_1 = correct_rank == 1 if correct_rank else False
                    found_in_top_5 = correct_rank <= 5 if correct_rank else False
                    reciprocal_rank = 1.0 / correct_rank if correct_rank else 0.0

                    results.append(
                        EvaluationResult(
                            query=query,
                            correct_memory_id=correct_memory_id,
                            correct_rank=correct_rank,
                            top_5_ids=top_5_ids[:5],
                            found_in_top_1=found_in_top_1,
                            found_in_top_5=found_in_top_5,
                            reciprocal_rank=reciprocal_rank,
                            similarity_score=similarity_score,
                            difficulty=difficulty,
                            category=query_category,
                        )
                    )

                    queries_evaluated += 1

                except Exception as e:
                    logger.error(f"Search error for query '{query[:50]}': {e}")

                # Check max queries limit
                if max_queries and queries_evaluated >= max_queries:
                    break

            if max_queries and queries_evaluated >= max_queries:
                break

        elapsed = time.time() - start_time
        logger.info(
            f"Evaluated {queries_evaluated} queries in {elapsed:.2f}s "
            f"({queries_evaluated/elapsed:.1f} queries/sec)"
        )

        # Calculate aggregate metrics
        metrics = self._calculate_aggregate_metrics(results)

        return results, metrics

    def _calculate_aggregate_metrics(
        self, results: List[EvaluationResult]
    ) -> AggregateMetrics:
        """Calculate aggregate metrics from individual results."""
        if not results:
            return AggregateMetrics(
                recall_at_1=0.0,
                recall_at_5=0.0,
                mrr=0.0,
                avg_similarity=0.0,
                total_queries=0,
            )

        total = len(results)
        recall_1 = sum(1 for r in results if r.found_in_top_1) / total
        recall_5 = sum(1 for r in results if r.found_in_top_5) / total
        mrr = sum(r.reciprocal_rank for r in results) / total

        similarities = [r.similarity_score for r in results if r.similarity_score]
        avg_sim = sum(similarities) / len(similarities) if similarities else 0.0

        # Breakdown by difficulty
        by_difficulty: Dict[str, Dict[str, float]] = {}
        for difficulty in ["easy", "medium", "hard"]:
            subset = [r for r in results if r.difficulty == difficulty]
            if subset:
                by_difficulty[difficulty] = {
                    "recall@1": sum(1 for r in subset if r.found_in_top_1)
                    / len(subset),
                    "recall@5": sum(1 for r in subset if r.found_in_top_5)
                    / len(subset),
                    "mrr": sum(r.reciprocal_rank for r in subset) / len(subset),
                    "count": len(subset),
                }

        # Breakdown by category
        by_category: Dict[str, Dict[str, float]] = {}
        categories = set(r.category for r in results)
        for category in categories:
            subset = [r for r in results if r.category == category]
            if subset:
                by_category[category] = {
                    "recall@1": sum(1 for r in subset if r.found_in_top_1)
                    / len(subset),
                    "recall@5": sum(1 for r in subset if r.found_in_top_5)
                    / len(subset),
                    "mrr": sum(r.reciprocal_rank for r in subset) / len(subset),
                    "count": len(subset),
                }

        return AggregateMetrics(
            recall_at_1=recall_1,
            recall_at_5=recall_5,
            mrr=mrr,
            avg_similarity=avg_sim,
            total_queries=total,
            by_difficulty=by_difficulty,
            by_category=by_category,
        )

    def compare_models(
        self,
        baseline_model,
        finetuned_model,
        output_path: str,
        max_queries: Optional[int] = None,
    ) -> Dict:
        """Compare baseline vs fine-tuned model.

        Args:
            baseline_model: Baseline embedding model
            finetuned_model: Fine-tuned embedding model
            output_path: Path to save comparison results
            max_queries: Maximum queries to evaluate per model

        Returns:
            Comparison results dictionary
        """
        # Evaluate baseline
        baseline_results, baseline_metrics = self.evaluate_model(
            baseline_model, "Baseline (nomic-embed-text-v1.5)", max_queries
        )

        # Evaluate fine-tuned
        finetuned_results, finetuned_metrics = self.evaluate_model(
            finetuned_model, "Fine-tuned (all-MiniLM-L6-v2)", max_queries
        )

        # Calculate improvement
        improvement = {
            "recall@1": (
                (finetuned_metrics.recall_at_1 - baseline_metrics.recall_at_1)
                / baseline_metrics.recall_at_1
                * 100
                if baseline_metrics.recall_at_1 > 0
                else 0
            ),
            "recall@5": (
                (finetuned_metrics.recall_at_5 - baseline_metrics.recall_at_5)
                / baseline_metrics.recall_at_5
                * 100
                if baseline_metrics.recall_at_5 > 0
                else 0
            ),
            "mrr": (
                (finetuned_metrics.mrr - baseline_metrics.mrr)
                / baseline_metrics.mrr
                * 100
                if baseline_metrics.mrr > 0
                else 0
            ),
        }

        # Prepare output
        comparison = {
            "metadata": {
                "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_queries_evaluated": baseline_metrics.total_queries,
            },
            "baseline": {
                "model": "nomic-embed-text-v1.5",
                "recall@1": baseline_metrics.recall_at_1,
                "recall@5": baseline_metrics.recall_at_5,
                "mrr": baseline_metrics.mrr,
                "avg_similarity": baseline_metrics.avg_similarity,
                "by_difficulty": baseline_metrics.by_difficulty,
                "by_category": baseline_metrics.by_category,
            },
            "finetuned": {
                "model": "all-MiniLM-L6-v2-finetuned",
                "recall@1": finetuned_metrics.recall_at_1,
                "recall@5": finetuned_metrics.recall_at_5,
                "mrr": finetuned_metrics.mrr,
                "avg_similarity": finetuned_metrics.avg_similarity,
                "by_difficulty": finetuned_metrics.by_difficulty,
                "by_category": finetuned_metrics.by_category,
            },
            "improvement": improvement,
        }

        # Save results
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"Comparison results saved to {output_path}")

        # Print summary
        logger.info("\n=== EVALUATION SUMMARY ===")
        logger.info(f"Baseline Recall@1: {baseline_metrics.recall_at_1:.3f}")
        logger.info(f"Fine-tuned Recall@1: {finetuned_metrics.recall_at_1:.3f}")
        logger.info(f"Improvement: {improvement['recall@1']:+.1f}%")
        logger.info(f"\nBaseline Recall@5: {baseline_metrics.recall_at_5:.3f}")
        logger.info(f"Fine-tuned Recall@5: {finetuned_metrics.recall_at_5:.3f}")
        logger.info(f"Improvement: {improvement['recall@5']:+.1f}%")
        logger.info(f"\nBaseline MRR: {baseline_metrics.mrr:.3f}")
        logger.info(f"Fine-tuned MRR: {finetuned_metrics.mrr:.3f}")
        logger.info(f"Improvement: {improvement['mrr']:+.1f}%")

        return comparison


def main():
    """CLI entry point for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned embedding model"
    )
    parser.add_argument(
        "--test-suite",
        default="rag_training/test_suite/enhanced_test_suite.json",
        help="Path to test suite",
    )
    parser.add_argument(
        "--finetuned-model",
        default="ace/embedding_finetuning/models/ace_finetuned",
        help="Path to fine-tuned model",
    )
    parser.add_argument(
        "--output",
        default="rag_training/optimization_results/v5_finetuned_embeddings.json",
        help="Output path for results",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        help="Maximum queries to evaluate (None = all)",
    )
    parser.add_argument(
        "--qdrant-url", default="http://localhost:6333", help="Qdrant URL"
    )
    parser.add_argument(
        "--collection", default="ace_memories_hybrid", help="Collection name"
    )
    parser.add_argument(
        "--baseline-url",
        default="http://localhost:1234",
        help="LM Studio URL for baseline embeddings",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create evaluator
    evaluator = EmbeddingEvaluator(
        test_suite_path=args.test_suite,
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
    )
    evaluator.load_test_suite()

    # Load models
    logger.info("Loading baseline model (LM Studio)...")
    baseline_model = BaselineEmbeddingClient(
        embedding_url=args.baseline_url,
        model="text-embedding-qwen3-embedding-8b",
    )

    logger.info(f"Loading fine-tuned model from {args.finetuned_model}...")
    finetuned_model = SentenceTransformer(args.finetuned_model)

    try:
        # Compare models
        evaluator.compare_models(
            baseline_model=baseline_model,
            finetuned_model=finetuned_model,
            output_path=args.output,
            max_queries=args.max_queries,
        )

        logger.info(f"\nResults saved to: {args.output}")

    finally:
        baseline_model.close()


if __name__ == "__main__":
    main()
