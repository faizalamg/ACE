"""
RAG Testing Framework for Qdrant Vector Database

Production-ready framework for testing retrieval performance in hybrid search scenarios.
Measures Recall@K, MRR, NDCG, and provides detailed failure analysis.

Author: RocketRag.ai
License: See repository LICENSE file
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict

import httpx
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rag_tests.log')
    ]
)
logger = logging.getLogger(__name__)


class QueryCategory(str, Enum):
    """Categories for query variation types."""
    DIRECT = "direct"  # Direct rephrasing of memory content
    SEMANTIC = "semantic"  # Semantically equivalent but different words
    KEYWORD = "keyword"  # Keyword-based variations
    PARAPHRASE = "paraphrase"  # Natural paraphrasing
    QUESTION = "question"  # Question format variations
    IMPLICIT = "implicit"  # Contextual/implicit references
    PARTIAL = "partial"  # Subset of concepts
    TECHNICAL = "technical"  # Technical terminology
    CASUAL = "casual"  # Casual/informal phrasing
    EDGE_SHORT = "edge_short"  # Very short queries
    EDGE_LONG = "edge_long"  # Very long queries


class Difficulty(str, Enum):
    """Test case difficulty levels."""
    EASY = "easy"  # Direct matches, obvious keywords
    MEDIUM = "medium"  # Requires semantic understanding
    HARD = "hard"  # Complex reasoning, implicit references


@dataclass
class QueryVariant:
    """Individual query variant with metadata."""
    query: str
    category: QueryCategory
    expected_rank: int = 1  # Expected rank (1 = top result)
    min_similarity: float = 0.5  # Minimum expected similarity score


@dataclass
class TestCase:
    """Test case for RAG retrieval evaluation."""
    memory_id: int
    expected_content: str  # The memory content we expect to retrieve
    test_queries: List[QueryVariant] = field(default_factory=list)
    difficulty: Difficulty = Difficulty.MEDIUM
    category: str = "general"  # Memory category (e.g., "coding", "debugging")
    tags: List[str] = field(default_factory=list)

    def add_query(
        self,
        query: str,
        category: QueryCategory,
        expected_rank: int = 1,
        min_similarity: float = 0.5
    ) -> None:
        """Add a query variant to this test case."""
        self.test_queries.append(
            QueryVariant(query, category, expected_rank, min_similarity)
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "memory_id": self.memory_id,
            "expected_content": self.expected_content,
            "difficulty": self.difficulty.value,
            "category": self.category,
            "tags": self.tags,
            "test_queries": [
                {
                    "query": q.query,
                    "category": q.category.value,
                    "expected_rank": q.expected_rank,
                    "min_similarity": q.min_similarity
                }
                for q in self.test_queries
            ]
        }


@dataclass
class QueryResult:
    """Result of a single query execution."""
    query: str
    query_category: QueryCategory
    expected_memory_id: int
    retrieved_ids: List[int]
    scores: List[float]
    rank: Optional[int]  # None if not found
    score: Optional[float]  # None if not found
    success: bool
    latency_ms: float
    false_positives: List[Tuple[int, float]]  # (id, score) pairs above target


@dataclass
class TestCaseResult:
    """Results for all queries in a test case."""
    test_case_id: int
    difficulty: Difficulty
    category: str
    total_queries: int
    successful_queries: int
    query_results: List[QueryResult] = field(default_factory=list)

    # Metrics
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mean_reciprocal_rank: float = 0.0
    ndcg_at_5: float = 0.0
    avg_latency_ms: float = 0.0

    # Failure analysis
    failures_by_category: Dict[str, int] = field(default_factory=dict)
    avg_false_positives: float = 0.0


@dataclass
class TestSuiteResult:
    """Aggregated results for entire test suite."""
    timestamp: str
    total_test_cases: int
    total_queries: int
    successful_queries: int
    test_case_results: List[TestCaseResult] = field(default_factory=list)

    # Overall metrics
    overall_recall_at_1: float = 0.0
    overall_recall_at_5: float = 0.0
    overall_mrr: float = 0.0
    overall_ndcg: float = 0.0

    # Breakdown by difficulty
    results_by_difficulty: Dict[str, dict] = field(default_factory=dict)
    results_by_category: Dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total_test_cases": self.total_test_cases,
                "total_queries": self.total_queries,
                "successful_queries": self.successful_queries,
                "success_rate": self.successful_queries / self.total_queries if self.total_queries > 0 else 0.0
            },
            "overall_metrics": {
                "recall_at_1": self.overall_recall_at_1,
                "recall_at_5": self.overall_recall_at_5,
                "mrr": self.overall_mrr,
                "ndcg_at_5": self.overall_ndcg
            },
            "by_difficulty": self.results_by_difficulty,
            "by_category": self.results_by_category,
            "detailed_results": [
                {
                    "test_case_id": tc.test_case_id,
                    "difficulty": tc.difficulty.value,
                    "category": tc.category,
                    "metrics": {
                        "recall_at_1": tc.recall_at_1,
                        "recall_at_5": tc.recall_at_5,
                        "mrr": tc.mean_reciprocal_rank,
                        "ndcg_at_5": tc.ndcg_at_5,
                        "avg_latency_ms": tc.avg_latency_ms
                    },
                    "failures_by_category": tc.failures_by_category,
                    "avg_false_positives": tc.avg_false_positives,
                    "queries": [
                        {
                            "query": qr.query,
                            "category": qr.query_category.value,
                            "success": qr.success,
                            "rank": qr.rank,
                            "score": qr.score,
                            "latency_ms": qr.latency_ms,
                            "false_positives_count": len(qr.false_positives)
                        }
                        for qr in tc.query_results
                    ]
                }
                for tc in self.test_case_results
            ]
        }


class QdrantClient:
    """Lightweight Qdrant HTTP client for search operations."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "ace_memories_hybrid"
    ):
        self.base_url = f"http://{host}:{port}"
        self.collection_name = collection_name
        self.client = httpx.Client(timeout=30.0)
        logger.info(f"Initialized Qdrant client: {self.base_url}/{collection_name}")

    def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Execute hybrid search query.

        Args:
            query: Search query string
            limit: Maximum number of results
            score_threshold: Minimum score threshold

        Returns:
            List of search results with id, score, and payload
        """
        # Note: This assumes the collection has hybrid search enabled
        # with text field configured for BM25 + dense vector search
        payload = {
            "query": query,
            "limit": limit,
            "with_payload": True,
            "with_vector": False
        }

        if score_threshold is not None:
            payload["score_threshold"] = score_threshold

        try:
            response = self.client.post(
                f"{self.base_url}/collections/{self.collection_name}/points/query",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data.get("result", {}).get("points", [])
        except httpx.HTTPError as e:
            logger.error(f"Qdrant search failed: {e}")
            raise

    def close(self):
        """Close HTTP client."""
        self.client.close()


class RAGTestRunner:
    """Test runner for RAG retrieval evaluation."""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        top_k: int = 10,
        verbose: bool = True
    ):
        self.qdrant = qdrant_client
        self.top_k = top_k
        self.verbose = verbose

    def run_query(
        self,
        query_variant: QueryVariant,
        expected_memory_id: int
    ) -> QueryResult:
        """
        Execute single query and evaluate results.

        Args:
            query_variant: Query with metadata
            expected_memory_id: Expected memory ID to retrieve

        Returns:
            QueryResult with metrics
        """
        start_time = time.time()

        try:
            results = self.qdrant.search(query_variant.query, limit=self.top_k)
        except Exception as e:
            logger.error(f"Query failed: {query_variant.query[:50]}... - {e}")
            return QueryResult(
                query=query_variant.query,
                query_category=query_variant.category,
                expected_memory_id=expected_memory_id,
                retrieved_ids=[],
                scores=[],
                rank=None,
                score=None,
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                false_positives=[]
            )

        latency_ms = (time.time() - start_time) * 1000

        # Extract IDs and scores
        retrieved_ids = [r["id"] for r in results]
        scores = [r["score"] for r in results]

        # Find target memory
        rank = None
        score = None
        if expected_memory_id in retrieved_ids:
            rank = retrieved_ids.index(expected_memory_id) + 1
            score = scores[retrieved_ids.index(expected_memory_id)]

        # Identify false positives (results above target)
        false_positives = []
        if rank is not None:
            false_positives = [
                (retrieved_ids[i], scores[i])
                for i in range(rank - 1)
            ]

        # Evaluate success
        success = (
            rank is not None and
            rank <= query_variant.expected_rank and
            score is not None and
            score >= query_variant.min_similarity
        )

        if self.verbose:
            status = "✓" if success else "✗"
            logger.info(
                f"{status} Query: {query_variant.query[:60]}... | "
                f"Rank: {rank}/{self.top_k} | Score: {score:.3f if score else 0:.3f}"
            )

        return QueryResult(
            query=query_variant.query,
            query_category=query_variant.category,
            expected_memory_id=expected_memory_id,
            retrieved_ids=retrieved_ids,
            scores=scores,
            rank=rank,
            score=score,
            success=success,
            latency_ms=latency_ms,
            false_positives=false_positives
        )

    def run_test_case(self, test_case: TestCase) -> TestCaseResult:
        """
        Run all queries for a test case.

        Args:
            test_case: Test case to execute

        Returns:
            TestCaseResult with aggregated metrics
        """
        logger.info(
            f"\n{'='*80}\n"
            f"Running Test Case: ID={test_case.memory_id} | "
            f"Difficulty={test_case.difficulty.value} | "
            f"Queries={len(test_case.test_queries)}\n"
            f"{'='*80}"
        )

        query_results = []
        for query_variant in test_case.test_queries:
            result = self.run_query(query_variant, test_case.memory_id)
            query_results.append(result)

        # Calculate metrics
        result = TestCaseResult(
            test_case_id=test_case.memory_id,
            difficulty=test_case.difficulty,
            category=test_case.category,
            total_queries=len(query_results),
            successful_queries=sum(1 for r in query_results if r.success),
            query_results=query_results
        )

        # Recall@K
        result.recall_at_1 = sum(1 for r in query_results if r.rank == 1) / len(query_results)
        result.recall_at_5 = sum(1 for r in query_results if r.rank and r.rank <= 5) / len(query_results)
        result.recall_at_10 = sum(1 for r in query_results if r.rank and r.rank <= 10) / len(query_results)

        # Mean Reciprocal Rank
        reciprocal_ranks = [1/r.rank for r in query_results if r.rank]
        result.mean_reciprocal_rank = sum(reciprocal_ranks) / len(query_results) if reciprocal_ranks else 0.0

        # NDCG@5
        result.ndcg_at_5 = self._calculate_ndcg(query_results, k=5)

        # Average latency
        result.avg_latency_ms = sum(r.latency_ms for r in query_results) / len(query_results)

        # Failure analysis
        failures_by_category = defaultdict(int)
        for qr in query_results:
            if not qr.success:
                failures_by_category[qr.query_category.value] += 1
        result.failures_by_category = dict(failures_by_category)

        # False positives
        total_fp = sum(len(r.false_positives) for r in query_results)
        result.avg_false_positives = total_fp / len(query_results)

        logger.info(
            f"\nTest Case Results:\n"
            f"  Success Rate: {result.successful_queries}/{result.total_queries} "
            f"({result.successful_queries/result.total_queries*100:.1f}%)\n"
            f"  Recall@1: {result.recall_at_1:.3f}\n"
            f"  Recall@5: {result.recall_at_5:.3f}\n"
            f"  MRR: {result.mean_reciprocal_rank:.3f}\n"
            f"  NDCG@5: {result.ndcg_at_5:.3f}\n"
            f"  Avg Latency: {result.avg_latency_ms:.1f}ms"
        )

        return result

    def run_test_suite(
        self,
        test_cases: List[TestCase],
        output_file: Optional[Path] = None
    ) -> TestSuiteResult:
        """
        Run entire test suite.

        Args:
            test_cases: List of test cases to execute
            output_file: Optional path to save results JSON

        Returns:
            TestSuiteResult with aggregated metrics
        """
        logger.info(f"\n{'#'*80}\n# Starting Test Suite: {len(test_cases)} test cases\n{'#'*80}\n")

        suite_result = TestSuiteResult(
            timestamp=datetime.now().isoformat(),
            total_test_cases=len(test_cases),
            total_queries=sum(len(tc.test_queries) for tc in test_cases),
            successful_queries=0
        )

        for tc in test_cases:
            tc_result = self.run_test_case(tc)
            suite_result.test_case_results.append(tc_result)
            suite_result.successful_queries += tc_result.successful_queries

        # Calculate overall metrics
        all_results = [
            qr for tcr in suite_result.test_case_results
            for qr in tcr.query_results
        ]

        suite_result.overall_recall_at_1 = sum(1 for r in all_results if r.rank == 1) / len(all_results)
        suite_result.overall_recall_at_5 = sum(1 for r in all_results if r.rank and r.rank <= 5) / len(all_results)

        reciprocal_ranks = [1/r.rank for r in all_results if r.rank]
        suite_result.overall_mrr = sum(reciprocal_ranks) / len(all_results) if reciprocal_ranks else 0.0

        suite_result.overall_ndcg = self._calculate_ndcg(all_results, k=5)

        # Breakdown by difficulty
        by_difficulty = defaultdict(lambda: {"queries": [], "success": 0})
        for tcr in suite_result.test_case_results:
            diff = tcr.difficulty.value
            by_difficulty[diff]["queries"].extend(tcr.query_results)
            by_difficulty[diff]["success"] += tcr.successful_queries

        for diff, data in by_difficulty.items():
            total = len(data["queries"])
            suite_result.results_by_difficulty[diff] = {
                "total_queries": total,
                "successful_queries": data["success"],
                "success_rate": data["success"] / total if total > 0 else 0.0,
                "recall_at_1": sum(1 for r in data["queries"] if r.rank == 1) / total,
                "recall_at_5": sum(1 for r in data["queries"] if r.rank and r.rank <= 5) / total
            }

        # Breakdown by category
        by_category = defaultdict(lambda: {"queries": [], "success": 0})
        for tcr in suite_result.test_case_results:
            cat = tcr.category
            by_category[cat]["queries"].extend(tcr.query_results)
            by_category[cat]["success"] += tcr.successful_queries

        for cat, data in by_category.items():
            total = len(data["queries"])
            suite_result.results_by_category[cat] = {
                "total_queries": total,
                "successful_queries": data["success"],
                "success_rate": data["success"] / total if total > 0 else 0.0,
                "recall_at_1": sum(1 for r in data["queries"] if r.rank == 1) / total,
                "recall_at_5": sum(1 for r in data["queries"] if r.rank and r.rank <= 5) / total
            }

        # Print summary
        logger.info(
            f"\n{'#'*80}\n"
            f"# Test Suite Summary\n"
            f"{'#'*80}\n"
            f"Total Queries: {suite_result.total_queries}\n"
            f"Successful: {suite_result.successful_queries} "
            f"({suite_result.successful_queries/suite_result.total_queries*100:.1f}%)\n"
            f"\nOverall Metrics:\n"
            f"  Recall@1: {suite_result.overall_recall_at_1:.3f}\n"
            f"  Recall@5: {suite_result.overall_recall_at_5:.3f}\n"
            f"  MRR: {suite_result.overall_mrr:.3f}\n"
            f"  NDCG@5: {suite_result.overall_ndcg:.3f}\n"
        )

        # Save results
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(suite_result.to_dict(), f, indent=2)
            logger.info(f"\nResults saved to: {output_file}")

        return suite_result

    @staticmethod
    def _calculate_ndcg(results: List[QueryResult], k: int = 5) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain.

        Args:
            results: List of query results
            k: Top-k results to consider

        Returns:
            NDCG@k score
        """
        ndcg_scores = []
        for result in results:
            if result.rank is None or result.rank > k:
                ndcg_scores.append(0.0)
                continue

            # DCG: relevance / log2(rank + 1)
            # Binary relevance: 1 if correct memory found, 0 otherwise
            dcg = 1.0 / (np.log2(result.rank + 1))

            # IDCG: Perfect ranking (correct memory at rank 1)
            idcg = 1.0 / np.log2(2)

            ndcg_scores.append(dcg / idcg)

        return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0


class ResultComparator:
    """Compare before/after test results."""

    @staticmethod
    def compare(
        before_file: Path,
        after_file: Path,
        output_file: Optional[Path] = None
    ) -> dict:
        """
        Compare two test suite results.

        Args:
            before_file: Path to before results JSON
            after_file: Path to after results JSON
            output_file: Optional path to save comparison

        Returns:
            Comparison dictionary
        """
        with open(before_file) as f:
            before = json.load(f)
        with open(after_file) as f:
            after = json.load(f)

        comparison = {
            "before_timestamp": before["timestamp"],
            "after_timestamp": after["timestamp"],
            "metric_changes": {
                "recall_at_1": {
                    "before": before["overall_metrics"]["recall_at_1"],
                    "after": after["overall_metrics"]["recall_at_1"],
                    "change": after["overall_metrics"]["recall_at_1"] - before["overall_metrics"]["recall_at_1"]
                },
                "recall_at_5": {
                    "before": before["overall_metrics"]["recall_at_5"],
                    "after": after["overall_metrics"]["recall_at_5"],
                    "change": after["overall_metrics"]["recall_at_5"] - before["overall_metrics"]["recall_at_5"]
                },
                "mrr": {
                    "before": before["overall_metrics"]["mrr"],
                    "after": after["overall_metrics"]["mrr"],
                    "change": after["overall_metrics"]["mrr"] - before["overall_metrics"]["mrr"]
                },
                "ndcg_at_5": {
                    "before": before["overall_metrics"]["ndcg_at_5"],
                    "after": after["overall_metrics"]["ndcg_at_5"],
                    "change": after["overall_metrics"]["ndcg_at_5"] - before["overall_metrics"]["ndcg_at_5"]
                }
            },
            "success_rate_change": {
                "before": before["summary"]["success_rate"],
                "after": after["summary"]["success_rate"],
                "change": after["summary"]["success_rate"] - before["summary"]["success_rate"]
            }
        }

        # Print comparison
        logger.info(
            f"\n{'='*80}\n"
            f"Comparison Results\n"
            f"{'='*80}\n"
            f"Before: {before['timestamp']}\n"
            f"After:  {after['timestamp']}\n\n"
            f"Metric Changes:\n"
        )

        for metric, values in comparison["metric_changes"].items():
            change_pct = (values["change"] / values["before"] * 100) if values["before"] > 0 else 0
            indicator = "↑" if values["change"] > 0 else "↓" if values["change"] < 0 else "→"
            logger.info(
                f"  {metric:15s}: {values['before']:.3f} → {values['after']:.3f} "
                f"({indicator} {abs(change_pct):+.1f}%)"
            )

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(comparison, f, indent=2)
            logger.info(f"\nComparison saved to: {output_file}")

        return comparison


# Numpy for NDCG calculation
try:
    import numpy as np
except ImportError:
    logger.warning("NumPy not found. NDCG calculation will use math.log2 fallback.")
    import math

    class np:
        """Fallback for NumPy if not installed."""
        @staticmethod
        def log2(x):
            return math.log2(x)


if __name__ == "__main__":
    # Example usage
    logger.info("RAG Test Framework - Example Usage")

    # Initialize client
    qdrant = QdrantClient()
    runner = RAGTestRunner(qdrant)

    # Create sample test case
    test_case = TestCase(
        memory_id=1,
        expected_content="Always use ThatOtherContextEngine for semantic code search",
        difficulty=Difficulty.MEDIUM,
        category="code_search"
    )

    # Add query variations
    test_case.add_query("How do I search code semantically?", QueryCategory.QUESTION)
    test_case.add_query("semantic code search tool", QueryCategory.KEYWORD)
    test_case.add_query("What's the best way to find code by meaning?", QueryCategory.PARAPHRASE)
    test_case.add_query("ThatOtherContextEngine usage", QueryCategory.KEYWORD)

    # Run single test case
    result = runner.run_test_case(test_case)

    logger.info("\nExample completed. See rag_tests.log for details.")

    qdrant.close()
