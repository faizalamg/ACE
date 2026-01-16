"""
Test Hybrid Scoring Weight Configurations
==========================================

Tests different BM25/dense weight combinations to improve retrieval precision
on architecture queries that currently fail at 33.3% precision.

Target query: "how is our system currently wired? qdrant option 2 with no local json playbook?"
Expected keywords: ["qdrant", "vector", "storage", "memory", "json", "playbook", "unified", "collection"]

Configurations tested:
1. Default RRF (current implementation - baseline)
2. BM25-heavy (higher prefetch limit for sparse, lower for dense)
3. Dense-heavy (higher prefetch limit for dense, lower for sparse)
4. Custom weighted fusion (manual score combination)

Metrics:
- Precision@K: How many of top K results contain expected keywords
- Recall@K: Did we find the ground truth document in top K
- MRR: Rank of the ground truth document
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import httpx
from dotenv import load_dotenv

load_dotenv()

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from ACE framework
from ace.unified_memory import UnifiedMemoryIndex, create_sparse_vector

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "ace_memories_hybrid"
EMBEDDING_DIM = 768

# Test query (failing architecture query)
TEST_QUERY = "how is our system currently wired? qdrant option 2 with no local json playbook?"
EXPECTED_KEYWORDS = ["qdrant", "vector", "storage", "memory", "json", "playbook", "unified", "collection"]

# Ground truth document ID (to be determined by manual inspection)
GROUND_TRUTH_ID = None  # Set after finding the correct document

# Retrieval limits
TEST_K_VALUES = [1, 3, 5, 10]


class HybridWeightTester:
    """Test different hybrid scoring configurations."""

    def __init__(self):
        """Initialize tester with Qdrant client."""
        self.client = httpx.Client(timeout=30.0, base_url=QDRANT_URL)
        self.memory_index = UnifiedMemoryIndex(
            collection_name=COLLECTION_NAME,
            qdrant_url=QDRANT_URL
        )

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding from UnifiedMemoryIndex."""
        return self.memory_index._get_embedding(text)

    def calculate_keyword_precision(self, results: List[Dict], k: int) -> float:
        """
        Calculate precision based on keyword presence in top K results.

        Returns: fraction of results containing at least 3 expected keywords
        """
        if not results or k == 0:
            return 0.0

        top_k = results[:k]
        matches = 0

        for result in top_k:
            payload = result.get("payload", {})
            # Combine all text fields
            text = " ".join([
                str(payload.get("content", "")),
                str(payload.get("category", "")),
                str(payload.get("lesson", "")),
            ]).lower()

            # Count keyword matches
            keyword_count = sum(1 for kw in EXPECTED_KEYWORDS if kw in text)

            # Consider it a match if at least 3 keywords present
            if keyword_count >= 3:
                matches += 1

        return matches / k

    def test_default_rrf(self, query: str, limit: int = 10) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Test 1: Default RRF (current implementation).

        Uses Qdrant's built-in RRF fusion with 3x prefetch multiplier.
        """
        start = time.perf_counter()

        embedding = self.get_embedding(query)
        sparse = create_sparse_vector(query)

        prefetch_queries = [
            {
                "query": embedding,
                "using": "dense",
                "limit": limit * 3,
            }
        ]

        if sparse.get("indices"):
            prefetch_queries.append({
                "query": {
                    "indices": sparse["indices"],
                    "values": sparse["values"],
                },
                "using": "sparse",
                "limit": limit * 3,
            })

        # Execute query
        response = self.client.post(
            f"/collections/{COLLECTION_NAME}/points/query",
            json={
                "prefetch": prefetch_queries,
                "query": {"fusion": "rrf"},
                "limit": limit,
                "with_payload": True,
            }
        )

        latency_ms = (time.perf_counter() - start) * 1000

        results = response.json().get("result", {})
        points = results if isinstance(results, list) else results.get("points", [])

        return points, {
            "config": "Default RRF (3x prefetch)",
            "latency_ms": latency_ms,
            "dense_prefetch": limit * 3,
            "sparse_prefetch": limit * 3,
        }

    def test_bm25_heavy(self, query: str, limit: int = 10) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Test 2: BM25-heavy configuration.

        Higher prefetch for sparse (5x), lower for dense (2x).
        Hypothesis: Better for exact keyword matching.
        """
        start = time.perf_counter()

        embedding = self.get_embedding(query)
        sparse = create_sparse_vector(query)

        prefetch_queries = [
            {
                "query": embedding,
                "using": "dense",
                "limit": limit * 2,  # Reduced dense prefetch
            }
        ]

        if sparse.get("indices"):
            prefetch_queries.append({
                "query": {
                    "indices": sparse["indices"],
                    "values": sparse["values"],
                },
                "using": "sparse",
                "limit": limit * 5,  # Increased sparse prefetch
            })

        response = self.client.post(
            f"/collections/{COLLECTION_NAME}/points/query",
            json={
                "prefetch": prefetch_queries,
                "query": {"fusion": "rrf"},
                "limit": limit,
                "with_payload": True,
            }
        )

        latency_ms = (time.perf_counter() - start) * 1000

        results = response.json().get("result", {})
        points = results if isinstance(results, list) else results.get("points", [])

        return points, {
            "config": "BM25-heavy (dense 2x, sparse 5x)",
            "latency_ms": latency_ms,
            "dense_prefetch": limit * 2,
            "sparse_prefetch": limit * 5,
        }

    def test_dense_heavy(self, query: str, limit: int = 10) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Test 3: Dense-heavy configuration.

        Higher prefetch for dense (5x), lower for sparse (2x).
        Hypothesis: Better for semantic understanding.
        """
        start = time.perf_counter()

        embedding = self.get_embedding(query)
        sparse = create_sparse_vector(query)

        prefetch_queries = [
            {
                "query": embedding,
                "using": "dense",
                "limit": limit * 5,  # Increased dense prefetch
            }
        ]

        if sparse.get("indices"):
            prefetch_queries.append({
                "query": {
                    "indices": sparse["indices"],
                    "values": sparse["values"],
                },
                "using": "sparse",
                "limit": limit * 2,  # Reduced sparse prefetch
            })

        response = self.client.post(
            f"/collections/{COLLECTION_NAME}/points/query",
            json={
                "prefetch": prefetch_queries,
                "query": {"fusion": "rrf"},
                "limit": limit,
                "with_payload": True,
            }
        )

        latency_ms = (time.perf_counter() - start) * 1000

        results = response.json().get("result", {})
        points = results if isinstance(results, list) else results.get("points", [])

        return points, {
            "config": "Dense-heavy (dense 5x, sparse 2x)",
            "latency_ms": latency_ms,
            "dense_prefetch": limit * 5,
            "sparse_prefetch": limit * 2,
        }

    def test_custom_weighted(self, query: str, limit: int = 10, dense_weight: float = 0.6) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Test 4: Custom weighted fusion.

        Manually combine dense and sparse scores with custom weights.
        Bypasses Qdrant's RRF fusion for more control.
        """
        start = time.perf_counter()

        embedding = self.get_embedding(query)
        sparse = create_sparse_vector(query)

        # Get dense results
        dense_response = self.client.post(
            f"/collections/{COLLECTION_NAME}/points/query",
            json={
                "query": embedding,
                "using": "dense",
                "limit": limit * 3,
                "with_payload": True,
            }
        )
        dense_results = dense_response.json().get("result", {})
        dense_points = dense_results if isinstance(dense_results, list) else dense_results.get("points", [])

        # Get sparse results
        sparse_points = []
        if sparse.get("indices"):
            sparse_response = self.client.post(
                f"/collections/{COLLECTION_NAME}/points/query",
                json={
                    "query": {
                        "indices": sparse["indices"],
                        "values": sparse["values"],
                    },
                    "using": "sparse",
                    "limit": limit * 3,
                    "with_payload": True,
                }
            )
            sparse_results = sparse_response.json().get("result", {})
            sparse_points = sparse_results if isinstance(sparse_results, list) else sparse_results.get("points", [])

        # Build score maps
        dense_scores = {str(p.get("id", "")): p.get("score", 0.0) for p in dense_points}
        sparse_scores = {str(p.get("id", "")): p.get("score", 0.0) for p in sparse_points}

        # Combine with custom weights
        all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        combined = []

        for doc_id in all_ids:
            dense_score = dense_scores.get(doc_id, 0.0)
            sparse_score = sparse_scores.get(doc_id, 0.0)

            # Weighted combination
            final_score = (dense_weight * dense_score) + ((1 - dense_weight) * sparse_score)

            # Get document (prefer dense, fallback to sparse)
            doc = None
            for p in dense_points:
                if str(p.get("id", "")) == doc_id:
                    doc = p
                    break
            if not doc:
                for p in sparse_points:
                    if str(p.get("id", "")) == doc_id:
                        doc = p
                        break

            if doc:
                result = doc.copy()
                result["score"] = final_score
                result["dense_score"] = dense_score
                result["sparse_score"] = sparse_score
                combined.append(result)

        # Sort by final score
        combined.sort(key=lambda x: x["score"], reverse=True)

        latency_ms = (time.perf_counter() - start) * 1000

        return combined[:limit], {
            "config": f"Custom weighted (dense={dense_weight:.2f}, sparse={1-dense_weight:.2f})",
            "latency_ms": latency_ms,
            "dense_weight": dense_weight,
            "sparse_weight": 1 - dense_weight,
        }

    def test_bm25_boosted(self, query: str, limit: int = 10, boost_factor: float = 2.0) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Test 5: BM25 boosted configuration.

        Increases sparse vector values by a boost factor to give more weight to exact keyword matches.
        """
        start = time.perf_counter()

        embedding = self.get_embedding(query)
        sparse = create_sparse_vector(query)

        prefetch_queries = [
            {
                "query": embedding,
                "using": "dense",
                "limit": limit * 3,
            }
        ]

        if sparse.get("indices"):
            # Apply boost to sparse values
            boosted_values = [v * boost_factor for v in sparse["values"]]
            prefetch_queries.append({
                "query": {
                    "indices": sparse["indices"],
                    "values": boosted_values,
                },
                "using": "sparse",
                "limit": limit * 3,
            })

        response = self.client.post(
            f"/collections/{COLLECTION_NAME}/points/query",
            json={
                "prefetch": prefetch_queries,
                "query": {"fusion": "rrf"},
                "limit": limit,
                "with_payload": True,
            }
        )

        latency_ms = (time.perf_counter() - start) * 1000

        results = response.json().get("result", {})
        points = results if isinstance(results, list) else results.get("points", [])

        return points, {
            "config": f"BM25 boosted (boost={boost_factor:.1f}x)",
            "latency_ms": latency_ms,
            "boost_factor": boost_factor,
        }

    def test_query_expansion(self, query: str, limit: int = 10) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Test 6: Query expansion.

        Adds related keywords to the query to improve sparse matching.
        """
        start = time.perf_counter()

        # Expand query with related terms
        expanded_query = query + " architecture design system wiring configuration setup"

        embedding = self.get_embedding(query)  # Original embedding
        sparse = create_sparse_vector(expanded_query)  # Expanded for BM25

        prefetch_queries = [
            {
                "query": embedding,
                "using": "dense",
                "limit": limit * 3,
            }
        ]

        if sparse.get("indices"):
            prefetch_queries.append({
                "query": {
                    "indices": sparse["indices"],
                    "values": sparse["values"],
                },
                "using": "sparse",
                "limit": limit * 3,
            })

        response = self.client.post(
            f"/collections/{COLLECTION_NAME}/points/query",
            json={
                "prefetch": prefetch_queries,
                "query": {"fusion": "rrf"},
                "limit": limit,
                "with_payload": True,
            }
        )

        latency_ms = (time.perf_counter() - start) * 1000

        results = response.json().get("result", {})
        points = results if isinstance(results, list) else results.get("points", [])

        return points, {
            "config": "Query expansion (related terms added)",
            "latency_ms": latency_ms,
            "expanded_query": expanded_query,
        }

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test configurations and compare results."""
        print("=" * 80)
        print("HYBRID SCORING WEIGHT TESTS")
        print("=" * 80)
        print(f"\nTest Query: {TEST_QUERY}")
        print(f"Expected Keywords: {EXPECTED_KEYWORDS}")
        print(f"Testing K values: {TEST_K_VALUES}")
        print("\n" + "=" * 80 + "\n")

        all_results = {}

        # Test 1: Default RRF
        print("Test 1: Default RRF (current implementation)")
        points, metadata = self.test_default_rrf(TEST_QUERY, limit=max(TEST_K_VALUES))
        all_results["default_rrf"] = self._evaluate_results(points, metadata)
        self._print_results(all_results["default_rrf"])

        # Test 2: BM25-heavy
        print("\nTest 2: BM25-heavy (sparse 5x, dense 2x)")
        points, metadata = self.test_bm25_heavy(TEST_QUERY, limit=max(TEST_K_VALUES))
        all_results["bm25_heavy"] = self._evaluate_results(points, metadata)
        self._print_results(all_results["bm25_heavy"])

        # Test 3: Dense-heavy
        print("\nTest 3: Dense-heavy (dense 5x, sparse 2x)")
        points, metadata = self.test_dense_heavy(TEST_QUERY, limit=max(TEST_K_VALUES))
        all_results["dense_heavy"] = self._evaluate_results(points, metadata)
        self._print_results(all_results["dense_heavy"])

        # Test 4: Custom weighted (multiple weights)
        for weight in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            print(f"\nTest 4.{int(weight*10)}: Custom weighted (dense={weight:.1f})")
            points, metadata = self.test_custom_weighted(TEST_QUERY, limit=max(TEST_K_VALUES), dense_weight=weight)
            key = f"custom_weighted_{weight:.1f}"
            all_results[key] = self._evaluate_results(points, metadata)
            self._print_results(all_results[key])

        # Test 5: BM25 boosted (increase sparse values)
        for boost in [1.5, 2.0, 3.0]:
            print(f"\nTest 5.{int(boost*10)}: BM25 boosted (boost={boost:.1f}x)")
            points, metadata = self.test_bm25_boosted(TEST_QUERY, limit=max(TEST_K_VALUES), boost_factor=boost)
            key = f"bm25_boosted_{boost:.1f}"
            all_results[key] = self._evaluate_results(points, metadata)
            self._print_results(all_results[key])

        # Test 6: Query expansion
        print("\nTest 6: Query expansion (architecture terms)")
        points, metadata = self.test_query_expansion(TEST_QUERY, limit=max(TEST_K_VALUES))
        all_results["query_expansion"] = self._evaluate_results(points, metadata)
        self._print_results(all_results["query_expansion"])

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY - PRECISION@5 COMPARISON")
        print("=" * 80)

        summary = []
        for key, result in all_results.items():
            summary.append({
                "config": result["metadata"]["config"],
                "precision@5": result["precision@5"],
                "precision@10": result["precision@10"],
                "latency_ms": result["metadata"]["latency_ms"],
            })

        # Sort by precision@5
        summary.sort(key=lambda x: x["precision@5"], reverse=True)

        print(f"\n{'Configuration':<50} {'P@5':<10} {'P@10':<10} {'Latency':<10}")
        print("-" * 80)
        for item in summary:
            print(f"{item['config']:<50} {item['precision@5']:<10.1%} {item['precision@10']:<10.1%} {item['latency_ms']:<10.1f}ms")

        print("\n" + "=" * 80)

        # Find best configuration
        best = max(summary, key=lambda x: x["precision@5"])
        print(f"\nBEST CONFIGURATION: {best['config']}")
        print(f"Precision@5: {best['precision@5']:.1%}")
        print(f"Improvement over baseline: {(best['precision@5'] - summary[-1]['precision@5']) / summary[-1]['precision@5']:.1%}")

        return {
            "query": TEST_QUERY,
            "expected_keywords": EXPECTED_KEYWORDS,
            "results": all_results,
            "summary": summary,
            "best_config": best,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _evaluate_results(self, points: List[Dict], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate results and calculate metrics."""
        metrics = {}

        # Calculate precision@K for each K value
        for k in TEST_K_VALUES:
            precision = self.calculate_keyword_precision(points, k)
            metrics[f"precision@{k}"] = precision

        # Add metadata
        metrics["metadata"] = metadata
        metrics["top_results"] = self._format_top_results(points[:5])

        return metrics

    def _format_top_results(self, points: List[Dict]) -> List[Dict[str, Any]]:
        """Format top results for display."""
        formatted = []
        for i, point in enumerate(points, 1):
            payload = point.get("payload", {})

            # Extract key fields
            content = str(payload.get("content", payload.get("lesson", "")))[:200]

            # Count keyword matches
            text = content.lower()
            matched_keywords = [kw for kw in EXPECTED_KEYWORDS if kw in text]

            formatted.append({
                "rank": i,
                "id": str(point.get("id", "")),
                "score": point.get("score", 0.0),
                "content_preview": content,
                "matched_keywords": matched_keywords,
                "keyword_count": len(matched_keywords),
            })

        return formatted

    def _print_results(self, result: Dict[str, Any]) -> None:
        """Print results for a single test."""
        print(f"Config: {result['metadata']['config']}")
        print(f"Latency: {result['metadata']['latency_ms']:.1f}ms")
        print(f"Precision@1: {result['precision@1']:.1%}")
        print(f"Precision@5: {result['precision@5']:.1%}")
        print(f"Precision@10: {result['precision@10']:.1%}")

        print("\nTop 5 Results:")
        for res in result["top_results"]:
            print(f"  {res['rank']}. Score={res['score']:.4f} | Matched KW: {res['keyword_count']}/{len(EXPECTED_KEYWORDS)} {res['matched_keywords']}")
            print(f"     {res['content_preview']}")

    def close(self):
        """Close HTTP client."""
        self.client.close()


def main():
    """Main execution."""
    tester = HybridWeightTester()

    try:
        results = tester.run_all_tests()

        # Save results
        output_path = Path(__file__).parent.parent / "rag_training" / "optimization_results" / "hybrid_weight_tests.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_path}")

    finally:
        tester.close()


if __name__ == "__main__":
    main()
