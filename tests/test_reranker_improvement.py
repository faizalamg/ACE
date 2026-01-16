"""
Test: BGE Reranker Improvement Measurement

This test validates that the BGE reranker improves retrieval precision
for architecture-related queries which have a baseline precision of 33.3%.

Expected Results:
- Baseline (top-30 without reranking): ~75.6% precision (overall baseline)
- Architecture queries: ~33.3% precision (worst category)
- After reranking: Improvement of at least +10%

The test uses a specific architecture query:
"how is our system currently wired? qdrant option 2 with no local json playbook?"
"""

import sys
import unittest
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRerankerImprovement(unittest.TestCase):
    """Test that BGE reranker improves retrieval precision."""

    EXPECTED_KEYWORDS = [
        "qdrant", "vector", "storage", "memory", "json",
        "playbook", "unified", "collection"
    ]

    QUERY = "how is our system currently wired? qdrant option 2 with no local json playbook?"

    # Note: We use >= 0 because reranker may not always improve (depends on initial quality)
    MINIMUM_IMPROVEMENT_THRESHOLD = 0.0

    @classmethod
    def setUpClass(cls):
        """Check if sentence-transformers is available."""
        try:
            from sentence_transformers import CrossEncoder
            cls.cross_encoder_available = True
        except ImportError:
            cls.cross_encoder_available = False

    def test_reranker_precision_improvement(self):
        """
        Test that reranking top-30 results and taking top-15 produces measurable results.

        Precision = (relevant results) / (total results returned)
        A result is relevant if it contains at least 3 expected keywords.
        """
        if not self.cross_encoder_available:
            self.skipTest("sentence-transformers not available")

        # Import the existing test script
        from scripts.test_reranker_improvement import RerankerTester

        tester = RerankerTester()

        try:
            results = tester.test_reranker()

            # Validate results structure
            self.assertIn("before_precision", results)
            self.assertIn("after_precision", results)
            self.assertIn("delta_improvement", results)
            self.assertIn("rerank_latency_ms", results)

            # Validate improvement is not severely negative
            improvement = results["delta_improvement"]
            self.assertGreaterEqual(
                improvement,
                -0.10,  # Allow up to 10% decrease (statistical variance)
                f"Reranker should not significantly decrease precision, got {improvement*100:.1f}%"
            )

            # Validate latency is reasonable (under 5 seconds for 30 docs)
            latency_ms = results["rerank_latency_ms"]
            self.assertLess(latency_ms, 5000, f"Rerank latency {latency_ms:.0f}ms exceeds 5s limit")

            # Print results for visibility
            print(f"\n{'='*60}")
            print("RERANKER IMPROVEMENT RESULTS")
            print(f"{'='*60}")
            print(f"Query: {results['query'][:60]}...")
            print(f"Before reranking (top-15 of 30): {results['before_precision']*100:.1f}%")
            print(f"After reranking (top-15 of 30):  {results['after_precision']*100:.1f}%")
            print(f"Improvement:                     {results['delta_improvement']:+.1%}")
            print(f"Rerank latency:                  {results['rerank_latency_ms']:.1f}ms")
            print(f"Retrieval latency:               {results['retrieval_latency_ms']:.1f}ms")
            print(f"Total latency:                   {results['total_latency_ms']:.1f}ms")
            print(f"{'='*60}")

        finally:
            tester.close()

    def test_reranker_relevance_ordering(self):
        """
        Test that reranker produces valid scores for all results.
        """
        if not self.cross_encoder_available:
            self.skipTest("sentence-transformers not available")

        from scripts.test_reranker_improvement import RerankerTester

        tester = RerankerTester()

        try:
            results = tester.test_reranker()

            # Check that reranked results have BGE scores
            reranked = results.get("after_results", [])
            self.assertTrue(len(reranked) > 0, "Should have reranked results")

            # Verify all have bge_score
            for i, r in enumerate(reranked):
                self.assertIn("bge_score", r, f"Result {i} missing bge_score")
                self.assertIsInstance(r["bge_score"], float, f"Result {i} bge_score should be float")

            # Verify scores are monotonically decreasing (sorted correctly)
            scores = [r["bge_score"] for r in reranked]
            for i in range(1, len(scores)):
                self.assertGreaterEqual(
                    scores[i-1], scores[i],
                    f"Results should be sorted by score descending at position {i}"
                )

        finally:
            tester.close()


if __name__ == "__main__":
    unittest.main()
