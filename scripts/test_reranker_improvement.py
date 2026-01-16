"""
Test script to measure BGE reranker improvement on precision.

This script:
1. Takes top-30 results for a specific query about Qdrant architecture
2. Applies BGE reranker to reorder by relevance
3. Takes top-15 after reranking
4. Measures precision improvement
5. Reports latency impact

Expected keywords: ["qdrant", "vector", "storage", "memory", "json", "playbook", "unified", "collection"]
"""

import time
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    print("ERROR: sentence-transformers not installed. Install with: pip install sentence-transformers")
    sys.exit(1)

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("ERROR: httpx not installed. Install with: pip install httpx")
    sys.exit(1)

from ace.unified_memory import UnifiedMemoryIndex, create_sparse_vector


# Configuration
QDRANT_URL = "http://localhost:6333"
EMBEDDING_URL = "http://localhost:1234"
EMBEDDING_MODEL = "text-embedding-qwen3-embedding-8b"
BGE_RERANKER_MODEL = "BAAI/bge-reranker-base"

# Test queries with expected keywords
TEST_QUERIES = [
    {
        "query": "how is our system currently wired? qdrant option 2 with no local json playbook?",
        "keywords": ["qdrant", "vector", "storage", "memory", "json", "playbook", "unified", "collection"],
        "category": "architecture"
    },
    {
        "query": "TDD test driven development protocol for code edits",
        "keywords": ["test", "tdd", "code", "edit", "production", "first", "failing", "write"],
        "category": "protocol"
    },
    {
        "query": "MCP tool syntax and parameter names for serena",
        "keywords": ["mcp", "serena", "tool", "parameter", "pattern", "symbol", "search", "find"],
        "category": "mcp_tools"
    },
    {
        "query": "excuse detection and correction protocol enforcement",
        "keywords": ["excuse", "correction", "protocol", "violation", "zen", "challenge", "response"],
        "category": "enforcement"
    },
    {
        "query": "ThatOtherContextEngine semantic search before grep or read tools",
        "keywords": ["ThatOtherContextEngine", "search", "semantic", "grep", "read", "first", "mandatory"],
        "category": "search_protocol"
    }
]

# Default test query (first one)
TEST_QUERY = TEST_QUERIES[0]["query"]
EXPECTED_KEYWORDS = TEST_QUERIES[0]["keywords"]

# Retrieval parameters
INITIAL_LIMIT = 30  # Retrieve 30 candidates
RERANKED_LIMIT = 15  # Return top 15 after reranking


class RerankerTester:
    """Test BGE reranker impact on precision."""

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        embedding_url: str = EMBEDDING_URL,
        embedding_model: str = EMBEDDING_MODEL,
        reranker_model: str = BGE_RERANKER_MODEL
    ):
        self.qdrant_url = qdrant_url
        self.embedding_url = embedding_url
        self.embedding_model = embedding_model
        self.client = httpx.Client(timeout=60.0)

        # Initialize UnifiedMemoryIndex
        self.unified_index = UnifiedMemoryIndex(qdrant_url=qdrant_url)

        # Load BGE reranker
        print(f"Loading BGE reranker model: {reranker_model}")
        self.reranker = CrossEncoder(reranker_model, max_length=512)
        print("BGE reranker loaded successfully")

    def get_embedding(self, text: str) -> List[float]:
        """Get dense embedding from LM Studio."""
        try:
            resp = self.client.post(
                f"{self.embedding_url}/v1/embeddings",
                json={"model": self.embedding_model, "input": text[:8000]}
            )
            if resp.status_code == 200:
                return resp.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"ERROR: Failed to get embedding: {e}")
        return []

    def hybrid_search(self, query: str, limit: int) -> Tuple[List[Dict], float]:
        """
        Perform hybrid search (dense + sparse RRF fusion) via Qdrant.

        Returns:
            Tuple of (results, latency_ms)
        """
        start = time.perf_counter()

        # Get dense embedding
        dense_embedding = self.get_embedding(query)
        if not dense_embedding:
            return [], 0.0

        # Create sparse vector
        sparse_vector = create_sparse_vector(query)

        # Build hybrid query with RRF fusion
        hybrid_query = {
            "prefetch": [
                {"query": dense_embedding, "using": "dense", "limit": limit * 2}
            ],
            "query": {"fusion": "rrf"},
            "limit": limit,
            "with_payload": True
        }

        # Add sparse vector if available
        if sparse_vector.get("indices"):
            hybrid_query["prefetch"].append({
                "query": {
                    "indices": sparse_vector["indices"],
                    "values": sparse_vector["values"]
                },
                "using": "sparse",
                "limit": limit * 2
            })

        # Execute search
        try:
            resp = self.client.post(
                f"{self.qdrant_url}/collections/ace_memories_hybrid/points/query",
                json=hybrid_query
            )
            if resp.status_code == 200:
                results = resp.json().get("result", {}).get("points", [])
                latency = (time.perf_counter() - start) * 1000
                return results, latency
        except Exception as e:
            print(f"ERROR: Hybrid search failed: {e}")

        return [], 0.0

    def rerank_results(self, query: str, candidates: List[Dict], limit: int) -> Tuple[List[Dict], float]:
        """
        Rerank candidates using BGE reranker.

        Args:
            query: Original search query
            candidates: Retrieved candidates from Qdrant
            limit: Number of results to return after reranking

        Returns:
            Tuple of (reranked_results, latency_ms)
        """
        if not candidates:
            return [], 0.0

        start = time.perf_counter()

        # Prepare query-document pairs
        pairs = []
        for candidate in candidates:
            payload = candidate.get("payload", {})
            doc_text = payload.get("content", "") or payload.get("lesson", "") or str(payload)
            pairs.append([query, doc_text[:1000]])  # Truncate to 1000 chars

        # Run reranker
        scores = self.reranker.predict(pairs)

        # Sort by score descending
        scored_candidates = list(zip(scores, candidates))
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Prepare reranked results
        reranked = []
        for score, candidate in scored_candidates[:limit]:
            candidate["original_score"] = candidate.get("score", 0)
            candidate["bge_score"] = float(score)
            candidate["score"] = float(score)
            candidate["reranked"] = True
            reranked.append(candidate)

        latency = (time.perf_counter() - start) * 1000
        return reranked, latency

    def calculate_precision(self, results: List[Dict], keywords: List[str]) -> float:
        """
        Calculate precision based on keyword matching.

        Precision = (# relevant results) / (# total results)

        A result is relevant if it contains at least 3 of the expected keywords.
        """
        if not results:
            return 0.0

        relevant_count = 0
        for result in results:
            payload = result.get("payload", {})
            content = (payload.get("content", "") or payload.get("lesson", "")).lower()

            # Count keyword matches
            matches = sum(1 for kw in keywords if kw.lower() in content)

            # Relevant if >= 3 keywords matched
            if matches >= 3:
                relevant_count += 1

        precision = relevant_count / len(results)
        return precision

    def test_reranker(self) -> Dict:
        """
        Run the reranker test.

        Returns:
            Dictionary with test results
        """
        print("\n" + "="*80)
        print("RERANKER PRECISION IMPROVEMENT TEST")
        print("="*80)
        print(f"Query: {TEST_QUERY}")
        print(f"Expected keywords: {EXPECTED_KEYWORDS}")
        print(f"Initial retrieval: top-{INITIAL_LIMIT}")
        print(f"After reranking: top-{RERANKED_LIMIT}")
        print("="*80 + "\n")

        # Step 1: Retrieve top-30 with hybrid search
        print(f"[1/3] Retrieving top-{INITIAL_LIMIT} results with hybrid search...")
        initial_results, retrieval_latency = self.hybrid_search(TEST_QUERY, INITIAL_LIMIT)
        print(f"  [OK] Retrieved {len(initial_results)} results in {retrieval_latency:.1f}ms")

        if not initial_results:
            print("ERROR: No results retrieved. Check Qdrant connection and collection.")
            return {}

        # Step 2: Calculate precision before reranking
        print(f"\n[2/3] Calculating precision (before reranking, top-{RERANKED_LIMIT})...")
        before_results = initial_results[:RERANKED_LIMIT]
        before_precision = self.calculate_precision(before_results, EXPECTED_KEYWORDS)
        print(f"  [OK] Precision before reranking: {before_precision:.1%}")

        # Step 3: Rerank and calculate precision after
        print(f"\n[3/3] Reranking with BGE and calculating precision...")
        reranked_results, rerank_latency = self.rerank_results(TEST_QUERY, initial_results, RERANKED_LIMIT)
        after_precision = self.calculate_precision(reranked_results, EXPECTED_KEYWORDS)
        print(f"  [OK] Reranking completed in {rerank_latency:.1f}ms")
        print(f"  [OK] Precision after reranking: {after_precision:.1%}")

        # Calculate improvement
        delta = after_precision - before_precision
        total_latency = retrieval_latency + rerank_latency

        # Prepare results
        results = {
            "query": TEST_QUERY,
            "expected_keywords": EXPECTED_KEYWORDS,
            "initial_limit": INITIAL_LIMIT,
            "reranked_limit": RERANKED_LIMIT,
            "before_precision": before_precision,
            "after_precision": after_precision,
            "delta_improvement": delta,
            "retrieval_latency_ms": retrieval_latency,
            "rerank_latency_ms": rerank_latency,
            "total_latency_ms": total_latency,
            "before_results": before_results,
            "after_results": reranked_results
        }

        return results

    def print_results(self, results: Dict):
        """Print formatted test results."""
        if not results:
            return

        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"Before reranking:  Precision = {results['before_precision']:.1%}")
        print(f"After reranking:   Precision = {results['after_precision']:.1%}")
        print(f"Delta improvement: {results['delta_improvement']:+.1%}")

        print(f"\nLatency breakdown:")
        print(f"  Retrieval:  {results['retrieval_latency_ms']:.1f}ms")
        print(f"  Reranking:  {results['rerank_latency_ms']:.1f}ms")
        print(f"  Total:      {results['total_latency_ms']:.1f}ms")

        print(f"\nLatency impact: +{results['rerank_latency_ms']:.1f}ms ({results['rerank_latency_ms']/results['total_latency_ms']:.1%} of total)")

        # Show top 5 results before and after
        print("\n" + "="*80)
        print("TOP 5 RESULTS (BEFORE RERANKING)")
        print("="*80)
        for i, result in enumerate(results['before_results'][:5], 1):
            payload = result.get("payload", {})
            content = payload.get("content", "") or payload.get("lesson", "")
            score = result.get("score", 0)
            print(f"\n{i}. Score: {score:.3f}")
            print(f"   Content: {content[:100]}...")

        print("\n" + "="*80)
        print("TOP 5 RESULTS (AFTER RERANKING)")
        print("="*80)
        for i, result in enumerate(results['after_results'][:5], 1):
            payload = result.get("payload", {})
            content = payload.get("content", "") or payload.get("lesson", "")
            bge_score = result.get("bge_score", 0)
            orig_score = result.get("original_score", 0)
            print(f"\n{i}. BGE Score: {bge_score:.4f} (original RRF: {orig_score:.4f})")
            print(f"   Content: {content[:100]}...")

        print("\n" + "="*80)

    def close(self):
        """Close HTTP client."""
        self.client.close()


def main():
    """Run the reranker test on all queries."""
    tester = RerankerTester()

    try:
        all_results = []

        print("\n" + "="*80)
        print("MULTI-QUERY RERANKER PRECISION TEST")
        print("="*80)
        print(f"Testing {len(TEST_QUERIES)} queries across different categories")
        print("="*80 + "\n")

        for i, test_case in enumerate(TEST_QUERIES, 1):
            query = test_case["query"]
            keywords = test_case["keywords"]
            category = test_case["category"]

            print(f"\n{'='*80}")
            print(f"QUERY {i}/{len(TEST_QUERIES)}: [{category.upper()}]")
            print(f"{'='*80}")
            print(f"Query: {query[:80]}...")
            print(f"Keywords: {keywords}")

            # Retrieve results
            initial_results, retrieval_latency = tester.hybrid_search(query, INITIAL_LIMIT)

            if not initial_results:
                print(f"  [SKIP] No results retrieved")
                continue

            # Calculate precision before
            before_results = initial_results[:RERANKED_LIMIT]
            before_precision = tester.calculate_precision(before_results, keywords)

            # Rerank and calculate precision after
            reranked_results, rerank_latency = tester.rerank_results(query, initial_results, RERANKED_LIMIT)
            after_precision = tester.calculate_precision(reranked_results, keywords)

            delta = after_precision - before_precision

            result = {
                "category": category,
                "query": query,
                "keywords": keywords,
                "before_precision": before_precision,
                "after_precision": after_precision,
                "delta": delta,
                "retrieval_latency_ms": retrieval_latency,
                "rerank_latency_ms": rerank_latency
            }
            all_results.append(result)

            print(f"  Before: {before_precision:.1%} | After: {after_precision:.1%} | Delta: {delta:+.1%}")
            print(f"  Latency: {retrieval_latency:.0f}ms retrieval + {rerank_latency:.0f}ms rerank = {retrieval_latency + rerank_latency:.0f}ms total")

        # Print aggregate summary
        print("\n" + "="*80)
        print("AGGREGATE SUMMARY")
        print("="*80)

        if all_results:
            avg_before = sum(r["before_precision"] for r in all_results) / len(all_results)
            avg_after = sum(r["after_precision"] for r in all_results) / len(all_results)
            avg_delta = avg_after - avg_before
            avg_rerank_latency = sum(r["rerank_latency_ms"] for r in all_results) / len(all_results)

            print(f"\nAverage Precision Before: {avg_before:.1%}")
            print(f"Average Precision After:  {avg_after:.1%}")
            print(f"Average Delta:            {avg_delta:+.1%}")
            print(f"Average Rerank Latency:   {avg_rerank_latency:.0f}ms")

            # Per-category breakdown
            print("\nPer-Category Results:")
            print("-" * 60)
            print(f"{'Category':<20} {'Before':>10} {'After':>10} {'Delta':>10}")
            print("-" * 60)
            for r in all_results:
                print(f"{r['category']:<20} {r['before_precision']:>10.1%} {r['after_precision']:>10.1%} {r['delta']:>+10.1%}")
            print("-" * 60)

            # Improvement count
            improved = sum(1 for r in all_results if r["delta"] > 0)
            unchanged = sum(1 for r in all_results if r["delta"] == 0)
            degraded = sum(1 for r in all_results if r["delta"] < 0)

            print(f"\nImproved: {improved} | Unchanged: {unchanged} | Degraded: {degraded}")

            # Final verdict
            print("\n" + "="*80)
            print("FINAL VERDICT")
            print("="*80)
            print(f"\nReranker Impact: {avg_delta:+.1%} precision with {avg_rerank_latency:.0f}ms latency")

            if avg_delta > 0.05:
                print("Recommendation: IMPLEMENT - Significant precision improvement")
            elif avg_delta > 0:
                print("Recommendation: CONSIDER - Marginal precision improvement")
            elif avg_delta == 0:
                print("Recommendation: SKIP - No precision change, latency overhead not justified")
            else:
                print("Recommendation: SKIP - Precision degradation")

    finally:
        tester.close()


if __name__ == "__main__":
    main()
