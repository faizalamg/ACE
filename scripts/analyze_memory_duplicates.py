"""
Analyze memory deduplication effectiveness in Qdrant.

BASELINE: Average Precision 75.6%, total memories: 2348

This script measures:
1. Duplicate rate in the full collection (random sampling)
2. Duplicate rate in retrieval results (top-15)
3. Potential precision gain from better deduplication

Key Hypothesis: High duplicate rate = wasted retrieval slots = lower precision
"""

import httpx
import random
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict


class DuplicateAnalyzer:
    """Analyze duplicate memories in Qdrant collection."""

    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self.qdrant_url = qdrant_url
        self.client = httpx.Client(timeout=30.0)
        self.collection_name = "ace_memories_hybrid"

    def get_collection_size(self) -> int:
        """Get total number of points in collection."""
        resp = self.client.get(
            f"{self.qdrant_url}/collections/{self.collection_name}"
        )
        resp.raise_for_status()
        return resp.json()["result"]["points_count"]

    def sample_random_memories(self, n: int = 100) -> List[Dict[str, Any]]:
        """
        Sample random memories from collection.

        Uses scroll API to get all points, then randomly samples from them.
        """
        total_size = self.get_collection_size()
        print(f"Total memories in collection: {total_size}")

        # Get all points using scroll
        all_points = []
        offset = None
        batch_size = 100

        print("Fetching all memories for sampling...")
        while True:
            req_body = {
                "limit": batch_size,
                "with_payload": True,
                "with_vector": ["dense"]
            }
            if offset is not None:
                req_body["offset"] = offset

            resp = self.client.post(
                f"{self.qdrant_url}/collections/{self.collection_name}/points/scroll",
                json=req_body
            )
            resp.raise_for_status()
            data = resp.json()

            points = data.get("result", {}).get("points", [])
            all_points.extend(points)

            next_offset = data.get("result", {}).get("next_page_offset")
            if not next_offset or len(all_points) >= total_size:
                break
            offset = next_offset
            print(f"  Fetched {len(all_points)} / {total_size}...")

        print(f"Total fetched: {len(all_points)}")

        # Deduplicate by ID (in case of any issues)
        seen_ids = set()
        unique_points = []
        for p in all_points:
            if p["id"] not in seen_ids:
                seen_ids.add(p["id"])
                unique_points.append(p)

        print(f"Unique points: {len(unique_points)}")

        # Random sample
        if len(unique_points) <= n:
            samples = unique_points
        else:
            samples = random.sample(unique_points, n)

        print(f"Sampled {len(samples)} random memories")
        return samples

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def find_duplicates_in_samples(
        self,
        samples: List[Dict[str, Any]],
        threshold: float = 0.95
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
        """
        Find near-duplicate pairs in sampled memories.

        Args:
            samples: List of memory points with vectors
            threshold: Cosine similarity threshold for duplicates

        Returns:
            List of (memory1, memory2, similarity) tuples
        """
        duplicates = []

        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                vec1 = samples[i].get("vector", {}).get("dense", [])
                vec2 = samples[j].get("vector", {}).get("dense", [])

                if not vec1 or not vec2:
                    continue

                similarity = self.cosine_similarity(vec1, vec2)

                if similarity >= threshold:
                    duplicates.append((samples[i], samples[j], similarity))

        return duplicates

    def retrieve_top_k(self, query: str, k: int = 15) -> List[Dict[str, Any]]:
        """
        Retrieve top-k memories for a query.

        Mimics the actual retrieval logic used in production.
        """
        # Get embedding for query (using LM Studio endpoint)
        embedding_url = "http://localhost:1234"
        embed_resp = self.client.post(
            f"{embedding_url}/v1/embeddings",
            json={"model": "text-embedding-qwen3-embedding-8b", "input": query + "</s>"}
        )
        embed_resp.raise_for_status()
        query_embedding = embed_resp.json()["data"][0]["embedding"]

        # Search Qdrant
        search_resp = self.client.post(
            f"{self.qdrant_url}/collections/{self.collection_name}/points/search",
            json={
                "vector": {
                    "name": "dense",
                    "vector": query_embedding
                },
                "limit": k,
                "with_payload": True,
                "with_vector": ["dense"]
            }
        )
        search_resp.raise_for_status()
        return search_resp.json()["result"]

    def analyze_retrieval_duplicates(
        self,
        query: str,
        k: int = 15,
        threshold: float = 0.95
    ) -> Dict[str, Any]:
        """
        Analyze duplicate rate in retrieval results.

        Returns:
            Dict with:
                - total_results: Number of results retrieved
                - duplicate_pairs: List of duplicate pairs found
                - duplicate_rate: Percentage of results that are duplicates
                - unique_information_rate: Percentage of results with unique info
        """
        results = self.retrieve_top_k(query, k)

        duplicates = []
        duplicate_ids = set()

        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                vec1 = results[i].get("vector", {}).get("dense", [])
                vec2 = results[j].get("vector", {}).get("dense", [])

                if not vec1 or not vec2:
                    continue

                similarity = self.cosine_similarity(vec1, vec2)

                if similarity >= threshold:
                    duplicates.append({
                        "id1": results[i]["id"],
                        "id2": results[j]["id"],
                        "similarity": similarity,
                        "content1": results[i].get("payload", {}).get("content", "")[:100],
                        "content2": results[j].get("payload", {}).get("content", "")[:100],
                    })
                    duplicate_ids.add(results[i]["id"])
                    duplicate_ids.add(results[j]["id"])

        total_results = len(results)
        num_duplicates = len(duplicate_ids)
        duplicate_rate = (num_duplicates / total_results * 100) if total_results > 0 else 0
        unique_info_rate = 100 - duplicate_rate

        return {
            "query": query,
            "total_results": total_results,
            "duplicate_pairs": duplicates,
            "num_duplicate_items": num_duplicates,
            "duplicate_rate": duplicate_rate,
            "unique_information_rate": unique_info_rate
        }

    def calculate_duplicate_rate(
        self,
        samples: List[Dict[str, Any]],
        threshold: float = 0.95
    ) -> Dict[str, Any]:
        """
        Calculate overall duplicate rate in sampled memories.

        Returns:
            Dict with:
                - total_samples: Number of memories sampled
                - duplicate_pairs: List of duplicate pairs
                - duplicate_rate: Estimated percentage of duplicates
        """
        duplicates = self.find_duplicates_in_samples(samples, threshold)

        # Count unique memories involved in duplicates
        duplicate_ids = set()
        for mem1, mem2, _ in duplicates:
            duplicate_ids.add(mem1["id"])
            duplicate_ids.add(mem2["id"])

        total_samples = len(samples)
        num_duplicates = len(duplicate_ids)
        duplicate_rate = (num_duplicates / total_samples * 100) if total_samples > 0 else 0

        return {
            "total_samples": total_samples,
            "duplicate_pairs": len(duplicates),
            "num_duplicate_items": num_duplicates,
            "duplicate_rate": duplicate_rate,
            "duplicates": [
                {
                    "id1": mem1["id"],
                    "id2": mem2["id"],
                    "similarity": sim,
                    "content1": mem1.get("payload", {}).get("content", "")[:100],
                    "content2": mem2.get("payload", {}).get("content", "")[:100],
                }
                for mem1, mem2, sim in duplicates[:10]  # Show top 10
            ]
        }

    def estimate_precision_gain(
        self,
        current_precision: float,
        duplicate_rate_in_retrieval: float
    ) -> Dict[str, Any]:
        """
        Estimate potential precision gain from deduplication.

        Assumptions:
        - Duplicates are wasting retrieval slots
        - Removing duplicates would allow more unique, relevant results
        - Conservative estimate: 50% of duplicate slots could be filled with relevant info

        Args:
            current_precision: Current Average Precision (e.g., 0.756 for 75.6%)
            duplicate_rate_in_retrieval: Percentage of results that are duplicates

        Returns:
            Dict with estimated precision gains
        """
        # Conservative estimate: deduplication recovers 50% of wasted slots as relevant
        recovery_rate = 0.5

        # Potential gain = duplicate_rate * recovery_rate
        potential_gain = (duplicate_rate_in_retrieval / 100) * recovery_rate

        estimated_new_precision = current_precision + potential_gain

        return {
            "current_precision": current_precision,
            "duplicate_rate": duplicate_rate_in_retrieval,
            "recovery_rate": recovery_rate,
            "potential_precision_gain": potential_gain,
            "estimated_new_precision": estimated_new_precision,
            "estimated_improvement_pct": (potential_gain / current_precision * 100) if current_precision > 0 else 0
        }


def main():
    """Run complete duplicate analysis."""
    print("=" * 80)
    print("MEMORY DEDUPLICATION ANALYSIS")
    print("=" * 80)
    print()

    analyzer = DuplicateAnalyzer()

    # 1. Sample random memories and check for duplicates
    print("STEP 1: Analyzing duplicate rate in collection (random sampling)")
    print("-" * 80)
    samples = analyzer.sample_random_memories(n=100)

    # Test multiple thresholds
    for threshold in [0.95, 0.92, 0.90, 0.85]:
        stats = analyzer.calculate_duplicate_rate(samples, threshold=threshold)
        print(f"  Threshold {threshold}: {stats['duplicate_rate']:.2f}% ({stats['duplicate_pairs']} pairs)")
    print()

    collection_stats = analyzer.calculate_duplicate_rate(samples, threshold=0.95)

    print(f"Total samples analyzed: {collection_stats['total_samples']}")
    print(f"Duplicate pairs found: {collection_stats['duplicate_pairs']}")
    print(f"Items involved in duplicates: {collection_stats['num_duplicate_items']}")
    print(f"Estimated duplicate rate: {collection_stats['duplicate_rate']:.2f}%")
    print()

    if collection_stats['duplicates']:
        print("Sample duplicate pairs (top 10):")
        for dup in collection_stats['duplicates']:
            print(f"  - Similarity: {dup['similarity']:.4f}")
            print(f"    ID1: {dup['id1']}")
            print(f"    Content1: {dup['content1']}...")
            print(f"    ID2: {dup['id2']}")
            print(f"    Content2: {dup['content2']}...")
            print()

    # 2. Test retrieval impact with multiple queries
    print()
    print("STEP 2: Analyzing duplicate rate in retrieval results")
    print("-" * 80)

    test_queries = [
        "how is our system currently wired?",
        "error handling best practices",
        "debugging strategies",
        "PowerShell bash path conversion",
        "MCP tool syntax",
    ]

    total_dup_rate = 0
    all_dup_pairs = []
    for query in test_queries:
        stats = analyzer.analyze_retrieval_duplicates(query, k=15, threshold=0.90)
        print(f"  Query: '{query[:40]}...' -> {stats['duplicate_rate']:.1f}% duplicates ({stats['num_duplicate_items']}/15)")
        total_dup_rate += stats['duplicate_rate']
        if stats['duplicate_pairs']:
            for pair in stats['duplicate_pairs']:
                all_dup_pairs.append((query, pair))

    avg_dup_rate = total_dup_rate / len(test_queries)
    print(f"\n  Average duplicate rate across {len(test_queries)} queries: {avg_dup_rate:.2f}%")

    if all_dup_pairs:
        print("\n  Duplicate pairs found:")
        for query, pair in all_dup_pairs[:5]:  # Show top 5
            print(f"    Query: '{query[:30]}...'")
            print(f"    Similarity: {pair['similarity']:.4f}")
            print(f"    Content1: {pair['content1'][:80]}...")
            print(f"    Content2: {pair['content2'][:80]}...")
            print()
    print()

    test_query = "how is our system currently wired?"
    retrieval_stats = analyzer.analyze_retrieval_duplicates(test_query, k=15, threshold=0.90)

    print(f"Query: '{retrieval_stats['query']}'")
    print(f"Total results retrieved: {retrieval_stats['total_results']}")
    print(f"Duplicate pairs in top-15: {len(retrieval_stats['duplicate_pairs'])}")
    print(f"Items involved in duplicates: {retrieval_stats['num_duplicate_items']}")
    print(f"Duplicate rate in results: {retrieval_stats['duplicate_rate']:.2f}%")
    print(f"Unique information rate: {retrieval_stats['unique_information_rate']:.2f}%")
    print()

    if retrieval_stats['duplicate_pairs']:
        print("Duplicate pairs found in top-15:")
        for dup in retrieval_stats['duplicate_pairs']:
            print(f"  - Similarity: {dup['similarity']:.4f}")
            print(f"    ID1: {dup['id1']}")
            print(f"    Content1: {dup['content1']}...")
            print(f"    ID2: {dup['id2']}")
            print(f"    Content2: {dup['content2']}...")
            print()

    # 3. Estimate precision gain using average duplicate rate
    print()
    print("STEP 3: Estimating potential precision gain from deduplication")
    print("-" * 80)

    current_precision = 0.756  # Baseline: 75.6%
    # Use average duplicate rate from multi-query analysis
    precision_gain = analyzer.estimate_precision_gain(
        current_precision,
        avg_dup_rate
    )

    print(f"Current Average Precision: {precision_gain['current_precision']:.1%}")
    print(f"Duplicate rate in retrieval: {precision_gain['duplicate_rate']:.2f}%")
    print(f"Recovery rate assumption: {precision_gain['recovery_rate']:.1%}")
    print(f"Potential precision gain: +{precision_gain['potential_precision_gain']:.1%}")
    print(f"Estimated new precision: {precision_gain['estimated_new_precision']:.1%}")
    print(f"Estimated improvement: +{precision_gain['estimated_improvement_pct']:.2f}%")
    print()

    # 4. Summary and recommendations
    print()
    print("=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    print()

    print(f"1. COLLECTION DUPLICATE RATE: {collection_stats['duplicate_rate']:.2f}%")
    print(f"   - Random sampling suggests ~{collection_stats['duplicate_rate']:.1f}% of memories are duplicates")
    print()

    print(f"2. RETRIEVAL DUPLICATE RATE (avg across {len(test_queries)} queries): {avg_dup_rate:.2f}%")
    print(f"   - At 0.90 similarity threshold")
    print(f"   - Wasting ~{avg_dup_rate:.1f}% of retrieval slots on near-duplicates")
    print()

    print(f"3. PRECISION IMPACT:")
    print(f"   - Current: {current_precision:.1%}")
    print(f"   - Potential: {precision_gain['estimated_new_precision']:.1%} (+{precision_gain['estimated_improvement_pct']:.2f}%)")
    print(f"   - Note: Duplicates found are semantically similar (same lesson, different wording)")
    print()

    # Recommendations based on average duplicate rate
    if avg_dup_rate > 20:
        print("[!] HIGH DUPLICATE RATE DETECTED")
        print("   Recommendations:")
        print("   - Increase dedup_threshold from 0.92 to 0.95 in index_bullet()")
        print("   - Enable more aggressive deduplication in retrieval pipeline")
        print("   - Consider post-retrieval deduplication before returning results")
    elif avg_dup_rate > 5:
        print("[!] MODERATE DUPLICATE RATE (~5%)")
        print("   Recommendations:")
        print("   - Fine-tune dedup_threshold (currently 0.92)")
        print("   - Add post-retrieval deduplication for top-k results")
        print("   - Consider semantic deduplication (same lesson, different wording)")
    else:
        print("[OK] DUPLICATE RATE ACCEPTABLE (<5%)")
        print("   Current deduplication strategy appears effective")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
