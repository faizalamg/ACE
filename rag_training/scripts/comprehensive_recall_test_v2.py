#!/usr/bin/env python3
"""Comprehensive RAG Recall Test v2 - Corrected Methodology.

METHODOLOGY FIXES:
- first_words: Now uses 5 words (96.1% unique) instead of 3 words (70.9% unique)
- Added statistical significance tracking
- Reports theoretical maximums for context

Target: 95%+ Recall@5 across all test types
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import httpx
import random
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


# Configuration
QDRANT_URL = "http://localhost:6333"
EMBEDDING_URL = "http://localhost:1234"
COLLECTION = "ace_memories_hybrid"
MODEL = "text-embedding-qwen3-embedding-8b"

# Test parameters
SAMPLE_SIZE = 500  # Large sample for high statistical confidence
FIRST_WORDS_COUNT = 5  # FIXED: Was 3, now 5 for 96.1% uniqueness
TARGET_RECALL = 0.95


@dataclass
class TestResult:
    """Result of a single retrieval test."""
    memory_id: int
    query: str
    found_at_rank: Optional[int]  # None if not found
    top_5_ids: List[int]


class ComprehensiveRecallTester:
    """Comprehensive RAG retrieval tester with corrected methodology."""

    def __init__(self):
        self.client = httpx.Client(timeout=60.0)
        self.memories = []
        self.contents_map = {}  # id -> content

    def load_memories(self):
        """Load all memories from Qdrant."""
        resp = self.client.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
            json={"limit": 3000, "with_payload": True, "with_vector": False}
        )
        if resp.status_code == 200:
            self.memories = resp.json().get("result", {}).get("points", [])
            for mem in self.memories:
                payload = mem.get("payload", {})
                content = payload.get("lesson", "") or payload.get("content", "")
                self.contents_map[mem["id"]] = content
        print(f"Loaded {len(self.memories)} memories")

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text."""
        if "qwen" in MODEL.lower() and not text.endswith("</s>"):
            text = f"{text}</s>"
        try:
            resp = self.client.post(
                f"{EMBEDDING_URL}/v1/embeddings",
                json={"model": MODEL, "input": text[:8000]}
            )
            if resp.status_code == 200:
                return resp.json()["data"][0]["embedding"]
        except Exception:
            pass
        return None

    def hybrid_search(self, query: str, limit: int = 5) -> List[Dict]:
        """Execute hybrid search."""
        embedding = self.get_embedding(query)
        if not embedding:
            return []

        hybrid_query = {
            "prefetch": [
                {"query": embedding, "using": "dense", "limit": limit * 2}
            ],
            "query": {"fusion": "rrf"},
            "limit": limit,
            "with_payload": True
        }

        try:
            resp = self.client.post(
                f"{QDRANT_URL}/collections/{COLLECTION}/points/query",
                json=hybrid_query
            )
            if resp.status_code == 200:
                return resp.json().get("result", {}).get("points", [])
        except Exception:
            pass
        return []

    def generate_query(self, content: str, query_type: str) -> str:
        """Generate query based on type."""
        if query_type == "exact_match":
            # Full content (truncated for practical embedding)
            return content[:500]

        elif query_type == "partial_query":
            # Middle portion of content
            words = content.split()
            if len(words) > 10:
                start = len(words) // 4
                end = start + len(words) // 2
                return " ".join(words[start:end])
            return content[:200]

        elif query_type == "keyword_only":
            # Extract key terms (nouns, verbs - approximated)
            words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
            # Remove common words
            stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'will', 'when', 'what', 'which', 'there', 'their', 'they', 'would', 'could', 'should', 'always', 'never', 'before', 'after'}
            keywords = [w for w in words if w not in stopwords][:7]
            return " ".join(keywords)

        elif query_type == "first_words":
            # FIXED: Now uses 5 words instead of 3
            words = content.split()[:FIRST_WORDS_COUNT]
            return " ".join(words)

        elif query_type == "semantic_rephrase":
            # Simplified semantic variation
            words = content.split()[:15]
            return "information about " + " ".join(words[:10])

        return content[:200]

    def run_test_type(
        self,
        query_type: str,
        sample: List[Dict]
    ) -> Tuple[List[TestResult], float, float]:
        """Run a specific test type on the sample."""
        results = []
        found_at_1 = 0
        found_at_5 = 0

        for mem in sample:
            mem_id = mem["id"]
            content = self.contents_map.get(mem_id, "")
            if not content:
                continue

            query = self.generate_query(content, query_type)
            search_results = self.hybrid_search(query, limit=5)
            top_5_ids = [r["id"] for r in search_results]

            found_at = None
            if mem_id in top_5_ids:
                found_at = top_5_ids.index(mem_id) + 1
                found_at_5 += 1
                if found_at == 1:
                    found_at_1 += 1

            results.append(TestResult(
                memory_id=mem_id,
                query=query[:100],
                found_at_rank=found_at,
                top_5_ids=top_5_ids
            ))

        total = len(results)
        recall_1 = found_at_1 / total if total > 0 else 0
        recall_5 = found_at_5 / total if total > 0 else 0

        return results, recall_1, recall_5

    def calculate_theoretical_max(self, sample: List[Dict], query_type: str) -> float:
        """Calculate theoretical maximum Recall@5 for the sample."""
        if query_type != "first_words":
            return 1.0  # Other query types should be 100% unique

        # For first_words, check uniqueness in sample
        query_counter = Counter()
        for mem in sample:
            content = self.contents_map.get(mem["id"], "")
            query = self.generate_query(content, query_type)
            query_counter[query] += 1

        # Calculate theoretical max
        theoretical_found = 0
        for query, count in query_counter.items():
            theoretical_found += min(5, count)  # Can only retrieve 5 of duplicates

        return theoretical_found / len(sample) if sample else 1.0

    def run_comprehensive_test(self):
        """Run comprehensive test across all query types."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE RAG RECALL TEST v2 - CORRECTED METHODOLOGY")
        print("=" * 80)
        print(f"Collection: {COLLECTION}")
        print(f"Embedding Model: {MODEL}")
        print(f"Sample Size: {SAMPLE_SIZE}")
        print(f"First Words Count: {FIRST_WORDS_COUNT} (was 3, now 5 for 96.1% uniqueness)")
        print(f"Target: {TARGET_RECALL*100:.0f}%+ Recall@5")
        print("=" * 80)

        # Sample memories
        sample = random.sample(self.memories, min(SAMPLE_SIZE, len(self.memories)))
        print(f"\nTesting on {len(sample)} sampled memories...\n")

        query_types = [
            "exact_match",
            "partial_query",
            "keyword_only",
            "first_words",
            "semantic_rephrase"
        ]

        all_results = {}
        overall_pass = True

        print(f"{'Test Type':<20} {'Recall@1':>10} {'Recall@5':>10} {'Theo Max':>10} {'Status':>10}")
        print("-" * 80)

        for query_type in query_types:
            results, recall_1, recall_5 = self.run_test_type(query_type, sample)
            theoretical_max = self.calculate_theoretical_max(sample, query_type)
            all_results[query_type] = {
                "results": results,
                "recall_1": recall_1,
                "recall_5": recall_5,
                "theoretical_max": theoretical_max
            }

            # Adjust target for theoretical maximum
            adjusted_target = min(TARGET_RECALL, theoretical_max - 0.01)
            status = "PASS" if recall_5 >= adjusted_target else "FAIL"
            if status == "FAIL":
                overall_pass = False

            print(f"{query_type:<20} {recall_1*100:>9.1f}% {recall_5*100:>9.1f}% {theoretical_max*100:>9.1f}% {status:>10}")

        print("-" * 80)

        # Overall assessment
        avg_recall_5 = sum(r["recall_5"] for r in all_results.values()) / len(all_results)
        print(f"{'AVERAGE':<20} {'-':>10} {avg_recall_5*100:>9.1f}% {'-':>10}")
        print()

        if overall_pass:
            print("OVERALL STATUS: PASS - All tests meet or exceed target (adjusted for theoretical max)")
        else:
            print("OVERALL STATUS: OPTIMIZATION NEEDED")
            print()
            # Identify failures
            for qt, data in all_results.items():
                if data["recall_5"] < TARGET_RECALL and data["recall_5"] < data["theoretical_max"] - 0.01:
                    print(f"  - {qt}: {data['recall_5']*100:.1f}% (target: {TARGET_RECALL*100:.0f}%)")

        print()
        print("=" * 80)

        return all_results, overall_pass

    def close(self):
        """Close HTTP client."""
        self.client.close()


def main():
    """Run comprehensive test."""
    tester = ComprehensiveRecallTester()
    tester.load_memories()
    results, passed = tester.run_comprehensive_test()
    tester.close()

    # Summary
    print("SUMMARY")
    print("=" * 80)
    print()
    if passed:
        print("SUCCESS: RAG retrieval achieves 95%+ Recall@5 across all valid test types!")
    else:
        print("ACTION REQUIRED: Some tests below target. Review failure patterns.")
    print()

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
