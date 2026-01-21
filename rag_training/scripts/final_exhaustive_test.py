#!/usr/bin/env python3
"""Final Exhaustive RAG Test - All 2003 Memories.

Tests every single memory in the collection for definitive results.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import httpx
import re
import time
from dataclasses import dataclass
from typing import List, Optional


QDRANT_URL = "http://localhost:6333"
EMBEDDING_URL = "http://localhost:1234"
COLLECTION = "ace_memories_hybrid"
MODEL = "text-embedding-qwen3-embedding-8b"
TARGET_RECALL = 0.95


class ExhaustiveTester:
    def __init__(self):
        self.client = httpx.Client(timeout=60.0)
        self.memories = []

    def load_memories(self):
        resp = self.client.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
            json={"limit": 3000, "with_payload": True, "with_vector": False}
        )
        if resp.status_code == 200:
            self.memories = resp.json().get("result", {}).get("points", [])
        print(f"Loaded {len(self.memories)} memories")

    def get_embedding(self, text: str):
        if "qwen" in MODEL.lower() and not text.endswith("</s>"):
            text = f"{text}</s>"
        try:
            resp = self.client.post(
                f"{EMBEDDING_URL}/v1/embeddings",
                json={"model": MODEL, "input": text[:8000]}
            )
            if resp.status_code == 200:
                return resp.json()["data"][0]["embedding"]
        except:
            pass
        return None

    def search(self, query: str, limit: int = 5):
        embedding = self.get_embedding(query)
        if not embedding:
            return []
        try:
            resp = self.client.post(
                f"{QDRANT_URL}/collections/{COLLECTION}/points/query",
                json={
                    "prefetch": [{"query": embedding, "using": "dense", "limit": limit * 2}],
                    "query": {"fusion": "rrf"},
                    "limit": limit,
                    "with_payload": True
                }
            )
            if resp.status_code == 200:
                return resp.json().get("result", {}).get("points", [])
        except:
            pass
        return []

    def run_exhaustive_test(self, query_type: str):
        """Test all memories with specified query type."""
        found_at_1 = 0
        found_at_5 = 0
        total = 0

        start = time.time()

        for i, mem in enumerate(self.memories):
            mem_id = mem["id"]
            payload = mem.get("payload", {})
            content = payload.get("lesson", "") or payload.get("content", "")
            if not content:
                continue

            # Generate query based on type
            if query_type == "exact":
                query = content[:500]
            elif query_type == "first_5_words":
                query = " ".join(content.split()[:5])
            elif query_type == "keywords":
                words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
                stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'will', 'when', 'what', 'which', 'there', 'their', 'they', 'would', 'could', 'should', 'always', 'never', 'before', 'after'}
                keywords = [w for w in words if w not in stopwords][:7]
                query = " ".join(keywords)
            else:
                query = content[:200]

            results = self.search(query, limit=5)
            found_ids = [r["id"] for r in results]

            if mem_id in found_ids:
                found_at_5 += 1
                if found_ids[0] == mem_id:
                    found_at_1 += 1

            total += 1

            if (i + 1) % 200 == 0:
                elapsed = time.time() - start
                eta = elapsed / (i + 1) * (len(self.memories) - i - 1)
                print(f"  Progress: {i+1}/{len(self.memories)} ({100*(i+1)/len(self.memories):.0f}%), ETA: {eta:.0f}s")

        recall_1 = found_at_1 / total if total > 0 else 0
        recall_5 = found_at_5 / total if total > 0 else 0

        return recall_1, recall_5, total

    def close(self):
        self.client.close()


def main():
    print("=" * 80)
    print("FINAL EXHAUSTIVE RAG TEST - ALL MEMORIES")
    print("=" * 80)
    print(f"Collection: {COLLECTION}")
    print(f"Model: {MODEL}")
    print(f"Target: {TARGET_RECALL*100:.0f}%+ Recall@5")
    print("=" * 80)
    print()

    tester = ExhaustiveTester()
    tester.load_memories()
    print()

    results = {}

    for query_type in ["exact", "first_5_words", "keywords"]:
        print(f"Testing: {query_type}")
        r1, r5, total = tester.run_exhaustive_test(query_type)
        results[query_type] = {"recall_1": r1, "recall_5": r5, "total": total}
        status = "PASS" if r5 >= TARGET_RECALL else "FAIL"
        print(f"  Recall@1: {r1*100:.2f}%, Recall@5: {r5*100:.2f}%, Status: {status}")
        print()

    tester.close()

    # Summary
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print()
    print(f"{'Query Type':<20} {'Recall@1':>12} {'Recall@5':>12} {'Status':>10}")
    print("-" * 60)

    all_pass = True
    for qt, data in results.items():
        status = "PASS" if data["recall_5"] >= TARGET_RECALL else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"{qt:<20} {data['recall_1']*100:>11.2f}% {data['recall_5']*100:>11.2f}% {status:>10}")

    print("-" * 60)
    avg_r5 = sum(r["recall_5"] for r in results.values()) / len(results)
    print(f"{'AVERAGE':<20} {'-':>12} {avg_r5*100:>11.2f}%")
    print()

    if all_pass:
        print("FINAL VERDICT: SUCCESS - 95%+ Recall@5 achieved on ALL query types!")
    else:
        print("FINAL VERDICT: OPTIMIZATION NEEDED")

    print("=" * 80)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
