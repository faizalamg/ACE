#!/usr/bin/env python3
"""Final Exhaustive RAG Test - ALL MEMORIES with FULL OPTIMIZATION.

Uses OptimizedRetriever with:
- Query expansion (4 variations)
- Multi-query RRF fusion
- BM25 hybrid search
- Cross-encoder re-ranking (if available)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent to path for imports
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))

import httpx
import re
import time
from ace.retrieval_optimized import OptimizedRetriever


QDRANT_URL = "http://localhost:6333"
COLLECTION = "ace_memories_hybrid"
TARGET_RECALL = 0.95


def load_memories():
    """Load all memories from Qdrant."""
    client = httpx.Client(timeout=60.0)
    resp = client.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
        json={"limit": 3000, "with_payload": True, "with_vector": False}
    )
    client.close()
    if resp.status_code == 200:
        return resp.json().get("result", {}).get("points", [])
    return []


def generate_query(content: str, query_type: str) -> str:
    """Generate query based on type."""
    if query_type == "exact":
        return content[:500]
    elif query_type == "first_5_words":
        return " ".join(content.split()[:5])
    elif query_type == "keywords":
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
        stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'will', 'when',
                     'what', 'which', 'there', 'their', 'they', 'would', 'could',
                     'should', 'always', 'never', 'before', 'after'}
        keywords = [w for w in words if w not in stopwords][:7]
        return " ".join(keywords)
    return content[:200]


def run_exhaustive_test(retriever: OptimizedRetriever, memories: list, query_type: str):
    """Test all memories with specified query type using full optimization."""
    found_at_1 = 0
    found_at_5 = 0
    total = 0

    start = time.time()

    for i, mem in enumerate(memories):
        mem_id = mem["id"]
        payload = mem.get("payload", {})
        content = payload.get("lesson", "") or payload.get("content", "")
        if not content:
            continue

        query = generate_query(content, query_type)

        # Use OptimizedRetriever with full features
        results = retriever.search(query, limit=5)
        found_ids = [r.id for r in results]

        if mem_id in found_ids:
            found_at_5 += 1
            if found_ids[0] == mem_id:
                found_at_1 += 1

        total += 1

        if (i + 1) % 200 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(memories) - i - 1) / rate
            print(f"  Progress: {i+1}/{len(memories)} ({100*(i+1)/len(memories):.0f}%), "
                  f"Current R@5: {100*found_at_5/total:.1f}%, ETA: {eta:.0f}s")

    recall_1 = found_at_1 / total if total > 0 else 0
    recall_5 = found_at_5 / total if total > 0 else 0

    return recall_1, recall_5, total


def main():
    print("=" * 80)
    print("FINAL EXHAUSTIVE RAG TEST - FULL OPTIMIZATION")
    print("=" * 80)
    print()
    print("FEATURES ENABLED:")
    print("  - Query expansion (4 variations)")
    print("  - Multi-query retrieval with RRF fusion")
    print("  - BM25 sparse + Dense hybrid search")
    print("  - Cross-encoder re-ranking (if available)")
    print()
    print(f"Target: {TARGET_RECALL*100:.0f}%+ Recall@5")
    print("=" * 80)
    print()

    # Load memories
    memories = load_memories()
    print(f"Loaded {len(memories)} memories")
    print()

    # Create optimized retriever
    print("Initializing OptimizedRetriever...")
    retriever = OptimizedRetriever({
        "collection_name": COLLECTION,
        "enable_reranking": True,  # Enable cross-encoder if available
        "num_expanded_queries": 4,
        "candidates_per_query": 20,
        "first_stage_k": 40,
        "final_k": 10,
    })
    print(f"  Cross-encoder: {'Enabled' if retriever.cross_encoder else 'Disabled'}")
    print()

    results = {}

    for query_type in ["exact", "first_5_words", "keywords"]:
        print(f"Testing: {query_type}")
        r1, r5, total = run_exhaustive_test(retriever, memories, query_type)
        results[query_type] = {"recall_1": r1, "recall_5": r5, "total": total}
        status = "PASS" if r5 >= TARGET_RECALL else "FAIL"
        print(f"  Final - Recall@1: {r1*100:.2f}%, Recall@5: {r5*100:.2f}%, Status: {status}")
        print()

    retriever.close()

    # Summary
    print("=" * 80)
    print("FINAL RESULTS - FULL OPTIMIZATION")
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
    overall_status = "PASS" if all_pass else "NEEDS WORK"
    print(f"{'AVERAGE':<20} {'-':>12} {avg_r5*100:>11.2f}% {overall_status:>10}")
    print()

    if all_pass:
        print("FINAL VERDICT: SUCCESS - Fortune 100 Grade RAG Achieved!")
        print("  - 95%+ Recall@5 on ALL query types")
        print("  - Full optimization pipeline active")
    else:
        print("FINAL VERDICT: Close but needs fine-tuning")
        for qt, data in results.items():
            if data["recall_5"] < TARGET_RECALL:
                gap = (TARGET_RECALL - data["recall_5"]) * 100
                print(f"  - {qt}: {gap:.2f}% below target")

    print("=" * 80)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
