#!/usr/bin/env python3
"""Diagnostic script to analyze first_words retrieval failure (76% Recall@5).

Identifies why queries with only first 3 words fail to retrieve their source memory.
"""

import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import httpx
import random
import json
from collections import Counter


QDRANT_URL = "http://localhost:6333"
EMBEDDING_URL = "http://localhost:1234"
COLLECTION = "ace_memories_hybrid"
MODEL = "text-embedding-qwen3-embedding-8b"


def get_all_memories():
    """Fetch all memories from Qdrant."""
    client = httpx.Client(timeout=60.0)
    resp = client.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
        json={"limit": 3000, "with_payload": True, "with_vector": False}
    )
    if resp.status_code == 200:
        return resp.json().get("result", {}).get("points", [])
    return []


def get_embedding(text: str, client: httpx.Client) -> list:
    """Get embedding for text."""
    if "qwen" in MODEL.lower() and not text.endswith("</s>"):
        text = f"{text}</s>"
    resp = client.post(
        f"{EMBEDDING_URL}/v1/embeddings",
        json={"model": MODEL, "input": text[:8000]}
    )
    if resp.status_code == 200:
        return resp.json()["data"][0]["embedding"]
    return None


def search_hybrid(query: str, client: httpx.Client, limit: int = 5) -> list:
    """Execute hybrid search."""
    embedding = get_embedding(query, client)
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

    resp = client.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/query",
        json=hybrid_query
    )

    if resp.status_code == 200:
        return resp.json().get("result", {}).get("points", [])
    return []


def get_first_n_words(text: str, n: int = 3) -> str:
    """Extract first N words from text."""
    words = text.split()[:n]
    return " ".join(words)


def analyze_first_words_patterns():
    """Analyze patterns in first 3 words of all memories."""
    memories = get_all_memories()
    print(f"Total memories: {len(memories)}")

    first_words_counter = Counter()

    for mem in memories:
        payload = mem.get("payload", {})
        content = payload.get("lesson", "") or payload.get("content", "")
        if content:
            first_words = get_first_n_words(content, 3)
            first_words_counter[first_words] += 1

    print("\n=== TOP 30 MOST COMMON FIRST 3 WORDS ===")
    for phrase, count in first_words_counter.most_common(30):
        print(f"  {count:3d}x: '{phrase}'")

    # Count unique vs duplicate
    unique = sum(1 for c in first_words_counter.values() if c == 1)
    duplicates = sum(1 for c in first_words_counter.values() if c > 1)
    total_duplicated = sum(c for c in first_words_counter.values() if c > 1)

    print(f"\n=== UNIQUENESS ANALYSIS ===")
    print(f"  Unique first-3-words: {unique} ({100*unique/len(memories):.1f}%)")
    print(f"  Duplicate patterns: {duplicates}")
    print(f"  Memories with duplicate first-words: {total_duplicated} ({100*total_duplicated/len(memories):.1f}%)")

    return first_words_counter, memories


def test_retrieval_on_samples(memories: list, sample_size: int = 100):
    """Test retrieval using first 3 words as query."""
    client = httpx.Client(timeout=60.0)

    # Sample memories
    sample = random.sample(memories, min(sample_size, len(memories)))

    results = {
        "found_at_1": 0,
        "found_at_5": 0,
        "not_found": 0,
        "failures": []
    }

    print(f"\n=== TESTING {len(sample)} SAMPLES ===")

    for i, mem in enumerate(sample):
        mem_id = mem["id"]
        payload = mem.get("payload", {})
        content = payload.get("lesson", "") or payload.get("content", "")

        if not content:
            continue

        query = get_first_n_words(content, 3)
        search_results = search_hybrid(query, client, limit=5)

        found_ids = [r["id"] for r in search_results]

        if mem_id in found_ids:
            position = found_ids.index(mem_id) + 1
            if position == 1:
                results["found_at_1"] += 1
            results["found_at_5"] += 1
        else:
            results["not_found"] += 1
            results["failures"].append({
                "id": mem_id,
                "query": query,
                "full_content": content[:200],
                "returned_ids": found_ids[:3]
            })

        if (i + 1) % 20 == 0:
            print(f"  Tested {i+1}/{len(sample)}...")

    total = results["found_at_1"] + results["found_at_5"] - results["found_at_1"] + results["not_found"]
    total = len(sample)

    print(f"\n=== RESULTS ===")
    print(f"  Recall@1: {100*results['found_at_1']/total:.1f}%")
    print(f"  Recall@5: {100*results['found_at_5']/total:.1f}%")
    print(f"  Not Found: {results['not_found']}")

    print(f"\n=== FAILURE ANALYSIS (first 10) ===")
    for fail in results["failures"][:10]:
        print(f"  Query: '{fail['query']}'")
        print(f"  Content: '{fail['full_content'][:100]}...'")
        print()

    client.close()
    return results


def analyze_failure_patterns(failures: list):
    """Analyze patterns in failed retrievals."""
    print("\n=== FAILURE PATTERN ANALYSIS ===")

    word_counts = Counter()
    for fail in failures:
        words = fail["query"].lower().split()
        for word in words:
            word_counts[word] += 1

    print("Most common words in failed queries:")
    for word, count in word_counts.most_common(20):
        print(f"  '{word}': {count}x")


if __name__ == "__main__":
    print("=" * 60)
    print("FIRST_WORDS RETRIEVAL FAILURE DIAGNOSTIC")
    print("=" * 60)

    # Analyze patterns
    first_words_counter, memories = analyze_first_words_patterns()

    # Test retrieval
    results = test_retrieval_on_samples(memories, sample_size=100)

    # Analyze failures
    if results["failures"]:
        analyze_failure_patterns(results["failures"])

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
