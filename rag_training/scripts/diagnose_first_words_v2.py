#!/usr/bin/env python3
"""Diagnose why first_words (5 words) fails at 91.5%."""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import httpx
import random
from collections import Counter


QDRANT_URL = "http://localhost:6333"
EMBEDDING_URL = "http://localhost:1234"
COLLECTION = "ace_memories_hybrid"
MODEL = "text-embedding-qwen3-embedding-8b"


def get_all_memories():
    client = httpx.Client(timeout=60.0)
    resp = client.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
        json={"limit": 3000, "with_payload": True, "with_vector": False}
    )
    client.close()
    if resp.status_code == 200:
        return resp.json().get("result", {}).get("points", [])
    return []


def get_embedding(text: str, client: httpx.Client):
    if "qwen" in MODEL.lower() and not text.endswith("</s>"):
        text = f"{text}</s>"
    resp = client.post(
        f"{EMBEDDING_URL}/v1/embeddings",
        json={"model": MODEL, "input": text[:8000]}
    )
    if resp.status_code == 200:
        return resp.json()["data"][0]["embedding"]
    return None


def search(query: str, client: httpx.Client, limit: int = 10):
    """Search with expanded limit to see ranking."""
    embedding = get_embedding(query, client)
    if not embedding:
        return []

    resp = client.post(
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
    return []


def main():
    memories = get_all_memories()
    print(f"Total memories: {len(memories)}")

    # Sample 100 memories
    sample = random.sample(memories, min(100, len(memories)))
    client = httpx.Client(timeout=60.0)

    failures = []
    successes = 0

    for mem in sample:
        mem_id = mem["id"]
        payload = mem.get("payload", {})
        content = payload.get("lesson", "") or payload.get("content", "")
        if not content:
            continue

        query = " ".join(content.split()[:5])
        results = search(query, client, limit=10)
        found_ids = [r["id"] for r in results]

        if mem_id in found_ids[:5]:
            successes += 1
        else:
            # Check if in top 10
            rank = found_ids.index(mem_id) + 1 if mem_id in found_ids else None
            failures.append({
                "id": mem_id,
                "query": query,
                "content": content[:150],
                "rank": rank,
                "top_5_snippets": [r.get("payload", {}).get("lesson", "")[:60] for r in results[:5]]
            })

    print(f"\nRecall@5: {100*successes/len(sample):.1f}%")
    print(f"Failures: {len(failures)}")

    print("\n=== FAILURE ANALYSIS ===\n")
    for i, fail in enumerate(failures[:15]):
        print(f"--- Failure {i+1} ---")
        print(f"Query: '{fail['query']}'")
        print(f"Target content: '{fail['content']}'")
        if fail['rank']:
            print(f"Found at rank: {fail['rank']} (just outside top 5)")
        else:
            print("NOT FOUND in top 10!")
        print(f"Top 5 returned:")
        for j, snippet in enumerate(fail['top_5_snippets']):
            print(f"  {j+1}. {snippet}...")
        print()

    # Analyze patterns in failures
    print("=== FAILURE PATTERN ANALYSIS ===")
    words = Counter()
    for fail in failures:
        for w in fail["query"].lower().split():
            words[w] += 1
    print("Most common words in failed queries:")
    for word, count in words.most_common(15):
        print(f"  '{word}': {count}x")

    # Check how many were just outside top 5
    just_outside = sum(1 for f in failures if f['rank'] and f['rank'] <= 10)
    not_found = sum(1 for f in failures if not f['rank'])
    print(f"\nJust outside top 5 (rank 6-10): {just_outside}")
    print(f"Not in top 10: {not_found}")

    client.close()


if __name__ == "__main__":
    main()
