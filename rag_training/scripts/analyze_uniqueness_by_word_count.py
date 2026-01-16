#!/usr/bin/env python3
"""Analyze uniqueness of content at different word counts."""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import httpx
from collections import Counter

QDRANT_URL = "http://localhost:6333"
COLLECTION = "ace_memories_hybrid"


def get_all_memories():
    """Fetch all memories from Qdrant."""
    client = httpx.Client(timeout=60.0)
    resp = client.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
        json={"limit": 3000, "with_payload": True, "with_vector": False}
    )
    client.close()
    if resp.status_code == 200:
        return resp.json().get("result", {}).get("points", [])
    return []


def get_first_n_words(text: str, n: int) -> str:
    """Extract first N words from text."""
    words = text.split()[:n]
    return " ".join(words)


def analyze_uniqueness():
    """Analyze uniqueness at different word counts."""
    memories = get_all_memories()
    print(f"Total memories: {len(memories)}")
    print()

    contents = []
    for mem in memories:
        payload = mem.get("payload", {})
        content = payload.get("lesson", "") or payload.get("content", "")
        if content:
            contents.append(content)

    print("=" * 70)
    print("UNIQUENESS ANALYSIS BY WORD COUNT")
    print("=" * 70)
    print()
    print(f"{'Words':<8} {'Unique':>10} {'%':>8} {'Duplicates':>12} {'Max Dup':>10} {'Theoretical Max R@5':>20}")
    print("-" * 70)

    for word_count in [3, 4, 5, 6, 7, 8, 10, 15, 20]:
        first_words_counter = Counter()
        for content in contents:
            prefix = get_first_n_words(content, word_count)
            first_words_counter[prefix] += 1

        unique = sum(1 for c in first_words_counter.values() if c == 1)
        duplicates = sum(1 for c in first_words_counter.values() if c > 1)
        max_dup = max(first_words_counter.values()) if first_words_counter else 1

        # Theoretical max Recall@5 calculation:
        # For unique entries: 100% retrievable
        # For duplicates: 5/count retrievable (capped at 100%)
        theoretical_max = 0
        for prefix, count in first_words_counter.items():
            if count == 1:
                theoretical_max += 1  # 100% retrievable
            else:
                theoretical_max += min(5, count)  # Max 5 of the duplicates

        theoretical_pct = 100 * theoretical_max / len(contents)

        print(f"{word_count:<8} {unique:>10} {100*unique/len(contents):>7.1f}% {duplicates:>12} {max_dup:>10} {theoretical_pct:>19.1f}%")

    print()
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print()
    print("The 'first_words' test with 3 words is INVALID because 29% of memories")
    print("share identical prefixes. Options:")
    print()
    print("  1. Use 7+ words for meaningful uniqueness (>95%)")
    print("  2. Use partial_query (middle portion) instead")
    print("  3. Skip first_words test entirely - it's unrealistic")
    print()


if __name__ == "__main__":
    analyze_uniqueness()
