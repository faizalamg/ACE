#!/usr/bin/env python3
"""
Quick inspection utility for test suite samples
"""

import json
from pathlib import Path
from collections import Counter

def main():
    test_file = Path(__file__).parent / "selected_memories.json"

    with open(test_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    print("\n" + "="*70)
    print("TEST SUITE QUICK INSPECTION")
    print("="*70)

    # Basic stats
    print(f"\nTotal Samples: {len(samples)}")
    total_queries = sum(len(s['sample_queries']) for s in samples)
    print(f"Total Test Queries: {total_queries}")
    print(f"Average Queries per Sample: {total_queries / len(samples):.2f}")

    # Category breakdown
    print("\n" + "-"*70)
    print("CATEGORY DISTRIBUTION")
    print("-"*70)
    cat_dist = Counter(s['category'] for s in samples)
    for cat, count in sorted(cat_dist.items(), key=lambda x: -x[1]):
        pct = (count / len(samples)) * 100
        print(f"{cat:25} {count:3} ({pct:5.1f}%)")

    # Severity breakdown
    print("\n" + "-"*70)
    print("SEVERITY DISTRIBUTION")
    print("-"*70)
    sev_dist = Counter(s['severity'] for s in samples)
    for sev in sorted(sev_dist.keys(), reverse=True):
        count = sev_dist[sev]
        pct = (count / len(samples)) * 100
        bar = "#" * int(pct / 2)
        print(f"Severity {sev:2}: {count:3} ({pct:5.1f}%) {bar}")

    # Feedback type breakdown
    print("\n" + "-"*70)
    print("FEEDBACK TYPE DISTRIBUTION")
    print("-"*70)
    ft_dist = Counter(s['feedback_type'] for s in samples)
    for ft, count in sorted(ft_dist.items(), key=lambda x: -x[1])[:10]:
        pct = (count / len(samples)) * 100
        print(f"{ft:25} {count:3} ({pct:5.1f}%)")

    # Sample preview
    print("\n" + "-"*70)
    print("SAMPLE PREVIEW (First 5)")
    print("-"*70)
    for i, sample in enumerate(samples[:5], 1):
        print(f"\n{i}. [{sample['category']}] Severity: {sample['severity']}")
        print(f"   ID: {sample['memory_id']}")
        print(f"   Content: {sample['content'][:80]}...")
        print(f"   Queries:")
        for q in sample['sample_queries']:
            print(f"     - '{q}'")

    # Query analysis
    print("\n" + "-"*70)
    print("QUERY ANALYSIS")
    print("-"*70)
    all_queries = [q for s in samples for q in s['sample_queries']]
    query_lengths = [len(q.split()) for q in all_queries]

    print(f"Total unique queries: {len(set(all_queries))}")
    print(f"Average query length: {sum(query_lengths) / len(query_lengths):.1f} words")
    print(f"Shortest query: {min(query_lengths)} words")
    print(f"Longest query: {max(query_lengths)} words")

    # Most common query terms
    all_terms = []
    for q in all_queries:
        all_terms.extend(q.lower().split())

    term_freq = Counter(all_terms)
    print(f"\nTop 15 query terms:")
    for term, count in term_freq.most_common(15):
        print(f"  {term:15} {count:3}")

    print("\n" + "="*70)
    print("END OF INSPECTION")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
