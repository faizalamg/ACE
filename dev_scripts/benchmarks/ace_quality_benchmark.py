#!/usr/bin/env python3
"""
Quick ACE-only benchmark to verify retrieval quality across 1000+ queries.
Tests ACE's code retrieval without waiting for slow ThatOtherContextEngine calls.
"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

from ace.code_retrieval import CodeRetrieval
from comprehensive_ace_ThatOtherContextEngine_benchmark import TEST_QUERIES


def run_ace_quality_benchmark(iterations: int = 4) -> dict:
    """
    Run ACE quality benchmark - verify ACE returns valid results.
    Success = ACE returns results with score >= 0.3 for each query.
    """
    r = CodeRetrieval()
    
    total_queries = 0
    successes = 0
    high_confidence = 0  # score >= 0.7
    medium_confidence = 0  # score >= 0.5
    failures = []
    
    all_queries = list(TEST_QUERIES) * iterations
    random.shuffle(all_queries)
    total = len(all_queries)
    
    print("=" * 80)
    print(f"ACE QUALITY BENCHMARK")
    print(f"Total queries: {total}")
    print("=" * 80)
    
    for i, (category, query) in enumerate(all_queries, 1):
        if i % 100 == 0:
            print(f"  Progress: {i}/{total} ({i/total*100:.0f}%)")
        
        try:
            results = r.search(query, limit=5)
            total_queries += 1
            
            if results:
                top_score = results[0]["score"]
                if top_score >= 0.7:
                    high_confidence += 1
                    successes += 1
                elif top_score >= 0.5:
                    medium_confidence += 1
                    successes += 1
                elif top_score >= 0.3:
                    successes += 1
                else:
                    failures.append({
                        "query": query,
                        "category": category,
                        "top_score": top_score,
                        "reason": f"Low score: {top_score:.3f}",
                    })
            else:
                failures.append({
                    "query": query,
                    "category": category,
                    "top_score": 0,
                    "reason": "No results",
                })
                
        except Exception as e:
            failures.append({
                "query": query,
                "category": category,
                "top_score": 0,
                "reason": str(e),
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("ACE QUALITY BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Total Queries: {total_queries}")
    print(f"Successes (score >= 0.3): {successes} ({successes/total_queries*100:.1f}%)")
    print(f"High Confidence (score >= 0.7): {high_confidence} ({high_confidence/total_queries*100:.1f}%)")
    print(f"Medium Confidence (score >= 0.5): {medium_confidence} ({medium_confidence/total_queries*100:.1f}%)")
    print(f"Failures: {len(failures)}")
    
    if failures[:10]:
        print("\nSample failures:")
        for f in failures[:10]:
            print(f"  [{f['category']}] {f['query'][:40]}... - {f['reason']}")
    
    return {
        "total": total_queries,
        "successes": successes,
        "high_confidence": high_confidence,
        "medium_confidence": medium_confidence,
        "failures": len(failures),
        "success_rate": successes/total_queries*100,
    }


if __name__ == "__main__":
    run_ace_quality_benchmark(iterations=4)
