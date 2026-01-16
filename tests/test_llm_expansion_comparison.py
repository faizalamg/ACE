#!/usr/bin/env python
"""
Test script to compare retrieval quality with and without LLM query expansion.

Compares:
1. Baseline (rule-based expansion only via _QUERY_EXPANSIONS)
2. LLM expansion enabled (GLM 4.6 semantic alternatives + rule-based)

Measures:
- Precision (keyword match in results)
- Latency (time per query)
- Result quality (relevance scoring)
"""

import os
import sys
import time
from typing import List, Dict, Any, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Disable LLM expansion initially for baseline
os.environ["ACE_LLM_EXPANSION"] = "false"

from ace.unified_memory import UnifiedMemoryIndex


# Test queries with expected keywords for relevance measurement
TEST_QUERIES = [
    # Technical queries
    ("API error handling", ["API", "error", "retry", "fail"]),
    ("TDD workflow", ["TDD", "test", "Test-Driven"]),
    ("rate limiting", ["rate", "limit", "throttl"]),
    ("SQL injection prevention", ["SQL", "inject", "sanitiz", "input"]),
    ("authentication JWT", ["auth", "JWT", "token"]),
    ("memory leak detection", ["memory", "leak"]),
    ("deadlock prevention", ["deadlock", "lock", "thread"]),
    ("logging best practices", ["log", "debug", "trace"]),
    # Vague queries
    ("fix it", ["error", "bug", "fix", "resolve", "issue"]),
    ("slow", ["performance", "optimization", "latency", "slow"]),
    ("pool", ["connection", "pool", "resource"]),
    ("lock", ["lock", "mutex", "concurrent", "thread"]),
    ("secure API", ["secur", "API", "auth", "encrypt"]),
]


def measure_retrieval(
    index: UnifiedMemoryIndex,
    query: str,
    expected_keywords: List[str],
    use_llm: bool,
) -> Tuple[int, int, float, List[str]]:
    """
    Measure retrieval quality for a single query.
    
    Returns:
        Tuple of (relevant_count, total_count, latency_ms, result_snippets)
    """
    start = time.time()
    results = index.retrieve(
        query=query,
        limit=5,
        use_llm_expansion=use_llm,
    )
    latency_ms = (time.time() - start) * 1000
    
    relevant = 0
    snippets = []
    for r in results:
        content = r.content.lower()
        is_relevant = any(kw.lower() in content for kw in expected_keywords)
        if is_relevant:
            relevant += 1
        snippets.append(r.content[:60] + "..." if len(r.content) > 60 else r.content)
    
    return relevant, len(results), latency_ms, snippets


def run_comparison():
    """Run comparison between baseline and LLM expansion."""
    print("=" * 80)
    print("LLM QUERY EXPANSION COMPARISON TEST")
    print("=" * 80)
    
    # Initialize index
    print("\nInitializing index...")
    index = UnifiedMemoryIndex()
    
    # Baseline test (LLM expansion disabled)
    print("\n" + "=" * 80)
    print("BASELINE: Rule-based expansion only (_QUERY_EXPANSIONS)")
    print("=" * 80)
    
    baseline_results = []
    baseline_latencies = []
    
    for query, keywords in TEST_QUERIES:
        relevant, total, latency, snippets = measure_retrieval(
            index, query, keywords, use_llm=False
        )
        baseline_results.append((query, relevant, total))
        baseline_latencies.append(latency)
        precision = relevant / total * 100 if total > 0 else 0
        print(f"\n[{precision:.0f}%] \"{query}\" -> {relevant}/{total} relevant ({latency:.0f}ms)")
        for snippet in snippets[:2]:
            kw_match = any(kw.lower() in snippet.lower() for kw in keywords)
            marker = "[OK]" if kw_match else "[MISS]"
            print(f"  {marker} {snippet}")
    
    baseline_total_relevant = sum(r[1] for r in baseline_results)
    baseline_total_count = sum(r[2] for r in baseline_results)
    baseline_precision = baseline_total_relevant / baseline_total_count * 100 if baseline_total_count > 0 else 0
    baseline_avg_latency = sum(baseline_latencies) / len(baseline_latencies)
    
    print(f"\nBASELINE SUMMARY:")
    print(f"  Precision: {baseline_total_relevant}/{baseline_total_count} ({baseline_precision:.1f}%)")
    print(f"  Avg Latency: {baseline_avg_latency:.0f}ms")
    
    # LLM expansion test
    print("\n" + "=" * 80)
    print("LLM EXPANSION: GLM 4.6 semantic alternatives + rule-based")
    print("=" * 80)
    
    llm_results = []
    llm_latencies = []
    
    for query, keywords in TEST_QUERIES:
        relevant, total, latency, snippets = measure_retrieval(
            index, query, keywords, use_llm=True
        )
        llm_results.append((query, relevant, total))
        llm_latencies.append(latency)
        precision = relevant / total * 100 if total > 0 else 0
        print(f"\n[{precision:.0f}%] \"{query}\" -> {relevant}/{total} relevant ({latency:.0f}ms)")
        for snippet in snippets[:2]:
            kw_match = any(kw.lower() in snippet.lower() for kw in keywords)
            marker = "[OK]" if kw_match else "[MISS]"
            print(f"  {marker} {snippet}")
    
    llm_total_relevant = sum(r[1] for r in llm_results)
    llm_total_count = sum(r[2] for r in llm_results)
    llm_precision = llm_total_relevant / llm_total_count * 100 if llm_total_count > 0 else 0
    llm_avg_latency = sum(llm_latencies) / len(llm_latencies)
    
    print(f"\nLLM EXPANSION SUMMARY:")
    print(f"  Precision: {llm_total_relevant}/{llm_total_count} ({llm_precision:.1f}%)")
    print(f"  Avg Latency: {llm_avg_latency:.0f}ms")
    
    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    precision_diff = llm_precision - baseline_precision
    latency_diff = llm_avg_latency - baseline_avg_latency
    
    print(f"\n{'Metric':<20} {'Baseline':<15} {'LLM Expansion':<15} {'Difference':<15}")
    print("-" * 65)
    print(f"{'Precision':<20} {baseline_precision:.1f}%{'':<10} {llm_precision:.1f}%{'':<10} {precision_diff:+.1f}%")
    print(f"{'Avg Latency':<20} {baseline_avg_latency:.0f}ms{'':<10} {llm_avg_latency:.0f}ms{'':<10} {latency_diff:+.0f}ms")
    
    # Per-query comparison
    print("\n" + "-" * 65)
    print("PER-QUERY COMPARISON:")
    print("-" * 65)
    
    improved = 0
    degraded = 0
    unchanged = 0
    
    for i, (query, keywords) in enumerate(TEST_QUERIES):
        baseline_rel = baseline_results[i][1]
        llm_rel = llm_results[i][1]
        diff = llm_rel - baseline_rel
        
        if diff > 0:
            status = "IMPROVED"
            improved += 1
        elif diff < 0:
            status = "WORSE"
            degraded += 1
        else:
            status = "SAME"
            unchanged += 1
        
        print(f"  \"{query}\": {baseline_rel} -> {llm_rel} ({status})")
    
    print(f"\nSUMMARY: {improved} improved, {degraded} degraded, {unchanged} unchanged")
    
    # Final verdict
    print("\n" + "=" * 80)
    if precision_diff > 0:
        print(f"[PASS] LLM expansion IMPROVES precision by {precision_diff:.1f}%")
        print(f"       Trade-off: {latency_diff:.0f}ms additional latency per query")
    elif precision_diff < 0:
        print(f"[FAIL] LLM expansion DECREASES precision by {abs(precision_diff):.1f}%")
    else:
        print(f"[NEUTRAL] LLM expansion has NO effect on precision")
        print(f"         Additional latency: {latency_diff:.0f}ms (not worth it)")
    
    return baseline_precision, llm_precision, precision_diff


if __name__ == "__main__":
    run_comparison()
