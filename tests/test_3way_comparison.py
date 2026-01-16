#!/usr/bin/env python
"""
3-Way Comparison Test: Current vs Semantic Scoring vs Structured Enhancement

Compares three approaches to query enhancement:
1. BASELINE: Current rule-based _QUERY_EXPANSIONS + optional LLM expansion
2. SEMANTIC: Use embedding similarity to score result relevance (no enhancement)
3. STRUCTURED: .enhancedprompt.md methodology with intent classification + domain expansion

Measures:
- Keyword Precision (legacy metric)
- Semantic Similarity (embedding-based relevance)
- Result Diversity (unique content retrieved)
"""

import os
import sys
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Disable LLM expansion for controlled comparison
os.environ["ACE_LLM_EXPANSION"] = "false"

# Import directly to avoid slow ace/__init__.py
from ace.unified_memory import UnifiedMemoryIndex
from ace.semantic_scorer import SemanticSimilarityScorer
from ace.structured_enhancer import StructuredQueryEnhancer


@dataclass
class ComparisonResult:
    """Results for a single query across all methods."""
    query: str
    # Baseline (current system)
    baseline_keyword_precision: float
    baseline_semantic_similarity: float
    baseline_results: List[str]
    # Structured enhancement
    enhanced_query: str
    enhanced_keyword_precision: float
    enhanced_semantic_similarity: float
    enhanced_results: List[str]


# Test queries with expected keywords
TEST_QUERIES = [
    ("fix database errors", ["database", "error", "query", "SQL", "fix"]),
    ("slow API response", ["API", "performance", "slow", "latency", "response"]),
    ("security vulnerability", ["security", "vulnerability", "auth", "inject"]),
    ("broken tests", ["test", "fail", "broken", "fix"]),
    ("memory leak", ["memory", "leak", "heap", "garbage"]),
    ("how to cache", ["cache", "caching", "performance", "store"]),
    ("debug production", ["debug", "production", "log", "trace"]),
    ("rate limiting", ["rate", "limit", "throttle", "API"]),
    ("microservices communication", ["microservice", "communication", "message", "API"]),
    ("authentication flow", ["auth", "flow", "token", "login"]),
]


def measure_keyword_precision(results: List[str], expected_keywords: List[str]) -> float:
    """Calculate keyword-based precision."""
    if not results:
        return 0.0
    
    relevant = 0
    for result in results:
        content_lower = result.lower()
        if any(kw.lower() in content_lower for kw in expected_keywords):
            relevant += 1
    
    return relevant / len(results)


def run_comparison():
    """Run the 3-way comparison."""
    print("=" * 80)
    print("3-WAY COMPARISON: BASELINE vs SEMANTIC vs STRUCTURED ENHANCEMENT")
    print("=" * 80)
    
    # Initialize components
    print("\nInitializing components...")
    index = UnifiedMemoryIndex()
    scorer = SemanticSimilarityScorer()
    enhancer = StructuredQueryEnhancer()
    
    results = []
    
    # Collect results for each query
    print("\nRunning queries...\n")
    print("-" * 80)
    
    for query, keywords in TEST_QUERIES:
        print(f"Query: \"{query}\"")
        
        # --- BASELINE ---
        baseline_results = index.retrieve(query, limit=5, use_llm_expansion=False)
        baseline_contents = [r.content for r in baseline_results]
        baseline_kw_prec = measure_keyword_precision(baseline_contents, keywords)
        baseline_sem_sim, _, _ = scorer.score_results(
            query, baseline_contents, threshold=0.5
        )
        
        # --- STRUCTURED ENHANCEMENT ---
        enhanced = enhancer.enhance(query)
        enhanced_results = index.retrieve(enhanced.enhanced_query, limit=5, use_llm_expansion=False)
        enhanced_contents = [r.content for r in enhanced_results]
        enhanced_kw_prec = measure_keyword_precision(enhanced_contents, keywords)
        enhanced_sem_sim, _, _ = scorer.score_results(
            query, enhanced_contents, threshold=0.5  # Score against ORIGINAL query
        )
        
        # Store results
        result = ComparisonResult(
            query=query,
            baseline_keyword_precision=baseline_kw_prec,
            baseline_semantic_similarity=baseline_sem_sim,
            baseline_results=baseline_contents[:2],
            enhanced_query=enhanced.enhanced_query,
            enhanced_keyword_precision=enhanced_kw_prec,
            enhanced_semantic_similarity=enhanced_sem_sim,
            enhanced_results=enhanced_contents[:2],
        )
        results.append(result)
        
        # Print per-query results
        kw_diff = enhanced_kw_prec - baseline_kw_prec
        sem_diff = enhanced_sem_sim - baseline_sem_sim
        kw_status = "+" if kw_diff > 0 else ("=" if kw_diff == 0 else "-")
        sem_status = "+" if sem_diff > 0 else ("=" if sem_diff == 0 else "-")
        
        print(f"  Baseline:   KW={baseline_kw_prec:.0%} SEM={baseline_sem_sim:.3f}")
        print(f"  Enhanced:   KW={enhanced_kw_prec:.0%} SEM={enhanced_sem_sim:.3f}")
        print(f"  Difference: KW={kw_status}{abs(kw_diff):.0%} SEM={sem_status}{abs(sem_diff):.3f}")
        print(f"  Enhanced Q: \"{enhanced.enhanced_query[:70]}...\"")
        print()
    
    # Summary statistics
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Calculate averages
    avg_baseline_kw = sum(r.baseline_keyword_precision for r in results) / len(results)
    avg_baseline_sem = sum(r.baseline_semantic_similarity for r in results) / len(results)
    avg_enhanced_kw = sum(r.enhanced_keyword_precision for r in results) / len(results)
    avg_enhanced_sem = sum(r.enhanced_semantic_similarity for r in results) / len(results)
    
    print(f"\n{'Metric':<25} {'Baseline':<15} {'Structured':<15} {'Difference':<15}")
    print("-" * 70)
    print(f"{'Keyword Precision':<25} {avg_baseline_kw:.1%}{'':<10} {avg_enhanced_kw:.1%}{'':<10} {avg_enhanced_kw - avg_baseline_kw:+.1%}")
    print(f"{'Semantic Similarity':<25} {avg_baseline_sem:.3f}{'':<10} {avg_enhanced_sem:.3f}{'':<10} {avg_enhanced_sem - avg_baseline_sem:+.3f}")
    
    # Count improvements/degradations
    kw_improved = sum(1 for r in results if r.enhanced_keyword_precision > r.baseline_keyword_precision)
    kw_degraded = sum(1 for r in results if r.enhanced_keyword_precision < r.baseline_keyword_precision)
    sem_improved = sum(1 for r in results if r.enhanced_semantic_similarity > r.baseline_semantic_similarity)
    sem_degraded = sum(1 for r in results if r.enhanced_semantic_similarity < r.baseline_semantic_similarity)
    
    print(f"\n{'Keyword':<25} {kw_improved} improved, {kw_degraded} degraded, {len(results) - kw_improved - kw_degraded} same")
    print(f"{'Semantic':<25} {sem_improved} improved, {sem_degraded} degraded, {len(results) - sem_improved - sem_degraded} same")
    
    # Result diversity (how many unique results between baseline and enhanced)
    total_unique = 0
    for r in results:
        baseline_set = set(r.baseline_results)
        enhanced_set = set(r.enhanced_results)
        unique = len(baseline_set ^ enhanced_set)  # Symmetric difference
        total_unique += unique
    
    avg_diversity = total_unique / len(results) / 4  # Normalize by max possible (4 results)
    print(f"\n{'Result Diversity':<25} {avg_diversity:.0%} of results are different between methods")
    
    # Final verdict
    print("\n" + "=" * 80)
    kw_change = avg_enhanced_kw - avg_baseline_kw
    sem_change = avg_enhanced_sem - avg_baseline_sem
    
    if kw_change > 0.05 and sem_change > 0.01:
        print("[PASS] Structured enhancement IMPROVES both keyword and semantic relevance")
    elif kw_change > 0.05 or sem_change > 0.01:
        print("[PARTIAL] Structured enhancement improves one metric")
        if kw_change > 0.05:
            print(f"  - Keyword precision: {kw_change:+.1%}")
        if sem_change > 0.01:
            print(f"  - Semantic similarity: {sem_change:+.3f}")
    elif kw_change < -0.05 or sem_change < -0.01:
        print("[FAIL] Structured enhancement DEGRADES retrieval quality")
    else:
        print("[NEUTRAL] Structured enhancement has minimal effect")
    
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    run_comparison()
