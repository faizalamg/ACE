#!/usr/bin/env python3
"""
FINAL RELEVANCY ASSESSMENT

Summary of structured enhancement impact on ACE retrieval system.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ace.unified_memory import UnifiedMemoryIndex
from ace.structured_enhancer import StructuredQueryEnhancer


def main():
    print("=" * 70)
    print("FINAL RELEVANCY ASSESSMENT - ACE Retrieval System")
    print("=" * 70)
    
    idx = UnifiedMemoryIndex()
    enhancer = StructuredQueryEnhancer()
    
    # Test suite covering different query types
    test_cases = [
        # Clear technical queries
        ("how to handle errors", ["error", "log", "handle", "context", "trace"]),
        ("configuration best practices", ["config", "env", "secret", "api"]),
        ("testing guidance", ["test", "ci", "cd", "automation", "regression"]),
        ("performance optimization", ["performance", "optimize", "benchmark", "cache"]),
        ("debugging tips", ["debug", "log", "error", "trace", "context"]),
        ("deployment workflow", ["deploy", "ci", "cd", "pipeline", "build"]),
        
        # Vague queries
        ("something is broken", ["error", "fix", "debug", "issue", "problem"]),
        ("make it faster", ["performance", "optimize", "cache", "speed"]),
        ("help with config", ["config", "env", "setting", "variable"]),
    ]
    
    enhanced_scores = []
    baseline_scores = []
    
    print("\n| Query | Enhanced | Baseline | Δ | Result |")
    print("|-------|----------|----------|---|--------|")
    
    for query, relevant_terms in test_cases:
        # With enhancement
        enhanced_results = idx.retrieve(query, limit=3, use_structured_enhancement=True)
        enhanced_relevant = 0
        for r in enhanced_results:
            content = r.content.lower()
            if any(t in content for t in relevant_terms):
                enhanced_relevant += 1
        enhanced_precision = enhanced_relevant / len(enhanced_results) if enhanced_results else 0
        enhanced_scores.append(enhanced_precision)
        
        # Without enhancement
        baseline_results = idx.retrieve(query, limit=3, use_structured_enhancement=False)
        baseline_relevant = 0
        for r in baseline_results:
            content = r.content.lower()
            if any(t in content for t in relevant_terms):
                baseline_relevant += 1
        baseline_precision = baseline_relevant / len(baseline_results) if baseline_results else 0
        baseline_scores.append(baseline_precision)
        
        delta = enhanced_precision - baseline_precision
        result = "↑" if delta > 0.05 else "↓" if delta < -0.05 else "="
        
        print(f"| {query[:25]:<25} | {enhanced_precision:>7.0%} | {baseline_precision:>7.0%} | {delta:>+.0%} | {result:^6} |")
    
    # Summary
    avg_enhanced = sum(enhanced_scores) / len(enhanced_scores)
    avg_baseline = sum(baseline_scores) / len(baseline_scores)
    
    print("\n" + "-" * 70)
    print(f"Average Enhanced:  {avg_enhanced:.1%}")
    print(f"Average Baseline:  {avg_baseline:.1%}")
    print(f"Improvement:       {avg_enhanced - avg_baseline:+.1%}")
    print("-" * 70)
    
    # Final verdict
    if avg_enhanced >= 0.8:
        grade = "A - EXCELLENT"
    elif avg_enhanced >= 0.7:
        grade = "B - GOOD"
    elif avg_enhanced >= 0.6:
        grade = "C - ACCEPTABLE"
    else:
        grade = "D - NEEDS WORK"
    
    print(f"\nGRADE: {grade}")
    print(f"Enhancement provides {avg_enhanced - avg_baseline:+.0%} improvement over baseline\n")


if __name__ == "__main__":
    main()
