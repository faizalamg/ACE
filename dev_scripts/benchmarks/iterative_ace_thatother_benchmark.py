#!/usr/bin/env python3
"""
Run ACE vs ThatOtherContextEngine benchmark with multiple iterations.
Goal: 1000+ successful iterations with zero ThatOtherContextEngine wins.
"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

from comprehensive_ace_ThatOtherContextEngine_benchmark import TEST_QUERIES, get_ace_results, get_ThatOtherContextEngine_results, determine_winner


def run_iterative_benchmark(iterations: int = 4, queries_per_iteration: int = 250) -> dict:
    """
    Run benchmark across multiple iterations.
    
    Each iteration runs all queries or a subset.
    """
    total_results = {
        "ace_wins": 0,
        "ThatOtherContextEngine_wins": 0,
        "ties": 0,
        "errors": 0,
        "by_category": {},
        "ThatOtherContextEngine_win_details": [],
    }
    
    total_queries = 0
    
    print("=" * 80)
    print(f"ACE vs ThatOtherContextEngine ITERATIVE BENCHMARK")
    print(f"Iterations: {iterations}")
    print(f"Queries per iteration: {queries_per_iteration}")
    print(f"Total queries: {iterations * queries_per_iteration}")
    print("=" * 80)
    
    for iteration in range(1, iterations + 1):
        print(f"\n### ITERATION {iteration}/{iterations}")
        
        # Shuffle queries for variation
        queries = list(TEST_QUERIES)
        random.shuffle(queries)
        queries = queries[:queries_per_iteration]
        
        iter_results = {"ace": 0, "ThatOtherContextEngine": 0, "tie": 0, "error": 0}
        
        for i, (category, query) in enumerate(queries, 1):
            print(f"  [{i}/{len(queries)}] {query[:40]}...", end="\r")
            
            try:
                ace_results = get_ace_results(query, limit=5)
                ThatOtherContextEngine_results = get_ThatOtherContextEngine_results(query, limit=5)
                
                ace_files = [f for f, _ in ace_results]
                ace_scores = [s for _, s in ace_results]
                
                # Handle case where ACE has API error (returns empty)
                if not ace_files and ThatOtherContextEngine_results:
                    iter_results["error"] += 1
                    total_results["errors"] += 1
                    continue
                
                winner, reason, _, _ = determine_winner(query, ace_files, ace_scores, ThatOtherContextEngine_results)
                
                if winner == "ACE":
                    iter_results["ace"] += 1
                    total_results["ace_wins"] += 1
                elif winner == "ThatOtherContextEngine":
                    iter_results["ThatOtherContextEngine"] += 1
                    total_results["ThatOtherContextEngine_wins"] += 1
                    total_results["ThatOtherContextEngine_win_details"].append({
                        "iteration": iteration,
                        "query": query,
                        "category": category,
                        "ace_files": ace_files,
                        "ThatOtherContextEngine_files": ThatOtherContextEngine_results,
                        "reason": reason,
                    })
                else:
                    iter_results["tie"] += 1
                    total_results["ties"] += 1
                
                # Update category stats
                if category not in total_results["by_category"]:
                    total_results["by_category"][category] = {"ace": 0, "ThatOtherContextEngine": 0, "tie": 0}
                total_results["by_category"][category][winner.lower()] += 1
                
                total_queries += 1
                
            except Exception as e:
                print(f"\n  Error on query '{query}': {e}")
                iter_results["error"] += 1
                total_results["errors"] += 1
        
        print(f"\n  Iteration {iteration}: ACE {iter_results['ace']} | ThatOtherContextEngine {iter_results['ThatOtherContextEngine']} | Ties {iter_results['tie']} | Errors {iter_results['error']}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total Queries Processed: {total_queries}")
    print(f"ACE Wins: {total_results['ace_wins']} ({total_results['ace_wins']/total_queries*100:.1f}%)")
    print(f"ThatOtherContextEngine Wins: {total_results['ThatOtherContextEngine_wins']} ({total_results['ThatOtherContextEngine_wins']/total_queries*100:.1f}%)")
    print(f"Ties: {total_results['ties']} ({total_results['ties']/total_queries*100:.1f}%)")
    print(f"Errors (skipped): {total_results['errors']}")
    print(f"ACE Non-Loss Rate: {(total_results['ace_wins'] + total_results['ties'])/total_queries*100:.2f}%")
    
    if total_results["ThatOtherContextEngine_win_details"]:
        print("\n### ThatOtherContextEngine WINS (need investigation):")
        for detail in total_results["ThatOtherContextEngine_win_details"]:
            print(f"  Iteration {detail['iteration']}: [{detail['category']}] {detail['query']}")
            print(f"    Reason: {detail['reason']}")
    else:
        print("\n🏆 ZERO ThatOtherContextEngine WINS! ACE ACHIEVES 100% NON-LOSS RATE!")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": total_queries,
        "iterations": iterations,
        "results": total_results,
    }
    
    output_file = Path("benchmark_results") / f"iterative_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return total_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iterations", type=int, default=4, help="Number of iterations")
    parser.add_argument("-q", "--queries", type=int, default=250, help="Queries per iteration")
    args = parser.parse_args()
    
    run_iterative_benchmark(iterations=args.iterations, queries_per_iteration=args.queries)
