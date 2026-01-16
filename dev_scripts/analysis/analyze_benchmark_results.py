#!/usr/bin/env python3
"""Analyze existing benchmark results to compare ACE vs ThatOtherContextEngine.

Since ThatOtherContextEngine credits are exhausted, we'll use the stored benchmark results
to analyze where ACE wins and where it loses.
"""
import json
import sys
sys.path.insert(0, ".")

def analyze_benchmark(filepath):
    """Analyze a benchmark result file."""
    with open(filepath) as f:
        data = json.load(f)
    
    results = data.get("results", [])
    
    ace_wins = []
    ThatOtherContextEngine_wins = []
    ties = []
    
    for r in results:
        winner = r.get("winner", "tie").upper()
        if winner == "ACE":
            ace_wins.append(r)
        elif winner == "ThatOtherContextEngine":
            ThatOtherContextEngine_wins.append(r)
        else:
            ties.append(r)
    
    return {
        "total": len(results),
        "ace_wins": ace_wins,
        "ThatOtherContextEngine_wins": ThatOtherContextEngine_wins,
        "ties": ties,
    }


def categorize_ThatOtherContextEngine_wins(ThatOtherContextEngine_wins):
    """Categorize ThatOtherContextEngine wins by root cause."""
    categories = {
        "test_files": [],
        "json_files": [],
        "examples": [],
        "other": [],
    }
    
    for result in ThatOtherContextEngine_wins:
        query = result.get("query", "")
        ace_files = result.get("ace_files", [])
        
        # Check if test files dominated ACE's top 3 results
        ace_test_files = sum(1 for f in ace_files[:3] 
                            if "test" in f.lower())
        
        ace_json_files = sum(1 for f in ace_files[:3] 
                            if f.endswith(".json"))
        
        ace_example_files = sum(1 for f in ace_files[:3] 
                               if "example" in f.lower())
        
        if ace_test_files >= 2:
            categories["test_files"].append(result)
        elif ace_json_files >= 1:
            categories["json_files"].append(result)
        elif ace_example_files >= 1:
            categories["examples"].append(result)
        else:
            categories["other"].append(result)
    
    return categories


def main():
    print("=" * 60)
    print("ACE vs ThatOtherContextEngine Benchmark Analysis")
    print("=" * 60)
    
    # Analyze both benchmark files
    benchmarks = [
        "benchmark_results/ace_ThatOtherContextEngine_1000_20260106_122712.json",
        "benchmark_results/ace_ThatOtherContextEngine_1000_20260106_113413.json",
    ]
    
    all_ThatOtherContextEngine_wins = []
    
    for filepath in benchmarks:
        print(f"\n### {filepath.split('/')[-1]}")
        
        try:
            analysis = analyze_benchmark(filepath)
        except FileNotFoundError:
            print(f"  File not found: {filepath}")
            continue
        except json.JSONDecodeError:
            print(f"  Invalid JSON: {filepath}")
            continue
        
        total = analysis["total"]
        ace_count = len(analysis["ace_wins"])
        ThatOtherContextEngine_count = len(analysis["ThatOtherContextEngine_wins"])
        tie_count = len(analysis["ties"])
        
        print(f"  Total queries: {total}")
        print(f"  ACE wins: {ace_count} ({100*ace_count/total:.1f}%)")
        print(f"  ThatOtherContextEngine wins: {ThatOtherContextEngine_count} ({100*ThatOtherContextEngine_count/total:.1f}%)")
        print(f"  Ties: {tie_count} ({100*tie_count/total:.1f}%)")
        
        all_ThatOtherContextEngine_wins.extend(analysis["ThatOtherContextEngine_wins"])
    
    # Categorize all ThatOtherContextEngine wins
    if all_ThatOtherContextEngine_wins:
        print("\n" + "=" * 60)
        print("Root Cause Analysis (ThatOtherContextEngine Wins)")
        print("=" * 60)
        
        categories = categorize_ThatOtherContextEngine_wins(all_ThatOtherContextEngine_wins)
        
        print(f"\nTest files dominated: {len(categories['test_files'])}")
        print(f"JSON files in results: {len(categories['json_files'])}")
        print(f"Example files ranked high: {len(categories['examples'])}")
        print(f"Other reasons: {len(categories['other'])}")
        
        # Show sample queries from each category
        for cat_name, cat_results in categories.items():
            if cat_results:
                print(f"\n### {cat_name.replace('_', ' ').title()} ({len(cat_results)} cases)")
                for r in cat_results[:3]:  # Show up to 3 examples
                    print(f"  - Query: {r.get('query', 'N/A')[:60]}...")
    
    # Now test ACE with CodeRetrieval directly
    print("\n" + "=" * 60)
    print("Validating ACE CodeRetrieval (with boosting)")
    print("=" * 60)
    
    from ace.code_retrieval import CodeRetrieval
    retrieval = CodeRetrieval()
    
    # Test queries where ThatOtherContextEngine won
    test_queries = [
        "semantic search",
        "embedding model voyage",
        "code indexer batch",
        "memory deduplication",
        "BM25 sparse vector",
    ]
    
    print("\nTest queries with CodeRetrieval (applies boosting):")
    for query in test_queries:
        results = retrieval.search(query, limit=5)
        print(f"\n  Query: '{query}'")
        for i, r in enumerate(results[:3]):
            fpath = r.get("file_path", "N/A")
            score = r.get("score", 0)
            is_test = "test" in fpath.lower()
            marker = " [TEST]" if is_test else ""
            print(f"    {i+1}. {fpath} ({score:.3f}){marker}")


if __name__ == "__main__":
    main()
