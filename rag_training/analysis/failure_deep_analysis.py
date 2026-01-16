"""
Deep Failure Analysis

Analyze the complete misses and low-rank results to understand root causes
and identify targeted optimizations.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple


def load_results(path: Path) -> Dict:
    """Load evaluation results."""
    with open(path) as f:
        return json.load(f)


def load_test_suite(path: Path) -> Dict:
    """Load test suite with queries."""
    with open(path) as f:
        return json.load(f)


def analyze_failures(results_path: Path, test_suite_path: Path):
    """Perform deep analysis of retrieval failures."""
    results = load_results(results_path)
    test_suite = load_test_suite(test_suite_path)

    # Build memory lookup
    memories = {tc["memory_id"]: tc for tc in test_suite["test_cases"]}

    print("=" * 80)
    print("DEEP FAILURE ANALYSIS")
    print("=" * 80)

    # Categorize failures
    complete_misses = []  # Not in top-10 at all
    low_rank = []  # In top-10 but not top-5
    false_positive_issues = []  # In results but with FPs above

    # Memory-level analysis
    memory_miss_counts = defaultdict(int)
    memory_total_queries = defaultdict(int)
    query_category_misses = defaultdict(lambda: {"miss": 0, "total": 0})
    difficulty_misses = defaultdict(lambda: {"miss": 0, "total": 0})

    for mem_result in results["memory_results"]:
        memory_id = mem_result["memory_id"]
        memory_info = memories.get(memory_id, {})

        # Find queries that failed
        for tc in test_suite["test_cases"]:
            if tc["memory_id"] == memory_id:
                for i, q in enumerate(tc.get("generated_queries", [])):
                    memory_total_queries[memory_id] += 1
                    query_category_misses[q["category"]]["total"] += 1
                    difficulty_misses[q["difficulty"]]["total"] += 1

                    # Estimate if this query missed based on memory recall
                    # (simplified - we'd need full query-level results for exact)

    # Analyze by memory category
    category_performance = defaultdict(lambda: {"miss": 0, "total": 0, "memories": []})
    for mem_result in results["memory_results"]:
        cat = mem_result["category"]
        category_performance[cat]["total"] += mem_result["total_queries"]
        miss_count = int(mem_result["total_queries"] * (1 - mem_result["recall_at_10"]))
        category_performance[cat]["miss"] += miss_count
        if mem_result["recall_at_1"] < 0.5:
            category_performance[cat]["memories"].append({
                "id": mem_result["memory_id"],
                "r@1": mem_result["recall_at_1"],
                "r@5": mem_result["recall_at_5"]
            })

    print("\n" + "=" * 60)
    print("PERFORMANCE BY MEMORY CATEGORY")
    print("=" * 60)
    for cat, data in sorted(category_performance.items(), key=lambda x: x[1]["miss"], reverse=True):
        miss_rate = data["miss"] / data["total"] * 100 if data["total"] > 0 else 0
        print(f"\n{cat}:")
        print(f"  Total queries: {data['total']}")
        print(f"  Miss rate @10: {miss_rate:.1f}%")
        if data["memories"]:
            print(f"  Struggling memories (<50% R@1):")
            for mem in data["memories"][:3]:
                print(f"    - ID {mem['id']}: R@1={mem['r@1']:.0%}, R@5={mem['r@5']:.0%}")

    # Find memories with highest miss rates
    print("\n" + "=" * 60)
    print("MEMORIES WITH HIGHEST MISS RATES")
    print("=" * 60)
    sorted_by_recall = sorted(results["memory_results"], key=lambda x: x["recall_at_10"])
    for mem in sorted_by_recall[:10]:
        memory_info = memories.get(mem["memory_id"], {})
        print(f"\nMemory {mem['memory_id']} ({mem['category']})")
        print(f"  R@1={mem['recall_at_1']:.0%}, R@5={mem['recall_at_5']:.0%}, R@10={mem['recall_at_10']:.0%}")
        print(f"  Content: {memory_info.get('content', 'N/A')[:80]}...")

    # Analyze semantic similarity issues (memories with similar content)
    print("\n" + "=" * 60)
    print("SEMANTIC SIMILARITY ANALYSIS")
    print("=" * 60)

    # Group memories by keywords
    keyword_groups = defaultdict(list)
    for tc in test_suite["test_cases"]:
        content_lower = tc["content"].lower()
        keywords = []
        if "validat" in content_lower or "sanitiz" in content_lower:
            keywords.append("validation")
        if "input" in content_lower:
            keywords.append("input")
        if "config" in content_lower:
            keywords.append("config")
        if "test" in content_lower:
            keywords.append("test")
        if "api" in content_lower:
            keywords.append("api")
        if "error" in content_lower or "exception" in content_lower:
            keywords.append("error")

        for kw in keywords:
            keyword_groups[kw].append({
                "id": tc["memory_id"],
                "content": tc["content"][:100],
                "category": tc["category"]
            })

    print("\nPotentially confusing memory groups (similar keywords):")
    for kw, mems in sorted(keyword_groups.items(), key=lambda x: len(x[1]), reverse=True):
        if len(mems) > 3:
            print(f"\n'{kw}' appears in {len(mems)} memories:")
            for mem in mems[:5]:
                print(f"  - [{mem['category']}] {mem['content'][:60]}...")

    # Difficulty analysis
    print("\n" + "=" * 60)
    print("PERFORMANCE BY DIFFICULTY")
    print("=" * 60)
    for diff, data in results["results_by_difficulty"].items():
        print(f"\n{diff.upper()}:")
        print(f"  Total: {data['total']}")
        print(f"  R@1: {data['recall_at_1']:.1%}")
        print(f"  R@5: {data['recall_at_5']:.1%}")
        print(f"  MRR: {data['mrr']:.3f}")

    # Query category analysis
    print("\n" + "=" * 60)
    print("PERFORMANCE BY QUERY CATEGORY")
    print("=" * 60)
    for cat, data in sorted(results["results_by_query_category"].items(),
                            key=lambda x: x[1]["recall_at_1"]):
        print(f"\n{cat}:")
        print(f"  Total: {data['total']}")
        print(f"  R@1: {data['recall_at_1']:.1%}")
        print(f"  R@5: {data['recall_at_5']:.1%}")
        print(f"  MRR: {data['mrr']:.3f}")

    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS FOR NEXT OPTIMIZATIONS")
    print("=" * 60)

    print("""
1. DUPLICATE/SIMILAR MEMORY HANDLING
   - Multiple validation/input memories with similar content
   - Consider: Deduplication or clustering similar memories
   - Consider: Memory-level metadata filtering

2. SEMANTIC GAP
   - Short queries like "validate input" match multiple memories
   - Consider: HyDE with LLM (generate hypothetical documents)
   - Consider: Query-specific context injection

3. EMBEDDING QUALITY
   - Hard and medium queries performing poorly
   - Consider: Fine-tune embeddings on query-memory pairs
   - Consider: Use stronger embedding model (e5-large, bge-large)

4. CATEGORY-SPECIFIC TUNING
   - Security and validation memories have low recall
   - Consider: Category-aware retrieval or filtering

5. RE-RANKER IMPROVEMENT
   - Current model: ms-marco-MiniLM-L-6-v2 (small)
   - Consider: Larger cross-encoder (ms-marco-MiniLM-L-12-v2)
   - Consider: Domain-specific re-ranker

6. SCORE THRESHOLDING
   - Complete misses might benefit from score-based filtering
   - Consider: Confidence-based fallback strategies
""")

    return results


def main():
    base_path = Path(__file__).parent.parent
    results_path = base_path / "optimization_results" / "v2_query_expansion.json"
    test_suite_path = base_path / "test_suite" / "enhanced_test_suite.json"

    analyze_failures(results_path, test_suite_path)


if __name__ == "__main__":
    main()
