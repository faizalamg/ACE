"""
Test Query Expansion Improvement on Failing Architecture Query

This script isolates the query expansion optimization and measures its impact
on the specific failing query: "how is our system currently wired? qdrant option 2 with no local json playbook?"

Expected improvement: Expansion should add architectural synonyms like:
- "wired" -> "configured", "setup", "architecture"
- "system" -> "storage", "memory", "vector"

Baseline Precision: 33.3% (architecture queries)
Target: Measure precision delta with query expansion
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import query expansion from optimizations
from rag_training.optimizations.v2_query_expansion import QueryExpander, QueryExpansionEvaluator


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

FAILING_QUERY = "how is our system currently wired? qdrant option 2 with no local json playbook?"
EXPECTED_KEYWORDS = [
    "qdrant", "vector", "storage", "memory", "json", "playbook",
    "unified", "collection", "architecture", "config", "setup"
]

# Known architecture-related memory IDs from test suite
# (These would need to be extracted from actual test suite)
ARCHITECTURE_MEMORY_IDS = [
    # Placeholder - will extract from test suite in actual test
]


# ============================================================================
# BASELINE SEARCH (NO EXPANSION)
# ============================================================================

def baseline_search(evaluator: QueryExpansionEvaluator, query: str, limit: int = 10) -> Tuple[List[Dict], float]:
    """Execute baseline hybrid search without query expansion."""
    start = time.perf_counter()
    results = evaluator.hybrid_search_single(query, limit)
    latency = (time.perf_counter() - start) * 1000
    return results, latency


# ============================================================================
# EXPANSION ANALYSIS
# ============================================================================

def analyze_expansion(expander: QueryExpander, query: str) -> None:
    """Analyze query expansion results."""
    print(f"\n{'='*80}")
    print("QUERY EXPANSION ANALYSIS")
    print(f"{'='*80}")
    print(f"\nOriginal Query: {query}")

    expanded = expander.expand_query(query, num_expansions=3)

    print(f"\nExpanded Queries ({len(expanded)} total):")
    for i, exp_query in enumerate(expanded):
        print(f"  [{i+1}] {exp_query}")

    print(f"\nKeyword Coverage:")
    for keyword in EXPECTED_KEYWORDS:
        found_in = []
        for i, exp_query in enumerate(expanded):
            if keyword.lower() in exp_query.lower():
                found_in.append(i)

        coverage = len(found_in) > 0
        symbol = "[OK]" if coverage else "[MISSING]"
        locations = f"(queries: {found_in})" if found_in else ""
        print(f"  {symbol} {keyword} {locations}")

    return expanded


# ============================================================================
# PRECISION CALCULATION
# ============================================================================

def calculate_precision(results: List[Dict], expected_ids: List[int]) -> Dict:
    """Calculate precision metrics."""
    retrieved_ids = [r["id"] for r in results]

    # Find ranks of expected IDs
    ranks = []
    for exp_id in expected_ids:
        if exp_id in retrieved_ids:
            ranks.append(retrieved_ids.index(exp_id) + 1)

    # Precision at K
    precision_at_1 = 1.0 if ranks and ranks[0] == 1 else 0.0
    precision_at_3 = len([r for r in ranks if r <= 3]) / min(3, len(expected_ids))
    precision_at_5 = len([r for r in ranks if r <= 5]) / min(5, len(expected_ids))
    precision_at_10 = len([r for r in ranks if r <= 10]) / min(10, len(expected_ids))

    # Mean Reciprocal Rank
    mrr = (1.0 / ranks[0]) if ranks else 0.0

    return {
        "retrieved_count": len(retrieved_ids),
        "expected_count": len(expected_ids),
        "found_count": len(ranks),
        "ranks": ranks,
        "precision_at_1": precision_at_1,
        "precision_at_3": precision_at_3,
        "precision_at_5": precision_at_5,
        "precision_at_10": precision_at_10,
        "mrr": mrr
    }


# ============================================================================
# CONTENT RELEVANCE SCORING
# ============================================================================

def sanitize_for_print(text: str) -> str:
    """Remove or replace non-ASCII characters for Windows console output."""
    return text.encode('ascii', 'replace').decode('ascii')


def score_content_relevance(results: List[Dict], keywords: List[str]) -> List[Dict]:
    """Score retrieved content by keyword relevance."""
    scored = []

    for result in results:
        payload = result.get("payload", {})
        content = payload.get("lesson", "") or payload.get("content", "") or str(payload)
        content_lower = content.lower()

        # Count keyword matches
        keyword_matches = [kw for kw in keywords if kw.lower() in content_lower]
        keyword_score = len(keyword_matches) / len(keywords)

        scored.append({
            "id": result["id"],
            "score": result.get("score", 0),
            "keyword_score": keyword_score,
            "matched_keywords": keyword_matches,
            "content_preview": sanitize_for_print(content[:100])
        })

    return scored


# ============================================================================
# MAIN TEST
# ============================================================================

def main():
    """Run query expansion improvement test."""
    print(f"\n{'='*80}")
    print("QUERY EXPANSION IMPROVEMENT TEST")
    print(f"{'='*80}")
    print(f"\nTarget Query: {FAILING_QUERY}")
    print(f"Expected Keywords: {', '.join(EXPECTED_KEYWORDS)}")

    # Initialize
    expander = QueryExpander()
    evaluator = QueryExpansionEvaluator()

    try:
        # 1. Analyze expansion
        expanded_queries = analyze_expansion(expander, FAILING_QUERY)

        # 2. Baseline search (no expansion)
        print(f"\n{'='*80}")
        print("BASELINE SEARCH (No Expansion)")
        print(f"{'='*80}")
        baseline_results, baseline_latency = baseline_search(evaluator, FAILING_QUERY)
        baseline_scored = score_content_relevance(baseline_results, EXPECTED_KEYWORDS)

        print(f"\nRetrieved: {len(baseline_results)} documents ({baseline_latency:.1f}ms)")
        print(f"\nTop 5 Results (Baseline):")
        for i, item in enumerate(baseline_scored[:5]):
            print(f"  [{i+1}] ID={item['id']}, Score={item['score']:.4f}, Keywords={item['keyword_score']:.1%}")
            print(f"      Matched: {', '.join(item['matched_keywords']) if item['matched_keywords'] else 'none'}")
            print(f"      Preview: {item['content_preview']}")

        # 3. Expanded search
        print(f"\n{'='*80}")
        print("EXPANDED SEARCH (With Query Expansion)")
        print(f"{'='*80}")
        expanded_results, expanded_queries_used, exp_lat, ret_lat, rerank_lat = evaluator.search_with_expansion(FAILING_QUERY)
        total_latency = exp_lat + ret_lat + rerank_lat
        expanded_scored = score_content_relevance(expanded_results, EXPECTED_KEYWORDS)

        print(f"\nRetrieved: {len(expanded_results)} documents ({total_latency:.1f}ms)")
        print(f"  Expansion: {exp_lat:.1f}ms")
        print(f"  Retrieval: {ret_lat:.1f}ms")
        print(f"  Rerank: {rerank_lat:.1f}ms")

        print(f"\nTop 5 Results (Expanded):")
        for i, item in enumerate(expanded_scored[:5]):
            print(f"  [{i+1}] ID={item['id']}, Score={item['score']:.4f}, Keywords={item['keyword_score']:.1%}")
            print(f"      Matched: {', '.join(item['matched_keywords']) if item['matched_keywords'] else 'none'}")
            print(f"      Preview: {item['content_preview']}")

        # 4. Calculate improvement
        print(f"\n{'='*80}")
        print("IMPROVEMENT ANALYSIS")
        print(f"{'='*80}")

        baseline_avg_keyword_score = sum(s["keyword_score"] for s in baseline_scored) / len(baseline_scored) if baseline_scored else 0
        expanded_avg_keyword_score = sum(s["keyword_score"] for s in expanded_scored) / len(expanded_scored) if expanded_scored else 0

        baseline_top3_keyword_score = sum(s["keyword_score"] for s in baseline_scored[:3]) / 3 if len(baseline_scored) >= 3 else 0
        expanded_top3_keyword_score = sum(s["keyword_score"] for s in expanded_scored[:3]) / 3 if len(expanded_scored) >= 3 else 0

        print(f"\nKeyword Coverage (All Results):")
        print(f"  Baseline: {baseline_avg_keyword_score:.1%}")
        print(f"  Expanded: {expanded_avg_keyword_score:.1%}")
        print(f"  Delta: {expanded_avg_keyword_score - baseline_avg_keyword_score:+.1%}")

        print(f"\nKeyword Coverage (Top 3):")
        print(f"  Baseline: {baseline_top3_keyword_score:.1%}")
        print(f"  Expanded: {expanded_top3_keyword_score:.1%}")
        print(f"  Delta: {expanded_top3_keyword_score - baseline_top3_keyword_score:+.1%}")

        print(f"\nLatency Overhead:")
        print(f"  Baseline: {baseline_latency:.1f}ms")
        print(f"  Expanded: {total_latency:.1f}ms")
        print(f"  Overhead: {total_latency - baseline_latency:+.1f}ms ({(total_latency/baseline_latency - 1)*100:+.1f}%)")

        # 5. Result overlap analysis
        baseline_ids = set(r["id"] for r in baseline_results)
        expanded_ids = set(r["id"] for r in expanded_results)

        new_docs = expanded_ids - baseline_ids
        same_docs = expanded_ids & baseline_ids
        lost_docs = baseline_ids - expanded_ids

        print(f"\nResult Set Changes:")
        print(f"  Same: {len(same_docs)} documents")
        print(f"  New: {len(new_docs)} documents (added by expansion)")
        print(f"  Lost: {len(lost_docs)} documents (filtered out)")

        if new_docs:
            print(f"\n  New Documents Added:")
            for doc_id in list(new_docs)[:3]:
                doc = next((r for r in expanded_results if r["id"] == doc_id), None)
                if doc:
                    payload = doc.get("payload", {})
                    content = payload.get("lesson", "") or payload.get("content", "") or str(payload)
                    print(f"    ID={doc_id}: {content[:80]}...")

        # 6. Save results
        output_path = Path(__file__).parent / "query_expansion_test_results.json"
        results_data = {
            "query": FAILING_QUERY,
            "expanded_queries": expanded_queries_used,
            "expected_keywords": EXPECTED_KEYWORDS,
            "baseline": {
                "retrieved_count": len(baseline_results),
                "avg_keyword_score": baseline_avg_keyword_score,
                "top3_keyword_score": baseline_top3_keyword_score,
                "latency_ms": baseline_latency,
                "top_5_ids": [r["id"] for r in baseline_results[:5]]
            },
            "expanded": {
                "retrieved_count": len(expanded_results),
                "avg_keyword_score": expanded_avg_keyword_score,
                "top3_keyword_score": expanded_top3_keyword_score,
                "latency_ms": total_latency,
                "expansion_latency_ms": exp_lat,
                "retrieval_latency_ms": ret_lat,
                "rerank_latency_ms": rerank_lat,
                "top_5_ids": [r["id"] for r in expanded_results[:5]]
            },
            "improvement": {
                "avg_keyword_score_delta": expanded_avg_keyword_score - baseline_avg_keyword_score,
                "top3_keyword_score_delta": expanded_top3_keyword_score - baseline_top3_keyword_score,
                "latency_overhead_ms": total_latency - baseline_latency,
                "new_docs_count": len(new_docs),
                "lost_docs_count": len(lost_docs)
            }
        }

        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*80}\n")

    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
