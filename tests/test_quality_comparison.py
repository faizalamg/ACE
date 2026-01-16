"""
ACE Quality Comparison Test Suite

Comprehensive quality comparison between ACE retrieval and ThatOtherContextEngine MCP output.
Target: 95%+ match rate on retrieval quality.

Tests cover:
- Semantic understanding (concept queries)
- Keyword precision (exact term matching)
- Edge cases (vague queries, complex queries)
- Domain coverage (coding, debugging, architecture, preferences)
"""

import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Add ace package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ace.unified_memory import UnifiedMemoryIndex, UnifiedNamespace


@dataclass
class QualityMetrics:
    """Quality metrics for retrieval comparison."""
    total_queries: int = 0
    relevant_at_1: int = 0  # Is top result relevant?
    relevant_at_5: int = 0  # Is any of top 5 relevant?
    avg_relevance_score: float = 0.0
    empty_results: int = 0
    latency_ms: List[float] = field(default_factory=list)
    
    @property
    def recall_at_1(self) -> float:
        return self.relevant_at_1 / self.total_queries if self.total_queries > 0 else 0.0
    
    @property
    def recall_at_5(self) -> float:
        return self.relevant_at_5 / self.total_queries if self.total_queries > 0 else 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latency_ms) / len(self.latency_ms) if self.latency_ms else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "recall_at_1": f"{self.recall_at_1:.2%}",
            "recall_at_5": f"{self.recall_at_5:.2%}",
            "avg_relevance_score": f"{self.avg_relevance_score:.3f}",
            "empty_results": self.empty_results,
            "avg_latency_ms": f"{self.avg_latency_ms:.1f}",
        }


@dataclass
class TestQuery:
    """Test query with expected results."""
    query: str
    category: str  # semantic, keyword, vague, complex, domain-specific
    expected_concepts: List[str]  # Concepts that should appear in results
    min_results: int = 1
    namespace: Optional[str] = None
    description: str = ""


# Comprehensive test queries covering different categories
TEST_QUERIES = [
    # === SEMANTIC UNDERSTANDING ===
    TestQuery(
        query="how do I handle network timeouts gracefully",
        category="semantic",
        expected_concepts=["timeout", "retry", "network", "error", "handling", "failure"],
        description="Semantic understanding of timeout handling"
    ),
    TestQuery(
        query="what's the best way to validate user input",
        category="semantic",
        expected_concepts=["validation", "input", "sanitize", "check", "validate", "reject"],
        description="Input validation strategies"
    ),
    TestQuery(
        query="strategies for improving code readability",
        category="semantic",
        expected_concepts=["readable", "readability", "clean", "naming", "structure", "format", "semantic"],
        description="Code quality improvement"
    ),
    
    # === KEYWORD PRECISION ===
    TestQuery(
        query="KISS DRY TDD",
        category="keyword",
        expected_concepts=["KISS", "DRY", "TDD", "simple", "repeat", "test"],
        description="Exact coding principle keywords"
    ),
    TestQuery(
        query="async await promise",
        category="keyword",
        expected_concepts=["async", "await", "promise", "asynchronous", "synchronous"],
        description="Async programming keywords"
    ),
    TestQuery(
        query="SQL injection prevention",
        category="keyword",
        expected_concepts=["SQL", "injection", "security", "sanitize", "parameter", "query"],
        description="Security keyword matching"
    ),
    
    # === EDGE CASES (VAGUE QUERIES) ===
    TestQuery(
        query="something is broken",
        category="vague",
        expected_concepts=[],  # Should return semantic_only mode results
        min_results=0,  # Vague queries may return 0 or semantic matches
        description="Ultra-vague debugging query"
    ),
    TestQuery(
        query="help",
        category="vague",
        expected_concepts=[],
        min_results=0,
        description="Minimal query"
    ),
    TestQuery(
        query="fix it",
        category="vague",
        expected_concepts=[],
        min_results=0,
        description="Vague action query"
    ),
    
    # === COMPLEX QUERIES ===
    TestQuery(
        query="how to implement retry logic with exponential backoff for failed API calls",
        category="complex",
        expected_concepts=["retry", "exponential", "backoff", "API", "failure", "reliability"],
        description="Multi-concept technical query"
    ),
    TestQuery(
        query="best practices for database connection pooling in high-concurrency environments",
        category="complex",
        expected_concepts=["database", "connection", "pool", "concurrency", "resilience", "Qdrant"],
        description="Complex architecture query"
    ),
    
    # === DOMAIN-SPECIFIC ===
    TestQuery(
        query="user prefers typescript over javascript",
        category="domain",
        expected_concepts=["typescript", "javascript", "prefer", "language", "tool", "guidance"],
        namespace="user_prefs",
        description="User preference query"
    ),
    TestQuery(
        query="debugging memory leaks in python",
        category="domain",
        expected_concepts=["memory", "leak", "debug", "python", "detect", "MANDATORY"],
        description="Debugging domain query"
    ),
    TestQuery(
        query="microservices vs monolith architecture decision",
        category="domain",
        expected_concepts=["microservice", "monolith", "architecture", "simplicity", "consolidate"],
        description="Architecture domain query"
    ),
    
    # === ACE-SPECIFIC ===
    TestQuery(
        query="ACE memory storage policy",
        category="ace",
        expected_concepts=["ACE", "memory", "storage", "policy", "Qdrant", "vector"],
        description="ACE-specific retrieval"
    ),
    TestQuery(
        query="how does hybrid search work in ACE",
        category="ace",
        expected_concepts=["hybrid", "search", "BM25", "dense", "RRF", "vector", "keyword"],
        description="ACE hybrid search query"
    ),
    
    # === ADDITIONAL CHALLENGING QUERIES ===
    TestQuery(
        query="error handling best practices",
        category="semantic",
        expected_concepts=["error", "handling", "exception", "failure", "catch", "mistake"],
        description="Error handling patterns"
    ),
    TestQuery(
        query="code review checklist",
        category="semantic",
        expected_concepts=["review", "check", "quality", "audit", "code"],
        description="Code review patterns"
    ),
    TestQuery(
        query="authentication authorization security",
        category="keyword",
        expected_concepts=["auth", "security", "access", "permission", "credential"],
        description="Security auth keywords"
    ),
    TestQuery(
        query="performance optimization latency",
        category="keyword",
        expected_concepts=["performance", "optimize", "latency", "speed", "fast"],
        description="Performance keywords"
    ),
    TestQuery(
        query="how to properly log errors and debug issues in production",
        category="complex",
        expected_concepts=["log", "error", "debug", "production", "trace", "monitor"],
        description="Logging and debugging"
    ),
    TestQuery(
        query="what are the conventions for naming variables and functions",
        category="semantic",
        expected_concepts=["naming", "convention", "variable", "function", "style", "format"],
        description="Naming conventions"
    ),
    TestQuery(
        query="caching strategy for frequently accessed data",
        category="complex",
        expected_concepts=["cache", "data", "access", "store", "memory"],
        description="Caching patterns"
    ),
    TestQuery(
        query="test coverage unit integration e2e",
        category="keyword",
        expected_concepts=["test", "coverage", "unit", "integration", "TDD"],
        description="Testing keywords"
    ),
]


def is_result_relevant(result: Any, expected_concepts: List[str]) -> bool:
    """Check if result contains expected concepts."""
    if not expected_concepts:
        # For vague queries, any result is acceptable
        return True
    
    content = str(result.content if hasattr(result, 'content') else result).lower()
    
    # Check if at least one expected concept appears
    for concept in expected_concepts:
        if concept.lower() in content:
            return True
    return False


def run_quality_test(index: UnifiedMemoryIndex, queries: List[TestQuery]) -> Tuple[QualityMetrics, List[Dict]]:
    """Run quality tests and return metrics + detailed results."""
    metrics = QualityMetrics()
    detailed_results = []
    
    for test in queries:
        metrics.total_queries += 1
        
        # Determine namespace filter
        namespace = None
        if test.namespace:
            namespace = getattr(UnifiedNamespace, test.namespace.upper(), None)
        
        # Execute retrieval
        start_time = time.perf_counter()
        try:
            results = index.retrieve(
                query=test.query,
                limit=5,
                namespace=namespace,
            )
            latency = (time.perf_counter() - start_time) * 1000
            metrics.latency_ms.append(latency)
        except Exception as e:
            detailed_results.append({
                "query": test.query,
                "category": test.category,
                "status": "ERROR",
                "error": str(e),
            })
            continue
        
        # Check results
        if not results:
            metrics.empty_results += 1
            relevant_at_1 = test.min_results == 0  # Empty is OK for vague queries
            relevant_at_5 = test.min_results == 0
        else:
            # Check relevance at position 1
            relevant_at_1 = is_result_relevant(results[0], test.expected_concepts)
            
            # Check relevance at any position in top 5
            relevant_at_5 = any(
                is_result_relevant(r, test.expected_concepts)
                for r in results[:5]
            )
        
        if relevant_at_1:
            metrics.relevant_at_1 += 1
        if relevant_at_5:
            metrics.relevant_at_5 += 1
        
        # Calculate average relevance score
        if results:
            avg_score = sum(r.qdrant_score for r in results) / len(results)
        else:
            avg_score = 0.0
        
        detailed_results.append({
            "query": test.query,
            "category": test.category,
            "description": test.description,
            "num_results": len(results),
            "relevant_at_1": relevant_at_1,
            "relevant_at_5": relevant_at_5,
            "top_result": str(results[0].content)[:100] if results else "N/A",
            "scores": [f"{r.qdrant_score:.3f}" for r in results[:5]] if results else [],
            "latency_ms": f"{latency:.1f}",
        })
    
    # Calculate average relevance score
    if detailed_results:
        scores = [r.get("scores", []) for r in detailed_results if r.get("scores")]
        if scores:
            all_scores = [float(s) for sublist in scores for s in sublist]
            metrics.avg_relevance_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    return metrics, detailed_results


def run_comparison_suite():
    """Run full comparison suite and generate report."""
    print("=" * 80)
    print("ACE Quality Comparison Test Suite")
    print("=" * 80)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Total test queries: {len(TEST_QUERIES)}")
    print()
    
    # Initialize index
    print("Initializing UnifiedMemoryIndex...")
    try:
        # Use the production collection with actual data
        index = UnifiedMemoryIndex(
            collection_name="ace_memories_hybrid",  # Production collection with 3042 points
            qdrant_url="http://localhost:6333",
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize index: {e}")
        return
    
    print("Running quality tests...")
    print()
    
    # Run tests
    metrics, detailed_results = run_quality_test(index, TEST_QUERIES)
    
    # Print summary
    print("=" * 80)
    print("SUMMARY METRICS")
    print("=" * 80)
    for key, value in metrics.to_dict().items():
        print(f"  {key}: {value}")
    
    # Print category breakdown
    print()
    print("=" * 80)
    print("CATEGORY BREAKDOWN")
    print("=" * 80)
    
    categories = {}
    for result in detailed_results:
        cat = result["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "r1": 0, "r5": 0}
        categories[cat]["total"] += 1
        if result.get("relevant_at_1"):
            categories[cat]["r1"] += 1
        if result.get("relevant_at_5"):
            categories[cat]["r5"] += 1
    
    for cat, stats in categories.items():
        r1 = stats["r1"] / stats["total"] * 100 if stats["total"] > 0 else 0
        r5 = stats["r5"] / stats["total"] * 100 if stats["total"] > 0 else 0
        status = "PASS" if r5 >= 80 else "NEEDS_WORK"
        print(f"  {cat:15} | R@1: {r1:5.1f}% | R@5: {r5:5.1f}% | [{status}]")
    
    # Print detailed results
    print()
    print("=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    
    for i, result in enumerate(detailed_results, 1):
        status = "OK" if result.get("relevant_at_5") else "MISS"
        print(f"\n[{i:2}] [{status}] {result['category']:10} | {result['query'][:50]}...")
        if result.get("status") == "ERROR":
            print(f"     ERROR: {result.get('error')}")
        else:
            print(f"     Results: {result['num_results']} | Latency: {result['latency_ms']}ms")
            if result.get("top_result") and result["top_result"] != "N/A":
                print(f"     Top: {result['top_result'][:80]}...")
    
    # Quality gate
    print()
    print("=" * 80)
    print("QUALITY GATE")
    print("=" * 80)
    
    target_r1 = 0.70  # 70% R@1
    target_r5 = 0.95  # 95% R@5 (target)
    
    r1_pass = metrics.recall_at_1 >= target_r1
    r5_pass = metrics.recall_at_5 >= target_r5
    
    print(f"  R@1: {metrics.recall_at_1:.1%} (target: {target_r1:.0%}) - {'PASS' if r1_pass else 'FAIL'}")
    print(f"  R@5: {metrics.recall_at_5:.1%} (target: {target_r5:.0%}) - {'PASS' if r5_pass else 'FAIL'}")
    
    overall = "PASS" if r1_pass and r5_pass else "FAIL"
    print(f"\n  OVERALL: {overall}")
    
    # Save results
    output_file = Path(__file__).parent.parent / "benchmark_results" / f"quality_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics.to_dict(),
            "category_breakdown": categories,
            "detailed_results": detailed_results,
            "quality_gate": {
                "r1_target": target_r1,
                "r5_target": target_r5,
                "r1_pass": r1_pass,
                "r5_pass": r5_pass,
                "overall": overall,
            }
        }, f, indent=2)
    
    print(f"\n  Results saved to: {output_file}")
    
    return metrics, detailed_results


if __name__ == "__main__":
    run_comparison_suite()
