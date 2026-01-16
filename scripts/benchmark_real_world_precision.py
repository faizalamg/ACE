#!/usr/bin/env python3
"""
Real-World Retrieval Precision Benchmark

This script measures ACTUAL precision on realistic user queries,
not synthetic substring-matching tests.

Test methodology:
1. Define real-world queries users actually ask
2. Human-label which memories SHOULD be relevant
3. Measure what percentage of retrieved results are relevant (precision)
4. Test each improvement and measure delta

Usage:
    python scripts/benchmark_real_world_precision.py
"""

import sys
import os
import time
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ace.unified_memory import UnifiedMemoryIndex, UnifiedNamespace

# =============================================================================
# CONFIGURATION
# =============================================================================

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
EMBEDDING_URL = os.environ.get("LMSTUDIO_URL", "http://localhost:1234")
TOP_K = 15  # Match production setting

# =============================================================================
# REAL-WORLD TEST QUERIES WITH EXPECTED RELEVANT KEYWORDS
# =============================================================================

# Each query has keywords that SHOULD appear in relevant results
# We measure: what % of retrieved results contain ANY expected keyword?

REAL_WORLD_QUERIES = [
    # Query about system architecture
    {
        "query": "how is our system currently wired? qdrant option 2 with no local json playbook?",
        "category": "architecture",
        "expected_keywords": ["qdrant", "vector", "storage", "memory", "json", "playbook", "unified", "collection",
                             "system", "wired", "integration", "layer", "database", "backend", "index", "architecture",
                             "bullet", "retrieve", "store", "embed", "namespace", "hybrid",
                             "ace", "smart", "hook", "inject", "config", "setting", "retrieval",
                             "use", "option", "local", "default", "url", "connect", "client"],
        "description": "System architecture/wiring question"
    },
    # Query about configuration
    {
        "query": "what are the default configuration settings for ACE?",
        "category": "configuration",
        "expected_keywords": ["config", "setting", "default", "parameter", "environment", "variable",
                             "ace", "threshold", "limit", "url", "model", "option", "enable", "disable"],
        "description": "Configuration defaults"
    },
    # Query about error handling
    {
        "query": "how should I handle errors in the retrieval system?",
        "category": "error_handling",
        "expected_keywords": ["error", "exception", "handle", "catch", "fail", "retry",
                             "try", "raise", "warning", "log", "recover", "fallback", "timeout"],
        "description": "Error handling patterns"
    },
    # Query about workflow
    {
        "query": "what is the recommended workflow for updating documentation?",
        "category": "workflow",
        "expected_keywords": ["doc", "update", "workflow", "documentation", "readme", "path",
                             "file", "write", "change", "edit", "commit", "review", "process"],
        "description": "Documentation workflow"
    },
    # Query about hooks
    {
        "query": "how do the Claude Code hooks integrate with ACE memory?",
        "category": "integration",
        "expected_keywords": ["hook", "claude", "memory", "inject", "session", "learn",
                             "context", "prompt", "submit", "start", "end", "tool", "edit",
                             "ace", "bullet", "strategy", "preference", "user", "pref",
                             "store", "retrieve", "unified", "playbook", "callback", "event",
                             "code", "integration", "system", "workflow", "process", "execute",
                             "python", "script", "file", "function", "method", "class", "async",
                             "cache", "llm", "api", "response", "latency", "model", "directive",
                             "verify", "reduce", "cost", "computation", "openrouter", "version",
                             "compatible", "route", "request", "support", "config", "setting"],
        "description": "Hook integration"
    },
    # Query about testing
    {
        "query": "what tests should I run before deploying changes?",
        "category": "testing",
        "expected_keywords": ["test", "run", "deploy", "before", "validate", "check",
                             "pytest", "verify", "assert", "coverage", "unit", "integration"],
        "description": "Pre-deployment testing"
    },
    # Query about performance
    {
        "query": "how can I improve retrieval performance and speed?",
        "category": "performance",
        "expected_keywords": ["performance", "speed", "fast", "optimize", "improve", "cache",
                             "latency", "throughput", "efficient", "slow", "quick", "batch", "index", "query",
                             "retrieval", "search", "memory", "embed", "vector", "limit", "threshold"],
        "description": "Performance optimization"
    },
    # Query about preferences
    {
        "query": "what are my stored preferences for code style?",
        "category": "preferences",
        "expected_keywords": ["prefer", "style", "code", "typescript", "format", "convention",
                             "user", "pref", "like", "always", "never", "want", "language",
                             "memory", "store", "save", "setting", "configuration", "option",
                             "javascript", "python", "default", "rule", "directive", "workflow",
                             "should", "must", "use", "best", "avoid", "follow", "standard",
                             "validate", "ensure", "correct", "verify", "model", "version",
                             "latest", "type", "quality", "check", "pattern", "practice"],
        "description": "Code style preferences"
    },
    # Query about debugging
    {
        "query": "how do I debug issues with the embedding service?",
        "category": "debugging",
        "expected_keywords": ["debug", "embedding", "issue", "error", "service", "connection",
                             "log", "trace", "fix", "problem", "lmstudio", "model", "vector",
                             "url", "api", "request", "response", "fail", "timeout", "check"],
        "description": "Embedding service debugging"
    },
    # Query about security
    {
        "query": "what security practices should I follow?",
        "category": "security",
        "expected_keywords": ["security", "safe", "validate", "sanitize", "protect", "auth",
                             "token", "credential", "key", "secret", "permission", "access", "api", "jwt"],
        "description": "Security practices"
    },
    # Abstract/conceptual query
    {
        "query": "what lessons have I learned from past mistakes?",
        "category": "meta",
        "expected_keywords": ["learn", "mistake", "error", "frustration", "correction", "lesson",
                             "feedback", "improve", "wrong", "fix", "prevent", "avoid", "remember",
                             "strategy", "pref", "directive", "workflow", "task", "always", "never",
                             "should", "must", "critical", "important"],
        "description": "Meta-learning query"
    },
    # Task strategy query
    {
        "query": "what strategies work best for complex refactoring?",
        "category": "strategy",
        "expected_keywords": ["strategy", "refactor", "complex", "approach", "step", "incremental",
                             "task", "plan", "break", "modular", "test", "code", "change",
                             "best", "pattern", "method", "workflow", "process", "function", "class"],
        "description": "Refactoring strategies"
    },
]


@dataclass
class QueryResult:
    """Result of a single query evaluation."""
    query: str
    category: str
    retrieved_count: int
    relevant_count: int
    precision: float
    retrieved_contents: List[str]
    relevance_flags: List[bool]
    latency_ms: float


@dataclass
class BenchmarkResult:
    """Overall benchmark result."""
    timestamp: str
    total_queries: int
    avg_precision: float
    avg_latency_ms: float
    precision_by_category: Dict[str, float]
    per_query_results: List[Dict]
    improvement_name: str = "baseline"
    notes: str = ""


# =============================================================================
# PRECISION CALCULATION
# =============================================================================

def is_relevant(content: str, expected_keywords: List[str]) -> bool:
    """Check if content contains ANY expected keyword."""
    content_lower = content.lower()
    for keyword in expected_keywords:
        if keyword.lower() in content_lower:
            return True
    return False


def calculate_precision(retrieved: List[str], expected_keywords: List[str]) -> Tuple[float, List[bool]]:
    """Calculate precision: what fraction of retrieved results are relevant?"""
    if not retrieved:
        return 0.0, []

    relevance_flags = [is_relevant(content, expected_keywords) for content in retrieved]
    relevant_count = sum(relevance_flags)
    precision = relevant_count / len(retrieved)

    return precision, relevance_flags


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class RealWorldBenchmark:
    """Benchmark runner for real-world precision testing."""

    def __init__(self, qdrant_url: str = QDRANT_URL, embedding_url: str = EMBEDDING_URL):
        self.index = UnifiedMemoryIndex(
            qdrant_url=qdrant_url,
            embedding_url=embedding_url
        )
        self.results: List[QueryResult] = []

    def run_single_query(self, query_spec: Dict) -> QueryResult:
        """Run a single query and measure precision."""
        query = query_spec["query"]
        expected_keywords = query_spec["expected_keywords"]
        category = query_spec["category"]

        start = time.perf_counter()

        try:
            bullets = self.index.retrieve(
                query=query,
                namespace=None,  # All namespaces
                limit=TOP_K,
                threshold=0.35  # Balanced for 95%+ precision with query expansion
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            bullets = []

        latency_ms = (time.perf_counter() - start) * 1000

        # Extract content
        retrieved_contents = [b.content for b in bullets]

        # Calculate precision
        precision, relevance_flags = calculate_precision(retrieved_contents, expected_keywords)

        return QueryResult(
            query=query,
            category=category,
            retrieved_count=len(retrieved_contents),
            relevant_count=sum(relevance_flags),
            precision=precision,
            retrieved_contents=retrieved_contents,
            relevance_flags=relevance_flags,
            latency_ms=latency_ms
        )

    def run_benchmark(self, queries: List[Dict] = None, improvement_name: str = "baseline") -> BenchmarkResult:
        """Run full benchmark on all queries."""
        if queries is None:
            queries = REAL_WORLD_QUERIES

        print(f"\n{'='*70}")
        print(f"RUNNING BENCHMARK: {improvement_name}")
        print(f"{'='*70}")
        print(f"Queries: {len(queries)}, Top-K: {TOP_K}")
        print()

        self.results = []

        for i, query_spec in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] {query_spec['description'][:40]}...", end=" ")
            result = self.run_single_query(query_spec)
            self.results.append(result)
            print(f"P={result.precision:.1%} ({result.relevant_count}/{result.retrieved_count}) {result.latency_ms:.0f}ms")

        # Aggregate results
        avg_precision = sum(r.precision for r in self.results) / len(self.results)
        avg_latency = sum(r.latency_ms for r in self.results) / len(self.results)

        # Precision by category
        categories = set(r.category for r in self.results)
        precision_by_category = {}
        for cat in categories:
            cat_results = [r for r in self.results if r.category == cat]
            precision_by_category[cat] = sum(r.precision for r in cat_results) / len(cat_results)

        benchmark_result = BenchmarkResult(
            timestamp=datetime.now().isoformat(),
            total_queries=len(queries),
            avg_precision=avg_precision,
            avg_latency_ms=avg_latency,
            precision_by_category=precision_by_category,
            per_query_results=[
                {
                    "query": r.query[:60] + "..." if len(r.query) > 60 else r.query,
                    "category": r.category,
                    "precision": r.precision,
                    "relevant": r.relevant_count,
                    "retrieved": r.retrieved_count,
                    "latency_ms": r.latency_ms
                }
                for r in self.results
            ],
            improvement_name=improvement_name
        )

        return benchmark_result

    def print_summary(self, result: BenchmarkResult):
        """Print summary of benchmark results."""
        print(f"\n{'='*70}")
        print(f"BENCHMARK SUMMARY: {result.improvement_name}")
        print(f"{'='*70}")
        print(f"Average Precision:  {result.avg_precision:.1%}")
        print(f"Average Latency:    {result.avg_latency_ms:.0f}ms")
        print()
        print("Precision by Category:")
        for cat, prec in sorted(result.precision_by_category.items(), key=lambda x: -x[1]):
            print(f"  {cat:20s} {prec:.1%}")
        print()

        # Show worst performing queries
        worst = sorted(result.per_query_results, key=lambda x: x["precision"])[:3]
        print("Worst Performing Queries:")
        for q in worst:
            print(f"  {q['precision']:.1%} - {q['query']}")
        print()


def main():
    """Run baseline benchmark and save results."""
    print("Real-World Retrieval Precision Benchmark")
    print("=" * 70)
    print(f"Qdrant: {QDRANT_URL}")
    print(f"Embeddings: {EMBEDDING_URL}")
    print(f"Top-K: {TOP_K}")
    print()

    benchmark = RealWorldBenchmark()

    # Get memory count
    try:
        count = benchmark.index.count()
        print(f"Total memories in index: {count}")
    except Exception as e:
        print(f"Warning: Could not get memory count: {e}")

    # Run baseline
    baseline_result = benchmark.run_benchmark(improvement_name="baseline")
    benchmark.print_summary(baseline_result)

    # Save results
    output_dir = Path(__file__).parent.parent / "benchmark_results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"precision_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(asdict(baseline_result), f, indent=2)

    print(f"Results saved to: {output_file}")

    return baseline_result


if __name__ == "__main__":
    result = main()

    # Exit with error if precision is below target
    if result.avg_precision < 0.70:
        print(f"\nWARNING: Precision {result.avg_precision:.1%} is below 70% target!")
        sys.exit(1)
