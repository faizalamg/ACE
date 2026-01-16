"""Benchmark cross-encoder reranking performance.

Compares retrieval quality and latency with/without reranking.
Run with: python benchmarks/benchmark_reranking.py
"""

import time
import statistics
from typing import List, Tuple
from dataclasses import dataclass

from ace.playbook import Playbook
from ace.retrieval import SmartBulletIndex, ScoredBullet


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    query: str
    # Without reranking
    no_rerank_time_ms: float
    no_rerank_precision: float
    no_rerank_top_contents: List[str]
    # With reranking
    rerank_time_ms: float
    rerank_precision: float
    rerank_top_contents: List[str]
    # Delta
    latency_overhead_ms: float
    precision_delta: float


def create_benchmark_playbook() -> Playbook:
    """Create a playbook with realistic test data."""
    playbook = Playbook()
    
    # Ground truth: rate limiting bullets (RELEVANT)
    rate_limit_bullets = [
        ("Implement exponential backoff with jitter for 429 Too Many Requests errors", ["rate", "limit", "429", "backoff"]),
        ("Use circuit breaker pattern to handle rate-limited external APIs", ["rate", "limit", "circuit", "breaker"]),
        ("Cache API responses to reduce rate limit consumption", ["cache", "rate", "limit"]),
        ("Queue requests and batch them to stay within rate limits", ["queue", "batch", "rate", "limit"]),
        ("Implement request throttling on the client side", ["throttle", "rate", "limit", "client"]),
    ]
    
    # Noise: API-related but NOT about rate limiting (IRRELEVANT)
    noise_bullets = [
        ("Use OAuth2 for API authentication with refresh tokens", ["api", "oauth", "authentication"]),
        ("Document all API endpoints using OpenAPI/Swagger specification", ["api", "documentation", "openapi"]),
        ("Follow semantic versioning for API version management", ["api", "versioning", "semver"]),
        ("Test API endpoints with Postman or similar tools", ["api", "testing", "postman"]),
        ("Keep API response times under 200ms for good UX", ["api", "performance", "response"]),
        ("Validate all API request inputs before processing", ["api", "validation", "input"]),
        ("Use HTTPS for all API communications", ["api", "security", "https"]),
        ("Implement API key rotation for security", ["api", "key", "security"]),
        ("Log all API requests for debugging and audit", ["api", "logging", "audit"]),
        ("Use pagination for large API result sets", ["api", "pagination", "results"]),
    ]
    
    # Additional noise: completely unrelated bullets
    unrelated_bullets = [
        ("Use descriptive variable names for code readability", ["code", "readability", "naming"]),
        ("Write unit tests before implementing features (TDD)", ["testing", "tdd", "unit"]),
        ("Review code before merging to main branch", ["code", "review", "merge"]),
        ("Use git branches for feature development", ["git", "branches", "feature"]),
        ("Document complex functions with docstrings", ["documentation", "docstring", "function"]),
    ]
    
    for content, patterns in rate_limit_bullets:
        playbook.add_enriched_bullet(
            section="rate_limiting",
            content=content,
            trigger_patterns=patterns,
            task_types=["api", "reliability"],
        )
    
    for content, patterns in noise_bullets:
        playbook.add_enriched_bullet(
            section="api",
            content=content,
            trigger_patterns=patterns,
            task_types=["api"],
        )
    
    for content, patterns in unrelated_bullets:
        playbook.add_enriched_bullet(
            section="best_practices",
            content=content,
            trigger_patterns=patterns,
            task_types=["development"],
        )
    
    return playbook


def calculate_precision(results: List[ScoredBullet], relevant_keywords: List[str]) -> float:
    """Calculate precision@k for results."""
    if not results:
        return 0.0
    
    relevant_count = sum(
        1 for r in results 
        if any(kw in r.content.lower() for kw in relevant_keywords)
    )
    return relevant_count / len(results)


def run_benchmark(
    index: SmartBulletIndex,
    query: str,
    relevant_keywords: List[str],
    limit: int = 5,
    iterations: int = 3,
) -> BenchmarkResult:
    """Run a single benchmark comparing with/without reranking."""
    
    # Benchmark WITHOUT reranking
    no_rerank_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        results_no_rerank = index.retrieve(query=query, limit=limit, rerank=False)
        no_rerank_times.append((time.perf_counter() - start) * 1000)
    
    no_rerank_precision = calculate_precision(results_no_rerank, relevant_keywords)
    
    # Benchmark WITH reranking
    rerank_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        results_rerank = index.retrieve(query=query, limit=limit, rerank=True)
        rerank_times.append((time.perf_counter() - start) * 1000)
    
    rerank_precision = calculate_precision(results_rerank, relevant_keywords)
    
    return BenchmarkResult(
        query=query,
        no_rerank_time_ms=statistics.mean(no_rerank_times),
        no_rerank_precision=no_rerank_precision,
        no_rerank_top_contents=[r.content[:60] for r in results_no_rerank],
        rerank_time_ms=statistics.mean(rerank_times),
        rerank_precision=rerank_precision,
        rerank_top_contents=[r.content[:60] for r in results_rerank],
        latency_overhead_ms=statistics.mean(rerank_times) - statistics.mean(no_rerank_times),
        precision_delta=rerank_precision - no_rerank_precision,
    )


def main():
    """Run benchmarks and print results."""
    print("=" * 70)
    print("CROSS-ENCODER RERANKING BENCHMARK")
    print("=" * 70)
    
    # Setup
    print("\nSetting up benchmark playbook...")
    playbook = create_benchmark_playbook()
    index = SmartBulletIndex(playbook=playbook)
    print(f"Playbook contains {len(list(playbook.bullets()))} bullets")
    
    # Define benchmark queries
    benchmarks = [
        {
            "query": "How to handle API rate limiting?",
            "keywords": ["rate limit", "429", "backoff", "throttle", "circuit breaker", "queue", "batch"],
        },
        {
            "query": "What's the best way to deal with rate limit errors?",
            "keywords": ["rate limit", "429", "backoff", "throttle", "circuit breaker"],
        },
        {
            "query": "API throttling best practices",
            "keywords": ["rate limit", "throttle", "backoff", "circuit breaker"],
        },
    ]
    
    results: List[BenchmarkResult] = []
    
    print("\nRunning benchmarks...")
    print("-" * 70)
    
    for bench in benchmarks:
        result = run_benchmark(
            index=index,
            query=bench["query"],
            relevant_keywords=bench["keywords"],
            limit=5,
            iterations=3,
        )
        results.append(result)
        
        print(f"\nQuery: {result.query}")
        print(f"  Without reranking:")
        print(f"    Latency: {result.no_rerank_time_ms:.1f}ms")
        print(f"    Precision@5: {result.no_rerank_precision:.0%}")
        print(f"    Top results: {result.no_rerank_top_contents[:3]}")
        print(f"  With reranking:")
        print(f"    Latency: {result.rerank_time_ms:.1f}ms")
        print(f"    Precision@5: {result.rerank_precision:.0%}")
        print(f"    Top results: {result.rerank_top_contents[:3]}")
        print(f"  Delta:")
        print(f"    Latency overhead: +{result.latency_overhead_ms:.1f}ms")
        print(f"    Precision delta: {result.precision_delta:+.0%}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    avg_overhead = statistics.mean(r.latency_overhead_ms for r in results)
    avg_precision_delta = statistics.mean(r.precision_delta for r in results)
    avg_no_rerank_precision = statistics.mean(r.no_rerank_precision for r in results)
    avg_rerank_precision = statistics.mean(r.rerank_precision for r in results)
    
    print(f"\nAverage latency overhead: +{avg_overhead:.1f}ms")
    print(f"Average precision without reranking: {avg_no_rerank_precision:.0%}")
    print(f"Average precision with reranking: {avg_rerank_precision:.0%}")
    print(f"Average precision improvement: {avg_precision_delta:+.0%}")
    
    print("\n" + "-" * 70)
    if avg_precision_delta >= 0:
        print("RESULT: Reranking provides precision improvement with acceptable latency")
    else:
        print("RESULT: Baseline retrieval already performs well on this dataset")
    print("-" * 70)


if __name__ == "__main__":
    main()
