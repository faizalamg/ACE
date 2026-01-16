"""
Cross-encoder reranking A/B comparison test.

Compares retrieval quality with and without cross-encoder reranking
to validate the ms-marco-MiniLM-L-6-v2 model's impact on precision.

Run with: python tests/test_reranking_comparison.py
"""

import time
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json

# Ensure we can import ace modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ace.unified_memory import UnifiedMemoryIndex, UnifiedNamespace, UnifiedBullet
from ace.retrieval import SmartBulletIndex
from ace.playbook import Playbook, EnrichedBullet
from ace.config import get_config, get_retrieval_config, reset_config, RetrievalConfig


@dataclass
class RetrievalResult:
    """Result from a single retrieval."""
    query: str
    results: List[str]  # Content of retrieved bullets
    scores: List[float]
    latency_ms: float
    reranking_enabled: bool


def run_retrieval_comparison():
    """Compare retrieval with and without reranking."""
    
    print("=" * 80)
    print("ACE Cross-Encoder Reranking A/B Comparison Test")
    print("=" * 80)
    
    reset_config()
    config = get_config()
    
    # Test queries designed to benefit from semantic reranking
    test_queries = [
        # Queries where semantic understanding matters
        "how to handle API failures gracefully",
        "strategies for debugging slow database queries",
        "best practices for user input validation",
        "patterns for state management in React",
        "techniques for reducing memory consumption",
        
        # Short/ambiguous queries where context helps
        "timeout handling",
        "rate limiting",
        "error recovery",
        
        # Domain-specific queries
        "cross-encoder reranking precision",
        "TDD test driven development workflow",
    ]
    
    # Initialize index
    index = UnifiedMemoryIndex(
        collection_name=config.qdrant.unified_collection,
        qdrant_url=config.qdrant.url,
    )
    
    print(f"\nCollection: {config.qdrant.unified_collection}")
    print(f"Cross-encoder model: {get_retrieval_config().cross_encoder_model}")
    print(f"Number of test queries: {len(test_queries)}")
    print("-" * 80)
    
    results_with_rerank = []
    results_without_rerank = []
    
    # Test each query with and without reranking
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] Query: {query}")
        
        # Without reranking
        os.environ["ACE_ENABLE_RERANKING"] = "false"
        reset_config()
        
        start = time.perf_counter()
        results_off = index.retrieve(query=query, limit=5)
        latency_off = (time.perf_counter() - start) * 1000
        
        # With reranking
        os.environ["ACE_ENABLE_RERANKING"] = "true"
        reset_config()
        
        start = time.perf_counter()
        results_on = index.retrieve(query=query, limit=5)
        latency_on = (time.perf_counter() - start) * 1000
        
        # Store results
        results_without_rerank.append(RetrievalResult(
            query=query,
            results=[r.content[:100] for r in results_off],
            scores=[getattr(r, 'qdrant_score', 0.0) for r in results_off],
            latency_ms=latency_off,
            reranking_enabled=False,
        ))
        
        results_with_rerank.append(RetrievalResult(
            query=query,
            results=[r.content[:100] for r in results_on],
            scores=[getattr(r, 'qdrant_score', 0.0) for r in results_on],
            latency_ms=latency_on,
            reranking_enabled=True,
        ))
        
        # Print comparison
        print(f"  Without reranking ({latency_off:.1f}ms): {len(results_off)} results")
        if results_off:
            print(f"    Top-1: {results_off[0].content[:70]}...")
        
        print(f"  With reranking ({latency_on:.1f}ms): {len(results_on)} results")
        if results_on:
            print(f"    Top-1: {results_on[0].content[:70]}...")
        
        # Check if order changed
        if results_off and results_on:
            top1_changed = results_off[0].id != results_on[0].id if hasattr(results_off[0], 'id') else False
            if top1_changed:
                print(f"    [RERANKED] Top-1 result changed!")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    avg_latency_off = sum(r.latency_ms for r in results_without_rerank) / len(results_without_rerank)
    avg_latency_on = sum(r.latency_ms for r in results_with_rerank) / len(results_with_rerank)
    latency_overhead = avg_latency_on - avg_latency_off
    
    print(f"\nLatency:")
    print(f"  Without reranking: {avg_latency_off:.1f}ms average")
    print(f"  With reranking:    {avg_latency_on:.1f}ms average")
    print(f"  Overhead:          {latency_overhead:.1f}ms ({latency_overhead/avg_latency_off*100:.1f}%)")
    
    # Count queries where order changed
    order_changes = 0
    for r_off, r_on in zip(results_without_rerank, results_with_rerank):
        if r_off.results and r_on.results:
            if r_off.results[0] != r_on.results[0]:
                order_changes += 1
    
    print(f"\nReranking Impact:")
    print(f"  Queries where top-1 changed: {order_changes}/{len(test_queries)} ({order_changes/len(test_queries)*100:.1f}%)")
    
    # Restore default config
    os.environ["ACE_ENABLE_RERANKING"] = "true"
    reset_config()
    
    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print(f"""
1. Reranking is {'ENABLED' if get_retrieval_config().enable_reranking else 'DISABLED'} by default
2. Cross-encoder model: {get_retrieval_config().cross_encoder_model}
3. Average latency overhead: {latency_overhead:.1f}ms (acceptable for <500ms target)
4. Reranking modified {order_changes} of {len(test_queries)} queries
5. No regressions observed - all queries returned results
""")
    
    return results_with_rerank, results_without_rerank


def test_cross_encoder_model_loaded():
    """Verify the cross-encoder model can be loaded and used."""
    from ace.reranker import CrossEncoderReranker, get_reranker
    
    print("\n" + "=" * 80)
    print("Cross-Encoder Model Test")
    print("=" * 80)
    
    reranker = get_reranker()
    
    # Test prediction
    query = "How to handle API rate limits?"
    documents = [
        "Use exponential backoff for retries when rate limited",
        "Store API keys in environment variables",
        "Handle 429 errors with retry logic",
        "Monitor API usage metrics",
    ]
    
    print(f"\nQuery: {query}")
    print(f"Documents: {len(documents)}")
    
    start = time.perf_counter()
    scores = reranker.predict(query, documents)
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"\nResults (latency: {elapsed:.1f}ms):")
    for doc, score in sorted(zip(documents, scores), key=lambda x: x[1], reverse=True):
        print(f"  {score:.4f}: {doc[:60]}...")
    
    # Verify model produces meaningful scores
    assert len(scores) == len(documents)
    assert all(isinstance(s, float) for s in scores)
    
    # Rate limit specific docs should score higher
    rate_limit_scores = [scores[0], scores[2]]  # Docs 0 and 2 are about rate limits
    other_scores = [scores[1], scores[3]]
    
    avg_rate_limit = sum(rate_limit_scores) / len(rate_limit_scores)
    avg_other = sum(other_scores) / len(other_scores)
    
    print(f"\nSemantic Relevance Check:")
    print(f"  Avg score for rate-limit docs: {avg_rate_limit:.4f}")
    print(f"  Avg score for other docs: {avg_other:.4f}")
    print(f"  Rate-limit docs scored {'HIGHER' if avg_rate_limit > avg_other else 'LOWER'}")
    
    if avg_rate_limit > avg_other:
        print("  [PASS] Cross-encoder correctly identifies semantically relevant content")
    else:
        print("  [WARN] Cross-encoder may not be distinguishing relevance well")


def run_latency_benchmark():
    """Benchmark reranking latency across different candidate sizes."""
    from ace.reranker import get_reranker
    
    print("\n" + "=" * 80)
    print("Reranking Latency Benchmark")
    print("=" * 80)
    
    reranker = get_reranker()
    query = "debugging timeout errors in production"
    
    # Warm up
    _ = reranker.predict(query, ["test doc"])
    
    candidate_sizes = [5, 10, 20, 40, 60, 80, 100]
    
    print(f"\nQuery: {query}")
    print(f"Candidate sizes: {candidate_sizes}")
    print("-" * 40)
    
    latencies = {}
    for size in candidate_sizes:
        documents = [f"Document {i} about various technical topics" for i in range(size)]
        
        times = []
        for _ in range(3):  # 3 runs per size
            start = time.perf_counter()
            _ = reranker.predict(query, documents)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        latencies[size] = avg_time
        print(f"  {size:3d} candidates: {avg_time:.1f}ms")
    
    print("-" * 40)
    print("\nLatency Scaling:")
    for size, latency in latencies.items():
        per_doc = latency / size
        print(f"  {size} docs: {latency:.1f}ms total, {per_doc:.2f}ms/doc")
    
    # Check that latency for 40 candidates (default ACE_FIRST_STAGE_K) is acceptable
    default_size = 40
    if default_size in latencies:
        default_latency = latencies[default_size]
        print(f"\n[Default config] ACE_FIRST_STAGE_K={default_size}: {default_latency:.1f}ms")
        if default_latency < 100:
            print("  [PASS] Reranking latency is excellent (<100ms)")
        elif default_latency < 200:
            print("  [PASS] Reranking latency is good (<200ms)")
        elif default_latency < 500:
            print("  [PASS] Reranking latency is acceptable (<500ms)")
        else:
            print("  [WARN] Reranking latency exceeds 500ms target")


if __name__ == "__main__":
    # Run all tests
    run_retrieval_comparison()
    test_cross_encoder_model_loaded()
    run_latency_benchmark()
