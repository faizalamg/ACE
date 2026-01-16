# -*- coding: utf-8 -*-
"""Benchmark multi-stage retrieval vs standard retrieval.

Compares:
1. Accuracy (R@1, R@5, P@3) using cross-encoder scoring
2. Latency (mean, p95)
3. Stage-by-stage metrics

Ensures multi-stage does NOT degrade existing performance.
"""
import sys
import os
import time

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from ace.unified_memory import UnifiedMemoryIndex
from ace.config import reset_config, MultiStageConfig, get_retrieval_config

reset_config()

# Cross-encoder for honest relevance scoring
from sentence_transformers import CrossEncoder
ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

# Test queries (same as honest_assessment.py)
TEST_QUERIES = [
    'is this wired up and working in production',
    'how does hybrid search work',
    'what is the token limit',
    'where is error handling',
    'how to configure the LLM',
    'what database is used',
    'how does memory work',
    'how to run tests',
    'what is cross-encoder',
    'how does the retrieval work',
    'what is BM25',
    'where are hooks defined',
    'how to debug issues',
    'what is the embedding model',
    'where is the config file',
    'how does Qdrant work',
    'what is the ACE framework',
    'how to store memories',
    'what is vector search',
    'how does deduplication work',
]

# Cross-encoder threshold (centralized in ace/config.py)
# Default: -11.5 targets 95%+ P@3 with high recall
RELEVANCE_THRESHOLD = get_retrieval_config().cross_encoder_threshold


def evaluate_results(query: str, results: list) -> dict:
    """Evaluate results using cross-encoder."""
    if not results:
        return {'r1': 0, 'r5': 0, 'p3': 0.0, 'relevant_count': 0}

    # Score with cross-encoder
    pairs = [[query, r.content[:500]] for r in results[:5]]
    ce_scores = ce_model.predict(pairs)

    # Count relevant
    relevant_top1 = 1 if ce_scores[0] > RELEVANCE_THRESHOLD else 0
    relevant_top5 = sum(1 for s in ce_scores[:5] if s > RELEVANCE_THRESHOLD)
    relevant_top3 = sum(1 for s in ce_scores[:3] if s > RELEVANCE_THRESHOLD)

    return {
        'r1': relevant_top1,
        'r5': 1 if relevant_top5 > 0 else 0,
        'p3': relevant_top3 / 3.0,
        'relevant_count': relevant_top5,
        'ce_scores': ce_scores[:5].tolist(),
    }


def benchmark_method(index, method_name: str, queries: list, **kwargs) -> dict:
    """Benchmark a retrieval method."""
    results = {
        'method': method_name,
        'total_r1': 0,
        'total_r5': 0,
        'total_p3': 0.0,
        'latencies': [],
        'stage_metadata': [],
    }

    for query in queries:
        start = time.perf_counter()

        if method_name == 'retrieve_multistage':
            retrieval_results, metadata = index.retrieve_multistage(
                query, limit=5, return_metadata=True, **kwargs
            )
            results['stage_metadata'].append(metadata)
        else:
            retrieval_results = index.retrieve(
                query, limit=5, use_cross_encoder=True, **kwargs
            )

        latency = (time.perf_counter() - start) * 1000  # ms
        results['latencies'].append(latency)

        # Evaluate
        eval_result = evaluate_results(query, retrieval_results)
        results['total_r1'] += eval_result['r1']
        results['total_r5'] += eval_result['r5']
        results['total_p3'] += eval_result['p3']

    n = len(queries)
    results['r1_pct'] = results['total_r1'] / n * 100
    results['r5_pct'] = results['total_r5'] / n * 100
    results['p3_pct'] = results['total_p3'] / n * 100
    results['mean_latency'] = sum(results['latencies']) / n
    results['p95_latency'] = sorted(results['latencies'])[int(n * 0.95)] if n > 1 else results['latencies'][0]

    return results


def main():
    print('=' * 90)
    print('MULTI-STAGE RETRIEVAL BENCHMARK')
    print('Comparing retrieve() vs retrieve_multistage() - ensuring NO REGRESSION')
    print('=' * 90)

    # Initialize index
    index = UnifiedMemoryIndex(
        qdrant_url='http://localhost:6333',
        embedding_url='http://localhost:1234',
        collection_name='ace_memories_hybrid',
        embedding_dim=4096,
        embedding_model='text-embedding-qwen3-embedding-8b'
    )

    print(f'\nTest queries: {len(TEST_QUERIES)}')
    print(f'Cross-encoder threshold: {RELEVANCE_THRESHOLD}')

    # Benchmark standard retrieve
    print('\n' + '-' * 40)
    print('BASELINE: Standard retrieve() with cross-encoder')
    print('-' * 40)

    baseline = benchmark_method(index, 'retrieve', TEST_QUERIES)

    print(f'R@1:     {baseline["r1_pct"]:5.1f}%')
    print(f'R@5:     {baseline["r5_pct"]:5.1f}%')
    print(f'P@3:     {baseline["p3_pct"]:5.1f}%')
    print(f'Latency: {baseline["mean_latency"]:5.0f}ms mean, {baseline["p95_latency"]:5.0f}ms p95')

    # Benchmark multi-stage retrieve
    print('\n' + '-' * 40)
    print('MULTISTAGE: retrieve_multistage() (enabled)')
    print('-' * 40)

    multistage = benchmark_method(index, 'retrieve_multistage', TEST_QUERIES)

    print(f'R@1:     {multistage["r1_pct"]:5.1f}%')
    print(f'R@5:     {multistage["r5_pct"]:5.1f}%')
    print(f'P@3:     {multistage["p3_pct"]:5.1f}%')
    print(f'Latency: {multistage["mean_latency"]:5.0f}ms mean, {multistage["p95_latency"]:5.0f}ms p95')

    # Stage breakdown
    if multistage['stage_metadata']:
        stages = multistage['stage_metadata'][0]['stages']
        print(f'\nStage breakdown (first query):')
        print(f'  Stage 1 (coarse):     {stages["stage1_candidates"]} candidates')
        print(f'  Stage 2 (filtered):   {stages["stage2_filtered"]} candidates')
        print(f'  Stage 3 (reranked):   {stages["stage3_reranked"]} candidates')
        print(f'  Stage 4 (final):      {stages["stage4_final"]} results')

    # Benchmark multi-stage DISABLED (should match baseline)
    print('\n' + '-' * 40)
    print('MULTISTAGE DISABLED: retrieve_multistage(enable_multistage=False)')
    print('-' * 40)

    disabled_config = MultiStageConfig(enable_multistage=False)
    disabled = benchmark_method(
        index, 'retrieve_multistage', TEST_QUERIES,
        config=disabled_config
    )

    print(f'R@1:     {disabled["r1_pct"]:5.1f}%')
    print(f'R@5:     {disabled["r5_pct"]:5.1f}%')
    print(f'P@3:     {disabled["p3_pct"]:5.1f}%')
    print(f'Latency: {disabled["mean_latency"]:5.0f}ms mean, {disabled["p95_latency"]:5.0f}ms p95')

    # Summary comparison
    print('\n' + '=' * 90)
    print('COMPARISON SUMMARY')
    print('=' * 90)

    print(f'\n{"Method":<30} {"R@1":>8} {"R@5":>8} {"P@3":>8} {"Latency":>12}')
    print('-' * 70)
    print(f'{"Baseline (retrieve)":<30} {baseline["r1_pct"]:>7.1f}% {baseline["r5_pct"]:>7.1f}% {baseline["p3_pct"]:>7.1f}% {baseline["mean_latency"]:>10.0f}ms')
    print(f'{"Multi-stage (enabled)":<30} {multistage["r1_pct"]:>7.1f}% {multistage["r5_pct"]:>7.1f}% {multistage["p3_pct"]:>7.1f}% {multistage["mean_latency"]:>10.0f}ms')
    print(f'{"Multi-stage (disabled)":<30} {disabled["r1_pct"]:>7.1f}% {disabled["r5_pct"]:>7.1f}% {disabled["p3_pct"]:>7.1f}% {disabled["mean_latency"]:>10.0f}ms')

    # Regression check
    print('\n' + '-' * 70)
    print('REGRESSION CHECK')
    print('-' * 70)

    # Multi-stage should NOT degrade metrics
    r1_delta = multistage['r1_pct'] - baseline['r1_pct']
    r5_delta = multistage['r5_pct'] - baseline['r5_pct']
    p3_delta = multistage['p3_pct'] - baseline['p3_pct']
    latency_ratio = multistage['mean_latency'] / baseline['mean_latency']

    print(f'R@1 delta:     {r1_delta:+.1f}% {"PASS" if r1_delta >= -5 else "FAIL"}')
    print(f'R@5 delta:     {r5_delta:+.1f}% {"PASS" if r5_delta >= -5 else "FAIL"}')
    print(f'P@3 delta:     {p3_delta:+.1f}% {"PASS" if p3_delta >= -5 else "FAIL"}')
    print(f'Latency ratio: {latency_ratio:.2f}x {"PASS" if latency_ratio <= 2.0 else "FAIL (>2x)"}')

    # Overall pass/fail
    all_passed = (
        r1_delta >= -5 and
        r5_delta >= -5 and
        p3_delta >= -5 and
        latency_ratio <= 2.0
    )

    print('\n' + '=' * 90)
    if all_passed:
        print('RESULT: PASS - Multi-stage retrieval does NOT degrade performance')
    else:
        print('RESULT: FAIL - Multi-stage retrieval DEGRADES performance!')
    print('=' * 90)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
