# -*- coding: utf-8 -*-
"""Final benchmark with LLM relevance filtering using Z.ai GLM 4.6."""
import sys
import os
import re

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env for ZAI_API_KEY
from dotenv import load_dotenv
load_dotenv()

# Verify ZAI_API_KEY is set
if not os.environ.get("ZAI_API_KEY"):
    print("ERROR: ZAI_API_KEY not set. Set it in .env or environment.")
    sys.exit(1)

from ace.unified_memory import UnifiedMemoryIndex
from ace.retrieval_presets import llm_filter_and_rerank

index = UnifiedMemoryIndex(
    qdrant_url='http://localhost:6333',
    embedding_url='http://localhost:1234',
    collection_name='ace_memories_hybrid',
    embedding_dim=4096,
    embedding_model='text-embedding-qwen3-embedding-8b'
)

TEST_CASES = [
    {'cat': 'vector', 'query': 'How to implement hybrid search with vectors?', 'expect': ['hybrid', 'search', 'vector', 'keyword']},
    {'cat': 'vector', 'query': 'Validate embeddings before search', 'expect': ['embedding', 'valid', 'vector', 'search']},
    {'cat': 'vector', 'query': 'Vector and keyword deduplication', 'expect': ['vector', 'keyword', 'dedup', 'hybrid', 'stop', 'strip', 'word']},
    {'cat': 'vector', 'query': 'Qdrant vector database implementation', 'expect': ['qdrant', 'vector', 'database', 'implement']},
    {'cat': 'vector', 'query': 'Dense and sparse vector search', 'expect': ['dense', 'sparse', 'vector', 'search', 'hybrid']},
    {'cat': 'vector', 'query': 'Embedding dimension validation', 'expect': ['embedding', 'dimension', 'valid', 'model']},
    {'cat': 'validation', 'query': 'Validate metrics before deployment', 'expect': ['valid', 'metric', 'deploy', 'schema']},
    {'cat': 'validation', 'query': 'Input validation at boundaries', 'expect': ['valid', 'input', 'boundar', 'inject']},
    {'cat': 'validation', 'query': 'Object state validation before assignment', 'expect': ['valid', 'state', 'object', 'error']},
    {'cat': 'validation', 'query': 'Retrieval score validation', 'expect': ['retriev', 'score', 'valid', 'relev']},
    {'cat': 'validation', 'query': 'Test threshold validation', 'expect': ['test', 'threshold', 'valid', 'benchmark']},
    {'cat': 'validation', 'query': 'Data quality threshold enforcement', 'expect': ['quality', 'threshold', 'data', 'process']},
    {'cat': 'error', 'query': 'Error prevention strategies', 'expect': ['error', 'prevent', 'valid', 'check', 'exception', 'fallback']},
    {'cat': 'error', 'query': 'Override error prevention', 'expect': ['override', 'error', 'prevent', 'state', 'exception', 'handle', 'root', 'cause']},
    {'cat': 'error', 'query': 'Handling typo errors in input', 'expect': ['typo', 'error', 'input', 'check']},
    {'cat': 'error', 'query': 'Root cause analysis for failures', 'expect': ['root', 'cause', 'fail', 'debug', 'diagnos', 'analysis']},
    {'cat': 'error', 'query': 'Session-aware error fallbacks', 'expect': ['session', 'error', 'fallback', 'retriev']},
    {'cat': 'error', 'query': 'Retry failed operations', 'expect': ['retry', 'fail', 'operation', 'error']},
    {'cat': 'testing', 'query': 'Isolate test fixtures', 'expect': ['test', 'isolat', 'fixture', 'state']},
    {'cat': 'testing', 'query': 'Test shared state prevention', 'expect': ['test', 'shared', 'state', 'prevent']},
    {'cat': 'testing', 'query': 'Benchmark validation before release', 'expect': ['benchmark', 'valid', 'test', 'release']},
    {'cat': 'testing', 'query': 'Test threshold configuration', 'expect': ['test', 'threshold', 'config', 'valid']},
    {'cat': 'testing', 'query': 'Query performance benchmarking', 'expect': ['query', 'performance', 'benchmark', 'metric', 'test', 'load']},
    {'cat': 'testing', 'query': 'Retrieval metrics validation', 'expect': ['retriev', 'metric', 'valid', 'test']},
    {'cat': 'config', 'query': 'API configuration settings', 'expect': ['api', 'config', 'setting', 'model']},
    {'cat': 'config', 'query': 'GLM model configuration', 'expect': ['glm', 'model', 'config', 'api']},
    {'cat': 'config', 'query': 'Preference for concise responses', 'expect': ['prefer', 'concise', 'response', 'technical']},
    {'cat': 'config', 'query': 'Documentation change explanations', 'expect': ['document', 'change', 'explain', 'update']},
    {'cat': 'config', 'query': 'Quality threshold configuration', 'expect': ['quality', 'threshold', 'config', 'data']},
    {'cat': 'config', 'query': 'Relevance threshold settings', 'expect': ['relev', 'threshold', 'retriev', 'config']},
]

print('=' * 75)
print('BENCHMARK WITH NATURAL LANGUAGE LLM FILTERING')
print('LLM judges relevance directly - no numeric thresholds')
print('=' * 75)

metrics = {cat: {'r1': [], 'r5': [], 'p3': []} for cat in ['vector', 'validation', 'error', 'testing', 'config']}

import time  # For rate limiting

def has_relevant(content, expect):
    return any(kw.lower() in content.lower() for kw in expect)


for tc in TEST_CASES:
    cat = tc['cat']
    query = tc['query']
    expect = tc['expect']

    results = index.retrieve(
        query, limit=10, auto_detect_preset=True,
        use_llm_expansion=True, use_llm_rerank=False
    )

    if results:
        results_with_scores = [(b, getattr(b, 'qdrant_score', 0.5)) for b in results]
        # Uses Z.ai GLM 4.6 from config (ZAI_API_KEY, ZAI_API_BASE, ZAI_MODEL)
        results = llm_filter_and_rerank(
            query=query,
            results=results_with_scores,
            # All settings from ace.config.LLMConfig
        )

    r1 = 1.0 if results and has_relevant(results[0].content, expect) else 0.0
    r5 = 1.0 if any(has_relevant(r.content, expect) for r in results[:5]) else 0.0
    actual_top3 = results[:3] if results else []
    relevant_in_3 = sum(1 for r in actual_top3 if has_relevant(r.content, expect))
    p3 = relevant_in_3 / len(actual_top3) if actual_top3 else 0.0

    metrics[cat]['r1'].append(r1)
    metrics[cat]['r5'].append(r5)
    metrics[cat]['p3'].append(p3)

    r1_mark = 'Y' if r1 == 1.0 else 'N'
    r5_mark = 'Y' if r5 == 1.0 else 'N'
    n_ret = len(results) if results else 0
    print(f'[{cat[:4].upper()}] R@1:{r1_mark} R@5:{r5_mark} P@3:{p3 * 100:.0f}% ({n_ret}r) {query[:35]}')

    # Rate limit: small delay between queries to avoid 429 errors
    time.sleep(1.0)

print('')
print('=' * 75)
print('RESULTS')
print('=' * 75)

overall_r1, overall_r5, overall_p3 = [], [], []

for cat in ['vector', 'validation', 'error', 'testing', 'config']:
    m = metrics[cat]
    avg_r1 = sum(m['r1']) / len(m['r1']) * 100
    avg_r5 = sum(m['r5']) / len(m['r5']) * 100
    avg_p3 = sum(m['p3']) / len(m['p3']) * 100
    overall_r1.extend(m['r1'])
    overall_r5.extend(m['r5'])
    overall_p3.extend(m['p3'])
    r1_pass = 'PASS' if avg_r1 >= 95 else 'FAIL'
    r5_pass = 'PASS' if avg_r5 >= 95 else 'FAIL'
    p3_pass = 'PASS' if avg_p3 >= 95 else 'FAIL'
    print(f'{cat.upper():12} | R@1: {avg_r1:5.1f}% [{r1_pass}] | R@5: {avg_r5:5.1f}% [{r5_pass}] | P@3: {avg_p3:5.1f}% [{p3_pass}]')

print('=' * 75)

total_r1 = sum(overall_r1) / len(overall_r1) * 100
total_r5 = sum(overall_r5) / len(overall_r5) * 100
total_p3 = sum(overall_p3) / len(overall_p3) * 100

r1_status = "PASS" if total_r1 >= 95 else "FAIL"
r5_status = "PASS" if total_r5 >= 95 else "FAIL"
p3_status = "PASS" if total_p3 >= 95 else "FAIL"

print(f'Recall@1:    {total_r1:5.1f}% (95%+) {r1_status}')
print(f'Recall@5:    {total_r5:5.1f}% (95%+) {r5_status}')
print(f'Precision@3: {total_p3:5.1f}% (95%+) {p3_status}')

all_pass = total_r1 >= 95 and total_r5 >= 95 and total_p3 >= 95
print('')
if all_pass:
    print('ALL TARGETS MET!!!')
else:
    print('NEEDS MORE TUNING')
