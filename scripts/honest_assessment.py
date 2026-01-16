# -*- coding: utf-8 -*-
"""HONEST assessment of retrieval quality with cross-encoder scores."""
import sys
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from ace.unified_memory import UnifiedMemoryIndex
from ace.config import reset_config, get_retrieval_config

reset_config()

# Cross-encoder for relevance scoring
from sentence_transformers import CrossEncoder
ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

# Cross-encoder threshold (centralized in ace/config.py)
# Default: -11.5 targets 95%+ P@3 with high recall
RELEVANCE_THRESHOLD = get_retrieval_config().cross_encoder_threshold

index = UnifiedMemoryIndex(
    qdrant_url='http://localhost:6333',
    embedding_url='http://localhost:1234',
    collection_name='ace_memories_hybrid',
    embedding_dim=4096,
    embedding_model='text-embedding-qwen3-embedding-8b'
)

# Test queries - expanded mix of real user questions (20 queries)
test_queries = [
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

print('=' * 90)
print('HONEST RELEVANCE ASSESSMENT - Cross-Encoder Scores')
print('Score > -8: HIGHLY relevant | -8 to -10: RELEVANT | -10 to -12: WEAK | < -12: IRRELEVANT')
print('=' * 90)

total_r1 = 0
total_r5 = 0
total_relevant_top3 = 0
queries_tested = 0

for query in test_queries:
    # Cross-encoder reranking is done inside retrieve() when use_cross_encoder=True
    results = index.retrieve(query, limit=5, auto_detect_preset=True, use_llm_expansion=False, use_cross_encoder=True)

    if not results:
        continue

    # Score with cross-encoder
    pairs = [[query, r.content[:500]] for r in results]
    ce_scores = ce_model.predict(pairs)

    # Count relevant (using centralized threshold from config)
    relevant_top1 = 1 if ce_scores[0] > RELEVANCE_THRESHOLD else 0
    relevant_top5 = sum(1 for s in ce_scores[:5] if s > RELEVANCE_THRESHOLD)
    relevant_top3 = sum(1 for s in ce_scores[:3] if s > RELEVANCE_THRESHOLD)

    r1 = relevant_top1
    r5 = 1 if relevant_top5 > 0 else 0
    p3 = relevant_top3 / 3.0

    total_r1 += r1
    total_r5 += r5
    total_relevant_top3 += relevant_top3
    queries_tested += 1

    # Display
    print(f'\nQuery: "{query[:50]}..."')
    print(f'  R@1: {"PASS" if r1 else "FAIL"} | R@5: {"PASS" if r5 else "FAIL"} | P@3: {p3*100:.0f}%')
    for i, (r, score) in enumerate(zip(results[:5], ce_scores[:5])):
        status = 'REL' if score > RELEVANCE_THRESHOLD else 'IRR'
        content_clean = ''.join(c for c in r.content[:45] if ord(c) < 128).replace('\n', ' ')
        print(f'    {i+1}. [{status}] {score:6.2f} | {content_clean}...')

# Final metrics
print()
print('=' * 90)
print(f'HONEST RESULTS (Cross-Encoder Threshold: {RELEVANCE_THRESHOLD})')
print('=' * 90)

if queries_tested > 0:
    final_r1 = total_r1 / queries_tested * 100
    final_r5 = total_r5 / queries_tested * 100
    final_p3 = total_relevant_top3 / (queries_tested * 3) * 100

    print(f'Queries tested: {queries_tested}')
    print(f'Recall@1:    {final_r1:5.1f}% (target: 95%+) {"PASS" if final_r1 >= 95 else "FAIL"}')
    print(f'Recall@5:    {final_r5:5.1f}% (target: 95%+) {"PASS" if final_r5 >= 95 else "FAIL"}')
    print(f'Precision@3: {final_p3:5.1f}% (target: 95%+) {"PASS" if final_p3 >= 95 else "FAIL"}')
