# -*- coding: utf-8 -*-
"""Debug dense-only vs hybrid search for problematic query."""
import sys
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import httpx
from sentence_transformers import CrossEncoder

ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

QDRANT_URL = 'http://localhost:6333'
COLLECTION = 'ace_memories_hybrid'
EMBEDDING_URL = 'http://localhost:1234'

def get_embedding(text):
    response = httpx.post(
        f'{EMBEDDING_URL}/v1/embeddings',
        json={'model': 'text-embedding-qwen3-embedding-8b', 'input': text},
        timeout=30.0,
    )
    if response.status_code == 200:
        return response.json()['data'][0]['embedding']
    return None

query = 'is this wired up and working in production'
query_emb = get_embedding(query)

print('=' * 80)
print('DEBUG: Dense-Only vs Hybrid Search')
print('=' * 80)
print(f'Query: "{query}"')

# 1. DENSE-ONLY search
print('\n1. DENSE-ONLY SEARCH:')
dense_response = httpx.post(
    f'{QDRANT_URL}/collections/{COLLECTION}/points/query',
    json={
        "query": query_emb,
        "using": "dense",
        "limit": 5,
        "with_payload": True
    },
    timeout=30.0
)

if dense_response.status_code == 200:
    dense_results = dense_response.json().get('result', {}).get('points', [])
    for i, r in enumerate(dense_results):
        content = r.get('payload', {}).get('content', '')[:60]
        content_clean = ''.join(c for c in content if ord(c) < 128).replace('\n', ' ')
        qdrant_score = r.get('score', 0)
        ce_score = ce_model.predict([[query, r.get('payload', {}).get('content', '')[:500]]])[0]
        status = 'REL' if ce_score > -10 else 'IRR'
        print(f'  {i+1}. [{status}] CE:{ce_score:6.2f} Qdrant:{qdrant_score:.3f} | {content_clean}...')

# 2. Hybrid search (what UnifiedMemoryIndex does)
print('\n2. HYBRID SEARCH (RRF fusion):')

# Generate sparse vector
from ace.unified_memory import create_sparse_vector, boost_sparse_vector
query_sparse = create_sparse_vector(query)

# LOW BM25 boost for conversational query (adaptive)
query_sparse_boosted = boost_sparse_vector(query_sparse, 0.3)

hybrid_response = httpx.post(
    f'{QDRANT_URL}/collections/{COLLECTION}/points/query',
    json={
        "prefetch": [
            {"query": query_emb, "using": "dense", "limit": 20},
            {"query": query_sparse_boosted, "using": "sparse", "limit": 50}
        ],
        "query": {"fusion": "rrf"},
        "limit": 5,
        "with_payload": True
    },
    timeout=30.0
)

if hybrid_response.status_code == 200:
    hybrid_results = hybrid_response.json().get('result', {}).get('points', [])
    for i, r in enumerate(hybrid_results):
        content = r.get('payload', {}).get('content', '')[:60]
        content_clean = ''.join(c for c in content if ord(c) < 128).replace('\n', ' ')
        qdrant_score = r.get('score', 0)
        ce_score = ce_model.predict([[query, r.get('payload', {}).get('content', '')[:500]]])[0]
        status = 'REL' if ce_score > -10 else 'IRR'
        print(f'  {i+1}. [{status}] CE:{ce_score:6.2f} Qdrant:{qdrant_score:.3f} | {content_clean}...')

# 3. Hybrid with ZERO BM25 (dense only via hybrid)
print('\n3. HYBRID WITH ZERO BM25 (dense dominates):')

zero_bm25_response = httpx.post(
    f'{QDRANT_URL}/collections/{COLLECTION}/points/query',
    json={
        "prefetch": [
            {"query": query_emb, "using": "dense", "limit": 20},
            {"query": query_sparse_boosted, "using": "sparse", "limit": 1}  # Minimal BM25
        ],
        "query": {"fusion": "rrf"},
        "limit": 5,
        "with_payload": True
    },
    timeout=30.0
)

if zero_bm25_response.status_code == 200:
    zero_results = zero_bm25_response.json().get('result', {}).get('points', [])
    for i, r in enumerate(zero_results):
        content = r.get('payload', {}).get('content', '')[:60]
        content_clean = ''.join(c for c in content if ord(c) < 128).replace('\n', ' ')
        qdrant_score = r.get('score', 0)
        ce_score = ce_model.predict([[query, r.get('payload', {}).get('content', '')[:500]]])[0]
        status = 'REL' if ce_score > -10 else 'IRR'
        print(f'  {i+1}. [{status}] CE:{ce_score:6.2f} Qdrant:{qdrant_score:.3f} | {content_clean}...')

print()
print('=' * 80)
print('CONCLUSION')
print('=' * 80)
print('If Dense-Only returns RELEVANT results but Hybrid doesn\'t,')
print('the solution is to DISABLE BM25 completely for conversational queries.')
