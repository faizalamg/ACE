# -*- coding: utf-8 -*-
"""Diagnose why certain queries fail - check what content exists."""
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

def search_with_bm25(query, limit=20):
    """Direct BM25 search in Qdrant"""
    # First get query embedding
    query_emb = get_embedding(query)
    if not query_emb:
        return []

    # Hybrid search with heavy BM25 weight
    payload = {
        "query": {
            "prefetch": [
                {"query": query_emb, "using": "dense", "limit": limit * 3},
                {"query": {"indices": list(range(10)), "values": [1.0] * 10}, "using": "sparse", "limit": limit * 5}
            ],
            "query": {"fusion": "rrf"},
            "limit": limit,
            "with_payload": True
        }
    }

    # Actually let's do a simple scroll to see what content exists
    scroll_response = httpx.post(
        f'{QDRANT_URL}/collections/{COLLECTION}/points/scroll',
        json={"limit": 100, "with_payload": True},
        timeout=30.0
    )

    if scroll_response.status_code == 200:
        return scroll_response.json().get('result', {}).get('points', [])
    return []

# Failed queries from honest assessment
failed_queries = [
    'is this wired up and working in production',
    'explain the playbook format',
]

print('=' * 90)
print('DIAGNOSING FAILED QUERIES')
print('=' * 90)

# Get sample of what's in the database
points = search_with_bm25("test", limit=100)
print(f'\nTotal memories sampled: {len(points)}')

for query in failed_queries:
    print()
    print(f'QUERY: "{query}"')
    print('-' * 70)

    # Search for potentially relevant content
    keywords = query.lower().split()

    # Find any memories mentioning key terms
    relevant_content = []
    for p in points:
        content = p.get('payload', {}).get('content', '').lower()
        # Check for keyword matches
        matches = sum(1 for kw in keywords if kw in content)
        if matches > 0:
            relevant_content.append((p, matches))

    relevant_content.sort(key=lambda x: x[1], reverse=True)

    if relevant_content:
        print(f'Found {len(relevant_content)} memories with keyword overlap:')
        for p, match_count in relevant_content[:5]:
            content = p.get('payload', {}).get('content', '')[:70]
            content_clean = ''.join(c for c in content if ord(c) < 128).replace('\n', ' ')

            # Check cross-encoder score
            ce_score = ce_model.predict([[query, p.get('payload', {}).get('content', '')[:500]]])[0]
            status = 'REL' if ce_score > -10 else 'IRR'

            print(f'  [{status}] CE:{ce_score:6.2f} kw:{match_count} | {content_clean}...')
    else:
        print('  NO memories found with keyword overlap!')
        print('  This query may need SEMANTIC search, not keywords.')

    # Now try semantic search via embedding similarity
    print()
    print(f'Semantic search results for "{query}":')
    query_emb = get_embedding(query)
    if query_emb:
        # Search with dense vectors
        search_response = httpx.post(
            f'{QDRANT_URL}/collections/{COLLECTION}/points/query',
            json={
                "query": query_emb,
                "using": "dense",
                "limit": 10,
                "with_payload": True
            },
            timeout=30.0
        )

        if search_response.status_code == 200:
            results = search_response.json().get('result', {}).get('points', [])
            for i, r in enumerate(results[:5]):
                content = r.get('payload', {}).get('content', '')[:70]
                content_clean = ''.join(c for c in content if ord(c) < 128).replace('\n', ' ')
                qdrant_score = r.get('score', 0)

                # Cross-encoder score
                ce_score = ce_model.predict([[query, r.get('payload', {}).get('content', '')[:500]]])[0]
                status = 'REL' if ce_score > -10 else 'IRR'

                print(f'  {i+1}. [{status}] CE:{ce_score:6.2f} Qdrant:{qdrant_score:.3f} | {content_clean}...')

print()
print('=' * 90)
print('DIAGNOSIS COMPLETE')
print('=' * 90)
print()
print('Key insight: If NO memories have relevant content for a query,')
print('then retrieval CANNOT succeed - the information simply does not exist.')
print('Solution: Either add relevant memories, or accept certain queries will fail.')
