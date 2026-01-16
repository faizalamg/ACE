# -*- coding: utf-8 -*-
"""Find the RIGHT threshold for 95%+ REAL precision."""
import sys
import os
import math

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from ace.unified_memory import UnifiedMemoryIndex
from ace.config import reset_config
import httpx

reset_config()

def get_embedding(text, url='http://localhost:1234'):
    response = httpx.post(
        f'{url}/v1/embeddings',
        json={'model': 'text-embedding-qwen3-embedding-8b', 'input': text},
        timeout=30.0,
    )
    if response.status_code == 200:
        return response.json()['data'][0]['embedding']
    return None

def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

index = UnifiedMemoryIndex(
    qdrant_url='http://localhost:6333',
    embedding_url='http://localhost:1234',
    collection_name='ace_memories_hybrid',
    embedding_dim=4096,
    embedding_model='text-embedding-qwen3-embedding-8b'
)

# The query that exposed the problem
query = 'is this wired up and working in production'
query_emb = get_embedding(query)

results = index.retrieve(query, limit=15, auto_detect_preset=True, use_llm_expansion=True, use_llm_rerank=False)

print('=' * 80)
print('RAW SIMILARITY SCORES - HONEST ASSESSMENT')
print('=' * 80)
print(f'Query: "{query}"')
print()

scores = []
for i, r in enumerate(results[:15]):
    content_emb = get_embedding(r.content[:300])
    if content_emb:
        sim = cosine_sim(query_emb, content_emb)
        # Strip emojis for display
        content_clean = ''.join(c for c in r.content[:70] if ord(c) < 128).replace('\n', ' ')
        scores.append((sim, content_clean, r.content))
        print(f'{i+1:2d}. sim={sim:.3f} | {content_clean}...')

print()
print('=' * 80)
print('THRESHOLD ANALYSIS')
print('=' * 80)

# Test different thresholds
for thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
    above = sum(1 for s, _, _ in scores if s >= thresh)
    print(f'Threshold {thresh}: {above}/15 results pass ({above/15*100:.0f}%)')
