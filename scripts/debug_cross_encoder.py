# -*- coding: utf-8 -*-
"""Debug why cross-encoder is not executing in retrieve()."""
import sys
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from ace.unified_memory import UnifiedMemoryIndex
from ace.config import reset_config

reset_config()

# Test query that exposed the problem
query = 'is this wired up and working in production'

index = UnifiedMemoryIndex(
    qdrant_url='http://localhost:6333',
    embedding_url='http://localhost:1234',
    collection_name='ace_memories_hybrid',
    embedding_dim=4096,
    embedding_model='text-embedding-qwen3-embedding-8b'
)

print('=' * 80)
print('DEBUG: Cross-encoder execution trace')
print('=' * 80)

# Test 1: WITHOUT cross-encoder
print('\nTEST 1: WITHOUT cross-encoder')
results_no_ce = index.retrieve(query, limit=5, auto_detect_preset=True, use_llm_expansion=False, use_cross_encoder=False)
for i, r in enumerate(results_no_ce):
    content_clean = ''.join(c for c in r.content[:60] if ord(c) < 128).replace('\n', ' ')
    print(f'  {i+1}. {content_clean}...')

# Test 2: WITH cross-encoder
print('\nTEST 2: WITH cross-encoder (should reorder!)')
results_with_ce = index.retrieve(query, limit=5, auto_detect_preset=True, use_llm_expansion=False, use_cross_encoder=True)
for i, r in enumerate(results_with_ce):
    content_clean = ''.join(c for c in r.content[:60] if ord(c) < 128).replace('\n', ' ')
    print(f'  {i+1}. {content_clean}...')

# Check if order is same
no_ce_order = [r.content[:50] for r in results_no_ce]
with_ce_order = [r.content[:50] for r in results_with_ce]

print()
if no_ce_order == with_ce_order:
    print('ORDER: SAME (cross-encoder NOT working)')
else:
    print('ORDER: DIFFERENT (cross-encoder IS working!)')

# Direct cross-encoder test
print()
print('=' * 80)
print('DIRECT CROSS-ENCODER TEST (standalone)')
print('=' * 80)
try:
    from sentence_transformers import CrossEncoder
    ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

    # Use the actual results from retrieval
    pairs = [[query, r.content[:500]] for r in results_no_ce]
    ce_scores = ce_model.predict(pairs)

    print(f'Query: "{query}"')
    print()
    for i, (r, score) in enumerate(zip(results_no_ce, ce_scores)):
        content_clean = ''.join(c for c in r.content[:50] if ord(c) < 128).replace('\n', ' ')
        print(f'  {i+1}. score={score:.3f} | {content_clean}...')

    # Show what the order SHOULD be
    scored = list(zip(results_no_ce, ce_scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    print()
    print('CORRECT ORDER (by cross-encoder score):')
    for i, (r, score) in enumerate(scored):
        content_clean = ''.join(c for c in r.content[:50] if ord(c) < 128).replace('\n', ' ')
        print(f'  {i+1}. score={score:.3f} | {content_clean}...')

except ImportError as e:
    print(f'CrossEncoder import failed: {e}')
except Exception as e:
    print(f'CrossEncoder error: {e}')
