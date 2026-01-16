# -*- coding: utf-8 -*-
"""Verify cross-encoder reranking is working correctly."""
import sys
import os
import json

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from ace.unified_memory import UnifiedMemoryIndex
from ace.config import reset_config
from sentence_transformers import CrossEncoder

reset_config()

print('Loading cross-encoder...')
CE_MODEL = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

index = UnifiedMemoryIndex(
    qdrant_url='http://localhost:6333',
    embedding_url='http://localhost:1234',
    collection_name='ace_memories_hybrid',
    embedding_dim=4096,
    embedding_model='text-embedding-qwen3-embedding-8b'
)

# Test query that failed
query = "explain to me how ACE stacks up with ThatOtherContextEngine and other major players out there"

print(f'\nQuery: {query[:80]}...\n')
print('='*80)

# Test WITHOUT cross-encoder reranking
print('BEFORE reranking (use_cross_encoder=False):')
results_before = index.retrieve(
    query, limit=10, threshold=0.0,
    auto_detect_preset=True,
    use_cross_encoder=False
)

pairs = [[query, r.content[:500]] for r in results_before[:5]]
scores_before = CE_MODEL.predict(pairs)
scores_list_before = scores_before.tolist() if hasattr(scores_before, 'tolist') else list(scores_before)

for i, (r, s) in enumerate(zip(results_before[:5], scores_list_before)):
    print(f'  [{i+1}] CE={s:.2f} | {r.content[:60]}...')

print(f'\n  Best in top 5: CE={max(scores_list_before):.2f} at rank {scores_list_before.index(max(scores_list_before))+1}')

# Test WITH cross-encoder reranking
print('\n' + '='*80)
print('AFTER reranking (use_cross_encoder=True):')
results_after = index.retrieve(
    query, limit=10, threshold=0.0,
    auto_detect_preset=True,
    use_cross_encoder=True
)

pairs = [[query, r.content[:500]] for r in results_after[:5]]
scores_after = CE_MODEL.predict(pairs)
scores_list_after = scores_after.tolist() if hasattr(scores_after, 'tolist') else list(scores_after)

for i, (r, s) in enumerate(zip(results_after[:5], scores_list_after)):
    print(f'  [{i+1}] CE={s:.2f} | {r.content[:60]}...')

print(f'\n  Best in top 5: CE={max(scores_list_after):.2f} at rank {scores_list_after.index(max(scores_list_after))+1}')

# Verify reranking worked
print('\n' + '='*80)
print('VERIFICATION:')
if scores_list_after[0] >= scores_list_after[1]:
    print('  PASS: Top-1 has highest CE score')
else:
    print('  FAIL: Top-1 does NOT have highest CE score!')
    print(f'  -> Rank 1: {scores_list_after[0]:.2f}')
    print(f'  -> Rank 2: {scores_list_after[1]:.2f}')

# Check if scores are monotonically decreasing
is_sorted = all(scores_list_after[i] >= scores_list_after[i+1] for i in range(len(scores_list_after)-1))
if is_sorted:
    print('  PASS: Results are sorted by CE score')
else:
    print('  FAIL: Results are NOT sorted by CE score!')
