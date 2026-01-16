# -*- coding: utf-8 -*-
"""Trace what UnifiedMemoryIndex.retrieve() is actually returning."""
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

from sentence_transformers import CrossEncoder
ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

# Cross-encoder threshold (centralized in ace/config.py)
CE_THRESHOLD = get_retrieval_config().cross_encoder_threshold

query = 'is this wired up and working in production'

index = UnifiedMemoryIndex(
    qdrant_url='http://localhost:6333',
    embedding_url='http://localhost:1234',
    collection_name='ace_memories_hybrid',
    embedding_dim=4096,
    embedding_model='text-embedding-qwen3-embedding-8b'
)

print('=' * 80)
print('TRACE: UnifiedMemoryIndex.retrieve() output')
print('=' * 80)
print(f'Query: "{query}"')

# Test with different settings
settings = [
    {'auto_detect_preset': True, 'use_cross_encoder': False, 'use_llm_expansion': False},
    {'auto_detect_preset': True, 'use_cross_encoder': True, 'use_llm_expansion': False},
    {'auto_detect_preset': False, 'use_cross_encoder': True, 'use_llm_expansion': False},  # No preset
]

for i, s in enumerate(settings):
    print(f'\n{i+1}. Settings: {s}')
    results = index.retrieve(query, limit=5, **s)

    if results:
        for j, r in enumerate(results[:5]):
            content_clean = ''.join(c for c in r.content[:60] if ord(c) < 128).replace('\n', ' ')
            ce_score = ce_model.predict([[query, r.content[:500]]])[0]
            status = 'REL' if ce_score > -10 else 'IRR'
            qdrant_score = getattr(r, 'qdrant_score', 0)
            print(f'  {j+1}. [{status}] CE:{ce_score:6.2f} Qdrant:{qdrant_score:.3f} | {content_clean}...')
    else:
        print('  No results!')
