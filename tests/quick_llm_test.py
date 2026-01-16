#!/usr/bin/env python
"""Quick test of LLM expansion on challenging queries."""

from ace.unified_memory import UnifiedMemoryIndex
from ace.retrieval_presets import expand_query_with_llm

idx = UnifiedMemoryIndex()

queries = [
    'make it faster',
    'broken code',
    'how to test',
    'security issue',
    'config problem',
]

for query in queries:
    print(f'Query: "{query}"')
    expansions = expand_query_with_llm(query)
    print(f'LLM expansions: {expansions}')
    
    # Without LLM
    results_no = idx.retrieve(query, limit=2, use_llm_expansion=False)
    print('WITHOUT LLM:')
    for r in results_no:
        print(f'  - {r.content[:70]}...')
    
    # With LLM
    results_yes = idx.retrieve(query, limit=2, use_llm_expansion=True)
    print('WITH LLM:')
    for r in results_yes:
        print(f'  - {r.content[:70]}...')
    
    # Check if different
    ids_no = set(r.id for r in results_no)
    ids_yes = set(r.id for r in results_yes)
    same = ids_no == ids_yes
    print(f'SAME RESULTS: {same}')
    print()
