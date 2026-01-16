#!/usr/bin/env python3
"""Test code retrieval directly."""
from ace.code_retrieval import CodeRetrieval

cr = CodeRetrieval(collection_name='agentic-context-engine_code_context')
results = cr.search('UnifiedMemoryIndex implementation', limit=3)
print(f'Found {len(results)} results:')
for r in results:
    file_path = r.get('file_path', '?')
    score = r.get('score', 0)
    content = r.get('content', '')[:150]
    print(f'  - {file_path}: {score:.3f}')
    print(f'    {content}...')
    print()
