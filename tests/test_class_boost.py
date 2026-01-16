#!/usr/bin/env python3
"""Test class name boost fix."""
import sys
sys.path.insert(0, '.')
from ace.code_retrieval import CodeRetrieval

cr = CodeRetrieval()
queries = [
    'HyDEGenerator class generate method',
    'ASTChunker class chunk method',
    'BM25Config k1 b parameters',
]
for q in queries:
    print(f"\n=== {q} ===")
    r = cr.search(q, limit=5)
    for i, x in enumerate(r, 1):
        print(f"  {i}. [{x.get('score', 0):.3f}] {x.get('file_path', '')}")
