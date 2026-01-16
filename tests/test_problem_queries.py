#!/usr/bin/env python3
"""Test problematic queries."""
import sys
sys.path.insert(0, '.')
from ace.code_retrieval import CodeRetrieval

cr = CodeRetrieval()
queries = [
    ('email validation regex pattern', 'email_validator.py'),
    ('config from dict', 'ace/config.py'),
    ('error recovery strategy', 'ace/resilience.py'),
    ('file not found error handling', 'ace/code_indexer.py'),
    ('integration guide howto', 'docs/INTEGRATION_GUIDE.md'),
]
for q, expected in queries:
    print(f"\n=== {q} ===")
    print(f"    Expected: {expected}")
    r = cr.search(q, limit=5)
    for i, x in enumerate(r, 1):
        fp = x.get('file_path', '')
        score = x.get('score', 0)
        marker = " <-- EXPECTED" if expected.split('/')[-1] in fp else ""
        print(f"    {i}. [{score:.3f}] {fp}{marker}")
