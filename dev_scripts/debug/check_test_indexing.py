#!/usr/bin/env python3
"""Check test file indexing."""
import sys
sys.path.insert(0, ".")

from ace.code_retrieval import CodeRetrieval

r = CodeRetrieval()

# Check test-related queries
queries = [
    "test class pytest fixture",
    "def test_ unit test",
    "tests folder test files",
    "assert equal test",
    "mock patch unittest",
]

for q in queries:
    results = r.search(q, limit=3)
    print(f"\n{q}:")
    for res in results:
        print(f"  {res['file_path']} ({res['score']:.3f})")
