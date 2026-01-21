#!/usr/bin/env python
"""Test raw embedding scores without any boosting."""

from ace.code_retrieval import CodeRetrieval

cr = CodeRetrieval()

# Monkey-patch to disable boosting
cr._apply_filename_boost = lambda f, q, s, c: s

query = "CHANGELOG release notes"
results = cr.search(query, limit=15)

print(f"=== RAW SCORES for '{query}' (No Boosting) ===\n")
for r in results:
    score = r["score"]
    path = r["file_path"]
    print(f"[{score:.4f}] {path}")
