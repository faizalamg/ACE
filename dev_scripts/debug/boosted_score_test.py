#!/usr/bin/env python
"""Test boosted scores to see what CHANGELOG.md gets."""

from ace.code_retrieval import CodeRetrieval

cr = CodeRetrieval()

query = "CHANGELOG release notes"
results = cr.search(query, limit=15)

print(f"=== BOOSTED SCORES for '{query}' ===\n")
for r in results:
    score = r["score"]
    orig = r["original_score"]
    path = r["file_path"]
    boost = score - orig
    print(f"[{score:.4f}] ({orig:.4f} + {boost:+.3f}) {path}")
