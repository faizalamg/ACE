#!/usr/bin/env python3
"""Test blended code retrieval."""
import os
os.environ["VOYAGE_API_KEY"] = "pa-3OeAEQVzfUwm9e0UKyW3L1JST8HJ4WT2sReaIv8GdDr"

from ace.code_retrieval import CodeRetrieval

retriever = CodeRetrieval()
results = retriever.search('QdrantConfig dataclass', limit=3)
print('=== Blended Search Results ===')
for r in results:
    print(f"  {r['file_path']} (score: {r['score']:.3f})")
    print(f"    Lines {r['start_line']}-{r['end_line']}")
