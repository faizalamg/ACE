#!/usr/bin/env python3
"""Debug the httpx query where ThatOtherContextEngine wins."""
import sys
sys.path.insert(0, ".")

from ace.code_retrieval import CodeRetrieval

r = CodeRetrieval()
query = "import httpx async"
print(f"Query: {query}")
print("-" * 50)

results = r.search(query, limit=5)
for i, res in enumerate(results, 1):
    print(f"{i}. {res['file_path']} (score: {res['score']:.3f})")

# Check what files actually import httpx
print("\n" + "=" * 50)
print("Files that import httpx:")
import os
from pathlib import Path

for root, dirs, files in os.walk("ace"):
    dirs[:] = [d for d in dirs if not d.startswith(("__", "."))]
    for f in files:
        if f.endswith(".py"):
            fp = Path(root) / f
            content = fp.read_text(encoding='utf-8', errors='ignore')
            if "import httpx" in content:
                lines = [l.strip() for l in content.split('\n') if 'httpx' in l][:3]
                print(f"  {fp}: {lines[0] if lines else ''}")
