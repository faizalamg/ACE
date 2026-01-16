#!/usr/bin/env python3
"""Debug demo penalty."""
import sys
sys.path.insert(0, '.')
import logging
logging.basicConfig(level=logging.DEBUG)
from ace.code_retrieval import CodeRetrieval

cr = CodeRetrieval()
r = cr.search('email validation regex pattern', limit=5)
for i, x in enumerate(r, 1):
    print(f"{i}. [{x.get('score', 0):.3f}] {x.get('file_path', '')}")
