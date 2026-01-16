#!/usr/bin/env python3
"""Debug class name boost."""
import sys
sys.path.insert(0, '.')
import logging
logging.basicConfig(level=logging.DEBUG)
from ace.code_retrieval import CodeRetrieval

cr = CodeRetrieval()
r = cr.search('HyDEGenerator class generate method', limit=5)
for i, x in enumerate(r, 1):
    print(f"{i}. [{x.get('score', 0):.3f}] {x.get('file_path', '')}")
