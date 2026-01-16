#!/usr/bin/env python3
"""Test indexing speed after batch optimization."""
import time
import sys
import os
sys.path.insert(0, ".")

from ace.code_indexer import CodeIndexer

print("=== ACE Workspace Reindex Speed Test ===")
print(f"Workspace: {os.getcwd()}")
print()

# Create indexer
start_total = time.time()
indexer = CodeIndexer(workspace_path=".", respect_gitignore=True)

# Get current collection count
try:
    info = indexer._client.get_collection(indexer.collection_name)
    print(f"Current collection: {info.points_count} points")
except Exception as e:
    print(f"Collection info error: {e}")

# Reindex
print()
print("Starting reindex...")
start_index = time.time()
stats = indexer.index_workspace()
index_time = time.time() - start_index

# Results
print()
print("=== Results ===")
print(f"Files indexed: {stats['files_indexed']}")
print(f"Chunks indexed: {stats['chunks_indexed']}")
print(f"Files skipped: {stats['files_skipped']}")
print(f"Errors: {len(stats.get('errors', []))}")
print()
print(f"Index time: {index_time:.2f}s")
if index_time > 0 and stats['chunks_indexed'] > 0:
    print(f"Chunks/second: {stats['chunks_indexed'] / index_time:.1f}")
print()
total_time = time.time() - start_total
print(f"Total time: {total_time:.2f}s")

# Show batch config
from ace.config import VoyageCodeEmbeddingConfig
cfg = VoyageCodeEmbeddingConfig()
print()
print("=== Batch Config ===")
print(f"Batch size: {cfg.batch_size}")
print(f"Max tokens/batch: {cfg.batch_max_tokens}")
print(f"Concurrent batches: {cfg.parallel_batches}")
