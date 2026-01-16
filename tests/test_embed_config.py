"""Quick test for embedding model configuration query."""
import os
os.environ['VOYAGE_API_KEY'] = 'pa-3OeAEQVzfUwm9e0UKyW3L1JST8HJ4WT2sReaIv8GdDr'
from ace.code_retrieval import CodeRetrieval

r = CodeRetrieval()
results = r.search('embedding model configuration', limit=5)

print('Query: embedding model configuration')
print('=' * 70)
for i, res in enumerate(results):
    path = res.get('file_path', 'N/A')
    score = res.get('score', 0)
    orig = res.get('original_score', 0)
    print(f'{i+1}. {path}')
    print(f'   score={score:.4f} orig={orig:.4f} boost={score-orig:.4f}')
