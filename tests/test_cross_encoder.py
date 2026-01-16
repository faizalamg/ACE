"""Debug cross-encoder reranking query."""
import os
os.environ['VOYAGE_API_KEY'] = 'pa-3OeAEQVzfUwm9e0UKyW3L1JST8HJ4WT2sReaIv8GdDr'
from ace.code_retrieval import CodeRetrieval

r = CodeRetrieval()
results = r.search('cross-encoder reranking', limit=10)

print('Query: cross-encoder reranking')
print('=' * 80)
for i, res in enumerate(results):
    path = res.get('file_path', 'N/A')
    score = res.get('score', 0)
    orig = res.get('original_score', 0)
    boost = score - orig
    marker = '<-- TARGET' if 'reranker.py' in path else ''
    marker2 = '<-- RAG_TRAINING' if 'rag_training' in path else ''
    print(f'{i+1}. {path}')
    print(f'   score={score:.4f} orig={orig:.4f} boost={boost:+.4f} {marker}{marker2}')
