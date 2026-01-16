"""Test code retrieval with Voyage embeddings."""
import os
os.environ['VOYAGE_API_KEY'] = 'pa-3OeAEQVzfUwm9e0UKyW3L1JST8HJ4WT2sReaIv8GdDr'

from ace.code_retrieval import CodeRetrieval

cr = CodeRetrieval()
print('Testing code retrieval with Voyage embeddings...')
print(f'Collection: {cr.collection_name}')

results = cr.search('UnifiedMemoryIndex class retrieval implementation', limit=3)

if results:
    print(f'\nFound {len(results)} results:\n')
    for i, r in enumerate(results):
        print(f'--- Result {i+1} ---')
        print(f'Type: {type(r)}')
        
        # Handle both CodeResult dataclass and dict
        if hasattr(r, 'score'):
            print(f'Score: {r.score:.4f}')
            print(f'File: {r.file_path}')
            print(f'Lines: {r.start_line}-{r.end_line}')
            print(f'Symbols: {r.symbols}')
            print(f'Preview: {r.content[:250]}...')
        elif isinstance(r, dict):
            print(f'Dict keys: {list(r.keys())}')
            print(f'File: {r.get("file_path", "N/A")}')
            print(f'Score: {r.get("score", "N/A")}')
            print(f'Lines: {r.get("start_line", "?")}-{r.get("end_line", "?")}')
            content = r.get('content', '')[:250]
            print(f'Preview: {content}...')
        print()
else:
    print('No results')

# Test formatted output
print('\n=== ThatOtherContextEngine-STYLE FORMATTED OUTPUT ===\n')
formatted = cr.format_ThatOtherContextEngine_style(results)
print(formatted[:2000])
