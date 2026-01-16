"""Full ACE results for detailed quality comparison against ThatOtherContextEngine."""
from ace.code_retrieval import CodeRetrieval

# Test queries - need to beat ThatOtherContextEngine on ALL
queries = [
    "UnifiedMemoryIndex store method implementation",
    "CodeRetrieval class search method implementation",
    "_apply_filename_boost function in code retrieval",
]

cr = CodeRetrieval()
for query in queries:
    results = cr.search(query, limit=5)
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print(f"{'='*70}")
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['score']:.3f} | {r['file_path']} (lines {r['start_line']}-{r['end_line']})")
        # Show first 150 chars of content
        content_preview = r['content'][:150].replace('\n', ' ')
        print(f"    → {content_preview}...")
