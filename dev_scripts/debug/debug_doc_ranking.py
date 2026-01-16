"""Debug doc ranking for pattern queries."""

from ace.code_retrieval import CodeRetrieval

r = CodeRetrieval()

# Check if docs even get embedded/indexed
query = "error handling patterns in Python with try except"
print(f"Query: {query}")
print("=" * 60)

# Run search with more results to see docs
results = r.search(query, limit=50)

# Separate code and docs
code_files = [res for res in results if not res['file_path'].endswith('.md')]
doc_files = [res for res in results if res['file_path'].endswith('.md')]

print(f"\nTotal results: {len(results)}")
print(f"Code files: {len(code_files)}")
print(f"Doc files: {len(doc_files)}")

if doc_files:
    print("\nDoc files in results:")
    for doc in doc_files[:10]:
        print(f"  {doc['file_path']}: {doc['score']:.4f} (orig: {doc['original_score']:.4f})")
else:
    print("\nNo docs in results - checking raw embedding scores...")
    
    # Try direct search without boosting
    print("\nTop 10 code results (with scores):")
    for res in code_files[:10]:
        print(f"  {res['file_path']}: {res['score']:.4f} (orig: {res['original_score']:.4f})")
