#!/usr/bin/env python3
"""Debug embedding query ranking to understand why voyage-related files rank lower."""

from ace.code_retrieval import CodeRetrieval

r = CodeRetrieval()

# Test embedding query with detailed scores
query = "embedding vector generation voyage-code-3 model"
results = r.search(query, limit=10)

print("Query:", query)
print()
for i, res in enumerate(results, 1):
    score = res["score"]
    file_path = res["file_path"]
    start_line = res["start_line"]
    end_line = res["end_line"]
    
    # Check if voyage is in content
    content = res.get("content", "")
    voyage_count = content.lower().count("voyage")
    
    print(f"{i}. [{score:.4f}] {file_path}")
    print(f"   Lines: {start_line}-{end_line}")
    print(f"   'voyage' mentions: {voyage_count}")
    
    # Show first 200 chars of content
    snippet = content[:200].replace("\n", " ")
    print(f"   Snippet: {snippet}...")
    print()
