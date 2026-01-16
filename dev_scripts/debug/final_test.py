"""Final test of code retrieval."""
from ace.code_retrieval import CodeRetrieval

retriever = CodeRetrieval()
query = "CodeRetrieval class search method implementation"

print(f"Query: {query}")
print(f"\n=== ACE Code Retrieval Results ===")
results = retriever.search(query, limit=15)

for i, r in enumerate(results):
    marker = ""
    if r['file_path'] == 'ace/code_retrieval.py':
        if r['start_line'] == 149 or r['start_line'] == 148:
            marker = " <<< SEARCH METHOD"
        elif r['start_line'] == 40:
            marker = " <<< CLASS DEFINITION"
    print(f"  {i+1}. score={r['score']:.4f} | {r['file_path']}:{r['start_line']}-{r['end_line']}{marker}")

# Also test the ThatOtherContextEngine-style output
print("\n" + "="*60)
print("=== ThatOtherContextEngine-style formatted output ===")
print("="*60)
# FIX: format_ThatOtherContextEngine_style expects List[Dict], not Dict with "results" key
output = retriever.format_ThatOtherContextEngine_style(results[:5])
print(output[:2000] if len(output) > 2000 else output)
