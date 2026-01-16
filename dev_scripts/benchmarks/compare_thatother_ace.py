"""Compare ThatOtherContextEngine vs ACE code retrieval quality."""
import subprocess
import sys

query = "CodeRetrieval class search method"

# Get ThatOtherContextEngine results
print("=== ThatOtherContextEngine OUTPUT ===")
result = subprocess.run(f'ThatOtherContextEngine context "{query}"', capture_output=True, text=True, timeout=30, shell=True)
print(result.stdout[:3000] if result.stdout else f"Error: {result.stderr}")

# Get ACE results
print("\n=== ACE OUTPUT ===")
from ace.code_retrieval import CodeRetrieval
retriever = CodeRetrieval()
results = retriever.search(query, limit=5, exclude_tests=True)
ace_out = retriever.format_ThatOtherContextEngine_style(results)
print(ace_out[:3000])
