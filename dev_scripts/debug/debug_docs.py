import os
import re

# Simulate the actual function
file_path = "docs/INTEGRATION_GUIDE.md"
query = "try except error handling pattern"

stop_words = {'the', 'and', 'for', 'with', 'this', 'that', 'from', 'how', 'what',
              'where', 'when', 'why', 'can', 'will', 'method', 'function', 'class',
              'code', 'file', 'def', 'implementation', 'search', 'find', 'get', 'set',
              'pattern', 'error', 'handling', 'import', 'logging', 'logger', 'setup',
              'exception', 'try', 'except', 'connection', 'settings', 'configuration'}

path_lower = file_path.lower()
filename = os.path.basename(path_lower)
print(f"file_path: {file_path}")
print(f"path_lower: {path_lower}")
print(f"filename: {filename}")
ext = os.path.splitext(filename)[1].lower()
print(f"ext: {ext}")

# camel_terms extraction
camel_terms = set()
for match in re.finditer(r'[A-Z][a-z]+(?:[A-Z][a-z]+)*', query):
    term = match.group().lower()
    if len(term) >= 3:
        camel_terms.add(term)
print(f"camel_terms: {camel_terms}")

# Check conditions
has_code_entity = bool(camel_terms) or any(
    term in query.lower() for term in ['class', 'dataclass', 'method', 'function', 'def', 'implementation']
)
print(f"has_code_entity: {has_code_entity}")

has_specific_symbol = bool(re.search(r'[A-Z][a-z]+[A-Z]', query)) or \
                      bool(re.search(r'\b\w+_\w+\b', query) and not all(
                          w in stop_words for w in query.lower().split()))
print(f"has_specific_symbol: {has_specific_symbol}")

# Check path conditions
print(f"'docs/' in file_path.lower(): {'docs/' in file_path.lower()}")
print(f"'guide' in file_path.lower(): {'guide' in file_path.lower()}")

# Simulate content
content = "```python\ntry:\n    x = 1\nexcept Exception as e:\n    pass\n```"
print(f"'```' in content: {'```' in content}")
print(f"'try:' in content.lower(): {'try:' in content.lower()}")
