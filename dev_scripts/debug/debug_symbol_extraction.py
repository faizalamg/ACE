#!/usr/bin/env python3
"""Debug specific symbol extraction to understand why voyage-code-3 isn't matching."""

import re

query = "embedding vector generation voyage-code-3 model"
query_lower = query.lower()

# Stop words from code_retrieval.py
stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'and', 'or',
              'is', 'it', 'by', 'as', 'from', 'that', 'this', 'be', 'are', 'was', 'were',
              'method', 'function', 'class', 'module', 'variable', 'constant', 'value',
              'code', 'file', 'def', 'implementation', 'search', 'find', 'get', 'set',
              'pattern', 'error', 'handling', 'import', 'logging', 'logger', 'setup',
              'exception', 'try', 'except', 'connection', 'settings', 'configuration'}

# Extract query terms (same logic as code_retrieval.py)
query_terms = set()
for term in re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', query_lower):
    if len(term) >= 3 and term not in stop_words:
        query_terms.add(term)

print(f"Query: {query}")
print(f"Query terms: {query_terms}")

# Extract CamelCase terms
camel_terms = set()
for match in re.finditer(r'[A-Z][a-z]+(?:[A-Z][a-z]+)*', query):
    term = match.group().lower()
    if len(term) >= 3:
        camel_terms.add(term)
        
print(f"CamelCase terms: {camel_terms}")

# Extract specific symbols (defined names)
specific_symbols = set()
# Pattern 1: _underscore_names (like _expand_query)
for match in re.finditer(r'_[a-zA-Z][a-zA-Z0-9_]+', query):
    specific_symbols.add(match.group().lower())
# Pattern 2: snake_case names (like code_retrieval)  
for match in re.finditer(r'[a-z][a-z0-9]*_[a-z0-9_]+', query_lower):
    # Filter out common phrases
    phrase = match.group()
    if phrase not in stop_words and not any(w in stop_words for w in phrase.split('_')):
        specific_symbols.add(phrase)
# Pattern 3: PascalCase/CamelCase (like CodeRetrieval)
for match in re.finditer(r'[A-Z][a-z]+(?:[A-Z][a-z]+)+', query):
    specific_symbols.add(match.group().lower())
# Pattern 4: exact function signatures like "def method_name("
for match in re.finditer(r'def\s+(\w+)\(', query_lower):
    specific_symbols.add(match.group(1))
    
print(f"Specific symbols: {specific_symbols}")

# Check if voyage-code-3 is being captured anywhere
print(f"\n'voyage' in query_terms: {'voyage' in query_terms}")
print(f"'voyage-code-3' in query: {'voyage-code-3' in query_lower}")

# Test content matching
test_content = """
Uses Voyage-code-3 embeddings (1024d) for optimal code semantic
understanding compared to general-purpose embeddings.
"""
content_lower = test_content.lower()
print(f"\n'voyage' in content: {'voyage' in content_lower}")
print(f"'voyage-code-3' in content: {'voyage-code-3' in content_lower}")
print(f"Content 'voyage' count: {content_lower.count('voyage')}")
