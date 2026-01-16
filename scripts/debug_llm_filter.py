# -*- coding: utf-8 -*-
"""Debug LLM filter response parsing."""
import time
import sys
import json
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from ace.unified_memory import UnifiedMemoryIndex
import httpx

# Initialize index
index = UnifiedMemoryIndex(
    qdrant_url='http://localhost:6333',
    embedding_url='http://localhost:1234',
    collection_name='ace_memories_hybrid',
    embedding_dim=4096,
    embedding_model='text-embedding-qwen3-embedding-8b'
)

query = 'How does hybrid search work with vectors and keywords?'
results = index.retrieve(query, limit=5, auto_detect_preset=True, use_llm_expansion=False, use_llm_rerank=False)
results_with_scores = [(b, getattr(b, 'qdrant_score', 0.5)) for b in results]

print('=== DEBUG LLM FILTER ===')
print(f'Query: {query}')
print(f'Candidates: {len(results_with_scores)}')

# Build candidate texts
candidate_texts = []
for i, (result, score) in enumerate(results_with_scores):
    content = getattr(result, 'content', str(result))[:200]
    candidate_texts.append(f'{i+1}. {content}')

prompt = f'''You are a relevance filter. Given a search query and candidate results,
identify ONLY the results that are DIRECTLY RELEVANT to answering the query.

Query: "{query}"

Candidates:
{chr(10).join(candidate_texts)}

Return a JSON object with:
- "relevant": array of result numbers that ARE relevant, ordered by relevance
- "irrelevant": array of result numbers that are NOT relevant

JSON response:'''

print()
print('=== Calling Local LLM ===')
llm_url = 'http://localhost:1234'
endpoint = f'{llm_url}/v1/chat/completions'

t0 = time.time()
response = httpx.post(
    endpoint,
    headers={'Content-Type': 'application/json'},
    json={
        'model': 'gpt-oss-20b',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 200,
        'temperature': 0.1,
    },
    timeout=30.0,
)
elapsed = time.time() - t0

print(f'Status: {response.status_code}')
print(f'Time: {elapsed:.2f}s')

if response.status_code == 200:
    result_json = response.json()
    content = result_json.get('choices', [{}])[0].get('message', {}).get('content', '')
    print(f'Raw content: {repr(content[:500])}')

    # Try to parse
    try:
        if '```' in content:
            content = content.split('```')[1].replace('json', '').strip()
        judgment = json.loads(content)
        print(f'Parsed JSON: {judgment}')
        print(f'Relevant indices: {judgment.get("relevant", [])}')
    except Exception as e:
        print(f'JSON parse error: {e}')
        print(f'Content after cleanup: {repr(content[:300])}')
else:
    print(f'Error: {response.text[:500]}')
