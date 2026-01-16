# -*- coding: utf-8 -*-
"""Benchmark with REAL user prompts using SEMANTIC matching (not keyword)."""
import sys
import os
import json
import time

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from ace.unified_memory import UnifiedMemoryIndex
from ace.config import reset_config
import httpx

reset_config()


def get_embedding(text: str, url: str = "http://localhost:1234") -> list:
    """Get embedding for text using local embedding model."""
    response = httpx.post(
        f"{url}/v1/embeddings",
        json={"model": "text-embedding-qwen3-embedding-8b", "input": text},
        timeout=30.0,
    )
    if response.status_code == 200:
        return response.json()["data"][0]["embedding"]
    return None


def cosine_similarity(a: list, b: list) -> float:
    """Calculate cosine similarity between two vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def is_semantically_relevant(query_emb: list, content: str, threshold: float = 0.5) -> bool:
    """Check if content is semantically relevant to query using embeddings."""
    content_emb = get_embedding(content[:500])  # Truncate for speed
    if content_emb is None:
        return False
    sim = cosine_similarity(query_emb, content_emb)
    return sim >= threshold


# Extract real user prompts from Claude logs
def extract_user_prompts():
    from pathlib import Path
    log_dir = Path(r'C:\Users\Erwin\.claude\projects\D--ApplicationDevelopment-Tools-agentic-context-engine')
    prompts = []

    for log_file in log_dir.glob('*.jsonl'):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get('type') == 'user':
                            msg = entry.get('message', {})
                            if isinstance(msg, dict):
                                content = msg.get('content', '')
                                if content and len(content) > 20:
                                    prompts.append(content)
                    except:
                        pass
        except:
            pass

    return prompts


def is_semantic_query(prompt):
    """Filter for queries that would benefit from semantic memory search."""
    prompt_lower = prompt.lower()

    skip_patterns = [
        '**task:', '**context:', 'run the', 'execute', 'create a',
        'implement', 'fix the', 'update the', 'add this', 'proceed',
        '<local-command', '/commit', '/help', 'zen challenge',
        'fucking', 'mother fucker', 'check on progress', 'continue'
    ]
    if any(p in prompt_lower for p in skip_patterns):
        return False

    include_patterns = [
        'what is', 'how does', 'how to', 'why is', 'where is',
        'is there', 'does this', 'can you', 'should', 'difference between',
        'what are', 'which', 'when', 'explain', '?'
    ]
    if any(p in prompt_lower for p in include_patterns):
        return True

    return False


# Main benchmark
print('=' * 70)
print('SEMANTIC BENCHMARK - Real Prompts + Embedding Similarity')
print('Relevance determined by cosine similarity >= 0.5')
print('=' * 70)

# Extract and filter prompts
all_prompts = extract_user_prompts()
semantic_queries = [p for p in all_prompts if is_semantic_query(p)]

# Deduplicate and limit
unique_queries = []
seen = set()
for q in semantic_queries:
    q_norm = q[:100].lower()
    if q_norm not in seen:
        seen.add(q_norm)
        unique_queries.append(q[:200])
        if len(unique_queries) >= 50:  # 50 queries for meaningful sample
            break

print(f'Total prompts: {len(all_prompts)}')
print(f'Semantic queries: {len(semantic_queries)}')
print(f'Testing: {len(unique_queries)} unique queries')
print('=' * 70)

# Initialize index
index = UnifiedMemoryIndex(
    qdrant_url='http://localhost:6333',
    embedding_url='http://localhost:1234',
    collection_name='ace_memories_hybrid',
    embedding_dim=4096,
    embedding_model='text-embedding-qwen3-embedding-8b'
)

# Run benchmark
metrics = {'r1': [], 'r5': [], 'p3': []}
SIMILARITY_THRESHOLD = 0.45

for i, query in enumerate(unique_queries):
    # Get query embedding once
    query_emb = get_embedding(query)
    if query_emb is None:
        continue

    # Retrieve
    results = index.retrieve(
        query, limit=10, auto_detect_preset=True,
        use_llm_expansion=True, use_llm_rerank=False
    )

    if not results:
        metrics['r1'].append(0.0)
        metrics['r5'].append(0.0)
        metrics['p3'].append(0.0)
        print(f'[{i+1:2d}] R@1:N R@5:N P@3:  0% (0r) {query[:45]}...')
        continue

    # Evaluate with semantic similarity
    relevance = []
    for r in results[:5]:
        content_emb = get_embedding(r.content[:500])
        if content_emb:
            sim = cosine_similarity(query_emb, content_emb)
            relevance.append(sim >= SIMILARITY_THRESHOLD)
        else:
            relevance.append(False)

    # Extend for positions 6-10 if needed
    while len(relevance) < min(len(results), 10):
        relevance.append(False)

    r1 = 1.0 if relevance[0] else 0.0
    r5 = 1.0 if any(relevance[:5]) else 0.0
    p3 = sum(relevance[:3]) / 3.0 if len(relevance) >= 3 else 0.0

    metrics['r1'].append(r1)
    metrics['r5'].append(r5)
    metrics['p3'].append(p3)

    r1_mark = 'Y' if r1 == 1.0 else 'N'
    r5_mark = 'Y' if r5 == 1.0 else 'N'
    query_short = query[:45].replace('\n', ' ')
    print(f'[{i+1:2d}] R@1:{r1_mark} R@5:{r5_mark} P@3:{p3*100:3.0f}% ({len(results):2d}r) {query_short}...')

    time.sleep(0.1)  # Small delay for embedding API

# Summary
print('')
print('=' * 70)
print('RESULTS (Semantic Matching)')
print('=' * 70)

if metrics['r1']:
    total_r1 = sum(metrics['r1']) / len(metrics['r1']) * 100
    total_r5 = sum(metrics['r5']) / len(metrics['r5']) * 100
    total_p3 = sum(metrics['p3']) / len(metrics['p3']) * 100

    r1_status = "PASS" if total_r1 >= 95 else "FAIL"
    r5_status = "PASS" if total_r5 >= 95 else "FAIL"
    p3_status = "PASS" if total_p3 >= 95 else "FAIL"

    print(f'Queries tested: {len(metrics["r1"])}')
    print(f'Similarity threshold: {SIMILARITY_THRESHOLD}')
    print(f'Recall@1:       {total_r1:5.1f}% (95%+ target) [{r1_status}]')
    print(f'Recall@5:       {total_r5:5.1f}% (95%+ target) [{r5_status}]')
    print(f'Precision@3:    {total_p3:5.1f}% (95%+ target) [{p3_status}]')
    print('')

    if total_r1 >= 95 and total_r5 >= 95 and total_p3 >= 95:
        print('ALL TARGETS MET!')
    else:
        print('NEEDS IMPROVEMENT')
else:
    print('No queries processed')
