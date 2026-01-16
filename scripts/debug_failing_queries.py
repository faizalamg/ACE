# -*- coding: utf-8 -*-
"""Debug failing queries to understand why they miss R@1.

Goal: Identify patterns in the 10% of queries that fail cross-encoder relevance.
"""
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
from ace.config import reset_config, get_retrieval_config
from sentence_transformers import CrossEncoder

reset_config()

print('Loading cross-encoder...')
CE_MODEL = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
# Cross-encoder threshold (centralized in ace/config.py)
# Default: -11.5 targets 95%+ P@3 with high recall
CE_THRESHOLD = get_retrieval_config().cross_encoder_threshold


def extract_test_queries():
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
    prompt_lower = prompt.lower()
    
    skip_patterns = [
        '**task:', '**context:', 'run the', 'execute', 'create a',
        'implement', 'fix the', 'update the', 'add this', 'proceed',
        '<local-command', '/commit', '/help', 'zen challenge', 'continue'
    ]
    if any(p in prompt_lower for p in skip_patterns):
        return False
    
    include_patterns = [
        'what is', 'how does', 'how to', 'why is', 'where is',
        'is there', 'does this', 'can you', 'should', 'difference between',
        'what are', 'which', 'when', 'explain', '?'
    ]
    return any(p in prompt_lower for p in include_patterns)


index = UnifiedMemoryIndex(
    qdrant_url='http://localhost:6333',
    embedding_url='http://localhost:1234',
    collection_name='ace_memories_hybrid',
    embedding_dim=4096,
    embedding_model='text-embedding-qwen3-embedding-8b'
)

all_prompts = extract_test_queries()
semantic_queries = [p for p in all_prompts if is_semantic_query(p)]

unique_queries = []
seen = set()
for q in semantic_queries:
    q_norm = q[:100].lower()
    if q_norm not in seen:
        seen.add(q_norm)
        unique_queries.append(q[:200])
        if len(unique_queries) >= 50:
            break

print(f'Testing {len(unique_queries)} queries...\n')
print('='*80)
print('FAILING QUERIES ANALYSIS')
print('='*80)

failures = []

for i, query in enumerate(unique_queries):
    results = index.retrieve(
        query, limit=10, threshold=0.0,
        auto_detect_preset=True,
        use_cross_encoder=True
    )
    
    if not results:
        failures.append({
            'query': query,
            'reason': 'NO_RESULTS',
            'results': []
        })
        continue
    
    pairs = [[query, r.content[:500]] for r in results[:5]]
    scores = CE_MODEL.predict(pairs)
    scores_list = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
    
    r1 = scores_list[0] >= CE_THRESHOLD if scores_list else False
    
    if not r1:
        failures.append({
            'query': query,
            'reason': 'TOP1_NOT_RELEVANT',
            'ce_scores': scores_list,
            'top3_content': [r.content[:150] for r in results[:3]]
        })
    
    time.sleep(0.05)

print(f'\nTotal Failures: {len(failures)}/{len(unique_queries)} ({len(failures)/len(unique_queries)*100:.1f}%)\n')

for i, f in enumerate(failures):
    print(f'\n{"="*80}')
    print(f'FAILURE {i+1}: {f["reason"]}')
    print(f'{"="*80}')
    print(f'Query: {f["query"][:100]}...')
    
    if f['reason'] == 'NO_RESULTS':
        print('  -> No results returned at all')
    else:
        print(f'  -> CE Scores: {[round(s, 2) for s in f.get("ce_scores", [])[:3]]}')
        print(f'  -> Threshold: {CE_THRESHOLD}')
        print('\n  Top 3 Retrieved:')
        for j, content in enumerate(f.get('top3_content', [])[:3]):
            print(f'  [{j+1}] {content}...')

print('\n' + '='*80)
print('FAILURE PATTERN ANALYSIS')
print('='*80)

# Categorize failures
no_results = [f for f in failures if f['reason'] == 'NO_RESULTS']
top1_irrelevant = [f for f in failures if f['reason'] == 'TOP1_NOT_RELEVANT']

print(f'\n1. NO_RESULTS: {len(no_results)} queries')
for f in no_results:
    print(f'   - {f["query"][:60]}...')

print(f'\n2. TOP1_NOT_RELEVANT: {len(top1_irrelevant)} queries')
for f in top1_irrelevant:
    scores = f.get('ce_scores', [])
    best_in_top5 = max(scores[:5]) if scores else -999
    print(f'   - Best CE score: {best_in_top5:.2f} | {f["query"][:50]}...')

# Check if any failures have a good result in top 5
print('\n' + '='*80)
print('RERANKING OPPORTUNITIES')
print('='*80)

for f in top1_irrelevant:
    scores = f.get('ce_scores', [])
    if scores and len(scores) > 1:
        best_idx = scores.index(max(scores))
        if best_idx > 0 and scores[best_idx] >= CE_THRESHOLD:
            print(f'\n  Query: {f["query"][:60]}...')
            print(f'  -> Rank 1 score: {scores[0]:.2f} (BELOW threshold)')
            print(f'  -> Rank {best_idx+1} score: {scores[best_idx]:.2f} (ABOVE threshold)')
            print(f'  -> OPPORTUNITY: Better result exists at rank {best_idx+1}!')
