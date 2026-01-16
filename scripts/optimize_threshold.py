# -*- coding: utf-8 -*-
"""Test different cross-encoder thresholds to find optimal cutoff.

Goal: Find the threshold that maximizes precision while maintaining recall.
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
from ace.config import reset_config
from sentence_transformers import CrossEncoder

reset_config()

print('Loading cross-encoder...')
CE_MODEL = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)


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
    """IMPROVED filter - exclude pasted tables, metrics, and non-query content."""
    prompt_lower = prompt.lower()
    
    # Skip non-queries (tables, metrics, code)
    skip_patterns = [
        '**task:', '**context:', 'run the', 'execute', 'create a',
        'implement', 'fix the', 'update the', 'add this', 'proceed',
        '<local-command', '/commit', '/help', 'zen challenge', 'continue',
        # NEW: Skip pasted metrics/tables
        '| position', '| metric', 'r@1:', 'r@5:', 'p@3:', 'verdict:',
        '|---', '```', 'pass', 'fail', '\n\n'  # Multi-line tables
    ]
    if any(p in prompt_lower for p in skip_patterns):
        return False
    
    # Must look like a question
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

print(f'Testing {len(unique_queries)} queries with improved filter...\n')

# Collect all scores
all_scores = []

for i, query in enumerate(unique_queries):
    results = index.retrieve(
        query, limit=10, threshold=0.0,
        auto_detect_preset=True,
        use_cross_encoder=True
    )
    
    if not results:
        continue
    
    pairs = [[query, r.content[:500]] for r in results[:5]]
    scores = CE_MODEL.predict(pairs)
    scores_list = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
    
    all_scores.append({
        'query': query,
        'scores': scores_list
    })
    time.sleep(0.05)

print(f'Collected scores for {len(all_scores)} queries\n')

# Test different thresholds
thresholds = [-11.5, -11.0, -10.75, -10.5, -10.25, -10.0, -9.5, -9.0]

print('='*80)
print('THRESHOLD OPTIMIZATION')
print('='*80)
print(f'\n{"Threshold":<12} {"R@1":>8} {"R@5":>8} {"P@3":>8}')
print('-'*40)

best_threshold = None
best_score = 0

for threshold in thresholds:
    r1_hits = 0
    r5_hits = 0
    p3_sum = 0
    
    for item in all_scores:
        scores = item['scores']
        
        # R@1: Is top result relevant?
        if scores[0] >= threshold:
            r1_hits += 1
        
        # R@5: Is any result in top 5 relevant?
        if any(s >= threshold for s in scores[:5]):
            r5_hits += 1
        
        # P@3: What fraction of top 3 are relevant?
        p3 = sum(1 for s in scores[:3] if s >= threshold) / min(3, len(scores))
        p3_sum += p3
    
    r1 = r1_hits / len(all_scores) * 100
    r5 = r5_hits / len(all_scores) * 100
    p3 = p3_sum / len(all_scores) * 100
    
    combined = (r1 + r5 + p3) / 3
    if combined > best_score:
        best_score = combined
        best_threshold = threshold
    
    status = '***' if threshold == -10.5 else ''
    print(f'{threshold:<12} {r1:>7.1f}% {r5:>7.1f}% {p3:>7.1f}% {status}')

print(f'\nBest threshold: {best_threshold} (combined score: {best_score:.1f}%)')
print(f'Current threshold: -10.5')

# Analyze score distribution
print('\n' + '='*80)
print('SCORE DISTRIBUTION')
print('='*80)

all_top1_scores = [item['scores'][0] for item in all_scores]
all_top1_scores.sort()

print(f'\nTop-1 Score Percentiles:')
for p in [10, 25, 50, 75, 90]:
    idx = int(len(all_top1_scores) * p / 100)
    print(f'  {p}th percentile: {all_top1_scores[idx]:.2f}')

print(f'\nMin: {min(all_top1_scores):.2f}')
print(f'Max: {max(all_top1_scores):.2f}')
