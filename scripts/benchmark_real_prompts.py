# -*- coding: utf-8 -*-
"""Benchmark with REAL user prompts from Claude logs."""
import sys
import os
import json
import time
from pathlib import Path

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

if not os.environ.get("ZAI_API_KEY"):
    print("ERROR: ZAI_API_KEY not set")
    sys.exit(1)

from ace.unified_memory import UnifiedMemoryIndex
from ace.retrieval_presets import llm_filter_and_rerank
from ace.config import reset_config
from ace.query_preprocessor import QueryPreprocessor

reset_config()

# Extract real user prompts from Claude logs
def extract_user_prompts():
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

# Filter for semantic search-like queries (not commands)
def is_semantic_query(prompt):
    """Filter for queries that would benefit from semantic memory search."""
    prompt_lower = prompt.lower()

    # Skip command-like prompts
    skip_patterns = [
        '**task:', '**context:', 'run the', 'execute', 'create a',
        'implement', 'fix the', 'update the', 'add this', 'proceed',
        '<local-command', '/commit', '/help', 'zen challenge',
        'fucking', 'mother fucker', 'check on progress'
    ]
    if any(p in prompt_lower for p in skip_patterns):
        return False

    # Skip metric/table dumps and analysis output (not real queries)
    invalid_patterns = [
        'verdict:', '| position', '|-------', 'r@1:', 'r@5:', 'p@3:',
        '| relevant', '~60-70%', '~70-80%', '~80-90%', '| not relevant'
    ]
    if any(p in prompt_lower for p in invalid_patterns):
        return False

    # Include semantic queries
    include_patterns = [
        'what is', 'how does', 'how to', 'why is', 'where is',
        'is there', 'does this', 'can you', 'should', 'difference between',
        'what are', 'which', 'when', 'explain', '?'
    ]
    if any(p in prompt_lower for p in include_patterns):
        return True

    return False

# Define expected keywords for evaluation (auto-detected from query)
def get_expected_keywords(query):
    """Extract key terms from query for relevance evaluation."""
    # Common stop words to ignore
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'must', 'can',
                  'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
                  'it', 'we', 'they', 'what', 'which', 'who', 'whom', 'whose',
                  'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both',
                  'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                  'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                  'just', 'and', 'but', 'or', 'for', 'with', 'about', 'into',
                  'through', 'during', 'before', 'after', 'above', 'below',
                  'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                  'under', 'again', 'further', 'then', 'once', 'here', 'there',
                  'any', 'my', 'your', 'still', 'see'}

    # Extract words
    words = query.lower().replace('?', '').replace('.', '').replace(',', '').split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2]

    return keywords[:5]  # Top 5 keywords

def has_relevant(content, keywords):
    """Check if content contains any expected keywords."""
    content_lower = content.lower()
    return any(kw in content_lower for kw in keywords)

# Main benchmark
print('=' * 80)
print('BENCHMARK WITH REAL USER PROMPTS FROM CLAUDE LOGS')
print('Using Z.ai GLM 4.6 for LLM filtering')
print('=' * 80)

# Extract and filter prompts
all_prompts = extract_user_prompts()
semantic_queries = [p for p in all_prompts if is_semantic_query(p)]

# Deduplicate and take first 60
unique_queries = []
seen = set()
for q in semantic_queries:
    q_norm = q[:100].lower()
    if q_norm not in seen:
        seen.add(q_norm)
        unique_queries.append(q[:200])
        if len(unique_queries) >= 30:  # Reduced for faster testing
            break

print(f'Total prompts: {len(all_prompts)}')
print(f'Semantic queries: {len(semantic_queries)}')
print(f'Unique queries for benchmark: {len(unique_queries)}')
print('=' * 80)

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
results_log = []
preprocessor = QueryPreprocessor()

for i, query in enumerate(unique_queries):
    # Use CORRECTED query for keyword extraction (match what retrieval uses)
    preprocess_result = preprocessor.preprocess(query)
    if preprocess_result.is_valid_query:
        corrected_query = preprocessor.correct_typos(preprocess_result.cleaned_query)
    else:
        corrected_query = query
    keywords = get_expected_keywords(corrected_query)

    # Retrieve
    results = index.retrieve(
        query, limit=10, auto_detect_preset=True,
        use_llm_expansion=True, use_llm_rerank=False
    )

    # LLM filter
    if results:
        results_with_scores = [(b, getattr(b, 'qdrant_score', 0.5)) for b in results]
        results = llm_filter_and_rerank(query=query, results=results_with_scores)

    # Evaluate
    n_results = len(results) if results else 0
    r1 = 1.0 if results and has_relevant(results[0].content, keywords) else 0.0
    r5 = 1.0 if any(has_relevant(r.content, keywords) for r in (results or [])[:5]) else 0.0

    actual_top3 = results[:3] if results else []
    relevant_in_3 = sum(1 for r in actual_top3 if has_relevant(r.content, keywords))
    p3 = relevant_in_3 / len(actual_top3) if actual_top3 else 0.0

    metrics['r1'].append(r1)
    metrics['r5'].append(r5)
    metrics['p3'].append(p3)

    r1_mark = 'Y' if r1 == 1.0 else 'N'
    r5_mark = 'Y' if r5 == 1.0 else 'N'
    query_short = query[:50].replace('\n', ' ')
    print(f'[{i+1:2d}] R@1:{r1_mark} R@5:{r5_mark} P@3:{p3*100:3.0f}% ({n_results:2d}r) {query_short}...')

    results_log.append({
        'query': query[:100],
        'keywords': keywords,
        'n_results': n_results,
        'r1': r1, 'r5': r5, 'p3': p3
    })

    # Rate limit (reduced for faster testing)
    time.sleep(0.5)

# Summary
print('')
print('=' * 80)
print('RESULTS SUMMARY')
print('=' * 80)

total_r1 = sum(metrics['r1']) / len(metrics['r1']) * 100
total_r5 = sum(metrics['r5']) / len(metrics['r5']) * 100
total_p3 = sum(metrics['p3']) / len(metrics['p3']) * 100

r1_status = "PASS" if total_r1 >= 95 else "FAIL"
r5_status = "PASS" if total_r5 >= 95 else "FAIL"
p3_status = "PASS" if total_p3 >= 95 else "FAIL"

print(f'Queries tested: {len(unique_queries)}')
print(f'Recall@1:       {total_r1:5.1f}% (95%+ target) [{r1_status}]')
print(f'Recall@5:       {total_r5:5.1f}% (95%+ target) [{r5_status}]')
print(f'Precision@3:    {total_p3:5.1f}% (95%+ target) [{p3_status}]')
print('')

all_pass = total_r1 >= 95 and total_r5 >= 95 and total_p3 >= 95
if all_pass:
    print('ALL TARGETS MET!')
else:
    print('NEEDS IMPROVEMENT')
    # Show failures
    print('')
    print('Failed queries:')
    for log in results_log:
        if log['r1'] == 0 or log['p3'] < 0.67:
            print(f"  - R@1:{log['r1']:.0f} P@3:{log['p3']*100:.0f}% | {log['query'][:60]}...")

# Save detailed results
with open('scripts/real_prompts_benchmark_results.json', 'w') as f:
    json.dump({
        'total_queries': len(unique_queries),
        'r1': total_r1, 'r5': total_r5, 'p3': total_p3,
        'results': results_log
    }, f, indent=2)
print(f'\nDetailed results saved to scripts/real_prompts_benchmark_results.json')
