"""
Semantic Similarity Test - Fortune 100 Grade

Tests semantic retrieval with query variations:
- First sentence extraction
- Keyword extraction
- Synonym replacement
- Question reformulation
- Short queries

Target: 95%+ Recall@5 across ALL query types
"""

import sys
import time
import re
import random
import json
import httpx
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from ace.retrieval_optimized import OptimizedRetriever, GPU_RERANKER_AVAILABLE
from ace.config import QdrantConfig

# Technical synonyms for query transformation
SYNONYMS = {
    'error': ['issue', 'problem', 'bug', 'failure', 'exception'],
    'fix': ['resolve', 'repair', 'correct', 'patch', 'solve'],
    'create': ['generate', 'build', 'make', 'construct', 'initialize'],
    'delete': ['remove', 'destroy', 'clear', 'drop', 'eliminate'],
    'update': ['modify', 'change', 'edit', 'patch', 'alter'],
    'check': ['verify', 'validate', 'confirm', 'ensure', 'test'],
    'config': ['configuration', 'settings', 'options', 'parameters'],
    'api': ['endpoint', 'interface', 'service', 'route'],
    'database': ['db', 'datastore', 'storage', 'persistence'],
    'function': ['method', 'procedure', 'routine', 'handler'],
    'test': ['spec', 'unit test', 'verification', 'check'],
    'handle': ['process', 'manage', 'deal with', 'respond to'],
    'implement': ['add', 'create', 'build', 'develop'],
    'use': ['utilize', 'employ', 'leverage', 'apply'],
    'always': ['consistently', 'invariably', 'perpetually'],
    'never': ['avoid', 'do not', 'refrain from'],
    'should': ['must', 'need to', 'ought to'],
    'important': ['critical', 'essential', 'vital', 'crucial'],
}

STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
    'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have',
    'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can',
    'this', 'that', 'these', 'those', 'it', 'its', 'not', 'no', 'just', 'also',
    'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which', 'who', 'when', 'where',
    'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'now'
}


def generate_semantic_queries(content: str) -> list:
    """Generate semantic variations of content for testing."""
    queries = []

    # 1. First sentence only
    sentences = content.replace('\n', ' ').split('.')
    if sentences and len(sentences[0].strip()) > 10:
        queries.append(('first_sentence', sentences[0].strip()[:80]))

    # 2. Keywords only (5 key terms)
    words = re.findall(r'\b[a-z]{3,}\b', content.lower())
    keywords = [w for w in words if w not in STOPWORDS]
    seen = set()
    unique_kw = [w for w in keywords if not (w in seen or seen.add(w))]
    if unique_kw:
        queries.append(('keywords', ' '.join(unique_kw[:5])))

    # 3. Synonym replacement
    transformed = content.lower()[:100]
    for word, syns in SYNONYMS.items():
        if word in transformed:
            transformed = transformed.replace(word, random.choice(syns), 1)
            break
    if transformed != content.lower()[:100]:
        queries.append(('synonym', transformed))

    # 4. Question format
    if len(sentences) > 0:
        first = sentences[0].strip()
        if not first.endswith('?'):
            queries.append(('question_how', f'how to {first[:50].lower()}'))
            queries.append(('question_what', f'what is {first[:50].lower()}'))

    # 5. Short query (3-4 words)
    if unique_kw and len(unique_kw) >= 3:
        queries.append(('short', ' '.join(unique_kw[:3])))

    # 6. Paraphrase (middle section)
    if len(content) > 100:
        mid = content[50:150].strip()
        if mid:
            queries.append(('paraphrase', mid))

    return queries


def load_memories(limit: int = 2500) -> list:
    """Load memories from Qdrant with pagination."""
    config = QdrantConfig()
    client = httpx.Client(timeout=60.0)

    memories = []
    next_offset = None

    print(f'Loading up to {limit} memories from {config.memories_collection}...')

    while len(memories) < limit:
        batch_size = min(100, limit - len(memories))
        payload = {
            'limit': batch_size,
            'with_payload': True,
            'with_vector': False
        }
        if next_offset:
            payload['offset'] = next_offset

        resp = client.post(
            f'{config.url}/collections/{config.memories_collection}/points/scroll',
            json=payload
        )

        if resp.status_code != 200:
            print(f'Error: {resp.status_code}')
            break

        result = resp.json().get('result', {})
        points = result.get('points', [])

        if not points:
            break

        memories.extend(points)
        next_offset = result.get('next_page_offset')

        if len(memories) % 500 == 0:
            print(f'  Loaded {len(memories)} memories...')

        if not next_offset:
            break

    client.close()
    print(f'Total loaded: {len(memories)} memories')
    return memories


def main():
    print('=' * 80)
    print('SEMANTIC SIMILARITY TEST - FORTUNE 100 GRADE')
    print('=' * 80)
    print(f'GPU Reranker: {GPU_RERANKER_AVAILABLE}')

    # Load memories
    memories = load_memories(2500)  # Load ALL 2003 memories
    print(f'Loaded {len(memories)} memories')

    # Init retriever
    print('Initializing OptimizedRetriever...')
    retriever = OptimizedRetriever(config={
        'enable_reranking': True,
        'num_expanded_queries': 4,
        'candidates_per_query': 30,
        'first_stage_k': 30,
        'final_k': 10,
    })

    # Test
    by_type = defaultdict(lambda: {'total': 0, 'success_at_1': 0, 'success_at_5': 0, 'success_at_10': 0})
    failures = []
    total_queries = 0
    total_success_5 = 0

    start = time.time()

    test_count = len(memories)  # Test ALL memories - Fortune 100 quality
    for i, mem in enumerate(memories[:test_count]):
        content = mem.get('payload', {}).get('lesson', '') or mem.get('payload', {}).get('content', '')
        if not content or len(content) < 20:
            continue

        queries = generate_semantic_queries(content)

        for qtype, query in queries:
            results = retriever.search(query, limit=10)
            retrieved_ids = [r.id for r in results]

            rank = None
            if mem['id'] in retrieved_ids:
                rank = retrieved_ids.index(mem['id']) + 1

            by_type[qtype]['total'] += 1
            total_queries += 1

            if rank == 1:
                by_type[qtype]['success_at_1'] += 1
            if rank and rank <= 5:
                by_type[qtype]['success_at_5'] += 1
                total_success_5 += 1
            if rank and rank <= 10:
                by_type[qtype]['success_at_10'] += 1

            if not rank or rank > 5:
                failures.append({
                    'type': qtype,
                    'query': query[:60],
                    'rank': rank,
                    'content': content[:50],
                    'memory_id': mem['id']
                })

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - start
            r5 = total_success_5 / total_queries * 100 if total_queries > 0 else 0
            print(f'[{i+1}/{test_count}] Recall@5: {r5:.1f}% | Queries: {total_queries} | {elapsed:.0f}s')

    elapsed = time.time() - start

    print()
    print('=' * 80)
    print('SEMANTIC SIMILARITY RESULTS')
    print('=' * 80)
    print(f'Total: {total_queries} queries in {elapsed:.1f}s ({total_queries/elapsed:.1f} qps)')
    print()

    overall_r1 = sum(d['success_at_1'] for d in by_type.values()) / total_queries * 100
    overall_r5 = total_success_5 / total_queries * 100
    overall_r10 = sum(d['success_at_10'] for d in by_type.values()) / total_queries * 100

    print('OVERALL METRICS:')
    print(f'  Recall@1:  {overall_r1:.1f}%')
    print(f'  Recall@5:  {overall_r5:.1f}%  <-- TARGET 95%')
    print(f'  Recall@10: {overall_r10:.1f}%')
    print()

    print('BY QUERY TYPE:')
    type_results = {}
    for qtype, data in sorted(by_type.items()):
        if data['total'] > 0:
            r5 = data['success_at_5'] / data['total'] * 100
            status = 'PASS' if r5 >= 95 else 'FAIL'
            print(f'  {qtype:20s}: R@5={r5:.1f}% [{status}] (n={data["total"]})')
            type_results[qtype] = {'recall_at_5': r5, 'total': data['total'], 'status': status}

    print()
    if overall_r5 >= 95:
        print('>>> FORTUNE 100 QUALITY: ACHIEVED <<<')
    else:
        print('>>> FORTUNE 100 QUALITY: NOT ACHIEVED <<<')
        print(f'>>> GAP TO TARGET: {95 - overall_r5:.1f}% <<<')
        print()
        print('SAMPLE FAILURES:')
        for f in failures[:10]:
            print(f'  [{f["type"]}] rank={f["rank"]} | {f["query"]}')

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_queries': total_queries,
        'total_memories': test_count,
        'elapsed_seconds': elapsed,
        'overall': {
            'recall_at_1': overall_r1,
            'recall_at_5': overall_r5,
            'recall_at_10': overall_r10,
        },
        'by_type': type_results,
        'failures': failures[:50],
        'verdict': 'ACHIEVED' if overall_r5 >= 95 else 'NOT_ACHIEVED'
    }

    output_path = Path(__file__).parent / 'optimization_results' / f'semantic_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to: {output_path}')

    return results


if __name__ == '__main__':
    main()
