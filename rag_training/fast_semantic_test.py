"""
FAST Semantic Similarity Test - Batch Processing

Optimized for speed with:
- Fewer query variations (3 per memory instead of 7)
- Progress every 50 memories
- Reduced reranking overhead

Target: 95%+ Recall@5 across ALL 2003 memories
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
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from ace.retrieval_optimized import OptimizedRetriever, GPU_RERANKER_AVAILABLE
from ace.config import QdrantConfig

STOPWORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
             'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have',
             'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can',
             'this', 'that', 'these', 'those', 'it', 'its', 'not', 'no', 'just', 'also'}


def extract_keywords(text: str, n: int = 5) -> str:
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    keywords = [w for w in words if w not in STOPWORDS]
    seen = set()
    unique = [w for w in keywords if not (w in seen or seen.add(w))]
    return ' '.join(unique[:n])


def generate_queries(content: str) -> list:
    """Generate 3 semantic query variations."""
    queries = []

    # 1. Keywords (5 terms)
    kw = extract_keywords(content, 5)
    if kw:
        queries.append(('keywords', kw))

    # 2. First sentence
    sentences = content.replace('\n', ' ').split('.')
    if sentences and len(sentences[0].strip()) > 10:
        queries.append(('first_sentence', sentences[0].strip()[:80]))

    # 3. Short (3 keywords)
    kw3 = extract_keywords(content, 3)
    if kw3:
        queries.append(('short', kw3))

    return queries


def load_all_memories() -> list:
    """Load ALL memories from Qdrant."""
    config = QdrantConfig()
    client = httpx.Client(timeout=60.0)

    memories = []
    next_offset = None

    while True:
        payload = {'limit': 100, 'with_payload': True, 'with_vector': False}
        if next_offset:
            payload['offset'] = next_offset

        resp = client.post(
            f'{config.url}/collections/{config.memories_collection}/points/scroll',
            json=payload
        )

        if resp.status_code != 200:
            break

        result = resp.json().get('result', {})
        points = result.get('points', [])

        if not points:
            break

        memories.extend(points)
        next_offset = result.get('next_page_offset')

        if not next_offset:
            break

    client.close()
    return memories


def main():
    print('=' * 80)
    print('FAST SEMANTIC TEST - ALL MEMORIES - BATCH OPTIMIZED')
    print('=' * 80)
    print(f'GPU Reranker: {GPU_RERANKER_AVAILABLE}')

    # Load ALL memories
    print('Loading ALL memories...')
    memories = load_all_memories()
    print(f'Loaded {len(memories)} memories')

    # Pre-generate ALL queries
    print('Pre-generating queries...')
    all_tests = []
    for mem in memories:
        content = mem.get('payload', {}).get('lesson', '') or mem.get('payload', {}).get('content', '')
        if not content or len(content) < 20:
            continue
        queries = generate_queries(content)
        for qtype, query in queries:
            all_tests.append((mem['id'], qtype, query))

    print(f'Generated {len(all_tests)} test queries')

    # Init retriever with LESS expansion for speed
    print('Initializing OptimizedRetriever (speed-optimized)...')
    retriever = OptimizedRetriever(config={
        'enable_reranking': True,
        'num_expanded_queries': 2,  # Reduced from 4
        'candidates_per_query': 20,  # Reduced from 30
        'first_stage_k': 20,
        'final_k': 10,
    })

    # Run tests
    by_type = defaultdict(lambda: {'total': 0, 's1': 0, 's5': 0, 's10': 0})
    failures = []

    start = time.time()

    for i, (mem_id, qtype, query) in enumerate(all_tests):
        results = retriever.search(query, limit=10)
        retrieved_ids = [r.id for r in results]

        rank = None
        if mem_id in retrieved_ids:
            rank = retrieved_ids.index(mem_id) + 1

        by_type[qtype]['total'] += 1
        if rank == 1:
            by_type[qtype]['s1'] += 1
        if rank and rank <= 5:
            by_type[qtype]['s5'] += 1
        if rank and rank <= 10:
            by_type[qtype]['s10'] += 1

        if not rank or rank > 5:
            if len(failures) < 100:
                failures.append({'type': qtype, 'query': query[:50], 'rank': rank})

        if (i + 1) % 200 == 0:
            elapsed = time.time() - start
            total = sum(d['total'] for d in by_type.values())
            s5 = sum(d['s5'] for d in by_type.values())
            r5 = s5 / total * 100 if total > 0 else 0
            qps = (i + 1) / elapsed
            eta = (len(all_tests) - i - 1) / qps
            print(f'[{i+1}/{len(all_tests)}] R@5: {r5:.1f}% | {qps:.1f} qps | ETA: {eta:.0f}s')

    elapsed = time.time() - start

    # Results
    total = sum(d['total'] for d in by_type.values())
    s1 = sum(d['s1'] for d in by_type.values())
    s5 = sum(d['s5'] for d in by_type.values())
    s10 = sum(d['s10'] for d in by_type.values())

    print()
    print('=' * 80)
    print('RESULTS')
    print('=' * 80)
    print(f'Total: {total} queries in {elapsed:.0f}s ({total/elapsed:.1f} qps)')
    print()
    print(f'Recall@1:  {s1/total*100:.1f}%')
    print(f'Recall@5:  {s5/total*100:.1f}%  <-- TARGET 95%')
    print(f'Recall@10: {s10/total*100:.1f}%')
    print()

    print('BY QUERY TYPE:')
    for qtype, d in sorted(by_type.items()):
        if d['total'] > 0:
            r5 = d['s5'] / d['total'] * 100
            status = 'PASS' if r5 >= 95 else 'FAIL'
            print(f'  {qtype:15s}: R@5={r5:.1f}% [{status}] (n={d["total"]})')

    print()
    if s5/total >= 0.95:
        print('>>> FORTUNE 100 QUALITY: ACHIEVED <<<')
    else:
        print(f'>>> GAP TO TARGET: {95 - s5/total*100:.1f}% <<<')
        print()
        print('SAMPLE FAILURES:')
        for f in failures[:10]:
            print(f'  [{f["type"]}] rank={f["rank"]} | {f["query"]}')

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'total_queries': total,
        'elapsed_sec': elapsed,
        'recall_at_1': s1/total*100,
        'recall_at_5': s5/total*100,
        'recall_at_10': s10/total*100,
        'by_type': {k: {'total': v['total'], 'r5': v['s5']/v['total']*100 if v['total'] else 0} for k, v in by_type.items()},
        'failures': failures[:50]
    }

    out_path = Path(__file__).parent / 'optimization_results' / f'fast_semantic_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
