# -*- coding: utf-8 -*-
"""A/B Testing Framework for ACE Retrieval Configurations.

Compare different retrieval configurations to measure performance improvements.
Tests: baseline vs adaptive expansion vs cross-encoder vs combined optimizations.
"""
import sys
import os
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from ace.unified_memory import UnifiedMemoryIndex
from ace.config import reset_config, get_retrieval_config
from sentence_transformers import CrossEncoder

reset_config()


@dataclass
class ABTestConfig:
    """Configuration for an A/B test variant."""
    name: str
    use_cross_encoder: bool = True
    use_llm_expansion: bool = False
    use_llm_rerank: bool = False
    auto_detect_preset: bool = True
    description: str = ""


@dataclass 
class ABTestResult:
    """Results for a single test variant."""
    config: ABTestConfig
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    precision_at_3: float = 0.0
    avg_latency_ms: float = 0.0
    query_results: List[Dict] = field(default_factory=list)


# Cross-encoder for relevance evaluation (ground truth)
print('Loading cross-encoder for relevance evaluation...')
CE_MODEL = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
# Cross-encoder threshold (centralized in ace/config.py)
# Default: -11.5 targets 95%+ P@3 with high recall
CE_THRESHOLD = get_retrieval_config().cross_encoder_threshold


def extract_test_queries() -> List[str]:
    """Extract semantic queries from Claude logs."""
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
                                if content and len(content) > 20 and is_semantic_query(content):
                                    prompts.append(content)
                    except:
                        pass
        except:
            pass
    
    # Deduplicate
    unique = []
    seen = set()
    for q in prompts:
        q_norm = q[:100].lower()
        if q_norm not in seen:
            seen.add(q_norm)
            unique.append(q[:200])
            if len(unique) >= 30:  # Use 30 queries for faster A/B testing
                break
    return unique


def is_semantic_query(prompt: str) -> bool:
    """Filter for queries that would benefit from semantic memory search."""
    prompt_lower = prompt.lower()
    
    skip_patterns = [
        '**task:', '**context:', 'run the', 'execute', 'create a',
        'implement', 'fix the', 'update the', 'add this', 'proceed',
        '<local-command', '/commit', '/help', 'zen challenge',
        'check on progress', 'continue'
    ]
    if any(p in prompt_lower for p in skip_patterns):
        return False
    
    include_patterns = [
        'what is', 'how does', 'how to', 'why is', 'where is',
        'is there', 'does this', 'can you', 'should', 'difference between',
        'what are', 'which', 'when', 'explain', '?'
    ]
    return any(p in prompt_lower for p in include_patterns)


def evaluate_relevance(query: str, results: List) -> Tuple[float, float, float, List[bool]]:
    """Evaluate relevance using cross-encoder scores."""
    if not results:
        return 0.0, 0.0, 0.0, []
    
    # Score top 5 results with cross-encoder
    pairs = [[query, r.content[:500]] for r in results[:5]]
    scores = CE_MODEL.predict(pairs)
    relevance = [score >= CE_THRESHOLD for score in scores]
    
    # Extend for remaining positions
    while len(relevance) < min(len(results), 10):
        relevance.append(False)
    
    r1 = 1.0 if relevance[0] else 0.0
    r5 = 1.0 if any(relevance[:5]) else 0.0
    p3 = sum(relevance[:3]) / 3.0 if len(relevance) >= 3 else 0.0
    
    return r1, r5, p3, relevance


def run_ab_test(
    index: UnifiedMemoryIndex,
    queries: List[str],
    configs: List[ABTestConfig]
) -> List[ABTestResult]:
    """Run A/B test across all configurations."""
    results = []
    
    for config in configs:
        print(f'\n{"="*60}')
        print(f'Testing: {config.name}')
        print(f'Description: {config.description}')
        print(f'{"="*60}')
        
        test_result = ABTestResult(config=config)
        r1_scores, r5_scores, p3_scores = [], [], []
        latencies = []
        
        for i, query in enumerate(queries):
            start = time.time()
            
            try:
                retrieved = index.retrieve(
                    query,
                    limit=10,
                    threshold=0.0,
                    auto_detect_preset=config.auto_detect_preset,
                    use_llm_expansion=config.use_llm_expansion,
                    use_llm_rerank=config.use_llm_rerank,
                    use_cross_encoder=config.use_cross_encoder
                )
            except Exception as e:
                print(f'  [{i+1}] ERROR: {e}')
                retrieved = []
            
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            
            r1, r5, p3, relevance = evaluate_relevance(query, retrieved)
            r1_scores.append(r1)
            r5_scores.append(r5)
            p3_scores.append(p3)
            
            test_result.query_results.append({
                'query': query[:60],
                'r1': r1,
                'r5': r5,
                'p3': p3,
                'latency_ms': latency,
                'n_results': len(retrieved)
            })
            
            r1_mark = 'Y' if r1 else 'N'
            r5_mark = 'Y' if r5 else 'N'
            print(f'  [{i+1:2d}] R@1:{r1_mark} R@5:{r5_mark} P@3:{p3*100:3.0f}% {latency:5.0f}ms | {query[:40]}...')
            time.sleep(0.05)  # Rate limit
        
        test_result.recall_at_1 = sum(r1_scores) / len(r1_scores) * 100
        test_result.recall_at_5 = sum(r5_scores) / len(r5_scores) * 100
        test_result.precision_at_3 = sum(p3_scores) / len(p3_scores) * 100
        test_result.avg_latency_ms = sum(latencies) / len(latencies)
        
        results.append(test_result)
    
    return results


def print_comparison(results: List[ABTestResult]) -> None:
    """Print side-by-side comparison of all variants."""
    print('\n' + '='*80)
    print('A/B TEST RESULTS COMPARISON')
    print('='*80)
    
    # Header
    print(f'\n{"Configuration":<30} {"R@1":>8} {"R@5":>8} {"P@3":>8} {"Latency":>10}')
    print('-'*70)
    
    # Baseline for delta calculation
    baseline = results[0] if results else None
    
    for r in results:
        delta_r1 = f'({r.recall_at_1 - baseline.recall_at_1:+.1f})' if baseline and r != baseline else ''
        delta_r5 = f'({r.recall_at_5 - baseline.recall_at_5:+.1f})' if baseline and r != baseline else ''
        delta_p3 = f'({r.precision_at_3 - baseline.precision_at_3:+.1f})' if baseline and r != baseline else ''
        
        r1_status = 'PASS' if r.recall_at_1 >= 80 else 'FAIL'
        r5_status = 'PASS' if r.recall_at_5 >= 95 else 'FAIL'
        p3_status = 'PASS' if r.precision_at_3 >= 80 else 'FAIL'
        
        print(f'{r.config.name:<30} {r.recall_at_1:5.1f}%{delta_r1:>6} {r.recall_at_5:5.1f}%{delta_r5:>6} {r.precision_at_3:5.1f}%{delta_p3:>6} {r.avg_latency_ms:8.1f}ms')
    
    print('\n' + '='*80)
    print('TARGET METRICS: R@1 >= 80%, R@5 >= 95%, P@3 >= 80%')
    print('='*80)
    
    # Winner determination
    best = max(results, key=lambda r: r.recall_at_1 + r.recall_at_5 + r.precision_at_3)
    print(f'\nBEST CONFIGURATION: {best.config.name}')
    print(f'  R@1: {best.recall_at_1:.1f}%  R@5: {best.recall_at_5:.1f}%  P@3: {best.precision_at_3:.1f}%')


def main():
    print('='*80)
    print('ACE RETRIEVAL A/B TESTING FRAMEWORK')
    print('Comparing: Baseline vs Adaptive Expansion vs Cross-Encoder vs Combined')
    print('='*80)
    
    # Initialize index
    index = UnifiedMemoryIndex(
        qdrant_url='http://localhost:6333',
        embedding_url='http://localhost:1234',
        collection_name='ace_memories_hybrid',
        embedding_dim=4096,
        embedding_model='text-embedding-qwen3-embedding-8b'
    )
    
    # Extract test queries
    queries = extract_test_queries()
    print(f'\nLoaded {len(queries)} test queries')
    
    # Define A/B test variants
    configs = [
        ABTestConfig(
            name='A: Baseline',
            use_cross_encoder=False,
            use_llm_expansion=False,
            auto_detect_preset=False,
            description='Pure vector search, no optimizations'
        ),
        ABTestConfig(
            name='B: Cross-Encoder Only',
            use_cross_encoder=True,
            use_llm_expansion=False,
            auto_detect_preset=False,
            description='Vector search + cross-encoder reranking'
        ),
        ABTestConfig(
            name='C: Adaptive Expansion',
            use_cross_encoder=False,
            use_llm_expansion=False,
            auto_detect_preset=True,  # Enables adaptive expansion
            description='Vector search + adaptive query expansion'
        ),
        ABTestConfig(
            name='D: Combined (Production)',
            use_cross_encoder=True,
            use_llm_expansion=False,
            auto_detect_preset=True,
            description='Adaptive expansion + cross-encoder (recommended)'
        ),
    ]
    
    # Run A/B test
    results = run_ab_test(index, queries, configs)
    
    # Print comparison
    print_comparison(results)
    
    # Save results to file
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_queries': len(queries),
        'results': [
            {
                'config': {
                    'name': r.config.name,
                    'description': r.config.description,
                    'use_cross_encoder': r.config.use_cross_encoder,
                    'use_llm_expansion': r.config.use_llm_expansion,
                    'auto_detect_preset': r.config.auto_detect_preset
                },
                'metrics': {
                    'recall_at_1': r.recall_at_1,
                    'recall_at_5': r.recall_at_5,
                    'precision_at_3': r.precision_at_3,
                    'avg_latency_ms': r.avg_latency_ms
                }
            }
            for r in results
        ]
    }
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'benchmark_results', 'ab_test_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nResults saved to: {output_path}')


if __name__ == '__main__':
    main()
