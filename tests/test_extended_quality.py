#!/usr/bin/env python
"""Extended quality testing for ACE retrieval - targeted domains, edge cases, and reranking analysis."""

from typing import List, Tuple, Dict
from ace.unified_memory import UnifiedMemoryIndex
from ace.reranker import get_reranker
from ace.config import get_config


def run_domain_quality_tests():
    """Test quality across specific domains."""
    config = get_config()
    index = UnifiedMemoryIndex(
        collection_name=config.qdrant.unified_collection,
        qdrant_url=config.qdrant.url,
    )
    
    # Domain-specific test cases with stricter keyword expectations
    DOMAIN_TESTS = {
        'Security': [
            ('XSS cross-site scripting prevention', ['XSS', 'script', 'sanitiz', 'escap']),
            ('CSRF token validation', ['CSRF', 'token', 'valid']),
            ('SSL TLS certificate pinning', ['SSL', 'TLS', 'cert', 'pin']),
            ('OAuth 2.0 flow implementation', ['OAuth', 'auth', 'token', 'flow']),
            ('input sanitization whitelist', ['sanitiz', 'input', 'whitelist', 'valid']),
        ],
        'Debugging': [
            ('stack trace analysis techniques', ['stack', 'trace', 'debug', 'error']),
            ('breakpoint debugging strategy', ['breakpoint', 'debug', 'step']),
            ('log correlation distributed systems', ['log', 'correlat', 'distribut', 'trace']),
            ('root cause analysis methodology', ['root', 'cause', 'analys', 'debug']),
            ('memory profiler heap dump', ['memory', 'profil', 'heap', 'dump']),
        ],
        'Architecture': [
            ('microservices communication patterns', ['microservice', 'service', 'communicat', 'API']),
            ('event sourcing CQRS pattern', ['event', 'sourc', 'CQRS', 'pattern']),
            ('database sharding strategy', ['shard', 'database', 'partition', 'scale']),
            ('load balancer sticky sessions', ['load', 'balanc', 'session', 'sticky']),
            ('circuit breaker pattern', ['circuit', 'breaker', 'fallback', 'resilien']),
        ],
        'Performance': [
            ('N+1 query optimization', ['N+1', 'query', 'optim', 'join']),
            ('database index tuning', ['index', 'database', 'query', 'optim']),
            ('connection pool sizing', ['connection', 'pool', 'size']),
            ('lazy loading strategy', ['lazy', 'load', 'eager']),
            ('caching invalidation strategy', ['cach', 'invalid', 'TTL', 'expir']),
        ],
    }
    
    print('=' * 80)
    print('DOMAIN-SPECIFIC QUALITY TESTS')
    print('=' * 80)
    print()
    
    domain_results = {}
    
    for domain, test_cases in DOMAIN_TESTS.items():
        print(f'\n### {domain.upper()} DOMAIN ###')
        print('-' * 60)
        
        relevant = 0
        total = 0
        
        for query, keywords in test_cases:
            results = index.retrieve(query=query, limit=3)
            print(f'\nQuery: "{query}"')
            
            for r in results:
                content = r.content[:120]
                is_relevant = any(kw.lower() in content.lower() for kw in keywords)
                status = 'OK' if is_relevant else 'MISS'
                if is_relevant:
                    relevant += 1
                total += 1
                print(f'  [{status}] {content}...')
        
        precision = 100 * relevant / total if total > 0 else 0
        domain_results[domain] = precision
        print(f'\n{domain} Precision: {relevant}/{total} ({precision:.1f}%)')
    
    print('\n' + '=' * 80)
    print('DOMAIN SUMMARY')
    print('=' * 80)
    for domain, precision in domain_results.items():
        status = 'PASS' if precision >= 70 else 'FAIL'
        print(f'  {domain}: {precision:.1f}% [{status}]')
    
    avg_precision = sum(domain_results.values()) / len(domain_results)
    print(f'\nOverall Average: {avg_precision:.1f}%')
    return domain_results


def run_reranking_edge_case_analysis():
    """Analyze if reranking improves edge cases."""
    config = get_config()
    index = UnifiedMemoryIndex(
        collection_name=config.qdrant.unified_collection,
        qdrant_url=config.qdrant.url,
    )
    reranker = get_reranker()
    
    # Edge cases where base retrieval might fail
    EDGE_CASES = [
        ('memory leak detection', ['memory', 'leak']),  # Known failure case
        ('deadlock prevention', ['deadlock', 'lock', 'thread']),  # Known failure case
        ('JWT token expiration handling', ['JWT', 'token', 'expir']),  # Specific auth case
        ('race condition debugging', ['race', 'condition', 'concurrent', 'thread']),
        ('null pointer exception handling', ['null', 'pointer', 'exception', 'None']),
    ]
    
    print('\n' + '=' * 80)
    print('RERANKING EDGE CASE ANALYSIS')
    print('=' * 80)
    print()
    
    base_relevant = 0
    reranked_relevant = 0
    total = 0
    
    for query, keywords in EDGE_CASES:
        print(f'Query: "{query}"')
        print(f'Keywords: {keywords}')
        
        # Get more candidates (10) to give reranker material to work with
        results = index.retrieve(query=query, limit=10)
        
        if not results:
            print('  No results found\n')
            continue
        
        docs = [r.content for r in results]
        
        # Base top-3
        print('\n  BASE TOP-3:')
        for i, doc in enumerate(docs[:3]):
            is_relevant = any(kw.lower() in doc[:150].lower() for kw in keywords)
            status = 'OK' if is_relevant else 'MISS'
            if is_relevant:
                base_relevant += 1
            total += 1
            print(f'    [{status}] {doc[:80]}...')
        
        # Reranked top-3
        reranked = reranker.rerank(query, docs, top_k=10)
        print('\n  RERANKED TOP-3:')
        rerank_count = 0
        for idx, score in reranked[:3]:
            doc = docs[idx]
            is_relevant = any(kw.lower() in doc[:150].lower() for kw in keywords)
            status = 'OK' if is_relevant else 'MISS'
            if is_relevant:
                rerank_count += 1
            print(f'    [{status}] (score: {score:.2f}) {doc[:70]}...')
        reranked_relevant += rerank_count
        
        print()
    
    base_precision = 100 * base_relevant / total if total > 0 else 0
    rerank_precision = 100 * reranked_relevant / total if total > 0 else 0
    
    print('=' * 80)
    print('RERANKING IMPACT ON EDGE CASES')
    print('=' * 80)
    print(f'  Base Precision:     {base_precision:.1f}%')
    print(f'  Reranked Precision: {rerank_precision:.1f}%')
    print(f'  Improvement:        {rerank_precision - base_precision:+.1f}%')
    
    return {'base': base_precision, 'reranked': rerank_precision}


def run_challenging_query_tests():
    """Test with vague, ambiguous, and multi-domain queries."""
    config = get_config()
    index = UnifiedMemoryIndex(
        collection_name=config.qdrant.unified_collection,
        qdrant_url=config.qdrant.url,
    )
    
    # Challenging queries with expected topic alignment
    CHALLENGING_QUERIES = [
        # Vague/short queries
        ('fix it', ['fix', 'error', 'debug', 'repair']),
        ('help', ['help', 'assist', 'support']),
        ('slow', ['slow', 'performance', 'optim', 'speed']),
        
        # Ambiguous queries (multiple valid interpretations)
        ('pool', ['pool', 'connection', 'thread']),  # connection pool? thread pool?
        ('lock', ['lock', 'deadlock', 'mutex', 'synchron']),  # database lock? file lock?
        ('token', ['token', 'JWT', 'auth', 'session']),  # JWT? CSRF? session?
        
        # Multi-domain queries
        ('secure fast API', ['secure', 'API', 'fast', 'perform']),
        ('test database authentication', ['test', 'database', 'auth']),
        ('debug production memory', ['debug', 'production', 'memory']),
        
        # Misspellings/typos
        ('authetication', ['auth']),  # authentication misspelled
        ('databse', ['database']),  # database misspelled
        ('preformance', ['perform']),  # performance misspelled
        
        # Technical jargon
        ('SOLID principles', ['SOLID', 'principle', 'single', 'open']),
        ('DRY KISS YAGNI', ['DRY', 'KISS', 'YAGNI', 'repeat']),
    ]
    
    print('\n' + '=' * 80)
    print('CHALLENGING/AMBIGUOUS QUERY TESTS')
    print('=' * 80)
    print()
    
    results_summary = []
    
    for query, keywords in CHALLENGING_QUERIES:
        results = index.retrieve(query=query, limit=3)
        
        relevant_count = 0
        for r in results:
            if any(kw.lower() in r.content[:150].lower() for kw in keywords):
                relevant_count += 1
        
        precision = 100 * relevant_count / len(results) if results else 0
        status = 'PASS' if precision >= 50 else 'FAIL'  # Lower threshold for challenging queries
        
        results_summary.append({
            'query': query,
            'results': len(results),
            'relevant': relevant_count,
            'precision': precision,
            'status': status
        })
        
        print(f'[{status}] "{query}" -> {relevant_count}/{len(results)} relevant ({precision:.0f}%)')
        if results:
            print(f'       Top result: {results[0].content[:60]}...')
    
    print('\n' + '=' * 80)
    
    passing = sum(1 for r in results_summary if r['status'] == 'PASS')
    total = len(results_summary)
    
    print(f'Challenging queries: {passing}/{total} passed ({100*passing/total:.0f}%)')
    
    return results_summary


if __name__ == '__main__':
    print('ACE RETRIEVAL EXTENDED QUALITY TESTING')
    print('=' * 80)
    
    # 1. Domain-specific tests
    domain_results = run_domain_quality_tests()
    
    # 2. Reranking edge case analysis
    rerank_results = run_reranking_edge_case_analysis()
    
    # 3. Challenging query tests
    challenging_results = run_challenging_query_tests()
    
    # Final summary
    print('\n' + '=' * 80)
    print('FINAL SUMMARY')
    print('=' * 80)
    print(f'Domain Average Precision: {sum(domain_results.values())/len(domain_results):.1f}%')
    print(f'Reranking Improvement: {rerank_results["reranked"] - rerank_results["base"]:+.1f}%')
    passing = sum(1 for r in challenging_results if r['status'] == 'PASS')
    print(f'Challenging Query Pass Rate: {100*passing/len(challenging_results):.0f}%')
