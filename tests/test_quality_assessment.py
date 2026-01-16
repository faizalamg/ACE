#!/usr/bin/env python
"""Quality assessment for ACE retrieval - checks actual relevance, not just recall."""

from ace.unified_memory import UnifiedMemoryIndex
from ace.config import get_config


def run_quality_assessment():
    """Run quality assessment on query-result pairs."""
    config = get_config()
    index = UnifiedMemoryIndex(
        collection_name=config.qdrant.unified_collection,
        qdrant_url=config.qdrant.url,
    )

    # Test queries with expected topic alignment
    # Each tuple: (query, list of keywords that should appear in relevant results)
    TEST_CASES = [
        ('how to handle API failures gracefully', ['API', 'error', 'failure', 'retry', 'backoff']),
        ('TDD test driven development workflow', ['TDD', 'test', 'Test-Driven']),
        ('rate limiting', ['rate', 'limit', 'throttl']),
        ('secure password hashing', ['password', 'hash', 'secur', 'encrypt', 'bcrypt']),
        ('SQL injection prevention', ['SQL', 'inject', 'sanitiz', 'input', 'validat']),
        ('authentication JWT tokens', ['auth', 'JWT', 'token']),
        ('CI/CD pipeline setup', ['CI', 'CD', 'pipeline', 'deploy', 'automat']),
        ('memory leak detection', ['memory', 'leak']),
        ('deadlock prevention', ['deadlock', 'lock', 'concurrent', 'thread']),
        ('logging best practices', ['log', 'debug', 'trace']),
    ]

    print('QUALITY ASSESSMENT: Query-Result Relevance')
    print('='*80)
    print()

    relevant_count = 0
    total_results = 0
    irrelevant_examples = []

    for query, keywords in TEST_CASES:
        results = index.retrieve(query=query, limit=3)
        print(f'Query: "{query}"')
        print(f'Expected keywords: {keywords}')
        print('-'*80)
        
        for i, r in enumerate(results):
            content = r.content[:150]
            # Check if any keyword is present
            keyword_found = any(kw.lower() in content.lower() for kw in keywords)
            status = 'RELEVANT' if keyword_found else 'IRRELEVANT'
            
            if keyword_found:
                relevant_count += 1
            else:
                irrelevant_examples.append((query, content))
            total_results += 1
            
            print(f'  [{status}] {content}...')
        print()

    precision = 100 * relevant_count / total_results if total_results > 0 else 0
    print('='*80)
    print(f'PRECISION: {relevant_count}/{total_results} results contained expected keywords ({precision:.1f}%)')
    print()
    
    if irrelevant_examples:
        print('IRRELEVANT EXAMPLES:')
        for query, content in irrelevant_examples[:5]:
            print(f'  Query: "{query}"')
            print(f'  Result: {content[:80]}...')
            print()
    
    if precision >= 80:
        print('[PASS] Precision >= 80%')
    else:
        print('[FAIL] Precision < 80%')
        
    return precision


if __name__ == '__main__':
    run_quality_assessment()
