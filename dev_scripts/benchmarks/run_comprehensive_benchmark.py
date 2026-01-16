"""Comprehensive ACE benchmark - 250+ queries across 10 categories."""
import sys
sys.path.insert(0, '.')
from ace.code_retrieval import CodeRetrieval
import json
from datetime import datetime

# 250+ queries across 10 categories
QUERIES = [
    # Category 1: Class/function definitions (25)
    'CodeRetrieval class definition', 'UnifiedMemoryIndex class', 'Playbook class', 'Generator class', 'Reflector class', 'Curator class',
    'ASTChunker class', 'CodeIndexer class', 'QdrantRetrieval class', 'HyDE class', 'DeltaOperation class', 'Bullet class',
    'search method', 'retrieve method', 'index_bullet method', 'store method', 'expand_query method', '_apply_filename_boost method',
    'create_sparse_vector method', 'embed_batch method', 'chunk_file method', 'deduplicate_results method', 'format_ThatOtherContextEngine_style method',
    'index_workspace function', 'get_embedding function',

    # Category 2: Configuration patterns (25)
    'EmbeddingConfig class', 'QdrantConfig class', 'LLMConfig class', 'VoyageCodeEmbeddingConfig class', 'BM25Config class',
    'environment variables', 'dotenv loading', 'API key configuration', 'model selection', 'provider configuration',
    'vector dimension', 'batch size', 'timeout settings', 'retry policy', 'cache TTL',
    'collection name', 'database URL', 'embedding model', 'reranker path', 'max tokens',
    'rate limiting', 'connection pool', 'chunk size', 'overlap size', 'top_k results',

    # Category 3: Error handling patterns (25)
    'exception handling', 'try except pattern', 'error logging', 'retry mechanism', 'backoff strategy',
    'timeout handling', 'connection error', 'API error', 'validation error', 'graceful degradation',
    'error recovery', 'fallback strategy', 'circuit breaker', 'error propagation', 'exception hierarchy',
    'custom exceptions', 'error context', 'stack trace', 'debug logging', 'error reporting',
    'resilience pattern', 'fault tolerance', 'error boundary', 'panic handling', 'cleanup on error',

    # Category 4: Import patterns and dependencies (25)
    'from qdrant_client import', 'import voyageai', 'from ace import', 'litellm import', 'langchain import',
    'asyncio import', 'typing import', 'dataclasses import', 'pathlib import', 'subprocess import',
    'threading import', 'logging import', 'yaml import', 'json import', 'hashlib import',
    'datetime import', 'base64 import', 'enum import', 'abc import', 'functools import',
    'httpx import', 'pydantic import', 'pytest import', 'tree_sitter import', 'numpy import',

    # Category 5: API endpoints and data structures (25)
    'REST API endpoint', 'POST handler', 'GET handler', 'request validation', 'response formatting',
    'dataclass definition', 'TypedDict', 'Pydantic model', 'data validation', 'serialization',
    'JSON schema', 'API versioning', 'pagination', 'filtering', 'sorting',
    'endpoint routing', 'middleware', 'authentication', 'authorization', 'rate limiting',
    'request body', 'query parameters', 'path parameters', 'headers', 'cookies',
    'status codes', 'error response', 'success response', 'response headers', 'content negotiation',

    # Category 6: Async/await patterns (25)
    'async def', 'await asyncio', 'async context manager', 'async for loop', 'async generator',
    'asyncio.gather', 'asyncio.create_task', 'asyncio.run', 'async with statement', 'awaitable object',
    'concurrent execution', 'parallel processing', 'async iterator', 'async iterable', 'coroutine',
    'event loop', 'task scheduling', 'async cancellation', 'timeout in async', 'async semaphore',
    'async queue', 'async lock', 'async event', 'async condition', 'streaming async',
    'async HTTP client', 'async database', 'async file I/O', 'batch async processing', 'async retry',

    # Category 7: Testing patterns (25)
    'pytest fixture', 'unit test', 'integration test', 'test case', 'test suite',
    'assert statement', 'mock object', 'test double', 'stub', 'spy',
    'test setup', 'test teardown', 'pytest parametrize', 'test discovery', 'test runner',
    'coverage report', 'test mocking', 'patch decorator', 'fixture scope', 'test configuration',
    'assertion error', 'test skip', 'test expected failure', 'async test', 'property based test',
    'test helpers', 'test utilities', 'fake data', 'test factory', 'test assertions',

    # Category 8: Documentation patterns (25)
    'docstring format', 'type hints', 'module documentation', 'class documentation', 'function documentation',
    'README', 'CHANGELOG', 'CONTRIBUTING', 'LICENSE', 'API documentation',
    'installation guide', 'quickstart tutorial', 'examples', 'usage guide', 'troubleshooting',
    'design document', 'architecture overview', 'API reference', 'changelog entry', 'release notes',
    'inline comments', 'documentation string', 'parameter docs', 'return value docs', 'exception docs',
    'example code', 'tutorial', 'how to guide', 'best practices', 'design principles',

    # Category 9: Architecture patterns (25)
    'singleton pattern', 'factory pattern', 'observer pattern', 'strategy pattern', 'adapter pattern',
    'dependency injection', 'service locator', 'repository pattern', 'unit of work', 'CQRS',
    'event sourcing', 'CQRS pattern', 'layered architecture', 'hexagonal architecture', 'clean architecture',
    'microservices', 'monolith', 'modular architecture', 'plugin system', 'extension point',
    'pipeline pattern', 'middleware pattern', 'interceptor', 'chain of responsibility', 'template method',
    'MVC pattern', 'MVP pattern', 'MVVM pattern', 'component based', 'entity component',

    # Category 10: Edge cases (25)
    'empty input handling', 'null safety', 'boundary condition', 'race condition', 'deadlock',
    'memory leak', 'resource cleanup', 'file descriptor limit', 'stack overflow', 'heap overflow',
    'integer overflow', 'floating point precision', 'string encoding', 'unicode handling', 'time zone',
    'concurrent access', 'thread safety', 'atomic operation', 'memory barrier', 'volatile variable',
    'edge case', 'corner case', 'boundary value', 'off-by-one error', 'divide by zero',
    'NaN handling', 'infinity handling', 'negative zero', 'rounding error', 'precision loss',
]

def main():
    print(f'Running {len(QUERIES)} queries...')
    retriever = CodeRetrieval()

    results = []
    for i, query in enumerate(QUERIES, 1):
        search_results = retriever.search(query, limit=3)
        if search_results:
            top = search_results[0]
            results.append({
                'query': query,
                'top_file': top['file_path'],
                'top_score': top['score'],
            })
            print(f'{i}/{len(QUERIES)}: {query[:40]:40} -> {top["file_path"]:40} [{top["score"]:.2f}]')
        else:
            results.append({
                'query': query,
                'top_file': 'NO RESULTS',
                'top_score': 0,
            })
            print(f'{i}/{len(QUERIES)}: {query[:40]:40} -> NO RESULTS')

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'total_queries': len(QUERIES),
        'results': results,
    }
    out_path = f'benchmark_results/ace_comprehensive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'\nResults saved to: {out_path}')

    # Summary
    avg_score = sum(r['top_score'] for r in results) / len(results)
    no_results = sum(1 for r in results if r['top_score'] == 0)
    print(f'Average score: {avg_score:.3f}')
    print(f'No results: {no_results}/{len(QUERIES)}')

    # Score distribution
    scores = [r['top_score'] for r in results]
    print(f'Score range: {min(scores):.3f} - {max(scores):.3f}')

if __name__ == '__main__':
    main()
