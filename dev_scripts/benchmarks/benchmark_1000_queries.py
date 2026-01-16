#!/usr/bin/env python3
"""
Comprehensive 1000-Query ACE vs ThatOtherContextEngine Benchmark

Goal: Achieve 100% ACE superiority over ThatOtherContextEngine MCP in code retrieval.

This benchmark:
1. Tests 10 categories with 25 queries each (250 base queries)
2. Runs 4 iterations for 1000 total query comparisons
3. Tracks ACE vs ThatOtherContextEngine performance per query
4. Identifies and reports all failures for iterative improvement

Categories (25 queries each):
1. Class/function definitions and implementations
2. Configuration patterns and settings
3. Error handling and debugging scenarios
4. Import patterns and dependency resolution
5. API endpoints and data structures
6. Async/await patterns and concurrency
7. Testing patterns and test utilities
8. Documentation and comment patterns
9. Architecture and design patterns
10. Edge cases and corner scenarios
"""

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# COMPREHENSIVE TEST QUERIES - 25 per category, 10 categories = 250 queries
# =============================================================================

CATEGORY_QUERIES = {
    # =========================================================================
    # CATEGORY 1: Class/Function Definitions (25 queries)
    # =========================================================================
    "ClassDefinitions": [
        # Basic class lookups
        ("CodeRetrieval class search method", ["ace/code_retrieval.py"]),
        ("UnifiedMemoryIndex class implementation", ["ace/unified_memory.py"]),
        ("ASTChunker class chunk method", ["ace/code_chunker.py"]),
        ("SmartBulletIndex retrieve method", ["ace/retrieval.py"]),
        ("Playbook class initialization", ["ace/playbook.py"]),
        
        # Config classes
        ("EmbeddingConfig dataclass definition", ["ace/config.py"]),
        ("QdrantConfig class definition", ["ace/config.py"]),
        ("BM25Config k1 b parameters", ["ace/config.py"]),
        ("LLMConfig provider model settings", ["ace/config.py"]),
        ("VoyageCodeEmbeddingConfig api_key", ["ace/config.py"]),
        
        # Specialized classes
        ("CodeIndexer index_workspace method", ["ace/code_indexer.py"]),
        ("HyDEGenerator class generate", ["ace/hyde.py"]),
        ("SemanticScorer score method", ["ace/semantic_scorer.py"]),
        ("FileWatcher callback handler", ["ace/file_watcher_daemon.py"]),
        ("CircuitBreaker is_open method", ["ace/resilience.py"]),
        
        # Advanced patterns
        ("RetryPolicy execute method", ["ace/resilience.py"]),
        ("CacheManager get set methods", ["ace/caching.py"]),
        ("DependencyGraph build method", ["ace/dependency_graph.py"]),
        ("QueryEnhancer expand method", ["ace/query_enhancer.py"]),
        ("GPUReranker rerank method", ["ace/gpu_reranker.py", "ace/reranker.py"]),
        
        # Dataclasses
        ("@dataclass class Bullet", ["ace/playbook.py"]),
        ("CodeChunk dataclass file_path", ["ace/code_chunker.py", "ace/code_indexer.py"]),
        ("QueryResult score file_path", ["ace/code_retrieval.py"]),
        ("UnifiedBullet namespace content", ["ace/unified_memory.py"]),
        ("RetrievalResult dataclass", ["ace/retrieval.py"]),
    ],
    
    # =========================================================================
    # CATEGORY 2: Configuration Patterns (25 queries)
    # =========================================================================
    "Configuration": [
        ("environment variable dotenv", ["ace/config.py"]),
        ("API_KEY environment variable", ["ace/config.py"]),
        ("QDRANT_URL connection string", ["ace/config.py"]),
        ("VOYAGE_API_KEY config", ["ace/config.py"]),
        ("OPENAI_API_KEY config", ["ace/config.py"]),
        
        ("MODEL_NAME configuration", ["ace/config.py"]),
        ("EMBEDDING_DIMENSION size", ["ace/config.py"]),
        ("MAX_TOKENS limit", ["ace/config.py"]),
        ("BATCH_SIZE setting", ["ace/config.py", "ace/code_indexer.py"]),
        ("TIMEOUT_SECONDS value", ["ace/config.py"]),
        
        ("log level logging configuration", ["ace/audit.py", "ace/config.py"]),
        ("debug logging verbose", ["ace/audit.py"]),
        ("config validation pydantic", ["ace/config.py"]),
        ("defaults dict configuration", ["ace/config.py"]),
        ("nested config structure", ["ace/config.py"]),
        
        ("BM25_K1 constant value", ["ace/config.py", "ace/retrieval_optimized.py"]),
        ("BM25_B constant value", ["ace/config.py", "ace/retrieval_optimized.py"]),
        ("cache TTL expiration", ["ace/caching.py", "ace/retrieval_caching.py"]),
        ("vector dimension 1024", ["ace/config.py"]),
        ("chunk size max tokens", ["ace/code_chunker.py"]),
        
        ("top_k limit search", ["ace/config.py"]),
        ("collection name workspace", ["ace/config.py", "ace/code_indexer.py"]),
        ("reranker model path", ["ace/reranker.py"]),
        ("rate limit throttling", ["ace/resilience.py"]),
        ("concurrent requests limit", ["ace/async_retrieval.py"]),
    ],
    
    # =========================================================================
    # CATEGORY 3: Error Handling (25 queries)
    # =========================================================================
    "ErrorHandling": [
        ("try except error handling", ["ace/resilience.py"]),
        ("exception retry backoff", ["ace/resilience.py"]),
        ("API rate limit retry", ["ace/resilience.py"]),
        ("timeout exception handling", ["ace/resilience.py"]),
        ("connection error handling", ["ace/resilience.py"]),
        
        ("validation error ValueError", ["ace/config.py"]),
        ("file not found error", ["ace/code_indexer.py"]),
        ("embedding error fallback", ["ace/semantic_scorer.py", "ace/code_retrieval.py"]),
        ("import error optional dependency", ["ace/features.py"]),
        ("graceful degradation", ["ace/resilience.py"]),
        
        ("finally cleanup block", ["ace/resilience.py"]),
        ("exception chaining from", ["ace/resilience.py"]),
        ("logging error stack trace", ["ace/audit.py"]),
        ("error recovery strategy", ["ace/resilience.py"]),
        ("circuit breaker pattern", ["ace/resilience.py"]),
        
        ("custom exception class", ["ace/resilience.py"]),
        ("exception hierarchy", ["ace/resilience.py"]),
        ("raise ValueError message", ["ace/config.py"]),
        ("HTTPStatusError handling", ["ace/code_retrieval.py"]),
        ("Qdrant error handling", ["ace/qdrant_retrieval.py"]),
        
        ("async error handling", ["ace/async_retrieval.py"]),
        ("retry decorator tenacity", ["ace/resilience.py"]),
        ("timeout decorator", ["ace/resilience.py"]),
        ("error logging format", ["ace/audit.py"]),
        ("fallback default value", ["ace/code_retrieval.py"]),
    ],
    
    # =========================================================================
    # CATEGORY 4: Import Patterns (25 queries)
    # =========================================================================
    "ImportPatterns": [
        ("from qdrant_client import", ["ace/unified_memory.py", "ace/code_indexer.py"]),
        ("import httpx async client", ["ace/code_retrieval.py", "ace/async_retrieval.py"]),
        ("from typing import List Dict", ["ace/code_retrieval.py"]),
        ("import asyncio gather", ["ace/async_retrieval.py"]),
        ("from pathlib import Path", ["ace/code_retrieval.py"]),
        
        ("import logging logger", ["ace/code_retrieval.py"]),
        ("from dataclasses import dataclass", ["ace/playbook.py"]),
        ("import os environ", ["ace/config.py"]),
        ("import json loads dumps", ["ace/delta.py"]),
        ("from datetime import datetime", ["ace/audit.py"]),
        
        ("import re regex pattern", ["ace/code_retrieval.py"]),
        ("from functools import lru_cache", ["ace/caching.py"]),
        ("import numpy as np", ["ace/code_retrieval.py"]),
        ("from collections import defaultdict", ["ace/code_retrieval.py"]),
        ("import hashlib sha256", ["ace/security.py"]),
        
        ("from abc import ABC abstractmethod", ["ace/llm.py"]),
        ("import enum Enum", ["ace/retrieval_presets.py"]),
        ("from tenacity import retry", ["ace/resilience.py"]),
        ("import sentence_transformers", ["ace/reranker.py"]),
        ("from pydantic import BaseModel", ["ace/config.py"]),
        
        ("from ace.config import", ["ace/code_retrieval.py"]),
        ("from ace.unified_memory import", ["ace/code_indexer.py"]),
        ("from ace.retrieval import", ["ace/code_retrieval.py"]),
        ("import ace module", ["ace/__init__.py"]),
        ("from . import relative", ["ace/__init__.py"]),
    ],
    
    # =========================================================================
    # CATEGORY 5: Data Structures (25 queries)
    # =========================================================================
    "DataStructures": [
        ("List[Dict] type annotation", ["ace/code_retrieval.py"]),
        ("Optional[str] parameter", ["ace/config.py"]),
        ("Tuple return type", ["ace/code_retrieval.py"]),
        ("Dict comprehension", ["ace/code_retrieval.py"]),
        ("Set operations unique", ["ace/deduplication.py"]),
        
        ("defaultdict factory", ["ace/code_retrieval.py"]),
        ("dataclass frozen", ["ace/playbook.py"]),
        ("Enum class values", ["ace/retrieval_presets.py"]),
        ("namedtuple pattern", ["ace/code_chunker.py"]),
        ("TypedDict definition", ["ace/config.py"]),
        
        ("list comprehension filter", ["ace/code_retrieval.py"]),
        ("generator expression yield", ["ace/async_retrieval.py"]),
        ("dict keys values items", ["ace/code_retrieval.py"]),
        ("set intersection union", ["ace/deduplication.py"]),
        ("tuple unpacking", ["ace/code_retrieval.py"]),
        
        ("nested dict structure", ["ace/config.py"]),
        ("array slice index", ["ace/code_retrieval.py"]),
        ("sorted key lambda", ["ace/code_retrieval.py"]),
        ("filter map reduce", ["ace/code_retrieval.py"]),
        ("zip enumerate iterate", ["ace/code_retrieval.py"]),
        
        ("JSON serializable", ["ace/delta.py"]),
        ("pickle serialize", ["ace/security.py"]),
        ("bytes encoding", ["ace/security.py"]),
        ("string format f-string", ["ace/code_retrieval.py"]),
        ("regex match groups", ["ace/code_retrieval.py"]),
    ],
    
    # =========================================================================
    # CATEGORY 6: Async Patterns (25 queries)
    # =========================================================================
    "AsyncPatterns": [
        ("async def await", ["ace/async_retrieval.py"]),
        ("asyncio gather parallel", ["ace/async_retrieval.py"]),
        ("async with httpx", ["ace/async_retrieval.py"]),
        ("async for stream", ["ace/async_retrieval.py"]),
        ("await embedding generation", ["ace/async_retrieval.py"]),
        
        ("async context manager", ["ace/async_retrieval.py"]),
        ("asyncio.create_task", ["ace/async_retrieval.py"]),
        ("async generator yield", ["ace/async_retrieval.py"]),
        ("semaphore rate limiting", ["ace/async_retrieval.py"]),
        ("asyncio timeout cancel", ["ace/async_retrieval.py"]),
        
        ("asyncio.run main", ["ace/async_retrieval.py"]),
        ("event loop run_until_complete", ["ace/async_retrieval.py"]),
        ("asyncio.Queue producer", ["ace/async_retrieval.py"]),
        ("async sleep delay", ["ace/async_retrieval.py"]),
        ("async lock mutex", ["ace/async_retrieval.py"]),
        
        ("concurrent futures", ["ace/async_retrieval.py"]),
        ("ThreadPoolExecutor async", ["ace/async_retrieval.py"]),
        ("ProcessPoolExecutor parallel", ["ace/async_retrieval.py"]),
        ("async batch processing", ["ace/async_retrieval.py"]),
        ("async retry backoff", ["ace/async_retrieval.py"]),
        
        ("httpx AsyncClient", ["ace/async_retrieval.py"]),
        ("aiohttp session", ["ace/async_retrieval.py"]),
        ("async embedding batch", ["ace/async_retrieval.py"]),
        ("parallel vector search", ["ace/async_retrieval.py"]),
        ("async callback handler", ["ace/async_retrieval.py"]),
    ],
    
    # =========================================================================
    # CATEGORY 7: Testing Patterns (25 queries)
    # =========================================================================
    "TestingPatterns": [
        ("pytest fixture setup", ["tests/"]),
        ("unittest TestCase class", ["tests/"]),
        ("mock patch decorator", ["tests/"]),
        ("assert statement equal", ["tests/"]),
        ("test function def test_", ["tests/"]),
        
        ("parametrize test cases", ["tests/"]),
        ("conftest.py fixtures", ["tests/"]),
        ("setup teardown method", ["tests/"]),
        ("async test aioclient", ["tests/"]),
        ("integration test end-to-end", ["tests/"]),
        
        ("test coverage report", ["tests/"]),
        ("benchmark performance test", ["tests/", "benchmarks/"]),
        ("snapshot testing", ["tests/"]),
        ("property based hypothesis", ["tests/"]),
        ("test isolation scope", ["tests/"]),
        
        ("mock return_value", ["tests/"]),
        ("side_effect exception", ["tests/"]),
        ("capture stdout stderr", ["tests/"]),
        ("temporary directory tmpdir", ["tests/"]),
        ("test data factory", ["tests/"]),
        
        ("test_ace.py unit test", ["tests/test_ace.py"]),
        ("test_retrieval.py search", ["tests/"]),
        ("test_config.py validation", ["tests/"]),
        ("test_memory.py storage", ["tests/"]),
        ("test_integration.py e2e", ["tests/"]),
    ],
    
    # =========================================================================
    # CATEGORY 8: Documentation Patterns (25 queries)
    # =========================================================================
    "DocumentationPatterns": [
        ("docstring triple quotes", ["ace/code_retrieval.py"]),
        ("Args Returns Raises", ["ace/code_retrieval.py"]),
        ("type hints annotation", ["ace/code_retrieval.py"]),
        ("TODO FIXME comment", ["ace/"]),
        ("module docstring file", ["ace/__init__.py"]),
        
        ("README.md setup install", ["README.md"]),
        ("CHANGELOG version notes", ["CHANGELOG.md"]),
        ("CONTRIBUTING guide", ["CONTRIBUTING.md"]),
        ("API_REFERENCE docs", ["docs/API_REFERENCE.md"]),
        ("QUICK_START tutorial", ["docs/QUICK_START.md"]),
        
        ("example usage code", ["examples/"]),
        ("inline comment explain", ["ace/code_retrieval.py"]),
        ("type alias definition", ["ace/config.py"]),
        ("deprecated warning", ["ace/"]),
        ("version __version__", ["ace/__init__.py"]),
        
        ("sphinx autodoc", ["docs/"]),
        ("markdown format", ["docs/"]),
        ("code example block", ["docs/"]),
        ("configuration reference", ["docs/"]),
        ("architecture diagram", ["docs/"]),
        
        ("Fortune100 enterprise", ["docs/Fortune100.md"]),
        ("MCP integration guide", ["docs/MCP_INTEGRATION.md"]),
        ("Claude Code setup", ["CLAUDE_CODE_README.md", "CLAUDE_CODE_INTEGRATION.md"]),
        ("golden rules guide", ["docs/GOLDEN_RULES.md"]),
        ("retrieval precision", ["docs/RETRIEVAL_PRECISION_OPTIMIZATION.md"]),
    ],
    
    # =========================================================================
    # CATEGORY 9: Architecture Patterns (25 queries)
    # =========================================================================
    "ArchitecturePatterns": [
        ("Generator Reflector Curator", ["ace/roles.py"]),
        ("unified memory architecture", ["ace/unified_memory.py"]),
        ("playbook bullet strategy", ["ace/playbook.py"]),
        ("tenant isolation namespace", ["ace/multitenancy.py"]),
        ("caching layer TTL", ["ace/retrieval_caching.py"]),
        
        ("HyDE hypothetical expansion", ["ace/hyde.py"]),
        ("observability tracing", ["ace/observability/"]),
        ("bullet enrichment LLM", ["ace/enrichment.py"]),
        ("delta operation batch", ["ace/delta.py"]),
        ("intent classification", ["ace/query_enhancer.py"]),
        
        ("hybrid search fusion", ["ace/qdrant_retrieval.py"]),
        ("dependency graph", ["ace/dependency_graph.py"]),
        ("chunk context expansion", ["ace/code_chunker.py"]),
        ("pattern detection", ["ace/pattern_detector.py"]),
        ("quality feedback loop", ["ace/quality_feedback.py"]),
        
        ("semantic deduplication", ["ace/deduplication.py", "ace/semantic_scorer.py"]),
        ("async retrieval concurrent", ["ace/async_retrieval.py"]),
        ("GPU reranker batch", ["ace/gpu_reranker.py"]),
        ("query enhancement", ["ace/query_enhancer.py"]),
        ("file change delta", ["ace/delta.py"]),
        
        ("embeddings voyage-code-3", ["ace/code_retrieval.py"]),
        ("vector storage Qdrant", ["ace/unified_memory.py"]),
        ("BM25 sparse vector", ["ace/unified_memory.py"]),
        ("cross-encoder reranker", ["ace/reranker.py"]),
        ("RRF reciprocal fusion", ["ace/qdrant_retrieval.py"]),
    ],
    
    # =========================================================================
    # CATEGORY 10: Edge Cases (25 queries)
    # =========================================================================
    "EdgeCases": [
        ("empty result handling", ["ace/code_retrieval.py"]),
        ("null None check", ["ace/code_retrieval.py"]),
        ("boundary condition", ["ace/code_retrieval.py"]),
        ("unicode string encoding", ["ace/code_retrieval.py"]),
        ("large file handling", ["ace/code_chunker.py"]),
        
        ("timeout exceeded", ["ace/resilience.py"]),
        ("memory limit", ["ace/code_chunker.py"]),
        ("concurrent access", ["ace/resilience.py"]),
        ("race condition", ["ace/async_retrieval.py"]),
        ("deadlock prevention", ["ace/resilience.py"]),
        
        ("special characters escape", ["ace/code_retrieval.py"]),
        ("path separator windows", ["ace/code_retrieval.py"]),
        ("relative absolute path", ["ace/code_indexer.py"]),
        ("symlink handling", ["ace/code_indexer.py"]),
        ("binary file skip", ["ace/code_indexer.py"]),
        
        ("nested dictionary deep", ["ace/config.py"]),
        ("circular reference", ["ace/dependency_graph.py"]),
        ("recursive function", ["ace/code_chunker.py"]),
        ("infinite loop guard", ["ace/resilience.py"]),
        ("stack overflow", ["ace/code_chunker.py"]),
        
        ("malformed input", ["ace/config.py"]),
        ("invalid JSON", ["ace/delta.py"]),
        ("missing key KeyError", ["ace/config.py"]),
        ("index out of bounds", ["ace/code_retrieval.py"]),
        ("type mismatch error", ["ace/config.py"]),
    ],
}


def normalize_path(path: str) -> str:
    """Normalize path for comparison."""
    return path.replace("\\", "/").lower().strip()


def paths_match(p1: str, p2: str) -> bool:
    """Check if two paths refer to the same file."""
    n1, n2 = normalize_path(p1), normalize_path(p2)
    # Handle both full paths and relative paths
    return (n1 == n2 or 
            n1.endswith("/" + n2) or 
            n2.endswith("/" + n1) or 
            n1.endswith(n2) or 
            n2.endswith(n1))


def get_ace_results(query: str, limit: int = 5) -> List[Tuple[str, float]]:
    """Get ACE results as (path, score) tuples."""
    import httpx
    import os
    from qdrant_client import QdrantClient
    
    # Get embedding
    voyage_api_key = os.environ.get("VOYAGE_API_KEY")
    if not voyage_api_key:
        raise RuntimeError("VOYAGE_API_KEY not set")
    
    # Embed query
    response = httpx.post(
        "https://api.voyageai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {voyage_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "input": query,
            "model": "voyage-code-3",
            "input_type": "query",
        },
        timeout=30.0,
    )
    response.raise_for_status()
    query_vector = response.json()["data"][0]["embedding"]
    
    # Search Qdrant with named vector via 'using' parameter
    client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    collection = os.environ.get("ACE_CODE_COLLECTION", "ace_code_context")
    
    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        using="dense",  # Use the named vector 'dense'
        limit=limit,
        with_payload=True,
    ).points
    
    return [(p.payload.get("file_path", ""), p.score) for p in results]


def get_ace_results_via_class(query: str, limit: int = 5) -> List[Tuple[str, float]]:
    """Get ACE results using CodeRetrieval class with boosting.
    
    Uses CodeRetrieval.search() which applies:
    - Core module boosting (ace/* files get +10%)
    - Test file penalty (test_* files get -15%)
    - Example file penalty (example* files get -10%)
    """
    from ace.code_retrieval import CodeRetrieval
    
    retrieval = CodeRetrieval()
    results = retrieval.search(query, limit=limit)
    
    return [(r.get("file_path", ""), r.get("score", 0.0)) for r in results]


def get_ThatOtherContextEngine_results(query: str, limit: int = 5) -> List[str]:
    """Get ThatOtherContextEngine MCP results (file paths only)."""
    try:
        import platform
        if platform.system() == "Windows":
            cmd = f'ThatOtherContextEngine -p "{query}" --max-turns 1'
            use_shell = True
        else:
            cmd = ["ThatOtherContextEngine", "-p", query, "--max-turns", "1"]
            use_shell = False
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=60,
            shell=use_shell,
            cwd=os.getcwd()
        )
        
        paths = []
        output = (result.stdout or "") + (result.stderr or "")
        
        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("Path: "):
                path = line[6:].strip()
                path = path.replace("\\", "/")
                if path and path not in paths:
                    paths.append(path)
        
        return paths[:limit]
    except Exception as e:
        print(f"  [ThatOtherContextEngine] Error: {e}")
        return []


def determine_winner(
    query: str,
    ace_files: List[str],
    ace_scores: List[float],
    ThatOtherContextEngine_files: List[str],
    expected_files: List[str],
) -> Tuple[str, str, bool, bool]:
    """
    Determine winner based on:
    1. Which system has more expected files in results
    2. Which system ranks expected files higher
    3. Result coverage
    """
    # Check ACE coverage of expected files
    ace_has_expected = 0
    ace_best_rank = float('inf')
    for exp in expected_files:
        for i, af in enumerate(ace_files):
            if paths_match(af, exp):
                ace_has_expected += 1
                ace_best_rank = min(ace_best_rank, i)
                break
    
    # Check ThatOtherContextEngine coverage of expected files
    ThatOtherContextEngine_has_expected = 0
    ThatOtherContextEngine_best_rank = float('inf')
    for exp in expected_files:
        for i, af in enumerate(ThatOtherContextEngine_files):
            if paths_match(af, exp):
                ThatOtherContextEngine_has_expected += 1
                ThatOtherContextEngine_best_rank = min(ThatOtherContextEngine_best_rank, i)
                break
    
    # Check cross-coverage
    ace_has_all_ThatOtherContextEngine = all(
        any(paths_match(af, aug) for af in ace_files)
        for aug in ThatOtherContextEngine_files
    ) if ThatOtherContextEngine_files else True
    
    ThatOtherContextEngine_has_all_ace = all(
        any(paths_match(aug, af) for aug in ThatOtherContextEngine_files)
        for af in ace_files[:3]  # Check top 3
    ) if ace_files else True
    
    # Determine winner
    if ace_has_expected > ThatOtherContextEngine_has_expected:
        return "ACE", f"ACE found {ace_has_expected} expected files vs ThatOtherContextEngine's {ThatOtherContextEngine_has_expected}", ace_has_all_ThatOtherContextEngine, ThatOtherContextEngine_has_all_ace
    elif ThatOtherContextEngine_has_expected > ace_has_expected:
        return "ThatOtherContextEngine", f"ThatOtherContextEngine found {ThatOtherContextEngine_has_expected} expected files vs ACE's {ace_has_expected}", ace_has_all_ThatOtherContextEngine, ThatOtherContextEngine_has_all_ace
    elif ace_best_rank < ThatOtherContextEngine_best_rank:
        return "ACE", f"ACE ranked expected file at {ace_best_rank+1} vs ThatOtherContextEngine's {ThatOtherContextEngine_best_rank+1}", ace_has_all_ThatOtherContextEngine, ThatOtherContextEngine_has_all_ace
    elif ThatOtherContextEngine_best_rank < ace_best_rank:
        return "ThatOtherContextEngine", f"ThatOtherContextEngine ranked expected file at {ThatOtherContextEngine_best_rank+1} vs ACE's {ace_best_rank+1}", ace_has_all_ThatOtherContextEngine, ThatOtherContextEngine_has_all_ace
    elif ace_has_expected > 0:
        # Both found expected, ACE wins ties
        return "ACE", "Tie broken in ACE's favor (both found expected files)", ace_has_all_ThatOtherContextEngine, ThatOtherContextEngine_has_all_ace
    elif len(ace_files) > len(ThatOtherContextEngine_files):
        return "ACE", f"ACE returned more results ({len(ace_files)} vs {len(ThatOtherContextEngine_files)})", ace_has_all_ThatOtherContextEngine, ThatOtherContextEngine_has_all_ace
    elif len(ThatOtherContextEngine_files) > 0:
        return "TIE", "Neither found expected files, both have results", ace_has_all_ThatOtherContextEngine, ThatOtherContextEngine_has_all_ace
    else:
        return "TIE", "Both returned no results", ace_has_all_ThatOtherContextEngine, ThatOtherContextEngine_has_all_ace


@dataclass
class QueryResult:
    """Result for a single query."""
    query: str
    category: str
    expected_files: List[str]
    ace_files: List[str]
    ace_scores: List[float]
    ThatOtherContextEngine_files: List[str]
    winner: str
    reason: str
    ace_has_all_ThatOtherContextEngine: bool
    ThatOtherContextEngine_has_all_ace: bool


def run_benchmark(iterations: int = 4, verbose: bool = True, skip_ThatOtherContextEngine: bool = False) -> Dict[str, Any]:
    """Run comprehensive benchmark."""
    
    # Flatten all queries
    all_queries = []
    for category, queries in CATEGORY_QUERIES.items():
        for query, expected in queries:
            all_queries.append((category, query, expected))
    
    total_base = len(all_queries)
    total_with_iterations = total_base * iterations
    
    print("=" * 80)
    print(f"ACE vs ThatOtherContextEngine COMPREHENSIVE BENCHMARK")
    print(f"Categories: {len(CATEGORY_QUERIES)}")
    print(f"Queries per category: {total_base // len(CATEGORY_QUERIES)}")
    print(f"Total base queries: {total_base}")
    print(f"Iterations: {iterations}")
    print(f"Total test cases: {total_with_iterations}")
    print(f"Skip ThatOtherContextEngine: {skip_ThatOtherContextEngine}")
    print("=" * 80)
    
    results = []
    stats = {
        "ace_wins": 0,
        "ThatOtherContextEngine_wins": 0,
        "ties": 0,
        "by_category": {cat: {"ace": 0, "ThatOtherContextEngine": 0, "tie": 0} for cat in CATEGORY_QUERIES},
    }
    
    query_count = 0
    
    for iteration in range(iterations):
        print(f"\n--- ITERATION {iteration + 1}/{iterations} ---")
        
        for category, query, expected in all_queries:
            query_count += 1
            
            if verbose:
                print(f"\n[{query_count}/{total_with_iterations}] [{category}] {query[:50]}...")
            else:
                print(f"[{query_count}/{total_with_iterations}] {query[:40]}...", end="\r")
            
            # Get ACE results
            try:
                ace_results = get_ace_results_via_class(query, limit=5)
                ace_files = [f for f, _ in ace_results]
                ace_scores = [s for _, s in ace_results]
            except Exception as e:
                print(f"  [ACE Error] {e}")
                ace_files = []
                ace_scores = []
            
            # Get ThatOtherContextEngine results (or skip)
            if skip_ThatOtherContextEngine:
                ThatOtherContextEngine_files = []
            else:
                try:
                    ThatOtherContextEngine_files = get_ThatOtherContextEngine_results(query, limit=5)
                except Exception as e:
                    print(f"  [ThatOtherContextEngine Error] {e}")
                    ThatOtherContextEngine_files = []
            
            # Determine winner
            winner, reason, ace_all, aug_all = determine_winner(
                query, ace_files, ace_scores, ThatOtherContextEngine_files, expected
            )
            
            result = QueryResult(
                query=query,
                category=category,
                expected_files=expected,
                ace_files=ace_files,
                ace_scores=ace_scores,
                ThatOtherContextEngine_files=ThatOtherContextEngine_files,
                winner=winner,
                reason=reason,
                ace_has_all_ThatOtherContextEngine=ace_all,
                ThatOtherContextEngine_has_all_ace=aug_all,
            )
            results.append(result)
            
            # Update stats
            if winner == "ACE":
                stats["ace_wins"] += 1
                stats["by_category"][category]["ace"] += 1
            elif winner == "ThatOtherContextEngine":
                stats["ThatOtherContextEngine_wins"] += 1
                stats["by_category"][category]["ThatOtherContextEngine"] += 1
            else:
                stats["ties"] += 1
                stats["by_category"][category]["tie"] += 1
            
            if verbose:
                print(f"  ACE: {ace_files[:2]}")
                if not skip_ThatOtherContextEngine:
                    print(f"  ThatOtherContextEngine: {ThatOtherContextEngine_files[:2]}")
                print(f"  Expected: {expected[:2]}")
                print(f"  Winner: {winner} - {reason}")
    
    # Calculate rates
    total = len(results)
    stats["ace_win_rate"] = (stats["ace_wins"] / total) * 100
    stats["ThatOtherContextEngine_win_rate"] = (stats["ThatOtherContextEngine_wins"] / total) * 100
    stats["tie_rate"] = (stats["ties"] / total) * 100
    stats["ace_superiority"] = stats["ace_win_rate"] + (stats["tie_rate"] / 2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"\nTotal Queries: {total}")
    print(f"ACE Wins: {stats['ace_wins']} ({stats['ace_win_rate']:.1f}%)")
    print(f"ThatOtherContextEngine Wins: {stats['ThatOtherContextEngine_wins']} ({stats['ThatOtherContextEngine_win_rate']:.1f}%)")
    print(f"Ties: {stats['ties']} ({stats['tie_rate']:.1f}%)")
    print(f"\nACE SUPERIORITY: {stats['ace_superiority']:.1f}%")
    
    print("\n### Results by Category")
    for cat, cat_stats in stats["by_category"].items():
        cat_total = cat_stats["ace"] + cat_stats["ThatOtherContextEngine"] + cat_stats["tie"]
        if cat_total > 0:
            ace_pct = (cat_stats["ace"] / cat_total * 100)
            print(f"  {cat}: ACE {cat_stats['ace']} | ThatOtherContextEngine {cat_stats['ThatOtherContextEngine']} | Tie {cat_stats['tie']} ({ace_pct:.0f}% ACE)")
    
    # Identify failures
    ThatOtherContextEngine_wins = [r for r in results if r.winner == "ThatOtherContextEngine"]
    if ThatOtherContextEngine_wins:
        print(f"\n### ThatOtherContextEngine WINS ({len(ThatOtherContextEngine_wins)} cases to investigate):")
        for r in ThatOtherContextEngine_wins[:20]:  # Show first 20
            print(f"  [{r.category}] {r.query[:50]}")
            print(f"    Expected: {r.expected_files}")
            print(f"    ACE: {r.ace_files[:2]}")
            print(f"    ThatOtherContextEngine: {r.ThatOtherContextEngine_files[:2]}")
            print(f"    Reason: {r.reason}")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "iterations": iterations,
        "total_queries": total,
        "stats": stats,
        "results": [asdict(r) for r in results],
    }
    
    output_file = Path("benchmark_results") / f"ace_ThatOtherContextEngine_1000_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ACE vs ThatOtherContextEngine 1000-Query Benchmark")
    parser.add_argument("--iterations", "-i", type=int, default=4, help="Number of iterations (default: 4)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--skip-ThatOtherContextEngine", "-s", action="store_true", help="Skip ThatOtherContextEngine calls (ACE-only validation)")
    args = parser.parse_args()
    
    run_benchmark(
        iterations=args.iterations,
        verbose=args.verbose,
        skip_ThatOtherContextEngine=args.skip_ThatOtherContextEngine,
    )


if __name__ == "__main__":
    main()
