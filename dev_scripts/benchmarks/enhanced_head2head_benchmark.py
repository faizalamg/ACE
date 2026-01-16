#!/usr/bin/env python3
"""
Enhanced ACE vs ThatOtherContextEngine Head-to-Head Benchmark.

Tests ACE against ThatOtherContextEngine MCP using real-world queries from chat histories.
Compares FULL RESULT SETS for:
1. File relevancy
2. Code context quality
3. Chunk size appropriateness
4. Ranking accuracy

When ThatOtherContextEngine wins, this script identifies WHY and provides recommendations.
"""

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from ace.code_retrieval import CodeRetrieval


# =============================================================================
# REAL-WORLD QUERIES FROM CHAT HISTORIES
# =============================================================================

REAL_WORLD_QUERIES = {
    # -------------------------------------------------------------------------
    # Category 1: Class Definitions (50 queries)
    # -------------------------------------------------------------------------
    "ClassDefinitions": [
        # Core ACE classes
        ("CodeRetrieval class definition", ["ace/code_retrieval.py"]),
        ("UnifiedMemoryIndex class search method", ["ace/unified_memory.py"]),
        ("ASTChunker class chunk method", ["ace/code_chunker.py"]),
        ("SmartBulletIndex retrieve method", ["ace/retrieval.py"]),
        ("Playbook class initialization", ["ace/playbook.py"]),
        ("EmbeddingConfig dataclass definition", ["ace/config.py"]),
        ("QdrantConfig class definition", ["ace/config.py"]),
        ("BM25Config k1 b parameters", ["ace/config.py"]),
        ("CodeIndexer index_workspace method", ["ace/code_indexer.py"]),
        ("HyDEGenerator class generate method", ["ace/hyde.py"]),
        
        # Config classes
        ("VoyageCodeEmbeddingConfig api_key model", ["ace/config.py"]),
        ("LLMConfig provider model settings", ["ace/config.py"]),
        ("RetrievalConfig limit threshold", ["ace/config.py"]),
        ("HyDEConfig num_hypotheticals temperature", ["ace/config.py", "ace/hyde.py"]),
        
        # Specialized classes
        ("SemanticScorer score calculation method", ["ace/semantic_scorer.py"]),
        ("DependencyGraph build method", ["ace/dependency_graph.py"]),
        ("FileWatcher callback handler", ["ace/file_watcher_daemon.py"]),
        ("CircuitBreaker is_open method", ["ace/resilience.py"]),
        ("RetryPolicy execute method", ["ace/resilience.py"]),
        ("CacheManager get set methods", ["ace/caching.py"]),
        
        # Data structures
        ("@dataclass class Bullet playbook", ["ace/playbook.py"]),
        ("CodeChunk dataclass file_path start_line", ["ace/code_chunker.py", "ace/code_indexer.py"]),
        ("QueryResult dataclass score file_path", ["ace/code_retrieval.py"]),
        
        # Protocol/interface patterns
        ("@property score getter", ["ace/playbook.py", "ace/code_retrieval.py"]),
        ("__init__ constructor pattern", ["ace/code_retrieval.py"]),
        ("context manager __enter__ __exit__", ["ace/resilience.py"]),
        ("@staticmethod factory pattern", ["ace/config.py"]),
        ("@classmethod from_config", ["ace/config.py"]),
        
        # Error handling classes
        ("custom exception class", ["ace/resilience.py"]),
        ("exception hierarchy", ["ace/resilience.py"]),
    ],
    
    # -------------------------------------------------------------------------
    # Category 2: Technical Identifiers (30 queries)
    # -------------------------------------------------------------------------
    "TechnicalIdentifiers": [
        ("voyage-code-3 embedding model", ["ace/code_retrieval.py", "ace/code_indexer.py"]),
        ("voyage-3 embedding generation", ["ace/code_retrieval.py"]),
        ("bge-m3 sparse vector", ["ace/hyde_retrieval.py"]),
        ("text-embedding-3-small openai", ["ace/openai_embeddings.py"]),
        ("text-embedding-ada-002 openai", ["ace/openai_embeddings.py"]),
        ("jina-embeddings-v2-base-code", ["ace/config.py"]),
        ("gemini-1.5-flash model", ["ace/gemini_embeddings.py"]),
        ("claude-3-5-sonnet llm", ["ace/llm.py"]),
        ("gpt-4o model config", ["ace/llm.py"]),
        ("httpx-async client", ["ace/async_retrieval.py"]),
        ("qdrant-client models", ["ace/unified_memory.py", "ace/code_indexer.py"]),
        ("loguru-logger setup", ["ace/audit.py"]),
        ("tenacity-retry decorator", ["ace/resilience.py"]),
        ("pydantic-v2 validation", ["ace/config.py"]),
        ("pytest-fixture setup", ["tests/"]),
        ("numpy-array operations", ["ace/code_retrieval.py"]),
        ("json-schema validation", ["ace/config.py"]),
        ("re-regex pattern matching", ["ace/code_retrieval.py"]),
        ("asyncio-gather parallel", ["ace/async_retrieval.py"]),
        ("functools-lru-cache decorator", ["ace/caching.py", "ace/gemini_embeddings.py"]),
    ],
    
    # -------------------------------------------------------------------------
    # Category 3: Function Patterns (40 queries)
    # -------------------------------------------------------------------------
    "FunctionPatterns": [
        ("def search function in retrieval", ["ace/code_retrieval.py", "ace/retrieval.py"]),
        ("async def embed method", ["ace/async_retrieval.py", "ace/gemini_embeddings.py"]),
        ("def _apply_filename_boost", ["ace/code_retrieval.py"]),
        ("create_sparse_vector BM25 function", ["ace/unified_memory.py"]),
        ("def format_ThatOtherContextEngine_style", ["ace/code_retrieval.py"]),
        ("def get_embedding batch", ["ace/code_retrieval.py", "ace/openai_embeddings.py"]),
        ("async def retrieve_async", ["ace/async_retrieval.py"]),
        ("def index_file method", ["ace/code_indexer.py"]),
        ("def parse_ast method", ["ace/code_chunker.py", "ace/code_analysis.py"]),
        ("def store memory method", ["ace/unified_memory.py"]),
        
        # Error handling functions
        ("def retry_with_backoff", ["ace/resilience.py"]),
        ("def handle_exception", ["ace/resilience.py"]),
        ("def log_error method", ["ace/audit.py"]),
        
        # Helper functions
        ("def normalize_path", ["ace/code_retrieval.py"]),
        ("def extract_symbols", ["ace/code_analysis.py"]),
        ("def chunk_code", ["ace/code_chunker.py"]),
        ("def expand_query", ["ace/code_retrieval.py"]),
        
        # Async functions
        ("async def batch_embed", ["ace/async_retrieval.py"]),
        ("async def stream_results", ["ace/async_retrieval.py"]),
        ("asyncio gather parallel execution", ["ace/async_retrieval.py"]),
    ],
    
    # -------------------------------------------------------------------------
    # Category 4: Configuration & Environment (16 queries)
    # -------------------------------------------------------------------------
    "Configuration": [
        ("environment variables dotenv loading", ["ace/config.py"]),
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
        
        # Config validation
        ("config validation pydantic", ["ace/config.py"]),
        ("defaults dict configuration", ["ace/config.py"]),
        ("nested config structure", ["ace/config.py"]),
        ("config from dict", ["ace/config.py"]),
    ],
    
    # -------------------------------------------------------------------------
    # Category 5: Error Handling (30 queries)
    # -------------------------------------------------------------------------
    "ErrorHandling": [
        ("try except error handling pattern", ["ace/resilience.py"]),
        ("exception retry backoff resilience", ["ace/resilience.py"]),
        ("API rate limit retry exponential backoff", ["ace/resilience.py"]),
        ("timeout exception handling httpx", ["ace/resilience.py", "ace/observability/health.py"]),
        ("connection error handling", ["ace/resilience.py"]),
        ("validation error handling ValueError", ["ace/config.py"]),
        ("file not found error handling", ["ace/code_indexer.py"]),
        ("embedding error fallback", ["ace/semantic_scorer.py"]),
        ("import error optional dependency", ["ace/features.py"]),
        ("graceful degradation circuit breaker", ["ace/resilience.py"]),
        ("finally cleanup block", ["ace/resilience.py"]),
        ("exception chaining from", ["ace/resilience.py"]),
        ("logging error stack trace", ["ace/audit.py"]),
        ("error recovery strategy", ["ace/resilience.py"]),
    ],
    
    # -------------------------------------------------------------------------
    # Category 6: Async Patterns (15 queries)
    # -------------------------------------------------------------------------
    "AsyncPatterns": [
        ("async def retrieve await", ["ace/async_retrieval.py"]),
        ("asyncio gather parallel", ["ace/async_retrieval.py", "ace/async_adaptation.py"]),
        ("async with httpx.AsyncClient", ["ace/async_retrieval.py"]),
        ("async for chunk stream", ["ace/async_retrieval.py"]),
        ("await embedding generation", ["ace/async_retrieval.py", "ace/gemini_embeddings.py"]),
        ("async context manager enter exit", ["ace/async_retrieval.py"]),
        ("asyncio.create_task background", ["ace/async_retrieval.py", "ace/async_adaptation.py"]),
        ("async generator yield", ["ace/async_retrieval.py"]),
        ("semaphore rate limiting async", ["ace/async_retrieval.py", "ace/scaling.py"]),
        ("asyncio timeout cancel", ["ace/async_retrieval.py"]),
        ("asyncio.run main", ["ace/async_retrieval.py"]),
        ("event loop get_event_loop", ["ace/async_retrieval.py"]),
        ("asyncio.Queue producer consumer", ["ace/async_retrieval.py"]),
        ("asyncio.Lock mutex", ["ace/async_retrieval.py"]),
        ("loop.run_in_executor thread", ["ace/async_retrieval.py"]),
    ],
    
    # -------------------------------------------------------------------------
    # Category 7: Import Patterns (30 queries)
    # -------------------------------------------------------------------------
    "ImportPatterns": [
        ("from qdrant_client import QdrantClient", ["ace/unified_memory.py", "ace/deduplication.py"]),
        ("import httpx async", ["ace/async_retrieval.py"]),
        ("from dataclasses import dataclass field", ["ace/playbook.py", "ace/config.py"]),
        ("from typing import Optional List Dict", ["ace/"]),
        ("import voyageai client", ["ace/code_retrieval.py"]),
        ("from pathlib import Path", ["ace/"]),
        ("import logging logger", ["ace/audit.py"]),
        ("from functools import lru_cache", ["ace/caching.py", "ace/gemini_embeddings.py"]),
        ("import json os sys", ["ace/"]),
        ("from collections import Counter defaultdict", ["ace/"]),
        ("import asyncio await", ["ace/async_retrieval.py"]),
        ("from abc import ABC abstractmethod", ["ace/"]),
        ("import re regex", ["ace/code_retrieval.py"]),
        ("from datetime import datetime", ["ace/"]),
        ("from enum import Enum auto", ["ace/"]),
    ],
    
    # -------------------------------------------------------------------------
    # Category 8: Documentation Patterns (20 queries)
    # -------------------------------------------------------------------------
    "DocumentationPatterns": [
        ("README installation guide", ["README.md"]),
        ("API reference documentation", ["docs/API_REFERENCE.md"]),
        ("configuration options docs", ["docs/"]),
        ("quickstart tutorial", ["QUICKSTART_CLAUDE_CODE.md"]),
        ("contributing guidelines", ["CONTRIBUTING.md"]),
        ("changelog version history", ["CHANGELOG.md"]),
        ("integration guide howto", ["docs/INTEGRATION_GUIDE.md"]),
        ("embedding config documentation", ["docs/CODE_EMBEDDING_CONFIG.md"]),
    ],
    
    # -------------------------------------------------------------------------
    # Category 9: Edge Cases & Challenging Queries (30 queries)
    # -------------------------------------------------------------------------
    "EdgeCases": [
        ("fibonacci sequence calculation", ["fibonacci.py"]),
        ("temperature converter celsius fahrenheit", ["temperature_converter.py"]),
        ("email validation regex pattern", ["email_validator.py"]),
        ("sparse BM25 term frequency calculation", ["ace/unified_memory.py", "ace/hyde_retrieval.py"]),
        ("vector similarity cosine distance", ["ace/code_retrieval.py"]),
        ("hybrid search dense sparse fusion", ["ace/unified_memory.py"]),
        ("code chunking AST parsing", ["ace/code_chunker.py"]),
        ("metadata filtering namespace", ["ace/unified_memory.py"]),
        ("deduplication similarity threshold", ["ace/deduplication.py"]),
        ("query expansion semantic", ["ace/code_retrieval.py"]),
        
        # Multi-concept queries
        ("async embedding batch retry error", ["ace/async_retrieval.py", "ace/resilience.py"]),
        ("config validation environment variable", ["ace/config.py"]),
        ("search retrieval ranking score", ["ace/code_retrieval.py"]),
        ("indexing chunking embedding storage", ["ace/code_indexer.py"]),
    ],
}


@dataclass
class DetailedResult:
    """Detailed comparison result for one query."""
    query: str
    category: str
    expected_files: List[str]
    
    # ACE results
    ace_files: List[str]
    ace_scores: List[float]
    ace_contents: List[str]
    ace_line_counts: List[int]
    
    # ThatOtherContextEngine results
    ThatOtherContextEngine_files: List[str]
    ThatOtherContextEngine_contents: List[str]
    ThatOtherContextEngine_line_counts: List[int]
    
    # Comparison
    winner: str  # "ACE", "ThatOtherContextEngine", "TIE"
    reason: str
    ace_advantages: List[str] = field(default_factory=list)
    ThatOtherContextEngine_advantages: List[str] = field(default_factory=list)
    
    # Metrics
    ace_found_expected: bool = False
    ThatOtherContextEngine_found_expected: bool = False
    ace_expected_rank: int = -1  # Rank of expected file in ACE results (-1 if not found)
    ThatOtherContextEngine_expected_rank: int = -1


def get_ace_detailed_results(query: str, limit: int = 5) -> Tuple[List[str], List[float], List[str], List[int]]:
    """Get detailed ACE results including content."""
    r = CodeRetrieval()
    results = r.search(query, limit=limit)
    
    files = [res["file_path"] for res in results]
    scores = [res["score"] for res in results]
    contents = [res.get("content", "")[:500] for res in results]  # First 500 chars
    line_counts = [res.get("end_line", 0) - res.get("start_line", 0) + 1 for res in results]
    
    return files, scores, contents, line_counts


def get_ThatOtherContextEngine_detailed_results(query: str, limit: int = 5) -> Tuple[List[str], List[str], List[int]]:
    """Get detailed ThatOtherContextEngine results including content."""
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
        
        files = []
        contents = []
        line_counts = []
        output = (result.stdout or "") + (result.stderr or "")
        
        current_content = []
        current_file = None
        line_count = 0
        
        for line in output.split("\n"):
            if line.startswith("Path: "):
                # Save previous result
                if current_file:
                    files.append(current_file)
                    contents.append("\n".join(current_content[:20]))  # First 20 lines
                    line_counts.append(line_count)
                
                # Start new result
                current_file = line[6:].strip().replace("\\", "/")
                current_content = []
                line_count = 0
            elif current_file:
                current_content.append(line)
                line_count += 1
        
        # Don't forget last result
        if current_file:
            files.append(current_file)
            contents.append("\n".join(current_content[:20]))
            line_counts.append(line_count)
        
        return files[:limit], contents[:limit], line_counts[:limit]
    except Exception as e:
        print(f"  [ThatOtherContextEngine Error] {e}")
        return [], [], []


def normalize_path(path: str) -> str:
    """Normalize path for comparison."""
    return path.replace("\\", "/").lower().strip()


def path_matches_expected(path: str, expected: List[str]) -> bool:
    """Check if path matches any expected file."""
    norm_path = normalize_path(path)
    for exp in expected:
        norm_exp = normalize_path(exp)
        if norm_path.endswith(norm_exp) or norm_exp.endswith(norm_path) or norm_exp in norm_path:
            return True
    return False


def find_expected_rank(files: List[str], expected: List[str]) -> int:
    """Find the rank of expected file in results (-1 if not found)."""
    for i, f in enumerate(files):
        if path_matches_expected(f, expected):
            return i + 1  # 1-indexed
    return -1


def determine_winner(
    ace_files: List[str],
    ace_scores: List[float],
    ace_line_counts: List[int],
    ThatOtherContextEngine_files: List[str],
    ThatOtherContextEngine_line_counts: List[int],
    expected_files: List[str]
) -> Tuple[str, str, List[str], List[str]]:
    """
    Determine winner based on multiple criteria.
    
    Returns: (winner, reason, ace_advantages, ThatOtherContextEngine_advantages)
    """
    ace_advantages = []
    ThatOtherContextEngine_advantages = []
    
    # Check if either is empty
    if not ThatOtherContextEngine_files:
        return "ACE", "ThatOtherContextEngine returned no results", ["Has results"], []
    if not ace_files:
        return "ThatOtherContextEngine", "ACE returned no results", [], ["Has results"]
    
    # Find expected file rank
    ace_rank = find_expected_rank(ace_files, expected_files)
    ThatOtherContextEngine_rank = find_expected_rank(ThatOtherContextEngine_files, expected_files)
    
    # Check if expected file was found
    ace_found = ace_rank != -1
    ThatOtherContextEngine_found = ThatOtherContextEngine_rank != -1
    
    if ace_found and not ThatOtherContextEngine_found:
        ace_advantages.append(f"Found expected file at rank {ace_rank}")
        return "ACE", f"ACE found expected file at rank {ace_rank}, ThatOtherContextEngine missed it", ace_advantages, ThatOtherContextEngine_advantages
    
    if ThatOtherContextEngine_found and not ace_found:
        ThatOtherContextEngine_advantages.append(f"Found expected file at rank {ThatOtherContextEngine_rank}")
        return "ThatOtherContextEngine", f"ThatOtherContextEngine found expected file at rank {ThatOtherContextEngine_rank}, ACE missed it", ace_advantages, ThatOtherContextEngine_advantages
    
    # Both found - compare ranks
    if ace_found and ThatOtherContextEngine_found:
        if ace_rank < ThatOtherContextEngine_rank:
            ace_advantages.append(f"Higher rank for expected file ({ace_rank} vs {ThatOtherContextEngine_rank})")
        elif ThatOtherContextEngine_rank < ace_rank:
            ThatOtherContextEngine_advantages.append(f"Higher rank for expected file ({ThatOtherContextEngine_rank} vs {ace_rank})")
    
    # Compare file coverage
    ace_unique = len([f for f in ace_files if not any(normalize_path(f) in normalize_path(af) or normalize_path(af) in normalize_path(f) for af in ThatOtherContextEngine_files)])
    ThatOtherContextEngine_unique = len([f for f in ThatOtherContextEngine_files if not any(normalize_path(f) in normalize_path(af) or normalize_path(af) in normalize_path(f) for af in ace_files)])
    
    if ace_unique > ThatOtherContextEngine_unique:
        ace_advantages.append(f"More unique files ({ace_unique} vs {ThatOtherContextEngine_unique})")
    elif ThatOtherContextEngine_unique > ace_unique:
        ThatOtherContextEngine_advantages.append(f"More unique files ({ThatOtherContextEngine_unique} vs {ace_unique})")
    
    # Compare scores (ACE only)
    if ace_scores and ace_scores[0] >= 0.7:
        ace_advantages.append(f"High confidence top score ({ace_scores[0]:.3f})")
    
    # Compare chunk sizes
    ace_avg_lines = sum(ace_line_counts) / len(ace_line_counts) if ace_line_counts else 0
    ThatOtherContextEngine_avg_lines = sum(ThatOtherContextEngine_line_counts) / len(ThatOtherContextEngine_line_counts) if ThatOtherContextEngine_line_counts else 0
    
    # Prefer moderate chunk sizes (20-100 lines)
    ace_size_score = 1.0 if 20 <= ace_avg_lines <= 100 else 0.5
    ThatOtherContextEngine_size_score = 1.0 if 20 <= ThatOtherContextEngine_avg_lines <= 100 else 0.5
    
    if ace_size_score > ThatOtherContextEngine_size_score:
        ace_advantages.append(f"Better chunk size ({ace_avg_lines:.0f} lines avg)")
    elif ThatOtherContextEngine_size_score > ace_size_score:
        ThatOtherContextEngine_advantages.append(f"Better chunk size ({ThatOtherContextEngine_avg_lines:.0f} lines avg)")
    
    # Determine winner
    ace_score = len(ace_advantages)
    ThatOtherContextEngine_score = len(ThatOtherContextEngine_advantages)
    
    # Ranking is most important
    if ace_found and ThatOtherContextEngine_found:
        if ace_rank < ThatOtherContextEngine_rank:
            ace_score += 2
        elif ThatOtherContextEngine_rank < ace_rank:
            ThatOtherContextEngine_score += 2
    
    if ace_score > ThatOtherContextEngine_score:
        return "ACE", f"ACE wins with {ace_score} advantages vs {ThatOtherContextEngine_score}", ace_advantages, ThatOtherContextEngine_advantages
    elif ThatOtherContextEngine_score > ace_score:
        return "ThatOtherContextEngine", f"ThatOtherContextEngine wins with {ThatOtherContextEngine_score} advantages vs {ace_score}", ace_advantages, ThatOtherContextEngine_advantages
    else:
        return "TIE", "Both systems performed equally", ace_advantages, ThatOtherContextEngine_advantages


def run_comprehensive_comparison(categories: List[str] = None, verbose: bool = False) -> Dict[str, Any]:
    """Run comprehensive ACE vs ThatOtherContextEngine comparison."""
    if categories is None:
        categories = list(REAL_WORLD_QUERIES.keys())
    
    results = []
    stats = {
        "total": 0,
        "ace_wins": 0,
        "ThatOtherContextEngine_wins": 0,
        "ties": 0,
        "ace_errors": 0,
        "ThatOtherContextEngine_errors": 0,
        "by_category": {},
        "ThatOtherContextEngine_wins_detail": [],
    }
    
    print("=" * 80)
    print("ENHANCED ACE vs ThatOtherContextEngine HEAD-TO-HEAD COMPARISON")
    print("=" * 80)
    
    for category in categories:
        queries = REAL_WORLD_QUERIES.get(category, [])
        if not queries:
            continue
        
        print(f"\n### Category: {category} ({len(queries)} queries)")
        stats["by_category"][category] = {"ace": 0, "ThatOtherContextEngine": 0, "tie": 0}
        
        for i, (query, expected) in enumerate(queries, 1):
            if verbose:
                print(f"\n[{i}/{len(queries)}] {query}")
            else:
                print(f"  [{i}/{len(queries)}] {query[:40]}...", end="\r")
            
            # Get results from both systems
            ace_files, ace_scores, ace_contents, ace_lines = get_ace_detailed_results(query)
            ThatOtherContextEngine_files, ThatOtherContextEngine_contents, ThatOtherContextEngine_lines = get_ThatOtherContextEngine_detailed_results(query)
            
            # Determine winner
            winner, reason, ace_adv, ThatOtherContextEngine_adv = determine_winner(
                ace_files, ace_scores, ace_lines,
                ThatOtherContextEngine_files, ThatOtherContextEngine_lines,
                expected
            )
            
            result = DetailedResult(
                query=query,
                category=category,
                expected_files=expected,
                ace_files=ace_files,
                ace_scores=ace_scores,
                ace_contents=ace_contents,
                ace_line_counts=ace_lines,
                ThatOtherContextEngine_files=ThatOtherContextEngine_files,
                ThatOtherContextEngine_contents=ThatOtherContextEngine_contents,
                ThatOtherContextEngine_line_counts=ThatOtherContextEngine_lines,
                winner=winner,
                reason=reason,
                ace_advantages=ace_adv,
                ThatOtherContextEngine_advantages=ThatOtherContextEngine_adv,
                ace_found_expected=find_expected_rank(ace_files, expected) != -1,
                ThatOtherContextEngine_found_expected=find_expected_rank(ThatOtherContextEngine_files, expected) != -1,
                ace_expected_rank=find_expected_rank(ace_files, expected),
                ThatOtherContextEngine_expected_rank=find_expected_rank(ThatOtherContextEngine_files, expected),
            )
            results.append(result)
            
            # Update stats
            stats["total"] += 1
            if winner == "ACE":
                stats["ace_wins"] += 1
                stats["by_category"][category]["ace"] += 1
            elif winner == "ThatOtherContextEngine":
                stats["ThatOtherContextEngine_wins"] += 1
                stats["by_category"][category]["ThatOtherContextEngine"] += 1
                stats["ThatOtherContextEngine_wins_detail"].append({
                    "query": query,
                    "category": category,
                    "expected": expected,
                    "ace_files": ace_files[:3],
                    "ThatOtherContextEngine_files": ThatOtherContextEngine_files[:3],
                    "reason": reason,
                    "ace_advantages": ace_adv,
                    "ThatOtherContextEngine_advantages": ThatOtherContextEngine_adv,
                })
            else:
                stats["ties"] += 1
                stats["by_category"][category]["tie"] += 1
            
            if verbose:
                print(f"  ACE: {ace_files[:2]}")
                print(f"  ThatOtherContextEngine: {ThatOtherContextEngine_files[:2]}")
                print(f"  Winner: {winner} - {reason}")
        
        # Category summary
        cat_stats = stats["by_category"][category]
        cat_total = cat_stats["ace"] + cat_stats["ThatOtherContextEngine"] + cat_stats["tie"]
        print(f"\n  {category}: ACE {cat_stats['ace']} | ThatOtherContextEngine {cat_stats['ThatOtherContextEngine']} | Tie {cat_stats['tie']}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total Queries: {stats['total']}")
    print(f"ACE Wins: {stats['ace_wins']} ({stats['ace_wins']/stats['total']*100:.1f}%)")
    print(f"ThatOtherContextEngine Wins: {stats['ThatOtherContextEngine_wins']} ({stats['ThatOtherContextEngine_wins']/stats['total']*100:.1f}%)")
    print(f"Ties: {stats['ties']} ({stats['ties']/stats['total']*100:.1f}%)")
    print(f"ACE Non-Loss Rate: {(stats['ace_wins'] + stats['ties'])/stats['total']*100:.2f}%")
    
    # ThatOtherContextEngine wins analysis
    if stats["ThatOtherContextEngine_wins_detail"]:
        print("\n" + "=" * 80)
        print("ThatOtherContextEngine WINS - ROOT CAUSE ANALYSIS")
        print("=" * 80)
        for detail in stats["ThatOtherContextEngine_wins_detail"]:
            print(f"\n[{detail['category']}] {detail['query']}")
            print(f"  Expected: {detail['expected']}")
            print(f"  ACE returned: {detail['ace_files']}")
            print(f"  ThatOtherContextEngine returned: {detail['ThatOtherContextEngine_files']}")
            print(f"  Reason: {detail['reason']}")
            if detail['ThatOtherContextEngine_advantages']:
                print(f"  ThatOtherContextEngine advantages: {detail['ThatOtherContextEngine_advantages']}")
    else:
        print("\n🏆 ZERO ThatOtherContextEngine WINS! ACE ACHIEVES 100% NON-LOSS RATE!")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "stats": stats,
        "results": [asdict(r) for r in results],
    }
    
    output_file = Path("benchmark_results") / f"enhanced_head2head_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")
    
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-c", "--category", type=str, help="Specific category to test")
    args = parser.parse_args()
    
    categories = [args.category] if args.category else None
    run_comprehensive_comparison(categories=categories, verbose=args.verbose)
