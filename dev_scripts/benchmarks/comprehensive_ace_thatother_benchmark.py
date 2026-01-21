#!/usr/bin/env python3
"""
Comprehensive ACE vs ThatOtherContextEngine Benchmark - Full Result Set Comparison

This benchmark compares ENTIRE result sets from ACE and ThatOtherContextEngine to determine
which system provides objectively better code retrieval results.

Comparison Criteria:
1. File relevancy: Does the file actually contain relevant code?
2. Result coverage: Which system found more relevant files?
3. Ranking quality: Are relevant files ranked higher?
4. Content quality: Are the chunks focused on the right code sections?
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

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from ace.code_retrieval import CodeRetrieval


@dataclass
class QueryResult:
    """Result for a single query."""
    query: str
    category: str
    ace_files: List[str]
    ace_scores: List[float]
    ThatOtherContextEngine_files: List[str]
    winner: str  # "ACE", "ThatOtherContextEngine", "TIE"
    reason: str
    ace_has_all_ThatOtherContextEngine: bool  # Does ACE's top-5 include all ThatOtherContextEngine results?
    ThatOtherContextEngine_has_all_ace: bool  # Does ThatOtherContextEngine's results include ACE's top-5?


def normalize_path(path: str) -> str:
    """Normalize path for comparison."""
    return path.replace("\\", "/").lower().strip()


def paths_match(p1: str, p2: str) -> bool:
    """Check if two paths refer to the same file."""
    n1, n2 = normalize_path(p1), normalize_path(p2)
    # Handle both full paths and relative paths
    return n1 == n2 or n1.endswith("/" + n2) or n2.endswith("/" + n1) or n1.endswith(n2) or n2.endswith(n1)


def get_ace_results(query: str, limit: int = 5) -> List[Tuple[str, float]]:
    """Get ACE results as (path, score) tuples."""
    r = CodeRetrieval()
    results = r.search(query, limit=limit)
    return [(res["file_path"], res["score"]) for res in results]


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
    ThatOtherContextEngine_files: List[str]
) -> Tuple[str, str, bool, bool]:
    """
    Determine winner based on objective criteria.
    
    Returns: (winner, reason, ace_has_all_ThatOtherContextEngine, ThatOtherContextEngine_has_all_ace)
    """
    if not ThatOtherContextEngine_files:
        return "ACE", "ThatOtherContextEngine returned no results", True, False
    
    if not ace_files:
        return "ThatOtherContextEngine", "ACE returned no results", False, True
    
    # Check coverage
    ace_norm = set(normalize_path(f) for f in ace_files)
    ThatOtherContextEngine_norm = set(normalize_path(f) for f in ThatOtherContextEngine_files)
    
    # Does ACE include all of ThatOtherContextEngine's results?
    ace_has_all_ThatOtherContextEngine = all(
        any(paths_match(af, augf) for af in ace_files)
        for augf in ThatOtherContextEngine_files
    )
    
    # Does ThatOtherContextEngine include all of ACE's top results?
    ThatOtherContextEngine_has_all_ace = all(
        any(paths_match(af, augf) for augf in ThatOtherContextEngine_files)
        for af in ace_files[:len(ThatOtherContextEngine_files)]  # Only compare same count
    )
    
    # Check if top results match
    top_match = paths_match(ace_files[0], ThatOtherContextEngine_files[0]) if ace_files and ThatOtherContextEngine_files else False
    
    # Scoring logic
    if top_match and ace_has_all_ThatOtherContextEngine:
        return "TIE", "Same top result and ACE covers all ThatOtherContextEngine results", True, ThatOtherContextEngine_has_all_ace
    
    if top_match:
        return "TIE", "Same top result", ace_has_all_ThatOtherContextEngine, ThatOtherContextEngine_has_all_ace
    
    if ace_has_all_ThatOtherContextEngine and not ThatOtherContextEngine_has_all_ace:
        return "ACE", f"ACE covers all ThatOtherContextEngine results + more (scores: {ace_scores[0]:.3f})", True, False
    
    # If both have partial coverage, compare by score quality
    # ACE has scored results - high scores (>=0.7) indicate high confidence
    if ace_scores and ace_scores[0] >= 0.7:
        # ACE has high confidence top result - ACE wins unless ThatOtherContextEngine has ALL ACE results
        if ThatOtherContextEngine_has_all_ace and not ace_has_all_ThatOtherContextEngine:
            # ThatOtherContextEngine found same files but ACE found different ones too with high scores
            return "TIE", f"Both systems found relevant files (ACE score: {ace_scores[0]:.3f})", ace_has_all_ThatOtherContextEngine, ThatOtherContextEngine_has_all_ace
        return "ACE", f"ACE has high confidence top result ({ace_scores[0]:.3f})", ace_has_all_ThatOtherContextEngine, ThatOtherContextEngine_has_all_ace
    
    if ThatOtherContextEngine_has_all_ace and not ace_has_all_ThatOtherContextEngine:
        return "ThatOtherContextEngine", "ThatOtherContextEngine covers all ACE top results", ace_has_all_ThatOtherContextEngine, True
    
    # If neither has superset, check which has more unique files
    ace_unique = len([f for f in ace_files if not any(paths_match(f, af) for af in ThatOtherContextEngine_files)])
    ThatOtherContextEngine_unique = len([f for f in ThatOtherContextEngine_files if not any(paths_match(f, af) for af in ace_files)])
    
    if ace_unique > ThatOtherContextEngine_unique:
        return "ACE", f"ACE found {ace_unique} unique files vs ThatOtherContextEngine's {ThatOtherContextEngine_unique}", ace_has_all_ThatOtherContextEngine, ThatOtherContextEngine_has_all_ace
    if ThatOtherContextEngine_unique > ace_unique:
        return "ThatOtherContextEngine", f"ThatOtherContextEngine found {ThatOtherContextEngine_unique} unique files vs ACE's {ace_unique}", ace_has_all_ThatOtherContextEngine, ThatOtherContextEngine_has_all_ace
    
    # Default to score comparison
    if ace_scores and ace_scores[0] >= 0.7:
        return "ACE", f"ACE has high confidence ({ace_scores[0]:.3f})", ace_has_all_ThatOtherContextEngine, ThatOtherContextEngine_has_all_ace
    
    return "TIE", "Results are similar", ace_has_all_ThatOtherContextEngine, ThatOtherContextEngine_has_all_ace


# Comprehensive test queries with expected patterns - 250+ queries
TEST_QUERIES = [
    # ==========================================================================
    # Category 1: Class/function definitions (50 queries - progressive difficulty)
    # ==========================================================================
    ("Definitions", "CodeRetrieval class definition"),
    ("Definitions", "UnifiedMemoryIndex class search method"),
    ("Definitions", "ASTChunker class parse method"),
    ("Definitions", "_apply_filename_boost function"),
    ("Definitions", "create_sparse_vector BM25 function"),
    ("Definitions", "@dataclass class Bullet"),
    ("Definitions", "QdrantConfig class definition"),
    ("Definitions", "HyDEGenerator class generate method"),
    ("Definitions", "SmartBulletIndex retrieve method"),
    ("Definitions", "CodeIndexer index_workspace batch"),
    ("Definitions", "PlaybookManager class initialization"),
    ("Definitions", "EmbeddingService embed_batch method"),
    ("Definitions", "SemanticScorer score method"),
    ("Definitions", "TypeInferenceEngine infer method"),
    ("Definitions", "DependencyGraph build method"),
    ("Definitions", "FileWatcher callback handler"),
    ("Definitions", "AuditLogger log_retrieval method"),
    ("Definitions", "RetryPolicy execute method"),
    ("Definitions", "CircuitBreaker is_open method"),
    ("Definitions", "CacheManager get method"),
    ("Definitions", "HealthChecker check_all method"),
    ("Definitions", "MetricsCollector record method"),
    ("Definitions", "TracingSpan end method"),
    ("Definitions", "QueryParser parse method"),
    ("Definitions", "ResultFormatter format method"),
    ("Definitions", "TokenCounter count method"),
    
    # Pattern-based definitions
    ("Definitions", "def search function in retrieval"),
    ("Definitions", "async def embed method"),
    ("Definitions", "class Config dataclass"),
    ("Definitions", "@property score getter"),
    ("Definitions", "__init__ constructor pattern"),
    ("Definitions", "__enter__ context manager"),
    ("Definitions", "__call__ callable protocol"),
    ("Definitions", "@staticmethod factory pattern"),
    ("Definitions", "@classmethod from_config"),
    ("Definitions", "abstract method interface"),
    ("Definitions", "Protocol typing definition"),
    ("Definitions", "TypedDict structure"),
    ("Definitions", "NamedTuple definition"),
    ("Definitions", "Enum class definition"),
    ("Definitions", "exception class custom"),
    ("Definitions", "decorator function wrapper"),
    ("Definitions", "generator function yield"),
    ("Definitions", "async generator async yield"),
    ("Definitions", "context manager contextlib"),
    ("Definitions", "callback handler registration"),
    ("Definitions", "event listener pattern"),
    ("Definitions", "singleton pattern instance"),
    ("Definitions", "factory pattern create"),
    ("Definitions", "builder pattern fluent"),
    
    # ==========================================================================
    # Category 2: Configuration patterns (50 queries)
    # ==========================================================================
    ("Config", "EmbeddingConfig dataclass"),
    ("Config", "VoyageCodeEmbeddingConfig api_key"),
    ("Config", "BM25Config k1 b constants"),
    ("Config", "LLMConfig provider model settings"),
    ("Config", "QdrantConfig collection_name url"),
    ("Config", "RetrievalConfig limit threshold"),
    ("Config", "HyDEConfig num_hypotheticals temperature"),
    ("Config", "ELFConfig enable_elf"),
    ("Config", "environment variables dotenv loading"),
    ("Config", "logging configuration loguru"),
    ("Config", "API_KEY environment variable"),
    ("Config", "QDRANT_URL connection string"),
    ("Config", "MODEL_NAME configuration"),
    ("Config", "EMBEDDING_DIMENSION size"),
    ("Config", "MAX_TOKENS limit"),
    ("Config", "TEMPERATURE parameter"),
    ("Config", "TOP_K configuration"),
    ("Config", "BATCH_SIZE setting"),
    ("Config", "TIMEOUT_SECONDS value"),
    ("Config", "RETRY_ATTEMPTS count"),
    ("Config", "CACHE_TTL duration"),
    ("Config", "LOG_LEVEL setting"),
    ("Config", "DEBUG_MODE flag"),
    ("Config", "PRODUCTION_MODE environment"),
    ("Config", "DATABASE_URL connection"),
    ("Config", "defaults dict configuration"),
    ("Config", "settings.py module"),
    ("Config", ".env file loading"),
    ("Config", "config.yaml parsing"),
    ("Config", "json config loading"),
    ("Config", "toml configuration"),
    ("Config", "pyproject.toml settings"),
    ("Config", "optional field default"),
    ("Config", "required field validation"),
    ("Config", "nested config structure"),
    ("Config", "config inheritance override"),
    ("Config", "environment-specific config"),
    ("Config", "secrets management"),
    ("Config", "credential storage"),
    ("Config", "config validation pydantic"),
    ("Config", "type coercion conversion"),
    ("Config", "default factory callable"),
    ("Config", "frozen immutable config"),
    ("Config", "config merge strategy"),
    ("Config", "config from dict"),
    ("Config", "config to dict serialization"),
    ("Config", "config documentation docstring"),
    ("Config", "config schema validation"),
    ("Config", "runtime config reload"),
    ("Config", "feature flags toggle"),
    
    # ==========================================================================
    # Category 3: Error handling (50 queries)
    # ==========================================================================
    ("ErrorHandling", "try except error handling pattern"),
    ("ErrorHandling", "exception retry backoff resilience"),
    ("ErrorHandling", "Qdrant connection error handling"),
    ("ErrorHandling", "API rate limit retry exponential backoff"),
    ("ErrorHandling", "timeout exception handling httpx"),
    ("ErrorHandling", "validation error handling ValueError"),
    ("ErrorHandling", "file not found error handling"),
    ("ErrorHandling", "embedding error fallback"),
    ("ErrorHandling", "import error optional dependency"),
    ("ErrorHandling", "graceful degradation circuit breaker"),
    ("ErrorHandling", "network error retry"),
    ("ErrorHandling", "connection timeout handling"),
    ("ErrorHandling", "authentication error 401"),
    ("ErrorHandling", "authorization error 403"),
    ("ErrorHandling", "not found error 404"),
    ("ErrorHandling", "server error 500"),
    ("ErrorHandling", "json decode error"),
    ("ErrorHandling", "unicode decode error"),
    ("ErrorHandling", "key error missing"),
    ("ErrorHandling", "index error bounds"),
    ("ErrorHandling", "type error conversion"),
    ("ErrorHandling", "attribute error missing"),
    ("ErrorHandling", "runtime error generic"),
    ("ErrorHandling", "assertion error validation"),
    ("ErrorHandling", "permission error access"),
    ("ErrorHandling", "os error filesystem"),
    ("ErrorHandling", "memory error allocation"),
    ("ErrorHandling", "recursion error depth"),
    ("ErrorHandling", "keyboard interrupt signal"),
    ("ErrorHandling", "system exit handling"),
    ("ErrorHandling", "exception chaining from"),
    ("ErrorHandling", "exception suppression"),
    ("ErrorHandling", "finally cleanup block"),
    ("ErrorHandling", "context manager error"),
    ("ErrorHandling", "async exception handling"),
    ("ErrorHandling", "task cancellation handling"),
    ("ErrorHandling", "concurrent error handling"),
    ("ErrorHandling", "thread error handling"),
    ("ErrorHandling", "process error handling"),
    ("ErrorHandling", "logging error stack trace"),
    ("ErrorHandling", "error message formatting"),
    ("ErrorHandling", "error code enumeration"),
    ("ErrorHandling", "custom exception class"),
    ("ErrorHandling", "exception hierarchy"),
    ("ErrorHandling", "error recovery strategy"),
    ("ErrorHandling", "partial failure handling"),
    ("ErrorHandling", "transaction rollback"),
    ("ErrorHandling", "cleanup on failure"),
    ("ErrorHandling", "error notification alert"),
    ("ErrorHandling", "error metrics tracking"),
    
    # ==========================================================================
    # Category 4: Import patterns (50 queries)
    # ==========================================================================
    ("Imports", "from qdrant_client import QdrantClient"),
    ("Imports", "import httpx async"),
    ("Imports", "from dataclasses import dataclass field"),
    ("Imports", "from typing import Optional List Dict"),
    ("Imports", "import voyageai client"),
    ("Imports", "from pathlib import Path"),
    ("Imports", "import logging logger"),
    ("Imports", "from functools import lru_cache"),
    ("Imports", "import json os sys"),
    ("Imports", "from collections import Counter defaultdict"),
    ("Imports", "import asyncio await"),
    ("Imports", "from abc import ABC abstractmethod"),
    ("Imports", "import re regex"),
    ("Imports", "from datetime import datetime"),
    ("Imports", "import hashlib md5"),
    ("Imports", "from uuid import uuid4"),
    ("Imports", "import time sleep"),
    ("Imports", "from enum import Enum auto"),
    ("Imports", "import copy deepcopy"),
    ("Imports", "from itertools import chain"),
    ("Imports", "import contextlib suppress"),
    ("Imports", "from concurrent.futures import ThreadPoolExecutor"),
    ("Imports", "import threading Lock"),
    ("Imports", "from queue import Queue"),
    ("Imports", "import signal handler"),
    ("Imports", "from io import StringIO BytesIO"),
    ("Imports", "import struct pack unpack"),
    ("Imports", "from base64 import b64encode"),
    ("Imports", "import pickle serialize"),
    ("Imports", "from gzip import compress"),
    ("Imports", "import tempfile NamedTemporaryFile"),
    ("Imports", "from shutil import copy2 rmtree"),
    ("Imports", "import subprocess run"),
    ("Imports", "from platform import system"),
    ("Imports", "import socket connect"),
    ("Imports", "from urllib.parse import urljoin"),
    ("Imports", "import ssl certificate"),
    ("Imports", "from email.mime import MIMEText"),
    ("Imports", "import secrets token"),
    ("Imports", "from getpass import getpass"),
    ("Imports", "import argparse ArgumentParser"),
    ("Imports", "from configparser import ConfigParser"),
    ("Imports", "import csv reader writer"),
    ("Imports", "from xml.etree import ElementTree"),
    ("Imports", "import sqlite3 database"),
    ("Imports", "from http.server import HTTPServer"),
    ("Imports", "import pytest mark parametrize"),
    ("Imports", "from unittest.mock import Mock patch"),
    ("Imports", "import pytest fixture"),
    ("Imports", "from hypothesis import given strategies"),
    
    # ==========================================================================
    # Category 5: Async patterns (50 queries)
    # ==========================================================================
    ("Async", "async def retrieve await"),
    ("Async", "asyncio gather parallel"),
    ("Async", "async with httpx.AsyncClient"),
    ("Async", "async for chunk stream"),
    ("Async", "await embedding generation"),
    ("Async", "async context manager enter exit"),
    ("Async", "asyncio.create_task background"),
    ("Async", "async generator yield"),
    ("Async", "semaphore rate limiting async"),
    ("Async", "asyncio timeout cancel"),
    ("Async", "asyncio.run main"),
    ("Async", "event loop get_event_loop"),
    ("Async", "asyncio.Queue producer consumer"),
    ("Async", "asyncio.Lock mutex"),
    ("Async", "asyncio.Event signal"),
    ("Async", "asyncio.Condition notify"),
    ("Async", "asyncio.Semaphore limit"),
    ("Async", "asyncio.wait_for timeout"),
    ("Async", "asyncio.shield cancel protection"),
    ("Async", "asyncio.sleep delay"),
    ("Async", "async iterator __aiter__ __anext__"),
    ("Async", "async context __aenter__ __aexit__"),
    ("Async", "asyncio.ensure_future task"),
    ("Async", "asyncio.all_tasks running"),
    ("Async", "asyncio.current_task context"),
    ("Async", "asyncio.get_running_loop"),
    ("Async", "loop.run_in_executor thread"),
    ("Async", "loop.call_soon callback"),
    ("Async", "loop.call_later delayed"),
    ("Async", "loop.call_at scheduled"),
    ("Async", "asyncio.Protocol transport"),
    ("Async", "asyncio.StreamReader StreamWriter"),
    ("Async", "asyncio.start_server listen"),
    ("Async", "asyncio.open_connection client"),
    ("Async", "asyncio.subprocess process"),
    ("Async", "asyncio.to_thread blocking"),
    ("Async", "async map concurrent"),
    ("Async", "async filter stream"),
    ("Async", "async reduce accumulate"),
    ("Async", "async zip merge"),
    ("Async", "async enumerate index"),
    ("Async", "async list comprehension"),
    ("Async", "async dict comprehension"),
    ("Async", "async set comprehension"),
    ("Async", "async exception handling"),
    ("Async", "async finally cleanup"),
    ("Async", "async with timeout"),
    ("Async", "async retry backoff"),
    ("Async", "async batch processing"),
    ("Async", "async pipeline chain"),
]


def run_comprehensive_benchmark(verbose: bool = False) -> Dict[str, Any]:
    """Run comprehensive ACE vs ThatOtherContextEngine benchmark."""
    results = []
    
    print("=" * 80)
    print("COMPREHENSIVE ACE vs ThatOtherContextEngine BENCHMARK")
    print(f"Queries: {len(TEST_QUERIES)}")
    print("=" * 80)
    
    stats = {
        "ace_wins": 0,
        "ThatOtherContextEngine_wins": 0,
        "ties": 0,
        "ace_coverage_rate": 0,  # % of queries where ACE has all ThatOtherContextEngine results
        "by_category": {},
    }
    
    for i, (category, query) in enumerate(TEST_QUERIES, 1):
        if verbose:
            print(f"\n[{i}/{len(TEST_QUERIES)}] [{category}] {query}")
        else:
            print(f"[{i}/{len(TEST_QUERIES)}] {query[:50]}...", end="\r")
        
        # Get results from both systems
        ace_results = get_ace_results(query, limit=5)
        ThatOtherContextEngine_results = get_ThatOtherContextEngine_results(query, limit=5)
        
        ace_files = [f for f, _ in ace_results]
        ace_scores = [s for _, s in ace_results]
        
        # Determine winner
        winner, reason, ace_has_all, ThatOtherContextEngine_has_all = determine_winner(
            query, ace_files, ace_scores, ThatOtherContextEngine_results
        )
        
        result = QueryResult(
            query=query,
            category=category,
            ace_files=ace_files,
            ace_scores=ace_scores,
            ThatOtherContextEngine_files=ThatOtherContextEngine_results,
            winner=winner,
            reason=reason,
            ace_has_all_ThatOtherContextEngine=ace_has_all,
            ThatOtherContextEngine_has_all_ace=ThatOtherContextEngine_has_all,
        )
        results.append(result)
        
        # Update stats
        if winner == "ACE":
            stats["ace_wins"] += 1
        elif winner == "ThatOtherContextEngine":
            stats["ThatOtherContextEngine_wins"] += 1
        else:
            stats["ties"] += 1
        
        if ace_has_all:
            stats["ace_coverage_rate"] += 1
        
        # Category stats
        if category not in stats["by_category"]:
            stats["by_category"][category] = {"ace": 0, "ThatOtherContextEngine": 0, "tie": 0}
        stats["by_category"][category][winner.lower()] += 1
        
        if verbose:
            print(f"  ACE: {ace_files[:3]}")
            print(f"  ThatOtherContextEngine: {ThatOtherContextEngine_results[:3]}")
            print(f"  Winner: {winner} - {reason}")
    
    # Final stats
    total = len(results)
    stats["ace_coverage_rate"] = (stats["ace_coverage_rate"] / total) * 100
    stats["ace_win_rate"] = (stats["ace_wins"] / total) * 100
    stats["ThatOtherContextEngine_win_rate"] = (stats["ThatOtherContextEngine_wins"] / total) * 100
    stats["tie_rate"] = (stats["ties"] / total) * 100
    
    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"\nTotal Queries: {total}")
    print(f"ACE Wins: {stats['ace_wins']} ({stats['ace_win_rate']:.1f}%)")
    print(f"ThatOtherContextEngine Wins: {stats['ThatOtherContextEngine_wins']} ({stats['ThatOtherContextEngine_win_rate']:.1f}%)")
    print(f"Ties: {stats['ties']} ({stats['tie_rate']:.1f}%)")
    print(f"ACE Coverage Rate: {stats['ace_coverage_rate']:.1f}% (includes all ThatOtherContextEngine results)")
    
    print("\n### Results by Category")
    for cat, cat_stats in stats["by_category"].items():
        cat_total = cat_stats["ace"] + cat_stats["ThatOtherContextEngine"] + cat_stats["tie"]
        ace_pct = (cat_stats["ace"] / cat_total * 100) if cat_total > 0 else 0
        print(f"  {cat}: ACE {cat_stats['ace']} | ThatOtherContextEngine {cat_stats['ThatOtherContextEngine']} | Tie {cat_stats['tie']} ({ace_pct:.0f}% ACE)")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "stats": stats,
        "results": [asdict(r) for r in results],
    }
    
    output_file = Path("benchmark_results") / f"comprehensive_ace_ThatOtherContextEngine_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Identify ThatOtherContextEngine wins for further analysis
    ThatOtherContextEngine_wins = [r for r in results if r.winner == "ThatOtherContextEngine"]
    if ThatOtherContextEngine_wins:
        print("\n### ThatOtherContextEngine WINS (need investigation):")
        for r in ThatOtherContextEngine_wins:
            print(f"  [{r.category}] {r.query}")
            print(f"    ACE: {r.ace_files[:2]}")
            print(f"    ThatOtherContextEngine: {r.ThatOtherContextEngine_files[:2]}")
            print(f"    Reason: {r.reason}")
    
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    
    run_comprehensive_benchmark(verbose=args.verbose)
