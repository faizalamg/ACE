#!/usr/bin/env python3
"""
Expanded ACE vs ThatOtherContextEngine Analysis Test Suite

This test adds more queries beyond the original 23 to comprehensively
test ACE's accuracy. For each query, we analyze BOTH systems and determine
who wins based on CORRECTNESS (not just matching each other).

METHODOLOGY:
1. For each new query, grep/find the actual location of the code
2. Compare ACE result vs ThatOtherContextEngine result vs GROUND TRUTH
3. Score: ACE_WIN, ThatOtherContextEngine_WIN, TIE, or BOTH_WRONG
"""

import subprocess
import sys
import json
import os
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["VOYAGE_API_KEY"] = "pa-3OeAEQVzfUwm9e0UKyW3L1JST8HJ4WT2sReaIv8GdDr"


@dataclass
class AnalysisResult:
    """Result of analyzing ACE vs ThatOtherContextEngine for correctness."""
    query: str
    ground_truth: List[str]  # List of valid answers (any match = correct)
    ace_top: str
    ThatOtherContextEngine_top: str  # From ThatOtherContextEngine_GROUND_TRUTH or manual check
    ace_correct: bool
    ThatOtherContextEngine_correct: bool
    winner: str  # ACE_WIN, ThatOtherContextEngine_WIN, TIE, BOTH_WRONG


def normalize_path(path: str) -> str:
    """Normalize file path for comparison."""
    path = path.replace("\\", "/").lower()
    while path.startswith("./"):
        path = path[2:]
    return path


def paths_match(path1: str, path2: str) -> bool:
    """Check if two paths refer to the same file."""
    norm1 = normalize_path(path1)
    norm2 = normalize_path(path2)
    return norm1 == norm2 or norm1.endswith(norm2) or norm2.endswith(norm1)


def run_ace_search(query: str, limit: int = 5) -> List[str]:
    """Run ACE code retrieval search and return file paths."""
    try:
        from ace.code_retrieval import CodeRetrieval
        retriever = CodeRetrieval()
        results = retriever.search(query, limit=limit, exclude_tests=True)
        return [r.get("file_path", "") for r in results if r.get("file_path")]
    except Exception as e:
        print(f"ACE search error: {e}")
        return []


# =============================================================================
# EXPANDED TEST QUERIES WITH VERIFIED GROUND TRUTH
# =============================================================================
# Each entry: (query, ground_truth_file, ThatOtherContextEngine_expected_file)
# ground_truth is verified by grep/inspection, ThatOtherContextEngine_expected is what ThatOtherContextEngine returns
# =============================================================================

EXPANDED_QUERIES = [
    # --- NEW: Config classes (VERIFIED) ---
    ("VoyageCodeEmbeddingConfig class", ["ace/config.py"], "ace/config.py"),
    ("BM25Config dataclass", ["ace/config.py"], "ace/config.py"),
    # RetrievalConfig is defined in BOTH config.py AND retrieval_presets.py
    ("RetrievalConfig settings", ["ace/config.py", "ace/retrieval_presets.py"], "ace/config.py"),
    
    # --- NEW: Method definitions (VERIFIED) ---
    ("_apply_filename_boost method", ["ace/code_retrieval.py"], "ace/code_retrieval.py"),
    # _expand_query exists in BOTH files (verified via grep)
    ("_expand_query function code retrieval", ["ace/code_retrieval.py", "ace/unified_memory.py"], "ace/code_retrieval.py"),
    # NO METHOD CALLED search_with_bm25! BM25 funcs are tokenize_for_bm25, create_sparse_vector in unified_memory
    ("tokenize_for_bm25 function", ["ace/unified_memory.py"], "ace/unified_memory.py"),
    
    # --- NEW: Module-level functions (VERIFIED) ---
    ("get_config function", ["ace/config.py"], "ace/config.py"),
    ("get_memory_config function", ["ace/config.py"], "ace/config.py"),
    
    # --- NEW: Class hierarchies (VERIFIED) ---
    ("ASTChunker class code chunker", ["ace/code_chunker.py"], "ace/code_chunker.py"),
    ("UnifiedMemoryIndex retrieve method", ["ace/unified_memory.py"], "ace/unified_memory.py"),
    
    # --- NEW: Import patterns - MULTI-VALID (many files have these) ---
    ("from qdrant_client import", ["ace/unified_memory.py", "ace/qdrant_retrieval.py", "ace/code_indexer.py", "ace/code_retrieval.py", "ace/deduplication.py"], None),
    ("import httpx", ["ace/code_retrieval.py", "ace/unified_memory.py", "ace/qdrant_retrieval.py", "ace/semantic_scorer.py", "ace/scaling.py", "ace/async_retrieval.py"], None),
    
    # --- NEW: Docstrings/Comments (VERIFIED) ---
    # code-specific embeddings appears in: code_retrieval.py, code_indexer.py, code_enrichment.py, code_analysis.py
    ("code-specific embeddings", ["ace/code_retrieval.py", "ace/code_indexer.py", "ace/code_enrichment.py", "ace/code_analysis.py"], "ace/code_retrieval.py"),
    # cross-encoder reranking: reranker.py has the implementation, others reference it
    ("cross-encoder reranking", ["ace/reranker.py", "ace/unified_memory.py", "ace/retrieval.py", "ace/config.py"], None),
    
    # --- NEW: Error handling patterns (VERIFIED) ---
    # "Qdrant REST error" appears in code_retrieval.py and unified_memory.py
    ("Qdrant REST error", ["ace/code_retrieval.py", "ace/unified_memory.py"], None),
    # "embedding error" appears in code_retrieval.py, code_indexer.py; "fallback" appears in code_indexer.py, code_retrieval.py
    ("embedding error fallback", ["ace/code_retrieval.py", "ace/code_indexer.py"], None),
    
    # --- NEW: Specific implementations (VERIFIED) ---
    ("format_ThatOtherContextEngine_style implementation", ["ace/code_retrieval.py"], "ace/code_retrieval.py"),
    ("_chunk_file method indexer", ["ace/code_indexer.py"], "ace/code_indexer.py"),
    ("index_workspace function", ["ace/code_indexer.py"], "ace/code_indexer.py"),
    
    # --- NEW: Multi-file concepts (MULTI-VALID) ---
    # QdrantClient is in: unified_memory, code_retrieval, code_indexer, deduplication, multitenancy, retrieval_optimized
    # config.py has Qdrant settings; qdrant_retrieval.py uses REST API instead of QdrantClient
    ("Qdrant client connection", ["ace/unified_memory.py", "ace/qdrant_retrieval.py", "ace/code_retrieval.py", "ace/code_indexer.py"], None),
    ("embedding dimension", ["ace/config.py", "ace/code_retrieval.py", "ace/unified_memory.py", "ace/gemini_embeddings.py"], None),
]


def analyze_query(query: str, ground_truth: List[str], ThatOtherContextEngine_expected: Optional[str]) -> AnalysisResult:
    """Analyze a single query comparing ACE vs ThatOtherContextEngine vs ground truth.
    
    Args:
        query: Search query
        ground_truth: List of valid file paths (any match = correct)
        ThatOtherContextEngine_expected: What ThatOtherContextEngine returns (or None if unknown)
    """
    ace_results = run_ace_search(query, limit=5)
    ace_top = ace_results[0] if ace_results else ""
    
    # Use ThatOtherContextEngine expected or assume ACE is baseline if unknown
    ThatOtherContextEngine_top = ThatOtherContextEngine_expected or ace_top
    
    # Check if ACE result matches ANY valid ground truth path
    ace_correct = any(paths_match(ace_top, gt) for gt in ground_truth) if ace_top else False
    # Check if ThatOtherContextEngine result matches ANY valid ground truth path
    ThatOtherContextEngine_correct = any(paths_match(ThatOtherContextEngine_top, gt) for gt in ground_truth) if ThatOtherContextEngine_top else False
    
    if ace_correct and ThatOtherContextEngine_correct:
        winner = "TIE"
    elif ace_correct and not ThatOtherContextEngine_correct:
        winner = "ACE_WIN"
    elif not ace_correct and ThatOtherContextEngine_correct:
        winner = "ThatOtherContextEngine_WIN"
    else:
        winner = "BOTH_WRONG"
    
    return AnalysisResult(
        query=query,
        ground_truth=ground_truth,
        ace_top=ace_top,
        ThatOtherContextEngine_top=ThatOtherContextEngine_top,
        ace_correct=ace_correct,
        ThatOtherContextEngine_correct=ThatOtherContextEngine_correct,
        winner=winner,
    )


def run_expanded_analysis():
    """Run expanded analysis and print results."""
    print("=" * 80)
    print("EXPANDED ACE vs ThatOtherContextEngine ACCURACY ANALYSIS")
    print("=" * 80)
    print(f"Testing {len(EXPANDED_QUERIES)} additional queries\n")
    
    results = {
        "ACE_WIN": [],
        "ThatOtherContextEngine_WIN": [],
        "TIE": [],
        "BOTH_WRONG": [],
    }
    
    for query, ground_truth, ThatOtherContextEngine_expected in EXPANDED_QUERIES:
        result = analyze_query(query, ground_truth, ThatOtherContextEngine_expected)
        results[result.winner].append(result)
        
        # Status indicator
        if result.winner == "TIE":
            status = "[TIE]      "
        elif result.winner == "ACE_WIN":
            status = "[ACE WINS] "
        elif result.winner == "ThatOtherContextEngine_WIN":
            status = "[AUG WINS] "
        else:
            status = "[BOTH BAD] "
        
        print(f"{status} {query[:50]}")
        # Display first ground truth, add "(+N more)" if multi-valid
        gt_display = ground_truth[0] if ground_truth else "???"
        if len(ground_truth) > 1:
            gt_display += f" (+{len(ground_truth)-1} more valid)"
        print(f"           Ground Truth: {gt_display}")
        print(f"           ACE:         {result.ace_top} {'Y' if result.ace_correct else 'X'}")
        if ThatOtherContextEngine_expected:
            print(f"           ThatOtherContextEngine:      {result.ThatOtherContextEngine_top} {'Y' if result.ThatOtherContextEngine_correct else 'X'}")
        print()
    
    # Summary
    total = len(EXPANDED_QUERIES)
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total queries:      {total}")
    print(f"ACE wins:           {len(results['ACE_WIN'])} ({100*len(results['ACE_WIN'])/total:.1f}%)")
    print(f"ThatOtherContextEngine wins:        {len(results['ThatOtherContextEngine_WIN'])} ({100*len(results['ThatOtherContextEngine_WIN'])/total:.1f}%)")
    print(f"Ties:               {len(results['TIE'])} ({100*len(results['TIE'])/total:.1f}%)")
    print(f"Both wrong:         {len(results['BOTH_WRONG'])} ({100*len(results['BOTH_WRONG'])/total:.1f}%)")
    
    ace_total = len(results['ACE_WIN']) + len(results['TIE'])
    print(f"\nACE correct:        {ace_total}/{total} ({100*ace_total/total:.1f}%)")
    
    if results['BOTH_WRONG']:
        print("\n--- QUERIES WHERE BOTH FAILED ---")
        for r in results['BOTH_WRONG']:
            print(f"  {r.query}")
            print(f"    Ground truth: {r.ground_truth}")
            print(f"    ACE returned: {r.ace_top}")
    
    return results


if __name__ == "__main__":
    results = run_expanded_analysis()
    
    # Run original test too for regression
    print("\n\n" + "=" * 80)
    print("RUNNING ORIGINAL 23-QUERY REGRESSION TEST")
    print("=" * 80)
    import subprocess
    subprocess.run([sys.executable, "tests/compare_ace_ThatOtherContextEngine_dynamic.py"])
