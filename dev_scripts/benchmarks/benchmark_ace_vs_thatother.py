"""Expanded ACE vs ThatOtherContextEngine benchmark comparison.

Tests diverse query types across 5 categories:
1. Code queries (functions, classes, patterns)
2. Doc queries (guides, references, tutorials)
3. Architecture queries (design, components, patterns)
4. Config queries (settings, environment, credentials)
5. Edge cases (specific symbols, error messages, imports)

Runs ACE CodeRetrieval and optionally compares against ThatOtherContextEngine MCP.
"""

import subprocess
import sys
import os
import argparse
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ace.code_retrieval import CodeRetrieval


@dataclass
class BenchmarkResult:
    """Result for a single benchmark query."""
    category: str
    label: str
    query: str
    ace_results: List[Tuple[str, float]]
    ThatOtherContextEngine_results: List[str]
    ace_top_match: bool  # Does ACE top file match ThatOtherContextEngine top?
    doc_coverage: int  # How many results are docs (for doc queries)


def run_ThatOtherContextEngine_mcp(query: str, limit: int = 5, verbose: bool = False) -> List[str]:
    """Run ThatOtherContextEngine MCP query via ThatOtherContextEngine CLI print mode and return file paths.

    Uses `ThatOtherContextEngine -p "<query>" --max-turns 1` to trigger codebase retrieval.
    ThatOtherContextEngine outputs code context with "Path: <file>" prefixes.

    Args:
        query: Search query for codebase retrieval
        limit: Maximum number of file paths to return
        verbose: Print debug info about ThatOtherContextEngine call

    Returns:
        List of file paths from ThatOtherContextEngine's codebase-retrieval results
    """
    try:
        # Use ThatOtherContextEngine print mode without quiet - we need the full output including tool calls
        # ThatOtherContextEngine's codebase-retrieval outputs "Path: <filepath>" format
        # On Windows, use shell=True and proper encoding
        import platform
        if platform.system() == "Windows":
            cmd = f'ThatOtherContextEngine -p "{query}" --max-turns 1'
            use_shell = True
            encoding = 'utf-8'
        else:
            cmd = [
                "ThatOtherContextEngine", "-p",
                query,
                "--max-turns", "1"
            ]
            use_shell = False
            encoding = 'utf-8'

        if verbose:
            print(f"  [ThatOtherContextEngine] Running: {cmd if isinstance(cmd, str) else ' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding=encoding,
            errors='replace',  # Replace invalid chars instead of failing
            timeout=60,
            shell=use_shell,
            cwd=os.getcwd()
        )

        # Parse paths from ThatOtherContextEngine output (format: "Path: <filepath>")
        paths = []
        output = (result.stdout or "") + (result.stderr or "")

        if verbose and not output:
            print(f"  [ThatOtherContextEngine] Empty output")

        for line in output.split("\n"):
            line = line.strip()
            # ThatOtherContextEngine output format: "Path: ace\code_retrieval.py" or "Path: ace/code_retrieval.py"
            if line.startswith("Path: "):
                path = line[6:].strip()
                # Normalize path separators (Windows uses backslash)
                path = path.replace("\\", "/")
                # Remove any ANSI color codes or other decorations
                path = path.split("\x1b")[0] if "\x1b" in path else path
                if path and path not in paths:
                    paths.append(path)

        if verbose and paths:
            print(f"  [ThatOtherContextEngine] Found {len(paths)} paths: {paths[:3]}...")
        elif verbose:
            print(f"  [ThatOtherContextEngine] No paths found in output ({len(output)} chars)")

        return paths[:limit]
    except subprocess.TimeoutExpired:
        if verbose:
            print(f"  [ThatOtherContextEngine] Timeout after 60s")
        return []
    except Exception as e:
        if verbose:
            print(f"  [ThatOtherContextEngine] Error: {e}")
        return []


def run_ace(query: str, limit: int = 5) -> List[Tuple[str, float]]:
    """Run ACE query and return (path, score) tuples."""
    retriever = CodeRetrieval()
    results = retriever.search(query, limit=limit)
    return [(r["file_path"], r["score"]) for r in results]


def normalize_path(path: str) -> str:
    """Normalize path for comparison."""
    path = path.replace("\\", "/").lower()
    # Remove leading ./ or absolute paths
    if "/" in path:
        # Keep only the relative part
        parts = path.split("/")
        # Find first significant directory (ace, docs, tests, etc.)
        for i, part in enumerate(parts):
            if part in ("ace", "docs", "tests", "examples", "scripts"):
                return "/".join(parts[i:])
    return path


def paths_match(path1: str, path2: str) -> bool:
    """Check if two paths refer to the same file."""
    norm1 = normalize_path(path1)
    norm2 = normalize_path(path2)
    
    if norm1 == norm2:
        return True
    if norm1.endswith(norm2) or norm2.endswith(norm1):
        return True
    return False


# Comprehensive query set - 50 queries across 5 categories
BENCHMARK_QUERIES = [
    # ============================================================
    # Category 1: Code queries (core implementation) - 10 queries
    # ============================================================
    ("Code", "retrieval class", "CodeRetrieval class search method implementation"),
    ("Code", "error handling", "exception error handling retry resilience pattern"),
    ("Code", "embedding", "embedding vector generation voyage-code-3 model"),
    ("Code", "chunking", "ASTChunker code chunking tree-sitter parsing"),
    ("Code", "deduplication", "deduplicate overlapping results similarity"),
    ("Code", "reranker", "cross encoder reranker BGE ranking"),
    ("Code", "indexer", "CodeIndexer index_workspace batch embedding"),
    ("Code", "dense search", "dense vector search cosine similarity"),
    ("Code", "sparse BM25", "sparse vector BM25 term frequency"),
    ("Code", "file watcher", "file watcher daemon monitor changes"),
    
    # ============================================================
    # Category 2: Doc queries (documentation files) - 10 queries
    # ============================================================
    ("Docs", "quick start", "quick start guide installation setup tutorial"),
    ("Docs", "API reference", "API reference documentation endpoints methods"),
    ("Docs", "MCP integration", "MCP server integration copilot claude setup"),
    ("Docs", "architecture overview", "architecture design system overview diagram"),
    ("Docs", "configuration guide", "configuration environment variables settings"),
    ("Docs", "prompts guide", "prompts generator reflector curator templates"),
    ("Docs", "retrieval precision", "retrieval precision optimization tuning"),
    ("Docs", "integration guide", "integration guide framework agent workflow"),
    ("Docs", "golden rules", "golden rules best practices guidelines"),
    ("Docs", "changelog", "changelog version history release notes"),
    
    # ============================================================
    # Category 3: Architecture queries (design patterns) - 10 queries
    # ============================================================
    ("Arch", "ACE components", "Generator Reflector Curator architecture roles"),
    ("Arch", "memory system", "unified memory retrieval storage qdrant vector"),
    ("Arch", "playbook", "playbook bullet strategy learning adaptation"),
    ("Arch", "multitenancy", "tenant isolation namespace collection separation"),
    ("Arch", "caching", "cache retrieval performance optimization"),
    ("Arch", "HyDE", "hypothetical document embeddings query expansion"),
    ("Arch", "observability", "observability tracing metrics opik monitoring"),
    ("Arch", "enrichment", "bullet enrichment LLM context enhancement"),
    ("Arch", "delta operations", "delta operation batch update playbook"),
    ("Arch", "intent classifier", "intent classification query routing semantic"),
    
    # ============================================================
    # Category 4: Config queries (settings and configuration) - 10 queries
    # ============================================================
    ("Config", "embedding config", "embedding model configuration voyage openai"),
    ("Config", "qdrant settings", "qdrant collection vector database settings"),
    ("Config", "LLM config", "LLM provider configuration API key model"),
    ("Config", "BM25 constants", "BM25 k1 b constants sparse vector"),
    ("Config", "environment vars", "environment variable configuration dotenv"),
    ("Config", "rate limiting", "rate limit API throttling requests"),
    ("Config", "vector dimensions", "vector dimension size embedding model"),
    ("Config", "batch size", "batch size chunk embedding processing"),
    ("Config", "collection name", "collection name workspace code context"),
    ("Config", "reranker model", "reranker model BGE cross encoder"),
    
    # ============================================================
    # Category 5: Edge cases (challenging patterns) - 10 queries
    # ============================================================
    ("Edge", "async patterns", "async await asyncio concurrent retrieval"),
    ("Edge", "type hints", "typing Optional List Dict dataclass"),
    ("Edge", "imports", "from qdrant_client import QdrantClient"),
    ("Edge", "specific class", "UnifiedMemoryIndex namespace hybrid"),
    ("Edge", "error message", "Qdrant client not available error"),
    ("Edge", "dataclass", "dataclass field default_factory frozen"),
    ("Edge", "decorator", "decorator functools wraps retry backoff"),
    ("Edge", "context manager", "context manager with statement async"),
    ("Edge", "lambda filter", "lambda filter map comprehension"),
    ("Edge", "regex pattern", "regex pattern re compile match search"),
]


def run_benchmark(include_ThatOtherContextEngine: bool = False, verbose: bool = False) -> List[BenchmarkResult]:
    """Run full benchmark suite."""
    results = []
    
    print("=" * 80)
    print("ACE vs ThatOtherContextEngine EXPANDED BENCHMARK")
    print(f"Queries: {len(BENCHMARK_QUERIES)} | Categories: 5 (10 each)")
    print(f"Mode: {'ACE + ThatOtherContextEngine comparison' if include_ThatOtherContextEngine else 'ACE only'}")
    print("=" * 80)
    
    category_stats: Dict[str, Dict[str, int]] = {}
    
    for category, label, query in BENCHMARK_QUERIES:
        if verbose:
            print(f"\n### [{category}] {label}")
            print(f"Query: \"{query}\"")
            print("-" * 60)
        
        # Run ACE
        ace_results = run_ace(query, limit=5)
        
        # Run ThatOtherContextEngine if enabled
        ThatOtherContextEngine_results = run_ThatOtherContextEngine_mcp(query, limit=5) if include_ThatOtherContextEngine else []
        
        # Calculate metrics
        ace_top_match = False
        if ThatOtherContextEngine_results and ace_results:
            ace_top = ace_results[0][0] if ace_results else ""
            ThatOtherContextEngine_top = ThatOtherContextEngine_results[0] if ThatOtherContextEngine_results else ""
            ace_top_match = paths_match(ace_top, ThatOtherContextEngine_top)
        
        # Doc coverage (for doc queries)
        doc_coverage = 0
        if category == "Docs":
            doc_coverage = sum(1 for p, _ in ace_results if ".md" in p or "docs/" in p)
        
        result = BenchmarkResult(
            category=category,
            label=label,
            query=query,
            ace_results=ace_results,
            ThatOtherContextEngine_results=ThatOtherContextEngine_results,
            ace_top_match=ace_top_match,
            doc_coverage=doc_coverage,
        )
        results.append(result)
        
        # Track category stats
        if category not in category_stats:
            category_stats[category] = {"total": 0, "match": 0, "doc_coverage": 0}
        category_stats[category]["total"] += 1
        if ace_top_match:
            category_stats[category]["match"] += 1
        category_stats[category]["doc_coverage"] += doc_coverage
        
        if verbose:
            print("ACE Results:")
            for i, (path, score) in enumerate(ace_results, 1):
                marker = "*" if include_ThatOtherContextEngine and ThatOtherContextEngine_results and paths_match(path, ThatOtherContextEngine_results[0]) else " "
                print(f"  {marker}{i}. [{score:.3f}] {path}")
            
            if include_ThatOtherContextEngine and ThatOtherContextEngine_results:
                print("ThatOtherContextEngine Results:")
                for i, path in enumerate(ThatOtherContextEngine_results, 1):
                    print(f"   {i}. {path}")
            
            if category == "Docs":
                print(f"  -> Doc coverage: {doc_coverage}/5")
    
    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    print("\n### Results by Category")
    print("-" * 60)
    total_queries = len(results)
    total_matches = sum(1 for r in results if r.ace_top_match)
    total_doc_coverage = sum(r.doc_coverage for r in results if r.category == "Docs")
    
    for cat, stats in category_stats.items():
        if cat == "Docs":
            doc_pct = (stats["doc_coverage"] / (stats["total"] * 5)) * 100 if stats["total"] > 0 else 0
            print(f"  {cat}: {stats['total']} queries | Doc coverage: {doc_pct:.0f}%")
        else:
            match_pct = (stats["match"] / stats["total"]) * 100 if include_ThatOtherContextEngine and stats["total"] > 0 else 0
            if include_ThatOtherContextEngine:
                print(f"  {cat}: {stats['total']} queries | ThatOtherContextEngine match: {match_pct:.0f}%")
            else:
                print(f"  {cat}: {stats['total']} queries")
    
    print("\n### Top Results Preview")
    print("-" * 60)
    for r in results[:10]:
        top = r.ace_results[0] if r.ace_results else ("N/A", 0)
        print(f"  [{r.category[:4]}] {r.label}: {top[0]} [{top[1]:.2f}]")
    
    if len(results) > 10:
        print(f"  ... and {len(results) - 10} more")
    
    print("\n### Key Metrics")
    print("-" * 60)
    print(f"  Total queries: {total_queries}")
    if include_ThatOtherContextEngine:
        match_pct = (total_matches / total_queries) * 100 if total_queries > 0 else 0
        print(f"  ThatOtherContextEngine top-match rate: {match_pct:.1f}%")
    
    doc_queries = sum(1 for r in results if r.category == "Docs")
    doc_coverage_pct = (total_doc_coverage / (doc_queries * 5)) * 100 if doc_queries > 0 else 0
    print(f"  Doc query coverage: {doc_coverage_pct:.1f}%")
    
    # Score distribution
    all_scores = [score for r in results for _, score in r.ace_results]
    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        max_score = max(all_scores)
        min_score = min(all_scores)
        print(f"  Score range: {min_score:.3f} - {max_score:.3f} (avg: {avg_score:.3f})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="ACE vs ThatOtherContextEngine benchmark")
    parser.add_argument("--ThatOtherContextEngine", action="store_true", help="Include ThatOtherContextEngine MCP comparison")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    results = run_benchmark(include_ThatOtherContextEngine=args.ThatOtherContextEngine, verbose=args.verbose)
    
    # Return exit code based on doc coverage (should be > 50%)
    doc_results = [r for r in results if r.category == "Docs"]
    total_doc_coverage = sum(r.doc_coverage for r in doc_results)
    doc_coverage_pct = (total_doc_coverage / (len(doc_results) * 5)) * 100 if doc_results else 0
    
    if doc_coverage_pct < 50:
        print(f"\nWARNING: Doc coverage below 50% ({doc_coverage_pct:.1f}%)")
        sys.exit(1)


if __name__ == "__main__":
    main()
