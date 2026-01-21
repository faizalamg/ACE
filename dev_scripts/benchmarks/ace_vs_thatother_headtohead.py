#!/usr/bin/env python3
"""
ACE vs ThatOtherContextEngine MCP Head-to-Head Benchmark.

This script directly compares ACE CodeRetrieval against ThatOtherContextEngine MCP
using real-world queries. For each query:
1. Call ACE code retrieval
2. Parse ThatOtherContextEngine MCP results (provided manually or via MCP call)
3. Compare files retrieved, rankings, content relevancy
4. Determine winner based on code context quality

Key metrics:
- File coverage: Did both systems find the right file?
- Ranking: Which system ranked the best file higher?
- Content relevancy: Which returned more relevant code snippets?
- Chunk size: Which provided appropriate context (not too much/little)?

Usage:
    python ace_vs_ThatOtherContextEngine_headtohead.py [--query "specific query"]
    python ace_vs_ThatOtherContextEngine_headtohead.py --all  # Run all benchmark queries
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from ace.code_retrieval import CodeRetrieval


@dataclass 
class RetrievalResult:
    """Result from one retrieval system."""
    files: List[str]
    scores: List[float]
    contents: List[str]
    line_ranges: List[Tuple[int, int]]
    
    @property
    def top_file(self) -> Optional[str]:
        return self.files[0] if self.files else None
    
    @property
    def total_lines(self) -> int:
        return sum(end - start + 1 for start, end in self.line_ranges if start and end)


@dataclass
class ComparisonResult:
    """Result of comparing ACE vs ThatOtherContextEngine for one query."""
    query: str
    category: str
    ace: RetrievalResult
    ThatOtherContextEngine: RetrievalResult
    winner: str  # "ACE", "ThatOtherContextEngine", "TIE", "BOTH_MISS"
    reason: str
    ace_advantages: List[str] = field(default_factory=list)
    ThatOtherContextEngine_advantages: List[str] = field(default_factory=list)
    expected_file: Optional[str] = None  # Ground truth if known


# ============================================================
# BENCHMARK QUERIES - Real queries from chat histories
# Each has: (category, query, expected_file_pattern or None)
# ============================================================
BENCHMARK_QUERIES = [
    # === Core Code Retrieval ===
    ("Code", "CodeRetrieval class search method implementation", "code_retrieval.py"),
    ("Code", "UnifiedMemoryIndex class Qdrant namespace hybrid search", "unified_memory.py"),
    ("Code", "ASTChunker class parse Python code AST", "code_chunker.py"),
    ("Code", "VoyageCodeEmbeddingConfig class configuration", "config.py"),
    ("Code", "embedding vector generation voyage-code-3 model", "code_retrieval.py"),
    ("Code", "QdrantConfig class definition ace config", "config.py"),
    ("Code", "BM25 constants BM25_K1 BM25_B sparse vector", None),
    ("Code", "create_sparse_vector function BM25 term hash", None),
    
    # === Method/Function Queries ===
    ("Code", "CodeRetrieval class _expand_query method", "code_retrieval.py"),
    ("Code", "CodeRetrieval class _apply_filename_boost method", "code_retrieval.py"),
    ("Code", "_deduplicate_results method overlapping chunks", "code_retrieval.py"),
    ("Code", "search method dense vector query embedding", "code_retrieval.py"),
    
    # === Dataclass/Config Queries ===
    ("Config", "EmbeddingConfig dataclass model url dimension", "config.py"),
    ("Config", "CodeChunk dataclass file_path start_line end_line", "code_chunker.py"),
    ("Config", "Bullet dataclass section content helpful harmful", None),
    ("Config", "LLMConfig class ace config llm provider", "config.py"),
    
    # === Documentation Queries ===
    ("Docs", "ACE quick start guide installation setup", None),
    ("Docs", "MCP server integration copilot claude setup", None),
    ("Docs", "retrieval precision optimization tuning guide", None),
    
    # === Architecture Queries ===
    ("Arch", "Generator Reflector Curator architecture roles", None),
    ("Arch", "unified memory retrieval storage qdrant vector", "unified_memory.py"),
    ("Arch", "playbook bullet strategy learning adaptation", "playbook.py"),
    ("Arch", "HyDE hypothetical document embeddings query expansion", "hyde.py"),
    
    # === Error Handling / Edge Cases ===
    ("Edge", "try except error handling embedding failure retry", None),
    ("Edge", "async await asyncio concurrent retrieval", "async_retrieval.py"),
    ("Edge", "from qdrant_client import QdrantClient models", None),
    ("Edge", "Qdrant client not available error connection", None),
    ("Edge", "decorator functools wraps retry backoff", None),
    
    # === Specific Symbol Queries ===
    ("Symbol", "_compute_bm25_sparse function implementation", None),
    ("Symbol", "format_ThatOtherContextEngine_style method code results", "code_retrieval.py"),
    ("Symbol", "expand_context method surrounding lines", "code_retrieval.py"),
    ("Symbol", "retrieve_blended method code memory results", "code_retrieval.py"),
]


def normalize_path(path: str) -> str:
    """Normalize path for comparison."""
    if not path:
        return ""
    # Convert to lowercase, replace backslashes
    path = path.replace("\\", "/").lower()
    # Extract just the filename for comparison
    return os.path.basename(path)


def paths_match(path1: str, path2: str) -> bool:
    """Check if two paths refer to the same file."""
    if not path1 or not path2:
        return False
    n1 = normalize_path(path1)
    n2 = normalize_path(path2)
    if n1 == n2:
        return True
    # Also check if one contains the other (for partial paths)
    if n1 in path2.lower() or n2 in path1.lower():
        return True
    return False


def file_matches_pattern(file_path: str, pattern: str) -> bool:
    """Check if file path matches expected pattern."""
    if not pattern:
        return True  # No expectation = any result ok
    normalized = normalize_path(file_path)
    return pattern.lower() in normalized


def parse_ThatOtherContextEngine_output(ThatOtherContextEngine_text: str) -> RetrievalResult:
    """Parse ThatOtherContextEngine MCP output format to extract files and content.
    
    ThatOtherContextEngine format:
        The following code sections were retrieved:
        Path: file/path.py
             1  line content
             2  line content
        ...
    """
    files = []
    scores = []
    contents = []
    line_ranges = []
    
    current_file = None
    current_content = []
    current_start = None
    current_end = None
    
    for line in ThatOtherContextEngine_text.split("\n"):
        if line.startswith("Path: "):
            # Save previous file
            if current_file:
                files.append(current_file)
                scores.append(1.0 / (len(files)))  # ThatOtherContextEngine doesn't give scores, use rank
                contents.append("\n".join(current_content))
                line_ranges.append((current_start or 1, current_end or 1))
            
            current_file = line[6:].strip()
            current_content = []
            current_start = None
            current_end = None
        elif current_file and line.strip():
            # Parse line number from format "   123	content"
            match = re.match(r'\s*(\d+)\t(.*)$', line)
            if match:
                line_num = int(match.group(1))
                content = match.group(2)
                current_content.append(content)
                if current_start is None:
                    current_start = line_num
                current_end = line_num
            elif line.strip() != "...":
                current_content.append(line)
    
    # Save last file
    if current_file:
        files.append(current_file)
        scores.append(1.0 / (len(files) + 1))
        contents.append("\n".join(current_content))
        line_ranges.append((current_start or 1, current_end or 1))
    
    return RetrievalResult(
        files=files,
        scores=scores,
        contents=contents,
        line_ranges=line_ranges
    )


def get_ace_results(query: str, limit: int = 5) -> RetrievalResult:
    """Get results from ACE CodeRetrieval."""
    retriever = CodeRetrieval()
    results = retriever.search(query, limit=limit)
    
    return RetrievalResult(
        files=[r["file_path"] for r in results],
        scores=[r["score"] for r in results],
        contents=[r["content"] for r in results],
        line_ranges=[(r.get("start_line", 1), r.get("end_line", 1)) for r in results]
    )


def compare_results(
    query: str,
    category: str,
    ace: RetrievalResult,
    ThatOtherContextEngine: RetrievalResult,
    expected_file: Optional[str] = None,
) -> ComparisonResult:
    """Compare ACE vs ThatOtherContextEngine results and determine winner."""
    ace_advantages = []
    ThatOtherContextEngine_advantages = []
    
    # === Metric 1: Did top result match expected file? ===
    ace_found_expected = False
    ThatOtherContextEngine_found_expected = False
    
    if expected_file:
        ace_found_expected = any(
            file_matches_pattern(f, expected_file) for f in ace.files[:3]
        )
        ThatOtherContextEngine_found_expected = any(
            file_matches_pattern(f, expected_file) for f in ThatOtherContextEngine.files[:3]
        )
        
        if ace_found_expected and not ThatOtherContextEngine_found_expected:
            ace_advantages.append(f"Found expected file {expected_file} (ThatOtherContextEngine missed)")
        elif ThatOtherContextEngine_found_expected and not ace_found_expected:
            ThatOtherContextEngine_advantages.append(f"Found expected file {expected_file} (ACE missed)")
    
    # === Metric 2: Top file match ===
    if ace.top_file and ThatOtherContextEngine.top_file:
        if paths_match(ace.top_file, ThatOtherContextEngine.top_file):
            pass  # Same top file = tie on this metric
        else:
            # Different top files - need to determine which is better
            # If one matches expected, that's better
            if expected_file:
                ace_top_matches = file_matches_pattern(ace.top_file, expected_file)
                ThatOtherContextEngine_top_matches = file_matches_pattern(ThatOtherContextEngine.top_file, expected_file)
                if ace_top_matches and not ThatOtherContextEngine_top_matches:
                    ace_advantages.append(f"Top result matches expected ({ace.top_file})")
                elif ThatOtherContextEngine_top_matches and not ace_top_matches:
                    ThatOtherContextEngine_advantages.append(f"Top result matches expected ({ThatOtherContextEngine.top_file})")
    
    # === Metric 3: File coverage overlap ===
    ace_files_set = set(normalize_path(f) for f in ace.files)
    ThatOtherContextEngine_files_set = set(normalize_path(f) for f in ThatOtherContextEngine.files)
    
    ace_unique = ace_files_set - ThatOtherContextEngine_files_set
    ThatOtherContextEngine_unique = ThatOtherContextEngine_files_set - ace_files_set
    
    if ace_unique:
        ace_advantages.append(f"Found unique files: {list(ace_unique)[:2]}")
    if ThatOtherContextEngine_unique:
        ThatOtherContextEngine_advantages.append(f"Found unique files: {list(ThatOtherContextEngine_unique)[:2]}")
    
    # === Metric 4: Content relevancy (keyword match) ===
    query_terms = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]+', query.lower()))
    query_terms = {t for t in query_terms if len(t) >= 3}
    
    def content_relevancy(contents: List[str]) -> float:
        if not contents:
            return 0.0
        all_content = " ".join(contents).lower()
        matched = sum(1 for t in query_terms if t in all_content)
        return matched / len(query_terms) if query_terms else 0.0
    
    ace_relevancy = content_relevancy(ace.contents)
    ThatOtherContextEngine_relevancy = content_relevancy(ThatOtherContextEngine.contents)
    
    if ace_relevancy > ThatOtherContextEngine_relevancy + 0.1:
        ace_advantages.append(f"Higher content relevancy ({ace_relevancy:.0%} vs {ThatOtherContextEngine_relevancy:.0%})")
    elif ThatOtherContextEngine_relevancy > ace_relevancy + 0.1:
        ThatOtherContextEngine_advantages.append(f"Higher content relevancy ({ThatOtherContextEngine_relevancy:.0%} vs {ace_relevancy:.0%})")
    
    # === Metric 5: Chunk size appropriateness ===
    # Not too small (< 10 lines) and not too large (> 200 lines)
    def chunk_score(line_ranges: List[Tuple[int, int]]) -> float:
        if not line_ranges:
            return 0.0
        total = sum(end - start + 1 for start, end in line_ranges if start and end)
        avg = total / len(line_ranges) if line_ranges else 0
        # Ideal range: 20-100 lines
        if 20 <= avg <= 100:
            return 1.0
        elif 10 <= avg <= 200:
            return 0.7
        else:
            return 0.4
    
    ace_chunk_score = chunk_score(ace.line_ranges)
    ThatOtherContextEngine_chunk_score = chunk_score(ThatOtherContextEngine.line_ranges)
    
    if ace_chunk_score > ThatOtherContextEngine_chunk_score:
        ace_advantages.append(f"Better chunk sizes (avg ~{ace.total_lines // max(1, len(ace.line_ranges))} lines)")
    elif ThatOtherContextEngine_chunk_score > ace_chunk_score:
        ThatOtherContextEngine_advantages.append(f"Better chunk sizes (avg ~{ThatOtherContextEngine.total_lines // max(1, len(ThatOtherContextEngine.line_ranges))} lines)")
    
    # === Determine winner ===
    ace_score = len(ace_advantages)
    ThatOtherContextEngine_score = len(ThatOtherContextEngine_advantages)
    
    # Prioritize finding expected file
    if expected_file:
        if ace_found_expected and not ThatOtherContextEngine_found_expected:
            ace_score += 2
        elif ThatOtherContextEngine_found_expected and not ace_found_expected:
            ThatOtherContextEngine_score += 2
    
    if not ace.files and not ThatOtherContextEngine.files:
        winner = "BOTH_MISS"
        reason = "Neither system returned results"
    elif ace_score > ThatOtherContextEngine_score:
        winner = "ACE"
        reason = f"ACE wins with {ace_score} advantages vs {ThatOtherContextEngine_score}"
    elif ThatOtherContextEngine_score > ace_score:
        winner = "ThatOtherContextEngine"
        reason = f"ThatOtherContextEngine wins with {ThatOtherContextEngine_score} advantages vs {ace_score}"
    else:
        winner = "TIE"
        reason = f"Tied with {ace_score} advantages each"
    
    return ComparisonResult(
        query=query,
        category=category,
        ace=ace,
        ThatOtherContextEngine=ThatOtherContextEngine,
        winner=winner,
        reason=reason,
        ace_advantages=ace_advantages,
        ThatOtherContextEngine_advantages=ThatOtherContextEngine_advantages,
        expected_file=expected_file,
    )


def run_single_comparison(
    query: str,
    category: str = "Code",
    expected_file: Optional[str] = None,
    ThatOtherContextEngine_output: Optional[str] = None,
    verbose: bool = True,
) -> ComparisonResult:
    """Run a single head-to-head comparison."""
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print(f"Category: {category}")
    if expected_file:
        print(f"Expected: {expected_file}")
    print("="*70)
    
    # Get ACE results
    ace = get_ace_results(query, limit=5)
    
    print(f"\n[ACE] Top 5 results:")
    for i, (f, s) in enumerate(zip(ace.files, ace.scores), 1):
        marker = "✓" if expected_file and file_matches_pattern(f, expected_file) else " "
        print(f"  {marker} {i}. [{s:.3f}] {f}")
    
    # Parse ThatOtherContextEngine output if provided
    if ThatOtherContextEngine_output:
        ThatOtherContextEngine = parse_ThatOtherContextEngine_output(ThatOtherContextEngine_output)
        print(f"\n[ThatOtherContextEngine] Top {len(ThatOtherContextEngine.files)} results:")
        for i, f in enumerate(ThatOtherContextEngine.files, 1):
            marker = "✓" if expected_file and file_matches_pattern(f, expected_file) else " "
            print(f"  {marker} {i}. {f}")
    else:
        ThatOtherContextEngine = RetrievalResult(files=[], scores=[], contents=[], line_ranges=[])
        print("\n[ThatOtherContextEngine] No output provided (run ThatOtherContextEngine MCP manually)")
    
    # Compare
    result = compare_results(query, category, ace, ThatOtherContextEngine, expected_file)
    
    print(f"\n{'─'*40}")
    print(f"WINNER: {result.winner}")
    print(f"Reason: {result.reason}")
    
    if result.ace_advantages:
        print(f"\nACE Advantages:")
        for adv in result.ace_advantages:
            print(f"  + {adv}")
    
    if result.ThatOtherContextEngine_advantages:
        print(f"\nThatOtherContextEngine Advantages:")
        for adv in result.ThatOtherContextEngine_advantages:
            print(f"  + {adv}")
    
    return result


def generate_report(results: List[ComparisonResult]) -> Dict[str, Any]:
    """Generate comprehensive comparison report."""
    total = len(results)
    ace_wins = sum(1 for r in results if r.winner == "ACE")
    ThatOtherContextEngine_wins = sum(1 for r in results if r.winner == "ThatOtherContextEngine")
    ties = sum(1 for r in results if r.winner == "TIE")
    both_miss = sum(1 for r in results if r.winner == "BOTH_MISS")
    
    # By category
    by_category: Dict[str, Dict[str, int]] = {}
    for r in results:
        if r.category not in by_category:
            by_category[r.category] = {"ACE": 0, "ThatOtherContextEngine": 0, "TIE": 0, "BOTH_MISS": 0}
        by_category[r.category][r.winner] += 1
    
    # ThatOtherContextEngine wins analysis (for fixing ACE)
    ThatOtherContextEngine_win_details = []
    for r in results:
        if r.winner == "ThatOtherContextEngine":
            ThatOtherContextEngine_win_details.append({
                "query": r.query,
                "category": r.category,
                "ace_top_file": r.ace.top_file,
                "ThatOtherContextEngine_top_file": r.ThatOtherContextEngine.top_file,
                "ThatOtherContextEngine_advantages": r.ThatOtherContextEngine_advantages,
                "expected_file": r.expected_file,
            })
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": total,
        "summary": {
            "ace_wins": ace_wins,
            "ThatOtherContextEngine_wins": ThatOtherContextEngine_wins,
            "ties": ties,
            "both_miss": both_miss,
            "ace_win_rate": f"{ace_wins / total * 100:.1f}%" if total else "0%",
        },
        "by_category": by_category,
        "ThatOtherContextEngine_wins_analysis": ThatOtherContextEngine_win_details,
    }
    
    return report


def print_report(report: Dict[str, Any]) -> None:
    """Print formatted report."""
    print("\n" + "="*70)
    print("HEAD-TO-HEAD BENCHMARK REPORT")
    print("="*70)
    
    summary = report["summary"]
    print(f"\nTotal Queries: {report['total_queries']}")
    print(f"ACE Wins:      {summary['ace_wins']} ({summary['ace_win_rate']})")
    print(f"ThatOtherContextEngine Wins:   {summary['ThatOtherContextEngine_wins']}")
    print(f"Ties:          {summary['ties']}")
    print(f"Both Miss:     {summary['both_miss']}")
    
    print("\n--- By Category ---")
    for cat, counts in report["by_category"].items():
        print(f"  {cat}: ACE={counts['ACE']}, ThatOtherContextEngine={counts['ThatOtherContextEngine']}, Tie={counts['TIE']}")
    
    if report["ThatOtherContextEngine_wins_analysis"]:
        print("\n--- ThatOtherContextEngine WINS (Need to fix ACE) ---")
        for detail in report["ThatOtherContextEngine_wins_analysis"]:
            print(f"\n  Query: {detail['query']}")
            print(f"  ACE returned: {detail['ace_top_file']}")
            print(f"  ThatOtherContextEngine returned: {detail['ThatOtherContextEngine_top_file']}")
            print(f"  ThatOtherContextEngine advantages: {detail['ThatOtherContextEngine_advantages']}")


def save_report(report: Dict[str, Any], output_dir: str = "benchmark_results") -> str:
    """Save report to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/ace_vs_ThatOtherContextEngine_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nReport saved to: {filename}")
    return filename


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ACE vs ThatOtherContextEngine Head-to-Head Benchmark")
    parser.add_argument("--query", "-q", help="Single query to test")
    parser.add_argument("--all", "-a", action="store_true", help="Run all benchmark queries")
    parser.add_argument("--category", "-c", help="Filter by category")
    parser.add_argument("--save", "-s", action="store_true", help="Save report to JSON")
    args = parser.parse_args()
    
    if args.query:
        # Single query mode
        result = run_single_comparison(args.query, "Manual")
        return
    
    if args.all:
        # Run all benchmark queries
        results = []
        queries = BENCHMARK_QUERIES
        
        if args.category:
            queries = [(c, q, e) for c, q, e in queries if c.lower() == args.category.lower()]
        
        for category, query, expected in queries:
            # For now, run ACE only (ThatOtherContextEngine output needs to be captured separately)
            result = run_single_comparison(query, category, expected, ThatOtherContextEngine_output=None, verbose=True)
            results.append(result)
        
        report = generate_report(results)
        print_report(report)
        
        if args.save:
            save_report(report)
    else:
        print("Usage: python ace_vs_ThatOtherContextEngine_headtohead.py --query 'your query' or --all")
        print("\nRun with --help for more options")


if __name__ == "__main__":
    main()
