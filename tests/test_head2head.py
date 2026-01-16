#!/usr/bin/env python3
"""
ACE vs ThatOtherContextEngine Head-to-Head Live Comparison

This script runs specific queries against ACE and formats output for
comparison with ThatOtherContextEngine MCP results.

Usage:
    python test_head2head.py "query string"
    python test_head2head.py --batch  # Run all benchmark queries
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

from ace.code_retrieval import CodeRetrieval


# 100+ queries across categories for comprehensive testing
BENCHMARK_QUERIES = [
    # === Category 1: Core Implementation (20 queries) ===
    ("Code", "CodeRetrieval class search method implementation"),
    ("Code", "exception error handling retry resilience pattern"),
    ("Code", "embedding vector generation voyage-code-3 model"),
    ("Code", "ASTChunker code chunking tree-sitter parsing"),
    ("Code", "deduplicate overlapping results similarity threshold"),
    ("Code", "cross encoder reranker BGE ranking score"),
    ("Code", "CodeIndexer index_workspace batch embedding"),
    ("Code", "dense vector search cosine similarity Qdrant"),
    ("Code", "sparse vector BM25 term frequency hashing"),
    ("Code", "file watcher daemon monitor changes filesystem"),
    ("Code", "UnifiedMemoryIndex store method implementation"),
    ("Code", "_apply_filename_boost function scoring"),
    ("Code", "expand_query method synonyms expansion"),
    ("Code", "create_sparse_vector BM25 implementation"),
    ("Code", "AsyncCodeRetrieval async search method"),
    ("Code", "normalize_scores method score normalization"),
    ("Code", "parse_code method AST parsing Python"),
    ("Code", "embed_text method embedding generation"),
    ("Code", "index_bullet method unified memory storage"),
    ("Code", "retrieve method Qdrant vector search"),
    
    # === Category 2: Configuration (15 queries) ===
    ("Config", "EmbeddingConfig class voyage openai model settings"),
    ("Config", "QdrantConfig collection vector database settings"),
    ("Config", "LLMConfig provider API key model configuration"),
    ("Config", "BM25_K1 BM25_B constants sparse vector tuning"),
    ("Config", "environment variable dotenv configuration loading"),
    ("Config", "rate limit API throttling requests per second"),
    ("Config", "vector dimension size 1024 768 embedding model"),
    ("Config", "batch size chunk processing embedding queue"),
    ("Config", "collection name workspace code context identifier"),
    ("Config", "reranker model BGE cross encoder path"),
    ("Config", "VOYAGE_API_KEY environment variable setup"),
    ("Config", "Qdrant localhost port 6333 connection"),
    ("Config", "top_k limit search results maximum"),
    ("Config", "cache TTL expiration time seconds"),
    ("Config", "chunk_size max_tokens code splitting"),
    
    # === Category 3: Architecture/Design (20 queries) ===
    ("Arch", "Generator Reflector Curator architecture pattern"),
    ("Arch", "unified memory retrieval storage architecture"),
    ("Arch", "playbook bullet strategy learning system"),
    ("Arch", "tenant isolation namespace collection multitenancy"),
    ("Arch", "caching layer retrieval performance TTL"),
    ("Arch", "HyDE hypothetical document embeddings expansion"),
    ("Arch", "observability tracing metrics opik monitoring"),
    ("Arch", "bullet enrichment LLM context enhancement"),
    ("Arch", "delta operation batch update playbook system"),
    ("Arch", "intent classification query routing semantic"),
    ("Arch", "hybrid search dense sparse fusion ranking"),
    ("Arch", "code analysis dependency graph import tree"),
    ("Arch", "chunk context expansion surrounding lines"),
    ("Arch", "pattern detection code smell antipattern"),
    ("Arch", "quality feedback loop learning adaptation"),
    ("Arch", "semantic deduplication similarity threshold"),
    ("Arch", "async retrieval concurrent embedding calls"),
    ("Arch", "GPU reranker batch processing CUDA"),
    ("Arch", "query enhancement reformulation expansion"),
    ("Arch", "file change delta incremental indexing"),
    
    # === Category 4: Documentation (15 queries) ===
    ("Docs", "quick start guide installation setup tutorial"),
    ("Docs", "API reference documentation endpoints methods"),
    ("Docs", "MCP server integration copilot claude setup"),
    ("Docs", "architecture design system overview diagram"),
    ("Docs", "configuration guide environment variables"),
    ("Docs", "prompts generator reflector curator templates"),
    ("Docs", "retrieval precision optimization tuning guide"),
    ("Docs", "integration guide framework agent workflow"),
    ("Docs", "golden rules best practices guidelines"),
    ("Docs", "changelog version history release notes"),
    ("Docs", "contributing guidelines pull request workflow"),
    ("Docs", "Claude Code integration MCP setup instructions"),
    ("Docs", "VSCode integration tasks launch json"),
    ("Docs", "deduplication strategy overlapping results"),
    ("Docs", "typo correction system learned patterns"),
    
    # === Category 5: Edge Cases (30 queries) ===
    ("Edge", "async await asyncio concurrent retrieval gather"),
    ("Edge", "typing Optional List Dict Tuple dataclass"),
    ("Edge", "from qdrant_client import QdrantClient models"),
    ("Edge", "UnifiedMemoryIndex namespace hybrid search"),
    ("Edge", "Qdrant client not available error exception"),
    ("Edge", "dataclass field default_factory frozen slots"),
    ("Edge", "decorator functools wraps retry backoff"),
    ("Edge", "context manager with statement async generator"),
    ("Edge", "lambda filter map list comprehension generator"),
    ("Edge", "regex pattern re compile match search findall"),
    ("Edge", "fibonacci.py error handling example"),
    ("Edge", "test_ace.py unit test pytest fixtures"),
    ("Edge", "openai_embeddings.py API client wrapper"),
    ("Edge", "gemini_embeddings.py Google AI integration"),
    ("Edge", "httpx async HTTP client requests timeout"),
    ("Edge", "json loads dumps serialization deserialization"),
    ("Edge", "pathlib Path resolve absolute path handling"),
    ("Edge", "logging logger info debug warning error"),
    ("Edge", "subprocess run Popen communicate shell"),
    ("Edge", "threading Thread Lock concurrent execution"),
    ("Edge", "abc ABC abstractmethod interface pattern"),
    ("Edge", "enum Enum IntEnum auto choices"),
    ("Edge", "hashlib sha256 md5 digest hexdigest"),
    ("Edge", "base64 encode decode b64encode urlsafe"),
    ("Edge", "datetime now strftime strptime timezone"),
    ("Edge", "pickle dump load serialization security"),
    ("Edge", "sqlite3 connect cursor execute fetchall"),
    ("Edge", "yaml safe_load dump configuration file"),
    ("Edge", "os.environ environment variable access"),
    ("Edge", "sys.path module import resolution"),
]


def run_ace_search(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Run ACE code retrieval and return detailed results."""
    retriever = CodeRetrieval()
    results = retriever.search(query, limit=limit)
    return results


def print_ace_results(query: str, results: List[Dict[str, Any]], show_content: bool = False):
    """Print ACE results in detailed format for comparison."""
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")
    print(f"\nACE Results ({len(results)} returned):")
    print("-" * 60)
    
    for i, r in enumerate(results, 1):
        file_path = r.get("file_path", "???")
        score = r.get("score", 0)
        start = r.get("start_line", 0)
        end = r.get("end_line", 0)
        chunk_type = r.get("chunk_type", "???")
        
        # Extract just filename for cleaner output
        filename = Path(file_path).name
        
        print(f"\n#{i} [{score:.3f}] {file_path}")
        print(f"   Lines: {start}-{end} | Type: {chunk_type}")
        
        if show_content:
            content = r.get("content", "")
            # Show first 500 chars
            preview = content[:500] + "..." if len(content) > 500 else content
            print(f"   Content preview:")
            for line in preview.split("\n")[:10]:
                print(f"      {line}")
    
    print("\n" + "-" * 60)


def run_single_query(query: str, show_content: bool = True):
    """Run a single query and show results."""
    results = run_ace_search(query, limit=10)
    print_ace_results(query, results, show_content)
    return results


def run_batch(categories: List[str] = None, show_content: bool = False):
    """Run all benchmark queries and generate report."""
    print(f"\n{'#'*80}")
    print(f"# ACE BENCHMARK - {len(BENCHMARK_QUERIES)} Queries")
    print(f"# Timestamp: {datetime.now().isoformat()}")
    print(f"{'#'*80}")
    
    results_by_category = {}
    all_results = []
    
    for category, query in BENCHMARK_QUERIES:
        if categories and category not in categories:
            continue
        
        if category not in results_by_category:
            results_by_category[category] = []
        
        results = run_ace_search(query, limit=5)
        
        # Extract top result info
        top_file = results[0]["file_path"] if results else "NO RESULTS"
        top_score = results[0]["score"] if results else 0
        
        result_summary = {
            "category": category,
            "query": query,
            "top_file": top_file,
            "top_score": top_score,
            "all_files": [r["file_path"] for r in results[:5]],
            "all_scores": [r["score"] for r in results[:5]],
        }
        
        results_by_category[category].append(result_summary)
        all_results.append(result_summary)
        
        # Print progress
        print(f"\n[{category}] {query[:50]}...")
        print(f"   #1: [{top_score:.3f}] {top_file}")
        if show_content and results:
            content = results[0].get("content", "")[:200]
            print(f"   Preview: {content[:100]}...")
    
    # Summary report
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    
    for category, results in results_by_category.items():
        avg_score = sum(r["top_score"] for r in results) / len(results) if results else 0
        print(f"\n{category}: {len(results)} queries, avg top score: {avg_score:.3f}")
        
        # Show distribution of top files
        top_files = {}
        for r in results:
            f = r["top_file"]
            top_files[f] = top_files.get(f, 0) + 1
        
        print(f"   Top files distribution:")
        for f, count in sorted(top_files.items(), key=lambda x: -x[1])[:5]:
            print(f"      {count}x: {f}")
    
    # Save to JSON
    output_path = Path("benchmark_results") / f"ace_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(all_results),
            "results": all_results,
            "by_category": {k: v for k, v in results_by_category.items()},
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ACE Head-to-Head Testing")
    parser.add_argument("query", nargs="?", help="Single query to test")
    parser.add_argument("--batch", action="store_true", help="Run all benchmark queries")
    parser.add_argument("--category", help="Filter by category (Code, Config, Arch, Docs, Edge)")
    parser.add_argument("--content", action="store_true", help="Show content preview")
    args = parser.parse_args()
    
    if args.batch:
        categories = [args.category] if args.category else None
        run_batch(categories, show_content=args.content)
    elif args.query:
        run_single_query(args.query, show_content=True)
    else:
        # Default: run a few test queries
        test_queries = [
            "exception error handling retry resilience pattern",
            "CodeRetrieval class search method implementation",
            "UnifiedMemoryIndex store method implementation",
        ]
        for q in test_queries:
            run_single_query(q, show_content=False)


if __name__ == "__main__":
    main()
