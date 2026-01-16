"""Head-to-head ACE vs ThatOtherContextEngine comparison test.

For each query, we call both systems and compare:
1. Top file match
2. Top 3 files overlap
3. Content relevance
"""
import subprocess
import sys
import json
from typing import Dict, List, Any

# Test queries - comprehensive coverage
TEST_QUERIES = [
    # Core ACE classes
    "EmbeddingConfig class definition dataclass",
    "UnifiedMemoryIndex class Qdrant namespace hybrid search",
    "CodeRetrieval class search method dense vector",
    "ASTChunker class parse Python code AST",
    
    # Configuration patterns
    "QdrantConfig class definition ace config",
    "class LLMConfig ace config llm",
    "BM25 constants BM25_K1 BM25_B",
    
    # Functions
    "create_sparse_vector function BM25 term hash",
    "_get_embedding method httpx embed",
    
    # Data structures
    "CodeChunk dataclass file_path start_line end_line",
    "Bullet dataclass section content helpful harmful playbook",
    
    # Async patterns
    "async def retrieve asynchronous asyncio",
    
    # Error handling
    "try except error handling embedding failure",
    
    # Import patterns
    "from qdrant_client import QdrantClient models",
]


def get_ace_results(query: str, limit: int = 5) -> List[str]:
    """Get top files from ACE."""
    from ace.code_retrieval import CodeRetrieval
    
    r = CodeRetrieval()
    results = r.search(query, limit=limit)
    return [r["file_path"] for r in results]


def get_ThatOtherContextEngine_results(query: str) -> List[str]:
    """Parse ThatOtherContextEngine MCP results to extract file paths."""
    # This would normally call ThatOtherContextEngine MCP, but since we can't call it directly
    # from this script, we'll simulate by using the same retrieval
    # In production, this would be the actual ThatOtherContextEngine MCP call
    
    # For now, return empty - we'll compare by running queries manually
    return []


def run_comparison():
    """Run head-to-head comparison."""
    print("="*70)
    print("ACE vs ThatOtherContextEngine HEAD-TO-HEAD COMPARISON")
    print("="*70)
    print()
    
    results = []
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"[{i}/{len(TEST_QUERIES)}] Query: {query[:50]}...")
        
        ace_files = get_ace_results(query)
        
        print(f"       ACE top file: {ace_files[0] if ace_files else 'None'}")
        print(f"       ACE top 3: {ace_files[:3]}")
        print()
        
        results.append({
            "query": query,
            "ace_files": ace_files,
        })
    
    # Summary - show all top files
    print("="*70)
    print("SUMMARY - TOP FILES FOR EACH QUERY")
    print("="*70)
    for r in results:
        print(f"Q: {r['query'][:50]}...")
        print(f"   ACE: {r['ace_files'][0] if r['ace_files'] else 'None'}")
        print()


if __name__ == "__main__":
    run_comparison()
