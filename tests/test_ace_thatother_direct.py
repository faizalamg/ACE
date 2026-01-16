"""Direct ACE vs ThatOtherContextEngine comparison for key code patterns.

This test directly compares what ACE code retrieval returns vs what
ThatOtherContextEngine MCP returns for the same queries, checking if the top-ranked
file from ThatOtherContextEngine is also in ACE's top results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ace.code_retrieval import CodeRetrieval


def test_key_patterns():
    """Test key code patterns that should match ThatOtherContextEngine behavior."""
    retriever = CodeRetrieval()
    
    # These queries were verified against ThatOtherContextEngine MCP
    test_cases = [
        # Query -> Expected file from ThatOtherContextEngine
        ("EmbeddingConfig class definition dataclass", "ace/config.py"),
        ("UnifiedMemoryIndex class Qdrant namespace hybrid", "ace/unified_memory.py"),
        ("ASTChunker class parse Python code AST", "ace/code_chunker.py"),
        ("CodeRetrieval class search method", "ace/code_retrieval.py"),
        ("create_sparse_vector function BM25 term hash", "ace/unified_memory.py"),
        ("QdrantConfig class definition ace config", "ace/config.py"),
        ("class LLMConfig ace config llm", "ace/config.py"),
        ("class Bullet dataclass playbook section content", "ace/playbook.py"),
        ("async def retrieve asynchronous retrieval", "ace/async_retrieval.py"),
    ]
    
    passed = 0
    failed = 0
    
    print("=" * 70)
    print("ACE vs ThatOtherContextEngine TOP FILE COMPARISON")
    print("=" * 70)
    
    for query, expected_file in test_cases:
        results = retriever.search(query, limit=5, exclude_tests=True)
        
        if not results:
            print(f"[FAIL] {query[:50]}...")
            print(f"       ACE returned no results")
            print(f"       Expected: {expected_file}")
            failed += 1
            continue
        
        # Check if expected file is in top 3 results
        top_files = [r["file_path"] for r in results[:3]]
        found = any(expected_file in f for f in top_files)
        
        if found:
            print(f"[PASS] {query[:50]}...")
            print(f"       ACE top: {results[0]['file_path']} (score: {results[0]['score']:.4f})")
            print(f"       Expected in top 3: {expected_file}")
            passed += 1
        else:
            print(f"[FAIL] {query[:50]}...")
            print(f"       ACE top 3: {top_files}")
            print(f"       Expected: {expected_file}")
            failed += 1
        print()
    
    print("=" * 70)
    print(f"RESULTS: {passed}/{passed + failed} ({100 * passed / (passed + failed):.1f}%)")
    print("=" * 70)
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = test_key_patterns()
    sys.exit(0 if failed == 0 else 1)
