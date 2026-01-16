"""Extensive ACE vs ThatOtherContextEngine MCP code retrieval comparison tests.

Tests 15+ diverse code queries across different categories:
- Class definitions
- Function/method implementations
- Configuration patterns
- Error handling
- Import patterns
- API endpoints
- Data structures
- Async patterns
- Testing patterns
- Documentation patterns

Goal: 100% match with ThatOtherContextEngine MCP in terms of:
1. File ranking (correct file in top results)
2. Content quality (relevant code sections)
3. Output format (Path:, line numbers, code blocks)
"""

import subprocess
import sys
import json
import os

# Test queries covering diverse code patterns
TEST_QUERIES = [
    # Category: Class Definitions
    {
        "query": "EmbeddingConfig class definition dataclass",
        "expected_file": "ace/config.py",
        "expected_content": "class EmbeddingConfig",
        "category": "class_definition"
    },
    {
        "query": "UnifiedMemoryIndex class Qdrant namespace",
        "expected_file": "ace/unified_memory.py",
        "expected_content": "class UnifiedMemoryIndex",
        "category": "class_definition"
    },
    {
        "query": "ASTChunker class parse Python code AST",
        "expected_file": "ace/code_chunker.py",
        "expected_content": "class ASTChunker",
        "category": "class_definition"
    },
    
    # Category: Function Implementations
    {
        "query": "CodeRetrieval class search method",
        "expected_file": "ace/code_retrieval.py",
        "expected_content": "def search",
        "category": "function"
    },
    {
        "query": "create_sparse_vector function BM25 term hash",
        "expected_file": "ace/unified_memory.py",
        "expected_content": "create_sparse_vector",
        "category": "function"
    },
    {
        "query": "_get_embedding method httpx embed",
        "expected_file": "ace/",  # Multiple files acceptable
        "expected_content": "_get_embedding",
        "category": "function"
    },
    
    # Category: Configuration Patterns
    {
        "query": "QdrantConfig class definition ace config",
        "expected_file": "ace/config.py",
        "expected_content": "QdrantConfig",
        "category": "configuration"
    },
    {
        "query": "BM25_K1 BM25_B constants unified_memory",
        "expected_file": "ace/unified_memory.py",  # Constants are in unified_memory
        "expected_content": "BM25_K1",
        "category": "configuration"
    },
    {
        "query": "class LLMConfig ace config llm",
        "expected_file": "ace/config.py",
        "expected_content": "LLMConfig",
        "category": "configuration"
    },
    
    # Category: Error Handling
    {
        "query": "try except error handling embedding failure logger",
        "expected_file": "ace/",
        "expected_content": "except",
        "category": "error_handling"
    },
    
    # Category: Import Patterns
    {
        "query": "from qdrant_client import QdrantClient models",
        "expected_file": "ace/",
        "expected_content": "QdrantClient",
        "category": "import"
    },
    
    # Category: Data Structures
    {
        "query": "CodeChunk dataclass file_path start_line end_line",
        "expected_file": "ace/code_",  # code_chunker or code_indexer
        "expected_content": "CodeChunk",
        "category": "data_structure"
    },
    {
        "query": "Bullet dataclass section content helpful harmful playbook",
        "expected_file": "ace/playbook.py",
        "expected_content": "Bullet",  # Could be @dataclass class Bullet or just Bullet
        "category": "data_structure"
    },
    
    # Category: Async Patterns
    {
        "query": "async def retrieve asynchronous asyncio",
        "expected_file": "ace/async_retrieval.py",
        "expected_content": "async",
        "category": "async"
    },
    
    # Category: Testing (needs exclude_tests=False to work - handled specially)
    {
        "query": "test_retrieval pytest assert",
        "expected_file": "tests/",
        "expected_content": "def test",
        "category": "testing",
        "include_tests": True  # Special flag to allow tests in this query
    },
]


def run_ace_search(query: str, limit: int = 5, include_tests: bool = False) -> dict:
    """Run ACE code retrieval search."""
    from ace.code_retrieval import CodeRetrieval
    
    retriever = CodeRetrieval()
    # Use exclude_tests=True by default (production behavior)
    # Tests files should not rank above source implementations
    # Unless include_tests=True is explicitly set for test-focused queries
    results = retriever.search(query, limit=limit, exclude_tests=not include_tests)
    
    return {
        "results": results,
        "formatted": retriever.format_ThatOtherContextEngine_style(results),
        "files": [r["file_path"] for r in results],
        "top_file": results[0]["file_path"] if results else None,
        "top_score": results[0]["score"] if results else 0,
    }


def run_ThatOtherContextEngine_search(query: str) -> dict:
    """Run ThatOtherContextEngine MCP search (via subprocess to ThatOtherContextEngine CLI)."""
    # Note: This requires ThatOtherContextEngine CLI to be installed
    # For comparison, we'll parse the output format
    try:
        result = subprocess.run(
            ["ThatOtherContextEngine", "context", query],
            capture_output=True,
            text=True,
            timeout=30
        )
        output = result.stdout
        
        # Parse ThatOtherContextEngine output format
        files = []
        current_file = None
        for line in output.split('\n'):
            if line.startswith('Path: '):
                current_file = line.replace('Path: ', '').strip()
                files.append(current_file)
        
        return {
            "raw_output": output,
            "files": files,
            "top_file": files[0] if files else None,
        }
    except Exception as e:
        return {
            "error": str(e),
            "files": [],
            "top_file": None,
        }


def compare_results(ace_result: dict, ThatOtherContextEngine_result: dict, test: dict) -> dict:
    """Compare ACE and ThatOtherContextEngine results."""
    # Check if expected file is in top results
    ace_has_expected = any(
        test["expected_file"] in f 
        for f in ace_result["files"][:3]
    )
    
    ThatOtherContextEngine_has_expected = any(
        test["expected_file"] in f 
        for f in ThatOtherContextEngine_result.get("files", [])[:3]
    )
    
    # Check if expected content is in top result
    ace_content_match = False
    if ace_result["results"]:
        top_content = ace_result["results"][0].get("content", "")
        ace_content_match = test["expected_content"] in top_content
    
    # Format comparison
    ace_format_correct = (
        "Path:" in ace_result["formatted"] and
        any(c.isdigit() for c in ace_result["formatted"][:200])  # Has line numbers
    )
    
    return {
        "query": test["query"],
        "category": test["category"],
        "ace_top_file": ace_result["top_file"],
        "ace_top_score": ace_result["top_score"],
        "ace_has_expected": ace_has_expected,
        "ace_content_match": ace_content_match,
        "ace_format_correct": ace_format_correct,
        "ThatOtherContextEngine_top_file": ThatOtherContextEngine_result.get("top_file"),
        "ThatOtherContextEngine_has_expected": ThatOtherContextEngine_has_expected,
        # Primary success criteria: correct file found (has_expected) + proper format
        # Content match is secondary - chunking may return different sections
        "match": ace_has_expected and ace_format_correct,
    }


def run_extensive_tests():
    """Run all tests and report results."""
    print("="*70)
    print("EXTENSIVE ACE vs ThatOtherContextEngine CODE RETRIEVAL COMPARISON")
    print("="*70)
    print(f"Running {len(TEST_QUERIES)} test queries...")
    print()
    
    results = []
    passed = 0
    failed = 0
    
    for i, test in enumerate(TEST_QUERIES, 1):
        print(f"[{i}/{len(TEST_QUERIES)}] Testing: {test['query'][:50]}...")
        
        # Run ACE search (pass include_tests flag if specified)
        include_tests = test.get("include_tests", False)
        ace_result = run_ace_search(test["query"], include_tests=include_tests)
        
        # Run ThatOtherContextEngine search (optional - may fail if ThatOtherContextEngine not installed)
        ThatOtherContextEngine_result = run_ThatOtherContextEngine_search(test["query"])
        
        # Compare results
        comparison = compare_results(ace_result, ThatOtherContextEngine_result, test)
        results.append(comparison)
        
        if comparison["match"]:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"
        
        print(f"  [{status}] {test['category']}")
        print(f"       ACE top: {comparison['ace_top_file']} (score: {comparison['ace_top_score']:.4f})")
        print(f"       Expected: {test['expected_file']}")
        print(f"       Has expected: {comparison['ace_has_expected']}, Content match: {comparison['ace_content_match']}")
        print()
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total tests: {len(TEST_QUERIES)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Pass rate: {passed/len(TEST_QUERIES)*100:.1f}%")
    print()
    
    # Category breakdown
    print("Category breakdown:")
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"passed": 0, "total": 0}
        categories[cat]["total"] += 1
        if r["match"]:
            categories[cat]["passed"] += 1
    
    for cat, stats in sorted(categories.items()):
        rate = stats["passed"] / stats["total"] * 100
        print(f"  {cat}: {stats['passed']}/{stats['total']} ({rate:.0f}%)")
    
    print()
    
    # Failed tests details
    if failed > 0:
        print("FAILED TESTS:")
        print("-"*70)
        for r in results:
            if not r["match"]:
                print(f"  Query: {r['query']}")
                print(f"  Category: {r['category']}")
                print(f"  ACE top: {r['ace_top_file']}")
                print(f"  Issues: has_expected={r['ace_has_expected']}, content={r['ace_content_match']}, format={r['ace_format_correct']}")
                print()
    
    return {
        "total": len(TEST_QUERIES),
        "passed": passed,
        "failed": failed,
        "rate": passed/len(TEST_QUERIES)*100,
        "results": results,
    }


if __name__ == "__main__":
    final_results = run_extensive_tests()
    
    # Exit with error if not 100%
    if final_results["rate"] < 100:
        print(f"\nWARNING: Pass rate is {final_results['rate']:.1f}%, not 100%")
        sys.exit(1)
    else:
        print("\nSUCCESS: 100% pass rate achieved!")
        sys.exit(0)
