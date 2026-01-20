"""Test that ACE excludes scripts/debug/benchmark files when exclude_tests=True."""

import pytest
from ace.code_retrieval import CodeRetrieval


class TestExcludeNonCoreFiles:
    """Verify _is_test_file excludes scripts/debug/benchmark files."""

    @pytest.fixture
    def retriever(self):
        return CodeRetrieval()

    # Test files that SHOULD be excluded (True = test/non-core file)
    @pytest.mark.parametrize("file_path,expected", [
        # Traditional test files
        ("tests/test_code.py", True),
        ("test/test_something.py", True),
        ("test_example.py", True),
        ("something_test.py", True),
        ("conftest.py", True),
        # Scripts directory
        ("scripts/trace_retrieve.py", True),
        ("scripts/ab_test.py", True),
        ("scripts/benchmark_cross_encoder.py", True),
        # Debug files
        ("debug_httpx_query.py", True),
        ("debug_retrieval.py", True),
        ("debug_something.py", True),
        # Benchmark files
        ("benchmark_1000_queries.py", True),
        ("benchmark_ace_quality.py", True),
        ("run_comprehensive_benchmark.py", True),
        # Dev scripts
        ("dev_scripts/analysis/analyze.py", True),
        ("dev_scripts/benchmarks/benchmark.py", True),
        # Examples
        ("examples/demo.py", True),
        ("example_usage.py", True),
    ])
    def test_should_exclude_non_core_files(self, retriever, file_path, expected):
        """These files should be classified as 'test' (non-core) files."""
        assert retriever._is_test_file(file_path) == expected, \
            f"{file_path} should be excluded (return True)"

    # Test files that SHOULD NOT be excluded (False = core implementation file)
    @pytest.mark.parametrize("file_path,expected", [
        # Core ace/ module files
        ("ace/code_retrieval.py", False),
        ("ace/unified_memory.py", False),
        ("ace/config.py", False),
        ("ace/code_indexer.py", False),
        ("ace/reranker.py", False),
        # Other valid source files
        ("src/main.py", False),
        ("lib/utils.py", False),
        ("app/service.py", False),
        # Top-level scripts that ARE implementations
        ("ace_mcp_server.py", False),
        ("setup.py", False),
        ("pyproject.toml", False),  # Not .py but still should not be excluded
    ])
    def test_should_not_exclude_core_files(self, retriever, file_path, expected):
        """These files should NOT be classified as 'test' files."""
        assert retriever._is_test_file(file_path) == expected, \
            f"{file_path} should NOT be excluded (return False)"


class TestSearchExcludesNonCoreFiles:
    """Integration test: search() with exclude_tests=True should not return scripts."""

    def test_index_workspace_returns_core_file(self):
        """Query for 'index_workspace function' should return ace/code_indexer.py not benchmark."""
        retriever = CodeRetrieval()
        results = retriever.search("index_workspace function", limit=5, exclude_tests=True)
        
        assert results, "Should return results"
        top_file = results[0].get("file_path", "")
        
        # Should NOT return benchmark/script files
        assert "benchmark" not in top_file.lower(), f"Got benchmark file: {top_file}"
        assert "script" not in top_file.lower(), f"Got script file: {top_file}"
        assert "debug" not in top_file.lower(), f"Got debug file: {top_file}"
        
        # Should return core implementation
        assert "ace/code_indexer.py" in top_file or "code_indexer" in top_file.lower()

    def test_expand_query_returns_core_file(self):
        """Query for '_expand_query' should return ace/code_retrieval.py or related files."""
        retriever = CodeRetrieval()
        results = retriever.search("_expand_query function code retrieval", limit=5, exclude_tests=True)
        
        assert results, "Should return results"
        top_file = results[0].get("file_path", "")
        
        # Should return core implementation file (includes retrieval_optimized which has expand variants)
        valid_files = ["ace/code_retrieval.py", "ace/unified_memory.py", "ace/retrieval_optimized.py"]
        assert any(vf in top_file for vf in valid_files), \
            f"Expected core file, got: {top_file}"

    def test_unified_memory_retrieve_returns_core_file(self):
        """Query for 'UnifiedMemoryIndex retrieve method' should return ace/unified_memory.py or docs."""
        retriever = CodeRetrieval()
        results = retriever.search("UnifiedMemoryIndex retrieve method", limit=5, exclude_tests=True)
        
        assert results, "Should return results"
        top_file = results[0].get("file_path", "")
        
        assert "scripts/" not in top_file.lower(), f"Got script file: {top_file}"
        assert "trace_retrieve" not in top_file.lower(), f"Got trace script: {top_file}"
        # Can return code OR docs (docs contain usage examples)
        valid = ("ace/unified_memory.py" in top_file or 
                 "unified_memory" in top_file.lower() or
                 "docs/" in top_file.lower())
        assert valid, f"Expected unified_memory or docs, got: {top_file}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
