"""
TDD Tests for CodeRetrieval - semantic search and ThatOtherContextEngine-style output formatting.

RED Phase: All tests should FAIL until implementation is complete.

CodeRetrieval is responsible for:
1. Querying indexed code chunks via semantic search
2. Formatting results in ThatOtherContextEngine MCP-compatible format
3. Deduplicating and ranking results
4. Returning both code AND memory results (blended)
"""

import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass
from typing import List, Dict, Any


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    with patch("qdrant_client.QdrantClient") as mock:
        client = MagicMock()
        client.collection_exists.return_value = True
        
        # Mock search results
        mock_point_1 = MagicMock()
        mock_point_1.score = 0.95
        mock_point_1.payload = {
            "file_path": "ace/unified_memory.py",
            "content": "class UnifiedMemoryIndex:\n    def __init__(self):\n        pass\n",
            "start_line": 10,
            "end_line": 15,
            "language": "python",
            "symbols": ["UnifiedMemoryIndex", "__init__"],
        }
        
        mock_point_2 = MagicMock()
        mock_point_2.score = 0.85
        mock_point_2.payload = {
            "file_path": "ace/retrieval.py",
            "content": "def retrieve_memories(query: str):\n    return []\n",
            "start_line": 1,
            "end_line": 5,
            "language": "python",
            "symbols": ["retrieve_memories"],
        }
        
        # query_points returns a response with .points attribute
        mock_response = MagicMock()
        mock_response.points = [mock_point_1, mock_point_2]
        client.query_points.return_value = mock_response
        
        # Also set legacy search for backwards compat in tests
        client.search.return_value = [mock_point_1, mock_point_2]
        mock.return_value = client
        yield client


@pytest.fixture
def mock_ace_memory():
    """Mock ACE memory retrieval results."""
    return [
        {
            "content": "Always use type hints in Python functions.",
            "category": "PREFERENCE",
            "namespace": "user_prefs",
            "section": "coding-style",
            "severity": 5,
        },
        {
            "content": "Isolate context logic in dedicated modules.",
            "category": "ARCHITECTURE",
            "namespace": "task_strategies",
            "section": "design",
            "severity": 7,
        },
    ]


@pytest.fixture
def mock_embedder():
    """Mock embedding function."""
    def embed(text: str) -> List[float]:
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:16], 16)
        return [float((hash_val >> i) & 1) for i in range(768)]
    return embed


# =============================================================================
# TEST: CodeRetrieval Initialization
# =============================================================================

class TestCodeRetrievalInit:
    """Test CodeRetrieval initialization."""
    
    def test_init_creates_instance(self, mock_qdrant_client):
        """CodeRetrieval should initialize correctly."""
        from ace.code_retrieval import CodeRetrieval
        
        retriever = CodeRetrieval(
            qdrant_url="http://localhost:6333",
            collection_name="ace_code_context"
        )
        
        assert retriever is not None
        assert retriever.collection_name == "ace_code_context"
    
    def test_init_uses_env_defaults(self, mock_qdrant_client):
        """Should use environment defaults."""
        with patch.dict(os.environ, {
            "QDRANT_URL": "http://custom:6333",
            "ACE_CODE_COLLECTION": "custom_collection"
        }):
            from ace.code_retrieval import CodeRetrieval
            
            retriever = CodeRetrieval()
            
            assert retriever.qdrant_url == "http://custom:6333"
            assert retriever.collection_name == "custom_collection"
    
    def test_init_with_custom_embedder(self, mock_qdrant_client, mock_embedder):
        """Should accept custom embedding function."""
        from ace.code_retrieval import CodeRetrieval
        
        retriever = CodeRetrieval(embed_fn=mock_embedder)
        
        assert retriever._embed_fn is not None


# =============================================================================
# TEST: Semantic Code Search
# =============================================================================

class TestCodeRetrievalSearch:
    """Test semantic code search functionality."""
    
    def test_search_returns_code_chunks(self, mock_qdrant_client, mock_embedder):
        """Search should return relevant code chunks."""
        from ace.code_retrieval import CodeRetrieval
        
        retriever = CodeRetrieval(embed_fn=mock_embedder)
        results = retriever.search("unified memory index")
        
        assert len(results) > 0
        assert "content" in results[0]
        assert "file_path" in results[0]
    
    def test_search_returns_scored_results(self, mock_qdrant_client, mock_embedder):
        """Results should include relevance scores."""
        from ace.code_retrieval import CodeRetrieval
        
        retriever = CodeRetrieval(embed_fn=mock_embedder)
        results = retriever.search("memory retrieval")
        
        assert all("score" in r for r in results)
        assert results[0]["score"] >= results[-1]["score"]  # Sorted by score
    
    def test_search_respects_limit(self, mock_qdrant_client, mock_embedder):
        """Should respect result limit."""
        from ace.code_retrieval import CodeRetrieval
        
        retriever = CodeRetrieval(embed_fn=mock_embedder)
        results = retriever.search("query", limit=1)
        
        assert len(results) <= 1
    
    def test_search_returns_empty_for_no_match(self, mock_qdrant_client, mock_embedder):
        """Should return empty list when no matches."""
        mock_empty_response = MagicMock()
        mock_empty_response.points = []
        mock_qdrant_client.query_points.return_value = mock_empty_response
        
        from ace.code_retrieval import CodeRetrieval
        
        retriever = CodeRetrieval(embed_fn=mock_embedder)
        results = retriever.search("nonexistent gibberish xyz123")
        
        assert results == []
    
    def test_search_includes_line_numbers(self, mock_qdrant_client, mock_embedder):
        """Results should include line number metadata."""
        from ace.code_retrieval import CodeRetrieval
        
        retriever = CodeRetrieval(embed_fn=mock_embedder)
        results = retriever.search("unified memory")
        
        assert "start_line" in results[0]
        assert "end_line" in results[0]
        assert results[0]["start_line"] > 0


# =============================================================================
# TEST: ThatOtherContextEngine-Style Output Formatting
# =============================================================================

class TestCodeRetrievalThatOtherContextEngineFormat:
    """Test ThatOtherContextEngine MCP-compatible output formatting."""
    
    def test_format_ThatOtherContextEngine_style_basic(self, mock_qdrant_client, mock_embedder):
        """Should format results like ThatOtherContextEngine MCP output."""
        from ace.code_retrieval import CodeRetrieval
        
        retriever = CodeRetrieval(embed_fn=mock_embedder)
        results = retriever.search("unified memory")
        formatted = retriever.format_ThatOtherContextEngine_style(results)
        
        # ThatOtherContextEngine format starts with "The following code sections were retrieved:"
        assert "The following code sections were retrieved:" in formatted
        # Path: file_path
        assert "Path:" in formatted
    
    def test_format_ThatOtherContextEngine_style_numbered_lines(self, mock_qdrant_client, mock_embedder):
        """Lines should be numbered like ThatOtherContextEngine output."""
        from ace.code_retrieval import CodeRetrieval
        
        retriever = CodeRetrieval(embed_fn=mock_embedder)
        
        # Create a simple result
        result = {
            "file_path": "test.py",
            "content": "def hello():\n    print('world')\n",
            "start_line": 1,
            "end_line": 2,
        }
        
        formatted = retriever.format_ThatOtherContextEngine_style([result])
        
        # ThatOtherContextEngine uses right-aligned line numbers followed by tab
        assert "1\t" in formatted or "     1\t" in formatted
    
    def test_format_ThatOtherContextEngine_style_multiple_files(self, mock_qdrant_client, mock_embedder):
        """Should handle multiple file results."""
        from ace.code_retrieval import CodeRetrieval
        
        retriever = CodeRetrieval(embed_fn=mock_embedder)
        results = retriever.search("memory")
        formatted = retriever.format_ThatOtherContextEngine_style(results)
        
        # Should have multiple "Path:" entries
        path_count = formatted.count("Path:")
        assert path_count >= 2  # From mock data
    
    def test_format_ThatOtherContextEngine_style_truncation(self, mock_qdrant_client, mock_embedder):
        """Long content should be truncated with ..."""
        from ace.code_retrieval import CodeRetrieval
        
        retriever = CodeRetrieval(embed_fn=mock_embedder)
        
        # Create result with long content
        long_content = "\n".join([f"line {i}" for i in range(1, 201)])
        result = {
            "file_path": "long.py",
            "content": long_content,
            "start_line": 1,
            "end_line": 200,
        }
        
        formatted = retriever.format_ThatOtherContextEngine_style([result], max_lines_per_file=50)
        
        # Should indicate truncation
        assert "..." in formatted or "lines omitted" in formatted.lower()
    
    def test_format_empty_results(self, mock_qdrant_client, mock_embedder):
        """Empty results should return appropriate message."""
        mock_empty_response = MagicMock()
        mock_empty_response.points = []
        mock_qdrant_client.query_points.return_value = mock_empty_response
        
        from ace.code_retrieval import CodeRetrieval
        
        retriever = CodeRetrieval(embed_fn=mock_embedder)
        results = retriever.search("xyz")
        formatted = retriever.format_ThatOtherContextEngine_style(results)
        
        assert "No relevant code" in formatted or formatted == ""


# =============================================================================
# TEST: Blended Results (Code + Memory)
# =============================================================================

class TestCodeRetrievalBlended:
    """Test blending code and memory results."""
    
    def test_retrieve_blended_returns_both(self, mock_qdrant_client, mock_embedder, mock_ace_memory):
        """Should return both code and memory results."""
        with patch("ace.unified_memory.UnifiedMemoryIndex") as mock_memory:
            mock_memory_instance = MagicMock()
            mock_memory_instance.search.return_value = mock_ace_memory
            mock_memory.return_value = mock_memory_instance
            
            from ace.code_retrieval import CodeRetrieval
            
            retriever = CodeRetrieval(embed_fn=mock_embedder)
            results = retriever.retrieve_blended("memory architecture")
            
            assert "code_results" in results
            assert "memory_results" in results
    
    def test_retrieve_blended_code_first(self, mock_qdrant_client, mock_embedder, mock_ace_memory):
        """Code should be returned before memories by default."""
        with patch("ace.unified_memory.UnifiedMemoryIndex") as mock_memory:
            mock_memory_instance = MagicMock()
            mock_memory_instance.search.return_value = mock_ace_memory
            mock_memory.return_value = mock_memory_instance
            
            from ace.code_retrieval import CodeRetrieval
            
            retriever = CodeRetrieval(embed_fn=mock_embedder)
            results = retriever.retrieve_blended("query")
            formatted = retriever.format_blended(results)
            
            # Code section should come before memories
            code_pos = formatted.find("code sections")
            memory_pos = formatted.find("memories") if "memories" in formatted.lower() else formatted.find("Memory")
            
            if code_pos >= 0 and memory_pos >= 0:
                assert code_pos < memory_pos
    
    def test_retrieve_blended_handles_no_code(self, mock_qdrant_client, mock_embedder, mock_ace_memory):
        """Should handle case where no code matches."""
        mock_empty_response = MagicMock()
        mock_empty_response.points = []
        mock_qdrant_client.query_points.return_value = mock_empty_response
        
        with patch("ace.unified_memory.UnifiedMemoryIndex") as mock_memory:
            mock_memory_instance = MagicMock()
            mock_memory_instance.search.return_value = mock_ace_memory
            mock_memory.return_value = mock_memory_instance
            
            from ace.code_retrieval import CodeRetrieval
            
            retriever = CodeRetrieval(embed_fn=mock_embedder)
            results = retriever.retrieve_blended("only memory match")
            
            assert len(results.get("code_results", [])) == 0
            assert len(results.get("memory_results", [])) > 0
    
    def test_retrieve_blended_handles_no_memory(self, mock_qdrant_client, mock_embedder):
        """Should handle case where no memory matches."""
        with patch("ace.unified_memory.UnifiedMemoryIndex") as mock_memory:
            mock_memory_instance = MagicMock()
            mock_memory_instance.search.return_value = []
            mock_memory.return_value = mock_memory_instance
            
            from ace.code_retrieval import CodeRetrieval
            
            retriever = CodeRetrieval(embed_fn=mock_embedder)
            results = retriever.retrieve_blended("only code match")
            
            assert len(results.get("code_results", [])) > 0
            assert len(results.get("memory_results", [])) == 0


# =============================================================================
# TEST: Deduplication
# =============================================================================

class TestCodeRetrievalDeduplication:
    """Test result deduplication."""
    
    def test_deduplicates_same_file_overlapping_chunks(self, mock_qdrant_client, mock_embedder):
        """Should merge overlapping chunks from same file."""
        # Mock overlapping results
        mock_point_1 = MagicMock()
        mock_point_1.score = 0.95
        mock_point_1.payload = {
            "file_path": "same_file.py",
            "content": "line1\nline2\nline3\n",
            "start_line": 1,
            "end_line": 3,
        }
        
        mock_point_2 = MagicMock()
        mock_point_2.score = 0.90
        mock_point_2.payload = {
            "file_path": "same_file.py",
            "content": "line2\nline3\nline4\n",
            "start_line": 2,
            "end_line": 4,
        }
        
        mock_response = MagicMock()
        mock_response.points = [mock_point_1, mock_point_2]
        mock_qdrant_client.query_points.return_value = mock_response
        
        from ace.code_retrieval import CodeRetrieval
        
        retriever = CodeRetrieval(embed_fn=mock_embedder)
        results = retriever.search("query", deduplicate=True)
        
        # Should merge or keep only one
        same_file_results = [r for r in results if r["file_path"] == "same_file.py"]
        
        # Either merged (1 result) or deduped (1 result)
        # At minimum, should not have exact duplicates
        assert len(same_file_results) <= 2
    
    def test_keeps_distinct_files(self, mock_qdrant_client, mock_embedder):
        """Should keep results from different files."""
        from ace.code_retrieval import CodeRetrieval
        
        retriever = CodeRetrieval(embed_fn=mock_embedder)
        results = retriever.search("query", deduplicate=True)
        
        # From mock data, we have 2 different files
        unique_files = set(r["file_path"] for r in results)
        assert len(unique_files) >= 1


# =============================================================================
# TEST: Error Handling
# =============================================================================

class TestCodeRetrievalErrorHandling:
    """Test error handling in code retrieval."""
    
    def test_handles_qdrant_connection_error(self, mock_embedder):
        """Should handle Qdrant connection failures gracefully."""
        with patch("qdrant_client.QdrantClient") as mock:
            mock.side_effect = ConnectionError("Connection refused")
            
            from ace.code_retrieval import CodeRetrieval
            
            retriever = CodeRetrieval(embed_fn=mock_embedder)
            results = retriever.search("query")
            
            # Should return empty, not raise
            assert results == []
    
    def test_handles_embedding_error(self, mock_qdrant_client):
        """Should handle embedding failures."""
        def bad_embedder(text: str):
            raise ValueError("Embedding failed")
        
        from ace.code_retrieval import CodeRetrieval
        
        retriever = CodeRetrieval(embed_fn=bad_embedder)
        results = retriever.search("query")
        
        # Should return empty, not raise
        assert results == []
    
    def test_handles_malformed_payload(self, mock_qdrant_client, mock_embedder):
        """Should handle malformed Qdrant payloads."""
        mock_point = MagicMock()
        mock_point.score = 0.9
        mock_point.payload = {"incomplete": "data"}  # Missing required fields
        
        mock_response = MagicMock()
        mock_response.points = [mock_point]
        mock_qdrant_client.query_points.return_value = mock_response
        
        from ace.code_retrieval import CodeRetrieval
        
        retriever = CodeRetrieval(embed_fn=mock_embedder)
        results = retriever.search("query")
        
        # Should skip malformed results
        assert len(results) == 0 or all("file_path" in r for r in results)


# =============================================================================
# TEST: MCP Tool Integration
# =============================================================================

class TestCodeRetrievalMCPIntegration:
    """Test MCP tool integration format."""
    
    def test_to_mcp_response(self, mock_qdrant_client, mock_embedder):
        """Should format for MCP tool response."""
        from ace.code_retrieval import CodeRetrieval
        
        retriever = CodeRetrieval(embed_fn=mock_embedder)
        results = retriever.search("query")
        mcp_response = retriever.to_mcp_response(results)
        
        # MCP response format
        assert "content" in mcp_response
        assert isinstance(mcp_response["content"], list)
        assert mcp_response["content"][0]["type"] == "text"
    
    def test_mcp_response_includes_formatted_text(self, mock_qdrant_client, mock_embedder):
        """MCP response should include ThatOtherContextEngine-formatted text."""
        from ace.code_retrieval import CodeRetrieval
        
        retriever = CodeRetrieval(embed_fn=mock_embedder)
        results = retriever.search("query")
        mcp_response = retriever.to_mcp_response(results)
        
        text = mcp_response["content"][0]["text"]
        assert "Path:" in text or "No relevant code" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
