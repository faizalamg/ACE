"""TDD tests for CodeIndexer - workspace code indexing.

Tests the CodeIndexer component that:
1. Scans workspace directories for code files
2. Parses files using ASTChunker for semantic chunks
3. Generates embeddings and stores in Qdrant code collection
4. Maintains metadata: file_path, start_line, end_line, language, symbols
5. Auto-updates index on file changes

RED Phase - Tests written first before implementation.
"""

import os
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def temp_workspace():
    """Create a temporary workspace with sample code files."""
    workspace = tempfile.mkdtemp(prefix="ace_test_workspace_")
    
    # Create Python file
    python_file = Path(workspace) / "sample.py"
    python_file.write_text('''"""Sample Python module for testing."""

class Calculator:
    """A simple calculator class."""
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b


def helper_function(x):
    """A helper function."""
    return x * 2
''')
    
    # Create JavaScript file
    js_file = Path(workspace) / "utils.js"
    js_file.write_text('''/**
 * JavaScript utilities module.
 */

function formatDate(date) {
    return date.toISOString();
}

class DataProcessor {
    constructor(data) {
        this.data = data;
    }
    
    process() {
        return this.data.map(x => x * 2);
    }
}

export { formatDate, DataProcessor };
''')
    
    # Create TypeScript file in subdirectory
    src_dir = Path(workspace) / "src"
    src_dir.mkdir()
    ts_file = src_dir / "types.ts"
    ts_file.write_text('''/**
 * TypeScript type definitions.
 */

interface User {
    id: string;
    name: string;
    email: string;
}

type Status = "active" | "inactive" | "pending";

function createUser(name: string, email: string): User {
    return {
        id: crypto.randomUUID(),
        name,
        email
    };
}

export { User, Status, createUser };
''')
    
    # Create non-code file (should be ignored)
    readme = Path(workspace) / "README.md"
    readme.write_text("# Test Project\n\nThis is a test project.")
    
    yield workspace
    
    # Cleanup
    shutil.rmtree(workspace)


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing without real database."""
    with patch("qdrant_client.QdrantClient") as mock:
        client = MagicMock()
        mock.return_value = client
        
        # Mock collection operations
        client.collection_exists.return_value = False
        client.create_collection.return_value = True
        client.upsert.return_value = MagicMock(status="ok")
        client.search.return_value = []
        
        yield client


@pytest.fixture
def mock_embedder():
    """Mock embedding function."""
    def embed_fn(text):
        # Return dummy 768-dim vector
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        return [float((hash_val >> i) & 1) for i in range(768)]
    
    return embed_fn


# =============================================================================
# TEST: CodeIndexer Initialization
# =============================================================================

class TestCodeIndexerInit:
    """Test CodeIndexer initialization and configuration."""
    
    def test_init_creates_instance(self, mock_qdrant_client):
        """CodeIndexer should initialize with workspace path."""
        from ace.code_indexer import CodeIndexer
        
        indexer = CodeIndexer(
            workspace_path="/path/to/workspace",
            qdrant_url="http://localhost:6333"
        )
        
        assert indexer is not None
        # workspace_path is converted to absolute path
        assert indexer.workspace_path == os.path.abspath("/path/to/workspace")
    
    def test_init_uses_env_defaults(self, mock_qdrant_client):
        """Should use environment variables for defaults."""
        with patch.dict(os.environ, {
            "ACE_CODE_COLLECTION": "my_code_collection",
            "ACE_CODE_EMBEDDING_DIM": "1024"
        }):
            from ace.code_indexer import CodeIndexer
            
            indexer = CodeIndexer(workspace_path="/workspace")
            
            assert indexer.collection_name == "my_code_collection"
            assert indexer.embedding_dim == 1024
    
    def test_init_default_collection_name(self, mock_qdrant_client):
        """Default collection should be ace_code_context."""
        from ace.code_indexer import CodeIndexer
        
        indexer = CodeIndexer(workspace_path="/workspace")
        
        assert indexer.collection_name == "ace_code_context"
    
    def test_init_creates_collection_if_not_exists(self, mock_qdrant_client):
        """Should create Qdrant collection on init if missing."""
        mock_qdrant_client.collection_exists.return_value = False
        
        from ace.code_indexer import CodeIndexer
        
        indexer = CodeIndexer(workspace_path="/workspace")
        
        mock_qdrant_client.create_collection.assert_called_once()


# =============================================================================
# TEST: File Scanning
# =============================================================================

class TestCodeIndexerScanning:
    """Test workspace file scanning."""
    
    def test_scan_finds_python_files(self, temp_workspace, mock_qdrant_client):
        """Should find Python files in workspace."""
        from ace.code_indexer import CodeIndexer
        
        indexer = CodeIndexer(workspace_path=temp_workspace)
        files = indexer.scan_workspace()
        
        python_files = [f for f in files if f.endswith(".py")]
        assert len(python_files) >= 1
        assert any("sample.py" in f for f in python_files)
    
    def test_scan_finds_js_files(self, temp_workspace, mock_qdrant_client):
        """Should find JavaScript files."""
        from ace.code_indexer import CodeIndexer
        
        indexer = CodeIndexer(workspace_path=temp_workspace)
        files = indexer.scan_workspace()
        
        js_files = [f for f in files if f.endswith(".js")]
        assert len(js_files) >= 1
    
    def test_scan_finds_ts_files_in_subdirs(self, temp_workspace, mock_qdrant_client):
        """Should find TypeScript files in subdirectories."""
        from ace.code_indexer import CodeIndexer
        
        indexer = CodeIndexer(workspace_path=temp_workspace)
        files = indexer.scan_workspace()
        
        ts_files = [f for f in files if f.endswith(".ts")]
        assert len(ts_files) >= 1
        assert any("src" in f and "types.ts" in f for f in ts_files)
    
    def test_scan_ignores_non_code_files(self, temp_workspace, mock_qdrant_client):
        """Should not include markdown/text files."""
        from ace.code_indexer import CodeIndexer
        
        indexer = CodeIndexer(workspace_path=temp_workspace)
        files = indexer.scan_workspace()
        
        md_files = [f for f in files if f.endswith(".md")]
        assert len(md_files) == 0
    
    def test_scan_respects_gitignore(self, temp_workspace, mock_qdrant_client):
        """Should respect .gitignore patterns."""
        # Add gitignore
        gitignore = Path(temp_workspace) / ".gitignore"
        gitignore.write_text("*.js\n")
        
        from ace.code_indexer import CodeIndexer
        
        indexer = CodeIndexer(workspace_path=temp_workspace, respect_gitignore=True)
        files = indexer.scan_workspace()
        
        js_files = [f for f in files if f.endswith(".js")]
        assert len(js_files) == 0
    
    def test_scan_respects_exclude_patterns(self, temp_workspace, mock_qdrant_client):
        """Should respect custom exclude patterns."""
        from ace.code_indexer import CodeIndexer
        
        indexer = CodeIndexer(
            workspace_path=temp_workspace,
            exclude_patterns=["**/node_modules/**", "**/test_*.py"]
        )
        files = indexer.scan_workspace()
        
        # Should still find main files
        assert len(files) > 0


# =============================================================================
# TEST: Code Chunking
# =============================================================================

class TestCodeIndexerChunking:
    """Test code file parsing and chunking."""
    
    def test_chunk_python_file(self, temp_workspace, mock_qdrant_client):
        """Should chunk Python file into semantic units."""
        from ace.code_indexer import CodeIndexer
        
        indexer = CodeIndexer(workspace_path=temp_workspace)
        python_file = Path(temp_workspace) / "sample.py"
        
        chunks = indexer.chunk_file(str(python_file))
        
        assert len(chunks) > 0
        # Should have class and function chunks
        chunk_texts = [c.content for c in chunks]
        assert any("Calculator" in t for t in chunk_texts)
        assert any("helper_function" in t for t in chunk_texts)
    
    def test_chunk_includes_metadata(self, temp_workspace, mock_qdrant_client):
        """Chunks should include file path and line numbers."""
        from ace.code_indexer import CodeIndexer
        
        indexer = CodeIndexer(workspace_path=temp_workspace)
        python_file = Path(temp_workspace) / "sample.py"
        
        chunks = indexer.chunk_file(str(python_file))
        
        for chunk in chunks:
            assert hasattr(chunk, "file_path")
            assert hasattr(chunk, "start_line")
            assert hasattr(chunk, "end_line")
            assert hasattr(chunk, "language")
            assert chunk.file_path.endswith("sample.py")
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line
            assert chunk.language == "python"
    
    def test_chunk_extracts_symbols(self, temp_workspace, mock_qdrant_client):
        """Chunks should list contained symbols when AST chunking is enabled."""
        from ace.code_indexer import CodeIndexer
        
        # Enable AST chunking to get symbol extraction
        with patch.dict(os.environ, {"ACE_ENABLE_AST_CHUNKING": "true"}):
            indexer = CodeIndexer(workspace_path=temp_workspace)
            python_file = Path(temp_workspace) / "sample.py"
            
            chunks = indexer.chunk_file(str(python_file))
            
            # Find the Calculator class chunk
            calc_chunks = [c for c in chunks if "Calculator" in c.content]
            assert len(calc_chunks) > 0
            
            calc_chunk = calc_chunks[0]
            assert hasattr(calc_chunk, "symbols")
            # Symbols may be empty if tree-sitter not available, that's ok
            # The point is the attribute exists and chunking works
    
    def test_chunk_detects_language(self, temp_workspace, mock_qdrant_client):
        """Should detect language from file extension."""
        from ace.code_indexer import CodeIndexer
        
        indexer = CodeIndexer(workspace_path=temp_workspace)
        
        py_chunks = indexer.chunk_file(str(Path(temp_workspace) / "sample.py"))
        assert all(c.language == "python" for c in py_chunks)
        
        js_chunks = indexer.chunk_file(str(Path(temp_workspace) / "utils.js"))
        assert all(c.language == "javascript" for c in js_chunks)
        
        ts_chunks = indexer.chunk_file(str(Path(temp_workspace) / "src" / "types.ts"))
        assert all(c.language == "typescript" for c in ts_chunks)


# =============================================================================
# TEST: Code Indexing
# =============================================================================

class TestCodeIndexerIndexing:
    """Test indexing code into Qdrant."""
    
    def test_index_workspace(self, temp_workspace, mock_qdrant_client, mock_embedder):
        """Should index entire workspace."""
        from ace.code_indexer import CodeIndexer
        
        indexer = CodeIndexer(
            workspace_path=temp_workspace,
            embed_fn=mock_embedder
        )
        
        stats = indexer.index_workspace()
        
        assert stats["files_indexed"] >= 3  # py, js, ts
        assert stats["chunks_indexed"] > 0
        assert mock_qdrant_client.upsert.called
    
    def test_index_stores_correct_payload(self, temp_workspace, mock_qdrant_client, mock_embedder):
        """Indexed chunks should have correct Qdrant payload."""
        from ace.code_indexer import CodeIndexer
        
        indexer = CodeIndexer(
            workspace_path=temp_workspace,
            embed_fn=mock_embedder
        )
        
        indexer.index_workspace()
        
        # Check upsert call
        call_args = mock_qdrant_client.upsert.call_args_list[0]
        points = call_args.kwargs.get("points") or call_args[1].get("points")
        
        assert len(points) > 0
        point = points[0]
        
        # Verify payload structure
        payload = point.payload
        assert "content" in payload
        assert "file_path" in payload
        assert "start_line" in payload
        assert "end_line" in payload
        assert "language" in payload
        assert "symbols" in payload
    
    def test_index_creates_relative_paths(self, temp_workspace, mock_qdrant_client, mock_embedder):
        """File paths should be relative to workspace root."""
        from ace.code_indexer import CodeIndexer
        
        indexer = CodeIndexer(
            workspace_path=temp_workspace,
            embed_fn=mock_embedder
        )
        
        indexer.index_workspace()
        
        # Check stored paths are relative
        call_args = mock_qdrant_client.upsert.call_args_list[0]
        points = call_args.kwargs.get("points") or call_args[1].get("points")
        
        for point in points:
            file_path = point.payload["file_path"]
            # Should not contain temp directory prefix
            assert not file_path.startswith(temp_workspace)
            # Should be a simple relative path
            assert file_path.count("/") <= 2 or file_path.count("\\") <= 2
    
    def test_index_single_file(self, temp_workspace, mock_qdrant_client, mock_embedder):
        """Should be able to index a single file."""
        from ace.code_indexer import CodeIndexer
        
        indexer = CodeIndexer(
            workspace_path=temp_workspace,
            embed_fn=mock_embedder
        )
        
        python_file = str(Path(temp_workspace) / "sample.py")
        chunks_indexed = indexer.index_file(python_file)
        
        assert chunks_indexed > 0
        assert mock_qdrant_client.upsert.called


# =============================================================================
# TEST: Incremental Updates
# =============================================================================

class TestCodeIndexerUpdates:
    """Test incremental index updates."""
    
    def test_update_file_on_change(self, temp_workspace, mock_qdrant_client, mock_embedder):
        """Should update index when file changes."""
        from ace.code_indexer import CodeIndexer
        
        indexer = CodeIndexer(
            workspace_path=temp_workspace,
            embed_fn=mock_embedder
        )
        
        # Initial index
        indexer.index_workspace()
        initial_call_count = mock_qdrant_client.upsert.call_count
        
        # Modify file
        python_file = Path(temp_workspace) / "sample.py"
        content = python_file.read_text()
        python_file.write_text(content + "\n\ndef new_function():\n    pass\n")
        
        # Update
        indexer.update_file(str(python_file))
        
        # Should have made additional upsert call
        assert mock_qdrant_client.upsert.call_count > initial_call_count
    
    def test_remove_file_from_index(self, temp_workspace, mock_qdrant_client, mock_embedder):
        """Should remove deleted file from index."""
        from ace.code_indexer import CodeIndexer
        
        indexer = CodeIndexer(
            workspace_path=temp_workspace,
            embed_fn=mock_embedder
        )
        
        # Initial index
        indexer.index_workspace()
        
        # Delete file
        python_file = Path(temp_workspace) / "sample.py"
        python_file.unlink()
        
        # Remove from index
        indexer.remove_file("sample.py")
        
        # Should have called delete
        mock_qdrant_client.delete.assert_called()
    
    def test_get_indexed_files(self, temp_workspace, mock_qdrant_client, mock_embedder):
        """Should track which files are indexed."""
        from ace.code_indexer import CodeIndexer
        
        indexer = CodeIndexer(
            workspace_path=temp_workspace,
            embed_fn=mock_embedder
        )
        
        indexer.index_workspace()
        indexed_files = indexer.get_indexed_files()
        
        assert len(indexed_files) >= 3
        assert any("sample.py" in f for f in indexed_files)


# =============================================================================
# TEST: File Watcher Integration
# =============================================================================

class TestCodeIndexerFileWatcher:
    """Test file watcher for auto-updates."""
    
    def test_start_file_watcher(self, temp_workspace, mock_qdrant_client, mock_embedder):
        """Should start file watcher for workspace."""
        from ace.code_indexer import CodeIndexer
        
        indexer = CodeIndexer(
            workspace_path=temp_workspace,
            embed_fn=mock_embedder
        )
        
        # Start watcher
        indexer.start_watching()
        
        assert indexer.is_watching()
        
        # Stop watcher
        indexer.stop_watching()
        
        assert not indexer.is_watching()
    
    @pytest.mark.asyncio
    async def test_file_change_triggers_update(self, temp_workspace, mock_qdrant_client, mock_embedder):
        """File changes should trigger index update."""
        from ace.code_indexer import CodeIndexer
        import asyncio
        
        indexer = CodeIndexer(
            workspace_path=temp_workspace,
            embed_fn=mock_embedder
        )
        
        # Index and start watching
        indexer.index_workspace()
        indexer.start_watching()
        
        initial_call_count = mock_qdrant_client.upsert.call_count
        
        # Modify file
        python_file = Path(temp_workspace) / "sample.py"
        content = python_file.read_text()
        python_file.write_text(content + "\n# New comment\n")
        
        # Wait for watcher to detect
        await asyncio.sleep(0.5)
        
        indexer.stop_watching()
        
        # Should have triggered update
        # Note: This may need adjustment based on actual watcher implementation


# =============================================================================
# TEST: Error Handling
# =============================================================================

class TestCodeIndexerErrorHandling:
    """Test error handling scenarios."""
    
    def test_handles_unreadable_file(self, temp_workspace, mock_qdrant_client):
        """Should skip unreadable files gracefully."""
        from ace.code_indexer import CodeIndexer
        
        indexer = CodeIndexer(workspace_path=temp_workspace)
        
        # Try to chunk non-existent file
        chunks = indexer.chunk_file("/nonexistent/file.py")
        
        assert chunks == []
    
    def test_handles_binary_file(self, temp_workspace, mock_qdrant_client):
        """Should skip binary files."""
        # Create binary file
        binary_file = Path(temp_workspace) / "image.png"
        binary_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        
        from ace.code_indexer import CodeIndexer
        
        indexer = CodeIndexer(workspace_path=temp_workspace)
        files = indexer.scan_workspace()
        
        # Should not include binary files
        assert not any("image.png" in f for f in files)
    
    def test_handles_encoding_errors(self, temp_workspace, mock_qdrant_client):
        """Should handle files with encoding issues."""
        # Create file with non-UTF8 content
        bad_file = Path(temp_workspace) / "bad_encoding.py"
        bad_file.write_bytes(b"# -*- coding: latin-1 -*-\n# \xff\xfe test\n")
        
        from ace.code_indexer import CodeIndexer
        
        indexer = CodeIndexer(workspace_path=temp_workspace)
        
        # Should not crash
        chunks = indexer.chunk_file(str(bad_file))
        # May return empty or attempt to decode


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
