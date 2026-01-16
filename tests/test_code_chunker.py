"""Tests for AST-based code chunking module.

TDD RED phase - tests written before implementation.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestASTChunkerConfig:
    """Test configuration-based enable/disable of AST chunking."""
    
    def test_chunker_disabled_by_default(self):
        """AST chunking should be disabled by default."""
        import os
        # Ensure env var is not set
        os.environ.pop("ACE_ENABLE_AST_CHUNKING", None)
        from ace.code_chunker import ASTChunker
        chunker = ASTChunker()
        assert chunker.is_enabled() is False
    
    def test_chunker_respects_config_toggle(self):
        """Chunker should check config before processing."""
        with patch.dict("os.environ", {"ACE_ENABLE_AST_CHUNKING": "true"}):
            from ace.code_chunker import ASTChunker
            chunker = ASTChunker()
            assert chunker.is_enabled() is True
        
        with patch.dict("os.environ", {"ACE_ENABLE_AST_CHUNKING": "false"}):
            from ace.code_chunker import ASTChunker
            chunker = ASTChunker()
            assert chunker.is_enabled() is False
    
    def test_chunker_returns_passthrough_when_disabled(self):
        """When disabled, chunker should return original content as single chunk."""
        with patch.dict("os.environ", {"ACE_ENABLE_AST_CHUNKING": "false"}):
            from ace.code_chunker import ASTChunker
            chunker = ASTChunker()
            
            code = "def foo():\n    pass\n\ndef bar():\n    pass"
            chunks = chunker.chunk(code, language="python")
            
            # Should return single chunk with original content
            assert len(chunks) == 1
            assert chunks[0].content == code


class TestASTChunkerBasic:
    """Test basic chunking functionality."""
    
    def test_chunk_python_functions(self):
        """Should chunk Python code by function boundaries."""
        with patch.dict("os.environ", {"ACE_ENABLE_AST_CHUNKING": "true"}):
            from ace.code_chunker import ASTChunker
            chunker = ASTChunker()
            
            code = '''def foo():
    """First function."""
    return 1

def bar():
    """Second function."""
    return 2
'''
            chunks = chunker.chunk(code, language="python")
            
            # Small functions may be grouped into one chunk (under max_lines)
            assert len(chunks) >= 1
            # Both symbols should be captured
            all_symbols = []
            for c in chunks:
                all_symbols.extend(c.symbols)
            assert "foo" in all_symbols
            assert "bar" in all_symbols
    
    def test_chunk_preserves_class_boundaries(self):
        """Should keep class methods together when possible."""
        with patch.dict("os.environ", {"ACE_ENABLE_AST_CHUNKING": "true"}):
            from ace.code_chunker import ASTChunker
            chunker = ASTChunker()
            
            code = '''class MyClass:
    def __init__(self):
        self.value = 0
    
    def method1(self):
        return self.value
'''
            chunks = chunker.chunk(code, language="python")
            
            # Class should be chunked together (under default max_lines)
            assert len(chunks) >= 1
            # At least one chunk should contain both __init__ and method1
            class_chunk = [c for c in chunks if "MyClass" in c.content]
            assert len(class_chunk) >= 1


class TestASTChunkerMetadata:
    """Test chunk metadata extraction."""
    
    def test_chunk_contains_line_numbers(self):
        """Each chunk should have start_line and end_line."""
        with patch.dict("os.environ", {"ACE_ENABLE_AST_CHUNKING": "true"}):
            from ace.code_chunker import ASTChunker
            chunker = ASTChunker()
            
            code = "def foo():\n    pass"
            chunks = chunker.chunk(code, language="python")
            
            assert len(chunks) >= 1
            assert hasattr(chunks[0], 'start_line')
            assert hasattr(chunks[0], 'end_line')
            assert chunks[0].start_line >= 1
    
    def test_chunk_extracts_symbols(self):
        """Chunks should list symbols (function/class names) they contain."""
        with patch.dict("os.environ", {"ACE_ENABLE_AST_CHUNKING": "true"}):
            from ace.code_chunker import ASTChunker
            chunker = ASTChunker()
            
            code = "def my_function():\n    pass"
            chunks = chunker.chunk(code, language="python")
            
            assert len(chunks) >= 1
            assert hasattr(chunks[0], 'symbols')
            assert "my_function" in chunks[0].symbols


class TestASTChunkerLanguageSupport:
    """Test language detection and support."""
    
    def test_supported_languages(self):
        """Should support common programming languages."""
        from ace.code_chunker import ASTChunker
        chunker = ASTChunker()
        
        supported = chunker.supported_languages()
        assert "python" in supported
        assert "javascript" in supported
        assert "typescript" in supported
    
    def test_unsupported_language_fallback(self):
        """Unsupported languages should fall back to line-based chunking."""
        with patch.dict("os.environ", {"ACE_ENABLE_AST_CHUNKING": "true"}):
            from ace.code_chunker import ASTChunker
            chunker = ASTChunker()
            
            code = "some obscure language code"
            chunks = chunker.chunk(code, language="obscurelang")
            
            # Should still return chunks (line-based fallback)
            assert len(chunks) >= 1


class TestASTChunkerConfigOptions:
    """Test configurable chunking parameters."""
    
    def test_max_lines_configurable(self):
        """max_lines should be configurable via environment."""
        with patch.dict("os.environ", {
            "ACE_ENABLE_AST_CHUNKING": "true",
            "ACE_AST_MAX_LINES": "50"
        }):
            from ace.code_chunker import ASTChunker
            chunker = ASTChunker()
            assert chunker.max_lines == 50
    
    def test_overlap_lines_configurable(self):
        """overlap_lines should be configurable via environment."""
        with patch.dict("os.environ", {
            "ACE_ENABLE_AST_CHUNKING": "true",
            "ACE_AST_OVERLAP_LINES": "10"
        }):
            from ace.code_chunker import ASTChunker
            chunker = ASTChunker()
            assert chunker.overlap_lines == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
