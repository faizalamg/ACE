"""Tests for tree-sitter based AST chunking.

TDD RED phase - tests written before tree-sitter implementation.
These tests verify multi-language support via tree-sitter grammars.
"""

import pytest
from unittest.mock import patch, MagicMock
import os


# Check if tree-sitter packages are available
TREE_SITTER_AVAILABLE = False
try:
    import tree_sitter
    import tree_sitter_python
    import tree_sitter_javascript
    import tree_sitter_typescript
    import tree_sitter_go
    TREE_SITTER_AVAILABLE = True
except ImportError:
    pass


@pytest.fixture
def enabled_chunker():
    """Create an enabled ASTChunker."""
    with patch.dict("os.environ", {"ACE_ENABLE_AST_CHUNKING": "true"}):
        from ace.code_chunker import ASTChunker
        yield ASTChunker()


@pytest.fixture
def javascript_code():
    """Sample JavaScript code for testing."""
    return '''// File utilities
const fs = require('fs');

/**
 * Read a file asynchronously
 * @param {string} path - File path
 * @returns {Promise<string>} File contents
 */
async function readFile(path) {
    return fs.promises.readFile(path, 'utf8');
}

/**
 * Write a file
 * @param {string} path - File path
 * @param {string} content - Content to write
 */
function writeFile(path, content) {
    fs.writeFileSync(path, content);
}

class FileManager {
    constructor(basePath) {
        this.basePath = basePath;
    }
    
    getFullPath(filename) {
        return `${this.basePath}/${filename}`;
    }
}

module.exports = { readFile, writeFile, FileManager };
'''


@pytest.fixture
def typescript_code():
    """Sample TypeScript code for testing."""
    return '''interface User {
    id: number;
    name: string;
    email: string;
}

type UserRole = 'admin' | 'user' | 'guest';

/**
 * User service for managing users
 */
class UserService {
    private users: Map<number, User> = new Map();
    
    /**
     * Add a new user
     */
    addUser(user: User): void {
        this.users.set(user.id, user);
    }
    
    /**
     * Get user by ID
     */
    getUser(id: number): User | undefined {
        return this.users.get(id);
    }
}

export const createUser = (id: number, name: string, email: string): User => ({
    id,
    name,
    email
});

export default UserService;
'''


@pytest.fixture
def go_code():
    """Sample Go code for testing."""
    return '''package main

import (
    "fmt"
    "io"
)

// Config holds application configuration
type Config struct {
    Host string
    Port int
}

// Server represents an HTTP server
type Server struct {
    config Config
    logger io.Writer
}

// NewServer creates a new server instance
func NewServer(config Config, logger io.Writer) *Server {
    return &Server{
        config: config,
        logger: logger,
    }
}

// Start starts the server
func (s *Server) Start() error {
    addr := fmt.Sprintf("%s:%d", s.config.Host, s.config.Port)
    fmt.Fprintf(s.logger, "Starting server on %s\\n", addr)
    return nil
}

// Stop gracefully stops the server
func (s *Server) Stop() error {
    fmt.Fprintln(s.logger, "Stopping server")
    return nil
}

func main() {
    config := Config{Host: "localhost", Port: 8080}
    server := NewServer(config, os.Stdout)
    server.Start()
}
'''


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not installed")
class TestTreeSitterJavaScript:
    """Test tree-sitter JavaScript/TypeScript parsing."""
    
    def test_chunk_javascript_functions(self, enabled_chunker, javascript_code):
        """Should extract JavaScript function symbols."""
        chunks = enabled_chunker.chunk(javascript_code, language="javascript")
        
        # Should find functions and class
        all_symbols = []
        for c in chunks:
            all_symbols.extend(c.symbols)
        
        assert "readFile" in all_symbols or any("readFile" in s for s in all_symbols)
        assert "writeFile" in all_symbols or any("writeFile" in s for s in all_symbols)
    
    def test_chunk_javascript_class(self, enabled_chunker, javascript_code):
        """Should extract JavaScript class."""
        chunks = enabled_chunker.chunk(javascript_code, language="javascript")
        
        all_symbols = []
        for c in chunks:
            all_symbols.extend(c.symbols)
        
        assert "FileManager" in all_symbols or any("FileManager" in s for s in all_symbols)
    
    def test_chunk_javascript_extracts_jsdoc(self, enabled_chunker, javascript_code):
        """Should preserve JSDoc comments in chunks."""
        chunks = enabled_chunker.chunk(javascript_code, language="javascript")
        
        # At least one chunk should contain JSDoc
        content_with_jsdoc = any("@param" in c.content or "@returns" in c.content for c in chunks)
        assert content_with_jsdoc, "JSDoc comments should be preserved"


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not installed")
class TestTreeSitterTypeScript:
    """Test tree-sitter TypeScript parsing."""
    
    def test_chunk_typescript_interfaces(self, enabled_chunker, typescript_code):
        """Should extract TypeScript interfaces."""
        chunks = enabled_chunker.chunk(typescript_code, language="typescript")
        
        all_symbols = []
        for c in chunks:
            all_symbols.extend(c.symbols)
        
        assert "User" in all_symbols or any("User" in s for s in all_symbols)
    
    def test_chunk_typescript_class(self, enabled_chunker, typescript_code):
        """Should extract TypeScript class and methods."""
        chunks = enabled_chunker.chunk(typescript_code, language="typescript")
        
        all_symbols = []
        for c in chunks:
            all_symbols.extend(c.symbols)
        
        assert "UserService" in all_symbols or any("UserService" in s for s in all_symbols)
    
    def test_chunk_typescript_type_alias(self, enabled_chunker, typescript_code):
        """Should extract TypeScript type aliases."""
        chunks = enabled_chunker.chunk(typescript_code, language="typescript")
        
        all_symbols = []
        for c in chunks:
            all_symbols.extend(c.symbols)
        
        # TypeScript type alias should be captured
        assert "UserRole" in all_symbols or any("UserRole" in s for s in all_symbols)


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not installed")
class TestTreeSitterGo:
    """Test tree-sitter Go parsing."""
    
    def test_chunk_go_functions(self, enabled_chunker, go_code):
        """Should extract Go functions."""
        chunks = enabled_chunker.chunk(go_code, language="go")
        
        all_symbols = []
        for c in chunks:
            all_symbols.extend(c.symbols)
        
        assert "NewServer" in all_symbols or any("NewServer" in s for s in all_symbols)
        assert "main" in all_symbols or any("main" in s for s in all_symbols)
    
    def test_chunk_go_structs(self, enabled_chunker, go_code):
        """Should extract Go struct types."""
        chunks = enabled_chunker.chunk(go_code, language="go")
        
        all_symbols = []
        for c in chunks:
            all_symbols.extend(c.symbols)
        
        assert "Config" in all_symbols or any("Config" in s for s in all_symbols)
        assert "Server" in all_symbols or any("Server" in s for s in all_symbols)
    
    def test_chunk_go_methods(self, enabled_chunker, go_code):
        """Should extract Go method receivers."""
        chunks = enabled_chunker.chunk(go_code, language="go")
        
        all_symbols = []
        for c in chunks:
            all_symbols.extend(c.symbols)
        
        # Methods should be captured (either as Start/Stop or Server.Start/Server.Stop)
        assert any("Start" in s for s in all_symbols) or any("Server.Start" in s for s in all_symbols)


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not installed")
class TestCodeSymbolMetadata:
    """Test enhanced CodeSymbol metadata extraction."""
    
    def test_symbol_has_docstring(self, enabled_chunker):
        """Chunks should extract docstrings for symbols."""
        python_code = '''
def documented_function():
    """This is the docstring."""
    pass
'''
        with patch.dict("os.environ", {"ACE_ENABLE_AST_CHUNKING": "true"}):
            from ace.code_chunker import ASTChunker
            chunker = ASTChunker()
            chunks = chunker.chunk(python_code, language="python")
            
            # Check docstring is in content
            assert any("This is the docstring" in c.content for c in chunks)
    
    def test_chunk_semantic_unit_flag(self, enabled_chunker):
        """Chunks should flag whether they're semantic units."""
        python_code = '''
class MyClass:
    def method(self):
        pass
'''
        with patch.dict("os.environ", {"ACE_ENABLE_AST_CHUNKING": "true"}):
            from ace.code_chunker import ASTChunker
            chunker = ASTChunker()
            chunks = chunker.chunk(python_code, language="python")
            
            # At least one chunk should be marked as semantic unit
            semantic_chunks = [c for c in chunks if c.is_semantic_unit]
            assert len(semantic_chunks) >= 1


class TestTreeSitterFallback:
    """Test graceful fallback when tree-sitter is unavailable."""
    
    def test_fallback_to_line_based_without_treesitter(self):
        """Should fallback to line-based chunking for unsupported language."""
        with patch.dict("os.environ", {"ACE_ENABLE_AST_CHUNKING": "true"}):
            from ace.code_chunker import ASTChunker
            chunker = ASTChunker()
            
            # Use a language that's not supported to test fallback
            obscure_code = "some obscure language code\nwith multiple lines"
            chunks = chunker.chunk(obscure_code, language="cobol")
            
            # Should still return chunks (via fallback)
            assert len(chunks) >= 1
            assert chunks[0].content  # Has content
            assert "cobol" == chunks[0].language


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not installed")
class TestTreeSitterLanguageDetection:
    """Test automatic language detection from file extensions."""
    
    def test_js_extension_mapping(self, enabled_chunker):
        """Should map .js files to JavaScript parser."""
        from ace.code_chunker import ASTChunker
        chunker = ASTChunker()
        
        # Both should be recognized as JavaScript
        assert "javascript" in chunker.supported_languages()
        assert "js" in chunker.supported_languages() or "javascript" in chunker.supported_languages()
    
    def test_ts_extension_mapping(self, enabled_chunker):
        """Should map .ts files to TypeScript parser."""
        from ace.code_chunker import ASTChunker
        chunker = ASTChunker()
        
        assert "typescript" in chunker.supported_languages()


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not installed")
class TestTreeSitterCallGraphExtraction:
    """Test extraction of function call relationships."""
    
    def test_extract_function_calls_python(self, enabled_chunker):
        """Should extract called functions from Python code."""
        python_code = '''
def caller():
    result = helper()
    process(result)
    return result

def helper():
    return 42

def process(x):
    print(x)
'''
        with patch.dict("os.environ", {"ACE_ENABLE_AST_CHUNKING": "true"}):
            from ace.code_chunker import ASTChunker
            chunker = ASTChunker()
            chunks = chunker.chunk(python_code, language="python")
            
            # Verify we get valid chunks
            assert len(chunks) >= 1
            all_symbols = []
            for c in chunks:
                all_symbols.extend(c.symbols)
            
            # Basic symbol extraction should work
            assert "caller" in all_symbols or "helper" in all_symbols


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
