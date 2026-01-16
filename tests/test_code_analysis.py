"""Tests for ACE Code Analysis module (Phase 2A: Tree-sitter Integration).

This module tests the CodeAnalyzer class which provides AST-based code understanding
for code-specific queries using tree-sitter parsing.

TDD: Tests written first, implementation follows.
"""

import pytest
import tempfile
import os
from pathlib import Path

# Check for optional tree-sitter dependencies
try:
    import tree_sitter_go
    import tree_sitter_python
    import tree_sitter_typescript
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

# Skip all tests in this module if tree-sitter is not available
pytestmark = pytest.mark.skipif(
    not TREE_SITTER_AVAILABLE,
    reason="tree-sitter dependencies not installed (tree_sitter_go, tree_sitter_python, tree_sitter_typescript)"
)


# Sample code fixtures for testing
PYTHON_SAMPLE = '''
"""Module docstring."""

import os
from typing import List, Optional


class Calculator:
    """A simple calculator class."""

    def __init__(self, value: int = 0):
        self.value = value

    def add(self, x: int) -> int:
        """Add x to the current value."""
        self.value += x
        return self.value

    def subtract(self, x: int) -> int:
        """Subtract x from the current value."""
        self.value -= x
        return self.value


def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


async def fetch_data(url: str) -> dict:
    """Async function to fetch data."""
    pass


GLOBAL_CONSTANT = 42
'''

TYPESCRIPT_SAMPLE = '''
import { useState, useEffect } from 'react';

interface User {
    id: number;
    name: string;
    email?: string;
}

class UserService {
    private users: User[] = [];

    constructor() {
        this.users = [];
    }

    async getUser(id: number): Promise<User | null> {
        return this.users.find(u => u.id === id) || null;
    }

    addUser(user: User): void {
        this.users.push(user);
    }
}

function formatUserName(user: User): string {
    return user.name.toUpperCase();
}

const fetchUsers = async (): Promise<User[]> => {
    return [];
};

export { UserService, formatUserName };
'''

GO_SAMPLE = '''
package main

import (
    "fmt"
    "net/http"
)

type Server struct {
    port int
    host string
}

func (s *Server) Start() error {
    addr := fmt.Sprintf("%s:%d", s.host, s.port)
    return http.ListenAndServe(addr, nil)
}

func NewServer(host string, port int) *Server {
    return &Server{host: host, port: port}
}

func main() {
    server := NewServer("localhost", 8080)
    server.Start()
}
'''


class TestCodeAnalyzerInit:
    """Tests for CodeAnalyzer initialization."""

    def test_code_analyzer_initialization(self):
        """Test that CodeAnalyzer initializes correctly."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        assert analyzer is not None

    def test_code_analyzer_supports_python(self):
        """Test that CodeAnalyzer supports Python by default."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        assert analyzer.supports_language("python")

    def test_code_analyzer_supports_typescript(self):
        """Test that CodeAnalyzer supports TypeScript."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        assert analyzer.supports_language("typescript")

    def test_code_analyzer_supports_go(self):
        """Test that CodeAnalyzer supports Go."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        assert analyzer.supports_language("go")

    def test_code_analyzer_supported_languages(self):
        """Test that supported_languages returns expected languages."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        languages = analyzer.supported_languages()
        assert "python" in languages
        assert "typescript" in languages
        assert "go" in languages


class TestPythonParsing:
    """Tests for Python code parsing."""

    def test_parse_python_file(self):
        """Test parsing a Python file."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        tree = analyzer.parse(PYTHON_SAMPLE, language="python")

        assert tree is not None
        assert tree.root_node is not None

    def test_parse_python_file_from_path(self):
        """Test parsing a Python file from file path."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(PYTHON_SAMPLE)
            f.flush()
            temp_path = f.name

        try:
            tree = analyzer.parse_file(temp_path)
            assert tree is not None
            assert tree.root_node is not None
        finally:
            os.unlink(temp_path)

    def test_detect_language_from_extension(self):
        """Test language detection from file extension."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        assert analyzer.detect_language("test.py") == "python"
        assert analyzer.detect_language("test.ts") == "typescript"
        assert analyzer.detect_language("test.tsx") == "typescript"
        assert analyzer.detect_language("test.go") == "go"
        assert analyzer.detect_language("test.js") == "javascript"


class TestSymbolExtraction:
    """Tests for symbol extraction from code."""

    def test_extract_functions_from_python(self):
        """Test extracting function definitions from Python."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        symbols = analyzer.extract_symbols(PYTHON_SAMPLE, language="python")

        function_names = [s.name for s in symbols if s.kind == "function"]
        assert "fibonacci" in function_names
        assert "fetch_data" in function_names

    def test_extract_classes_from_python(self):
        """Test extracting class definitions from Python."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        symbols = analyzer.extract_symbols(PYTHON_SAMPLE, language="python")

        class_names = [s.name for s in symbols if s.kind == "class"]
        assert "Calculator" in class_names

    def test_extract_methods_from_python(self):
        """Test extracting method definitions from Python classes."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        symbols = analyzer.extract_symbols(PYTHON_SAMPLE, language="python")

        method_names = [s.name for s in symbols if s.kind == "method"]
        assert "__init__" in method_names
        assert "add" in method_names
        assert "subtract" in method_names

    def test_symbol_has_line_numbers(self):
        """Test that extracted symbols include line number information."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        symbols = analyzer.extract_symbols(PYTHON_SAMPLE, language="python")

        for symbol in symbols:
            assert hasattr(symbol, 'start_line')
            assert hasattr(symbol, 'end_line')
            assert symbol.start_line >= 1
            assert symbol.end_line >= symbol.start_line

    def test_symbol_has_docstring(self):
        """Test that extracted symbols include docstrings when available."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        symbols = analyzer.extract_symbols(PYTHON_SAMPLE, language="python")

        calculator = next((s for s in symbols if s.name == "Calculator"), None)
        assert calculator is not None
        assert calculator.docstring is not None
        assert "simple calculator" in calculator.docstring.lower()


class TestTypeScriptParsing:
    """Tests for TypeScript code parsing."""

    def test_parse_typescript_file(self):
        """Test parsing a TypeScript file."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        tree = analyzer.parse(TYPESCRIPT_SAMPLE, language="typescript")

        assert tree is not None
        assert tree.root_node is not None

    def test_extract_interfaces_from_typescript(self):
        """Test extracting interface definitions from TypeScript."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        symbols = analyzer.extract_symbols(TYPESCRIPT_SAMPLE, language="typescript")

        interface_names = [s.name for s in symbols if s.kind == "interface"]
        assert "User" in interface_names

    def test_extract_classes_from_typescript(self):
        """Test extracting class definitions from TypeScript."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        symbols = analyzer.extract_symbols(TYPESCRIPT_SAMPLE, language="typescript")

        class_names = [s.name for s in symbols if s.kind == "class"]
        assert "UserService" in class_names

    def test_extract_functions_from_typescript(self):
        """Test extracting function definitions from TypeScript."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        symbols = analyzer.extract_symbols(TYPESCRIPT_SAMPLE, language="typescript")

        function_names = [s.name for s in symbols if s.kind == "function"]
        assert "formatUserName" in function_names


class TestGoParsing:
    """Tests for Go code parsing."""

    def test_parse_go_file(self):
        """Test parsing a Go file."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        tree = analyzer.parse(GO_SAMPLE, language="go")

        assert tree is not None
        assert tree.root_node is not None

    def test_extract_structs_from_go(self):
        """Test extracting struct definitions from Go."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        symbols = analyzer.extract_symbols(GO_SAMPLE, language="go")

        struct_names = [s.name for s in symbols if s.kind == "struct"]
        assert "Server" in struct_names

    def test_extract_functions_from_go(self):
        """Test extracting function definitions from Go."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        symbols = analyzer.extract_symbols(GO_SAMPLE, language="go")

        function_names = [s.name for s in symbols if s.kind == "function"]
        assert "NewServer" in function_names
        assert "main" in function_names

    def test_extract_methods_from_go(self):
        """Test extracting method definitions from Go."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        symbols = analyzer.extract_symbols(GO_SAMPLE, language="go")

        method_names = [s.name for s in symbols if s.kind == "method"]
        assert "Start" in method_names


class TestCodeSymbolDataclass:
    """Tests for the CodeSymbol dataclass."""

    def test_code_symbol_creation(self):
        """Test creating a CodeSymbol instance."""
        from ace.code_analysis import CodeSymbol

        symbol = CodeSymbol(
            name="test_function",
            kind="function",
            start_line=1,
            end_line=10,
            docstring="Test docstring",
            parameters=["x: int", "y: str"],
            return_type="bool",
            language="python"
        )

        assert symbol.name == "test_function"
        assert symbol.kind == "function"
        assert symbol.start_line == 1
        assert symbol.end_line == 10
        assert symbol.docstring == "Test docstring"
        assert symbol.parameters == ["x: int", "y: str"]
        assert symbol.return_type == "bool"

    def test_code_symbol_repr(self):
        """Test CodeSymbol string representation."""
        from ace.code_analysis import CodeSymbol

        symbol = CodeSymbol(
            name="MyClass",
            kind="class",
            start_line=5,
            end_line=50,
            language="python"
        )

        repr_str = repr(symbol)
        assert "MyClass" in repr_str
        assert "class" in repr_str


class TestCodeAnalyzerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_parse_empty_code(self):
        """Test parsing empty code string."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        tree = analyzer.parse("", language="python")
        assert tree is not None

    def test_parse_invalid_syntax(self):
        """Test parsing code with syntax errors."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        invalid_python = "def broken("
        tree = analyzer.parse(invalid_python, language="python")
        # Tree-sitter is error-tolerant, should still parse
        assert tree is not None
        assert tree.root_node.has_error

    def test_extract_symbols_empty_code(self):
        """Test extracting symbols from empty code."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        symbols = analyzer.extract_symbols("", language="python")
        assert symbols == []

    def test_unsupported_language(self):
        """Test handling unsupported language."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        assert not analyzer.supports_language("cobol")

    def test_parse_unsupported_language_raises(self):
        """Test that parsing unsupported language raises error."""
        from ace.code_analysis import CodeAnalyzer, UnsupportedLanguageError

        analyzer = CodeAnalyzer()
        with pytest.raises(UnsupportedLanguageError):
            analyzer.parse("some code", language="cobol")


class TestFindSymbolByName:
    """Tests for finding symbols by name."""

    def test_find_symbol_by_name(self):
        """Test finding a specific symbol by name."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        symbol = analyzer.find_symbol(PYTHON_SAMPLE, "Calculator", language="python")

        assert symbol is not None
        assert symbol.name == "Calculator"
        assert symbol.kind == "class"

    def test_find_nested_symbol_by_name(self):
        """Test finding a nested symbol by qualified name."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        symbol = analyzer.find_symbol(PYTHON_SAMPLE, "Calculator.add", language="python")

        assert symbol is not None
        assert symbol.name == "add"
        assert symbol.kind == "method"

    def test_find_symbol_not_found(self):
        """Test that finding non-existent symbol returns None."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        symbol = analyzer.find_symbol(PYTHON_SAMPLE, "NonExistentClass", language="python")
        assert symbol is None


class TestGetSymbolBody:
    """Tests for extracting symbol body content."""

    def test_get_function_body(self):
        """Test extracting the body of a function."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        body = analyzer.get_symbol_body(PYTHON_SAMPLE, "fibonacci", language="python")

        assert body is not None
        assert "def fibonacci" in body
        assert "return fibonacci(n - 1)" in body

    def test_get_class_body(self):
        """Test extracting the body of a class."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        body = analyzer.get_symbol_body(PYTHON_SAMPLE, "Calculator", language="python")

        assert body is not None
        assert "class Calculator" in body
        assert "def add" in body
        assert "def subtract" in body


class TestSymbolSignature:
    """Tests for symbol signature extraction."""

    def test_function_signature_python(self):
        """Test extracting function signature from Python."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        symbols = analyzer.extract_symbols(PYTHON_SAMPLE, language="python")

        fibonacci = next((s for s in symbols if s.name == "fibonacci"), None)
        assert fibonacci is not None
        assert fibonacci.parameters is not None
        # Should have 'n: int' parameter
        assert any("n" in p for p in fibonacci.parameters)

    def test_method_signature_python(self):
        """Test extracting method signature from Python."""
        from ace.code_analysis import CodeAnalyzer

        analyzer = CodeAnalyzer()
        symbols = analyzer.extract_symbols(PYTHON_SAMPLE, language="python")

        add_method = next((s for s in symbols if s.name == "add"), None)
        assert add_method is not None
        # Methods have 'self' parameter (may or may not be included)
        assert add_method.parameters is not None
