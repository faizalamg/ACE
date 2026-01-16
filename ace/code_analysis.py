"""ACE Code Analysis module (Phase 2A: Tree-sitter Integration).

This module provides AST-based code understanding for code-specific queries
using tree-sitter parsing. Supports Python, TypeScript, JavaScript, and Go.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import tree_sitter_go as tsgo
import tree_sitter_javascript as tsjs
import tree_sitter_python as tspython
import tree_sitter_typescript as tstype
from tree_sitter import Language, Parser, Tree


@dataclass
class CodeSymbol:
    """Represents a code symbol (function, class, method, etc.)."""

    name: str
    kind: str  # function, class, method, interface, struct
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    parameters: Optional[List[str]] = field(default_factory=list)
    return_type: Optional[str] = None
    language: str = "unknown"


class UnsupportedLanguageError(Exception):
    """Raised when attempting to parse an unsupported language."""

    pass


class CodeAnalyzer:
    """Multi-language code analyzer using tree-sitter."""

    def __init__(self):
        """Initialize language parsers."""
        self._parsers = {}
        self._languages = {}

        # Initialize Python
        self._languages["python"] = Language(tspython.language())
        self._parsers["python"] = Parser(self._languages["python"])

        # Initialize TypeScript (handles both .ts and .tsx)
        ts_language = Language(tstype.language_typescript())
        self._languages["typescript"] = ts_language
        self._parsers["typescript"] = Parser(ts_language)

        # Initialize JavaScript
        self._languages["javascript"] = Language(tsjs.language())
        self._parsers["javascript"] = Parser(self._languages["javascript"])

        # Initialize Go
        self._languages["go"] = Language(tsgo.language())
        self._parsers["go"] = Parser(self._languages["go"])

    def supports_language(self, language: str) -> bool:
        """Check if language is supported."""
        return language.lower() in self._parsers

    def supported_languages(self) -> List[str]:
        """Return list of supported languages."""
        return list(self._parsers.keys())

    def detect_language(self, filename: str) -> str:
        """Detect language from file extension."""
        suffix = Path(filename).suffix.lower()
        mapping = {
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".go": "go",
        }
        return mapping.get(suffix, "unknown")

    def parse(self, code: str, language: str) -> Tree:
        """Parse code string and return AST tree."""
        language = language.lower()
        if not self.supports_language(language):
            raise UnsupportedLanguageError(f"Language '{language}' is not supported")

        parser = self._parsers[language]
        return parser.parse(bytes(code, "utf8"))

    def parse_file(self, filepath: str) -> Tree:
        """Parse file and return AST tree (auto-detect language)."""
        language = self.detect_language(filepath)
        if not self.supports_language(language):
            raise UnsupportedLanguageError(
                f"Cannot detect supported language for file: {filepath}"
            )

        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()

        return self.parse(code, language)

    def extract_symbols(self, code: str, language: str) -> List[CodeSymbol]:
        """Extract symbols (functions, classes, methods) from code."""
        if not code.strip():
            return []

        tree = self.parse(code, language)
        language = language.lower()

        if language == "python":
            return self._extract_python_symbols(tree, code)
        elif language == "typescript":
            return self._extract_typescript_symbols(tree, code)
        elif language == "javascript":
            return self._extract_javascript_symbols(tree, code)
        elif language == "go":
            return self._extract_go_symbols(tree, code)

        return []

    def find_symbol(
        self, code: str, name: str, language: str
    ) -> Optional[CodeSymbol]:
        """Find symbol by name (supports qualified names like 'Class.method')."""
        symbols = self.extract_symbols(code, language)

        # Handle qualified names (e.g., "Calculator.add")
        if "." in name:
            parts = name.split(".")
            class_name = parts[0]
            method_name = parts[1]

            # Find the method within the class context
            for symbol in symbols:
                if symbol.name == method_name and symbol.kind == "method":
                    # Check if this method is within the specified class
                    # by checking if there's a class symbol that contains this method
                    for class_symbol in symbols:
                        if (
                            class_symbol.name == class_name
                            and class_symbol.kind == "class"
                            and class_symbol.start_line <= symbol.start_line
                            and class_symbol.end_line >= symbol.end_line
                        ):
                            return symbol
        else:
            # Simple name lookup
            for symbol in symbols:
                if symbol.name == name:
                    return symbol

        return None

    def get_symbol_body(
        self, code: str, name: str, language: str
    ) -> Optional[str]:
        """Get the source code body of a symbol."""
        symbol = self.find_symbol(code, name, language)
        if not symbol:
            return None

        lines = code.split("\n")
        # Extract lines from start_line to end_line (1-indexed)
        body_lines = lines[symbol.start_line - 1 : symbol.end_line]
        return "\n".join(body_lines)

    def _extract_python_symbols(self, tree: Tree, code: str) -> List[CodeSymbol]:
        """Extract symbols from Python code."""
        symbols = []
        root_node = tree.root_node

        def extract_docstring(node):
            """Extract docstring from function/class definition."""
            # Look for first string literal in body
            for child in node.children:
                if child.type == "block":
                    for stmt in child.children:
                        if stmt.type == "expression_statement":
                            for expr in stmt.children:
                                if expr.type == "string":
                                    # Remove quotes and triple quotes
                                    text = code[expr.start_byte : expr.end_byte]
                                    text = (
                                        text.strip('"""')
                                        .strip("'''")
                                        .strip('"')
                                        .strip("'")
                                    )
                                    return text.strip()
            return None

        def extract_parameters(node):
            """Extract parameters from function definition."""
            params = []
            for child in node.children:
                if child.type == "parameters":
                    for param_child in child.children:
                        if param_child.type in [
                            "identifier",
                            "typed_parameter",
                            "default_parameter",
                            "typed_default_parameter",
                        ]:
                            param_text = code[
                                param_child.start_byte : param_child.end_byte
                            ]
                            if param_text != "self":  # Skip 'self' for methods
                                params.append(param_text)
            return params if params else None

        def extract_return_type(node):
            """Extract return type annotation."""
            for child in node.children:
                if child.type == "type":
                    return code[child.start_byte : child.end_byte]
            return None

        def visit_node(node, parent_class=None):
            """Recursively visit AST nodes."""
            # Extract function definitions
            if node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = code[name_node.start_byte : name_node.end_byte]
                    kind = "method" if parent_class else "function"

                    symbol = CodeSymbol(
                        name=name,
                        kind=kind,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        docstring=extract_docstring(node),
                        parameters=extract_parameters(node),
                        return_type=extract_return_type(node),
                        language="python",
                    )
                    symbols.append(symbol)

            # Extract class definitions
            elif node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = code[name_node.start_byte : name_node.end_byte]

                    symbol = CodeSymbol(
                        name=name,
                        kind="class",
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        docstring=extract_docstring(node),
                        language="python",
                    )
                    symbols.append(symbol)

                    # Visit class body for methods
                    for child in node.children:
                        if child.type == "block":
                            for stmt in child.children:
                                visit_node(stmt, parent_class=name)
                    return  # Don't recurse into class body again

            # Recurse into children
            for child in node.children:
                visit_node(child, parent_class)

        visit_node(root_node)
        return symbols

    def _extract_typescript_symbols(self, tree: Tree, code: str) -> List[CodeSymbol]:
        """Extract symbols from TypeScript code."""
        symbols = []
        root_node = tree.root_node

        def extract_parameters(node):
            """Extract parameters from function/method."""
            params = []
            for child in node.children:
                if child.type == "formal_parameters":
                    for param in child.children:
                        if param.type in [
                            "required_parameter",
                            "optional_parameter",
                        ]:
                            param_text = code[param.start_byte : param.end_byte]
                            params.append(param_text)
            return params if params else None

        def extract_return_type(node):
            """Extract return type annotation."""
            for child in node.children:
                if child.type == "type_annotation":
                    return code[child.start_byte : child.end_byte].lstrip(":")
            return None

        def visit_node(node, parent_class=None):
            """Recursively visit AST nodes."""
            # Extract interface definitions
            if node.type == "interface_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = code[name_node.start_byte : name_node.end_byte]
                    symbol = CodeSymbol(
                        name=name,
                        kind="interface",
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language="typescript",
                    )
                    symbols.append(symbol)

            # Extract class definitions
            elif node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = code[name_node.start_byte : name_node.end_byte]
                    symbol = CodeSymbol(
                        name=name,
                        kind="class",
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language="typescript",
                    )
                    symbols.append(symbol)

                    # Visit class body for methods
                    for child in node.children:
                        if child.type == "class_body":
                            for stmt in child.children:
                                visit_node(stmt, parent_class=name)
                    return

            # Extract function declarations
            elif node.type == "function_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = code[name_node.start_byte : name_node.end_byte]
                    symbol = CodeSymbol(
                        name=name,
                        kind="function",
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        parameters=extract_parameters(node),
                        return_type=extract_return_type(node),
                        language="typescript",
                    )
                    symbols.append(symbol)

            # Extract method definitions
            elif node.type == "method_definition" and parent_class:
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = code[name_node.start_byte : name_node.end_byte]
                    symbol = CodeSymbol(
                        name=name,
                        kind="method",
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        parameters=extract_parameters(node),
                        return_type=extract_return_type(node),
                        language="typescript",
                    )
                    symbols.append(symbol)

            # Extract arrow functions assigned to variables
            elif node.type == "lexical_declaration":
                for child in node.children:
                    if child.type == "variable_declarator":
                        name_node = child.child_by_field_name("name")
                        value_node = child.child_by_field_name("value")
                        if (
                            name_node
                            and value_node
                            and value_node.type == "arrow_function"
                        ):
                            name = code[name_node.start_byte : name_node.end_byte]
                            symbol = CodeSymbol(
                                name=name,
                                kind="function",
                                start_line=node.start_point[0] + 1,
                                end_line=node.end_point[0] + 1,
                                parameters=extract_parameters(value_node),
                                return_type=extract_return_type(value_node),
                                language="typescript",
                            )
                            symbols.append(symbol)

            # Recurse into children
            for child in node.children:
                visit_node(child, parent_class)

        visit_node(root_node)
        return symbols

    def _extract_javascript_symbols(self, tree: Tree, code: str) -> List[CodeSymbol]:
        """Extract symbols from JavaScript code."""
        symbols = []
        root_node = tree.root_node

        def extract_parameters(node):
            """Extract parameters from function/method."""
            params = []
            for child in node.children:
                if child.type == "formal_parameters":
                    for param in child.children:
                        if param.type == "identifier":
                            param_text = code[param.start_byte : param.end_byte]
                            params.append(param_text)
            return params if params else None

        def visit_node(node, parent_class=None):
            """Recursively visit AST nodes."""
            # Extract class definitions
            if node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = code[name_node.start_byte : name_node.end_byte]
                    symbol = CodeSymbol(
                        name=name,
                        kind="class",
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language="javascript",
                    )
                    symbols.append(symbol)

                    # Visit class body for methods
                    for child in node.children:
                        if child.type == "class_body":
                            for stmt in child.children:
                                visit_node(stmt, parent_class=name)
                    return

            # Extract function declarations
            elif node.type == "function_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = code[name_node.start_byte : name_node.end_byte]
                    symbol = CodeSymbol(
                        name=name,
                        kind="function",
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        parameters=extract_parameters(node),
                        language="javascript",
                    )
                    symbols.append(symbol)

            # Extract method definitions
            elif node.type == "method_definition" and parent_class:
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = code[name_node.start_byte : name_node.end_byte]
                    symbol = CodeSymbol(
                        name=name,
                        kind="method",
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        parameters=extract_parameters(node),
                        language="javascript",
                    )
                    symbols.append(symbol)

            # Recurse into children
            for child in node.children:
                visit_node(child, parent_class)

        visit_node(root_node)
        return symbols

    def _extract_go_symbols(self, tree: Tree, code: str) -> List[CodeSymbol]:
        """Extract symbols from Go code."""
        symbols = []
        root_node = tree.root_node

        def extract_parameters(node):
            """Extract parameters from function/method."""
            params = []
            for child in node.children:
                if child.type == "parameter_list":
                    for param in child.children:
                        if param.type == "parameter_declaration":
                            param_text = code[param.start_byte : param.end_byte]
                            params.append(param_text)
            return params if params else None

        def extract_return_type(node):
            """Extract return type."""
            for child in node.children:
                if child.type in ["type_identifier", "pointer_type"]:
                    return code[child.start_byte : child.end_byte]
            return None

        def visit_node(node):
            """Recursively visit AST nodes."""
            # Extract struct definitions
            if node.type == "type_declaration":
                for child in node.children:
                    if child.type == "type_spec":
                        name_node = child.child_by_field_name("name")
                        type_node = child.child_by_field_name("type")
                        if (
                            name_node
                            and type_node
                            and type_node.type == "struct_type"
                        ):
                            name = code[name_node.start_byte : name_node.end_byte]
                            symbol = CodeSymbol(
                                name=name,
                                kind="struct",
                                start_line=node.start_point[0] + 1,
                                end_line=node.end_point[0] + 1,
                                language="go",
                            )
                            symbols.append(symbol)

            # Extract function declarations
            elif node.type == "function_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = code[name_node.start_byte : name_node.end_byte]
                    symbol = CodeSymbol(
                        name=name,
                        kind="function",
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        parameters=extract_parameters(node),
                        return_type=extract_return_type(node),
                        language="go",
                    )
                    symbols.append(symbol)

            # Extract method declarations
            elif node.type == "method_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = code[name_node.start_byte : name_node.end_byte]
                    symbol = CodeSymbol(
                        name=name,
                        kind="method",
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        parameters=extract_parameters(node),
                        return_type=extract_return_type(node),
                        language="go",
                    )
                    symbols.append(symbol)

            # Recurse into children
            for child in node.children:
                visit_node(child)

        visit_node(root_node)
        return symbols
