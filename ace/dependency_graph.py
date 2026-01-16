"""Dependency graph analysis for code understanding.

Extracts imports, function calls, and dependency relationships from source code
using tree-sitter for multiple programming languages.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import re


@dataclass
class Import:
    """Represents an import statement in source code."""

    module: str
    names: List[str] = field(default_factory=list)
    alias: Optional[str] = None
    line_number: int = 0


@dataclass
class CallEdge:
    """Represents a function call relationship."""

    caller: str
    callee: str
    line_number: int = 0


class DependencyGraph:
    """Analyzes code dependencies and call graphs across multiple languages."""

    def __init__(self, analyzer=None):
        """Initialize with optional CodeAnalyzer.

        Args:
            analyzer: Optional CodeAnalyzer instance (lazy-loaded if None)
        """
        self._analyzer = analyzer

    @property
    def analyzer(self):
        """Lazy-load CodeAnalyzer on first use."""
        if self._analyzer is None:
            try:
                from ace.code_analysis import CodeAnalyzer

                self._analyzer = CodeAnalyzer()
            except ImportError:
                # CodeAnalyzer not available yet - use regex fallback
                # Set sentinel value to indicate initialization was attempted
                self._analyzer = False
        return self._analyzer if self._analyzer is not False else None

    def extract_imports(self, code: str, language: str) -> List[Import]:
        """Extract import statements from code.

        Args:
            code: Source code string
            language: Programming language (python, javascript, go, etc.)

        Returns:
            List of Import objects
        """
        # Trigger lazy initialization
        _ = self.analyzer

        language = language.lower()

        if language == "python":
            return self._extract_python_imports(code)
        elif language in ("javascript", "typescript", "js", "ts"):
            return self._extract_js_imports(code)
        elif language == "go":
            return self._extract_go_imports(code)
        else:
            # Unsupported language
            return []

    def _extract_python_imports(self, code: str) -> List[Import]:
        """Extract Python import statements using regex."""
        imports = []
        lines = code.split("\n")

        for line_num, line in enumerate(lines, start=1):
            line = line.strip()

            # Match: import os
            # Match: import os.path
            match = re.match(r"^import\s+([\w.]+)(?:\s+as\s+(\w+))?", line)
            if match:
                module = match.group(1)
                alias = match.group(2)
                imports.append(
                    Import(
                        module=module, names=[], alias=alias, line_number=line_num
                    )
                )
                continue

            # Match: from typing import List, Optional
            # Match: from pathlib import Path as P
            match = re.match(r"^from\s+([\w.]+)\s+import\s+(.+)", line)
            if match:
                module = match.group(1)
                imports_str = match.group(2)

                # Handle: from x import *
                if imports_str.strip() == "*":
                    imports.append(
                        Import(module=module, names=["*"], line_number=line_num)
                    )
                    continue

                # Parse individual imports: List, Optional, Path as P
                for item in imports_str.split(","):
                    item = item.strip()
                    # Handle 'as' alias
                    if " as " in item:
                        name, alias = item.split(" as ")
                        imports.append(
                            Import(
                                module=module,
                                names=[name.strip()],
                                alias=alias.strip(),
                                line_number=line_num,
                            )
                        )
                    else:
                        imports.append(
                            Import(module=module, names=[item], line_number=line_num)
                        )

        return imports

    def _extract_js_imports(self, code: str) -> List[Import]:
        """Extract JavaScript/TypeScript imports (ES6 and CommonJS)."""
        imports = []
        lines = code.split("\n")

        for line_num, line in enumerate(lines, start=1):
            line = line.strip()

            # Match: import { useState, useEffect } from 'react'
            # Match: import * as React from 'react'
            match = re.match(
                r"^import\s+(?:\{([^}]+)\}|\*\s+as\s+(\w+)|(\w+))\s+from\s+['\"]([^'\"]+)['\"]",
                line,
            )
            if match:
                names_str = match.group(1)
                star_alias = match.group(2)
                default_import = match.group(3)
                module = match.group(4)

                if names_str:
                    # Named imports: { useState, useEffect }
                    names = [n.strip() for n in names_str.split(",")]
                    imports.append(
                        Import(module=module, names=names, line_number=line_num)
                    )
                elif star_alias:
                    # Wildcard import: * as React
                    imports.append(
                        Import(
                            module=module,
                            names=["*"],
                            alias=star_alias,
                            line_number=line_num,
                        )
                    )
                elif default_import:
                    # Default import: import axios from 'axios'
                    imports.append(
                        Import(
                            module=module, names=[default_import], line_number=line_num
                        )
                    )
                continue

            # Match: const fs = require('fs')
            # Match: const { readFile } = require('fs')
            match = re.match(
                r"^(?:const|let|var)\s+(?:\{([^}]+)\}|(\w+))\s*=\s*require\(['\"]([^'\"]+)['\"]\)",
                line,
            )
            if match:
                names_str = match.group(1)
                default_import = match.group(2)
                module = match.group(3)

                if names_str:
                    names = [n.strip() for n in names_str.split(",")]
                    imports.append(
                        Import(module=module, names=names, line_number=line_num)
                    )
                elif default_import:
                    imports.append(
                        Import(
                            module=module,
                            names=[],
                            alias=default_import,
                            line_number=line_num,
                        )
                    )

        return imports

    def _extract_go_imports(self, code: str) -> List[Import]:
        """Extract Go import statements."""
        imports = []
        lines = code.split("\n")

        in_import_block = False
        for line_num, line in enumerate(lines, start=1):
            line = line.strip()

            # Match: import "fmt"
            match = re.match(r'^import\s+"([^"]+)"', line)
            if match:
                module = match.group(1)
                imports.append(Import(module=module, line_number=line_num))
                continue

            # Match: import (
            if re.match(r"^import\s*\(", line):
                in_import_block = True
                continue

            # Match: )
            if in_import_block and line == ")":
                in_import_block = False
                continue

            # Inside import block
            if in_import_block:
                # Match: "fmt"
                # Match: _ "github.com/user/repo"
                # Match: alias "package"
                match = re.match(r'^(?:(\w+)\s+)?"([^"]+)"', line)
                if match:
                    alias = match.group(1)
                    module = match.group(2)
                    imports.append(
                        Import(module=module, alias=alias, line_number=line_num)
                    )

        return imports

    def build_call_graph(self, code: str, language: str) -> List[CallEdge]:
        """Build function call graph from code.

        Args:
            code: Source code string
            language: Programming language

        Returns:
            List of CallEdge objects representing caller->callee relationships
        """
        language = language.lower()

        if language == "python":
            return self._build_python_call_graph(code)
        elif language in ("javascript", "typescript", "js", "ts"):
            return self._build_js_call_graph(code)
        elif language == "go":
            return self._build_go_call_graph(code)
        else:
            return []

    def _build_python_call_graph(self, code: str) -> List[CallEdge]:
        """Build call graph for Python code using regex."""
        edges = []
        lines = code.split("\n")

        current_function = None

        for line_num, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Detect function definition
            func_match = re.match(r"^def\s+(\w+)\s*\(", stripped)
            if func_match:
                current_function = func_match.group(1)
                continue

            # Skip if not inside a function
            if current_function is None:
                continue

            # Detect function calls: word(...)
            # Match: helper(), Path("..."), print(...)
            call_matches = re.finditer(r"(\w+)\s*\(", stripped)
            for match in call_matches:
                callee = match.group(1)

                # Skip keywords and control structures
                if callee in ("if", "for", "while", "with", "elif", "except", "def"):
                    continue

                edges.append(
                    CallEdge(caller=current_function, callee=callee, line_number=line_num)
                )

        return edges

    def _build_js_call_graph(self, code: str) -> List[CallEdge]:
        """Build call graph for JavaScript/TypeScript code."""
        edges = []
        lines = code.split("\n")

        current_function = None

        for line_num, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Detect function definition
            # Match: function fetchData() {
            # Match: function App() {
            func_match = re.match(r"^function\s+(\w+)\s*\(", stripped)
            if func_match:
                current_function = func_match.group(1)
                continue

            # Match arrow functions: const App = () => {
            arrow_match = re.match(r"^(?:const|let|var)\s+(\w+)\s*=\s*\(.*\)\s*=>", stripped)
            if arrow_match:
                current_function = arrow_match.group(1)
                continue

            # Skip if not inside a function
            if current_function is None:
                continue

            # Detect function calls
            call_matches = re.finditer(r"(\w+)\s*\(", stripped)
            for match in call_matches:
                callee = match.group(1)

                # Skip keywords
                if callee in ("if", "for", "while", "switch", "catch", "function"):
                    continue

                edges.append(
                    CallEdge(caller=current_function, callee=callee, line_number=line_num)
                )

            # Detect method calls: axios.get(...), fmt.Println(...)
            method_matches = re.finditer(r"(\w+)\.(\w+)\s*\(", stripped)
            for match in method_matches:
                object_name = match.group(1)
                method_name = match.group(2)
                callee = f"{object_name}.{method_name}"

                edges.append(
                    CallEdge(caller=current_function, callee=callee, line_number=line_num)
                )

        return edges

    def _build_go_call_graph(self, code: str) -> List[CallEdge]:
        """Build call graph for Go code."""
        edges = []
        lines = code.split("\n")

        current_function = None

        for line_num, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Detect function definition: func helper() int {
            func_match = re.match(r"^func\s+(\w+)\s*\(", stripped)
            if func_match:
                current_function = func_match.group(1)
                continue

            # Skip if not inside a function
            if current_function is None:
                continue

            # Detect function calls
            call_matches = re.finditer(r"(\w+)\s*\(", stripped)
            for match in call_matches:
                callee = match.group(1)

                # Skip keywords
                if callee in ("if", "for", "switch", "func", "return"):
                    continue

                edges.append(
                    CallEdge(caller=current_function, callee=callee, line_number=line_num)
                )

            # Detect method calls: fmt.Println(...), os.Exit(...)
            method_matches = re.finditer(r"(\w+)\.(\w+)\s*\(", stripped)
            for match in method_matches:
                package_name = match.group(1)
                func_name = match.group(2)
                callee = f"{package_name}.{func_name}"

                edges.append(
                    CallEdge(caller=current_function, callee=callee, line_number=line_num)
                )

        return edges

    def find_callers(self, code: str, function_name: str, language: str) -> List[str]:
        """Find all functions that call the target function.

        Args:
            code: Source code string
            function_name: Name of function to find callers for
            language: Programming language

        Returns:
            List of function names that call the target
        """
        call_graph = self.build_call_graph(code, language)

        callers = [
            edge.caller
            for edge in call_graph
            if edge.callee == function_name
        ]

        # Remove duplicates while preserving order
        seen = set()
        unique_callers = []
        for caller in callers:
            if caller not in seen:
                seen.add(caller)
                unique_callers.append(caller)

        return unique_callers

    def find_callees(self, code: str, function_name: str, language: str) -> List[str]:
        """Find all functions called by the target function.

        Args:
            code: Source code string
            function_name: Name of function to find callees for
            language: Programming language

        Returns:
            List of function names called by the target
        """
        call_graph = self.build_call_graph(code, language)

        callees = [
            edge.callee
            for edge in call_graph
            if edge.caller == function_name
        ]

        # Remove duplicates while preserving order
        seen = set()
        unique_callees = []
        for callee in callees:
            if callee not in seen:
                seen.add(callee)
                unique_callees.append(callee)

        return unique_callees

    def analyze_file(self, filepath: str) -> Dict:
        """Analyze a complete file for imports, symbols, and call graph.

        Args:
            filepath: Path to source file

        Returns:
            Dict containing:
                - imports: List of Import objects
                - symbols: List of defined symbols (functions, classes)
                - call_graph: List of CallEdge objects

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Read file content
        code = path.read_text(encoding="utf-8")

        # Detect language from extension
        extension = path.suffix.lower()
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".go": "go",
        }

        language = language_map.get(extension, "unknown")

        # Extract imports
        imports = self.extract_imports(code, language)

        # Build call graph
        call_graph = self.build_call_graph(code, language)

        # Extract symbols (function/class definitions)
        symbols = self._extract_symbols(code, language)

        return {
            "imports": imports,
            "symbols": symbols,
            "call_graph": call_graph,
        }

    def _extract_symbols(self, code: str, language: str) -> List[str]:
        """Extract function and class definitions from code.

        Args:
            code: Source code string
            language: Programming language

        Returns:
            List of symbol names (functions, classes)
        """
        symbols = []
        lines = code.split("\n")

        if language == "python":
            for line in lines:
                stripped = line.strip()
                # Match: def function_name(
                # Match: class ClassName(
                match = re.match(r"^(?:def|class)\s+(\w+)\s*[\(:]", stripped)
                if match:
                    symbols.append(match.group(1))

        elif language in ("javascript", "typescript", "js", "ts"):
            for line in lines:
                stripped = line.strip()
                # Match: function functionName(
                match = re.match(r"^function\s+(\w+)\s*\(", stripped)
                if match:
                    symbols.append(match.group(1))

                # Match: const funcName = () =>
                # Match: class ClassName {
                match = re.match(
                    r"^(?:const|let|var|class)\s+(\w+)\s*(?:=|{)", stripped
                )
                if match:
                    symbols.append(match.group(1))

        elif language == "go":
            for line in lines:
                stripped = line.strip()
                # Match: func functionName(
                match = re.match(r"^func\s+(\w+)\s*\(", stripped)
                if match:
                    symbols.append(match.group(1))

        return symbols
