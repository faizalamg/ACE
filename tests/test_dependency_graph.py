"""Tests for dependency graph analysis.

Following TDD protocol - these tests are written FIRST and should FAIL.
"""

import pytest
from pathlib import Path
from ace.dependency_graph import (
    Import,
    CallEdge,
    DependencyGraph,
)


# Sample Python code for testing
SAMPLE_PYTHON = """
import os
from typing import List, Optional
from pathlib import Path

def helper():
    return 42

def main():
    result = helper()
    path = Path(".")
    print(result)
"""

SAMPLE_JAVASCRIPT = """
import { useState, useEffect } from 'react';
import axios from 'axios';
const fs = require('fs');

function fetchData() {
    return axios.get('/api/data');
}

function App() {
    const [data, setData] = useState(null);
    useEffect(() => {
        fetchData().then(setData);
    }, []);
    return data;
}
"""

SAMPLE_GO = """
package main

import (
    "fmt"
    "os"
    "github.com/user/repo/pkg"
)

func helper() int {
    return 42
}

func main() {
    result := helper()
    fmt.Println(result)
    os.Exit(0)
}
"""


class TestImportExtraction:
    """Test import extraction from different languages."""

    def test_extract_python_imports(self):
        """Extract Python import statements."""
        graph = DependencyGraph()
        imports = graph.extract_imports(SAMPLE_PYTHON, "python")

        # Should extract: os, typing.List, typing.Optional, pathlib.Path
        assert len(imports) >= 4

        # Check for 'import os'
        os_import = next((i for i in imports if i.module == "os"), None)
        assert os_import is not None
        assert os_import.alias is None

        # Check for 'from typing import List, Optional'
        typing_imports = [i for i in imports if i.module == "typing"]
        assert len(typing_imports) == 2
        imported_names = {i.names[0] for i in typing_imports if i.names}
        assert "List" in imported_names
        assert "Optional" in imported_names

        # Check for 'from pathlib import Path'
        pathlib_import = next((i for i in imports if i.module == "pathlib"), None)
        assert pathlib_import is not None
        assert "Path" in pathlib_import.names

    def test_extract_javascript_imports(self):
        """Extract JavaScript/TypeScript ES6 imports and CommonJS require."""
        graph = DependencyGraph()
        imports = graph.extract_imports(SAMPLE_JAVASCRIPT, "javascript")

        # Should extract: react, axios, fs
        assert len(imports) >= 3

        # Check for 'import { useState, useEffect } from 'react''
        react_import = next((i for i in imports if "react" in i.module), None)
        assert react_import is not None
        assert "useState" in react_import.names
        assert "useEffect" in react_import.names

        # Check for 'import axios from 'axios''
        axios_import = next((i for i in imports if "axios" in i.module), None)
        assert axios_import is not None

        # Check for 'const fs = require('fs')'
        fs_import = next((i for i in imports if "fs" in i.module), None)
        assert fs_import is not None

    def test_extract_go_imports(self):
        """Extract Go import statements."""
        graph = DependencyGraph()
        imports = graph.extract_imports(SAMPLE_GO, "go")

        # Should extract: fmt, os, github.com/user/repo/pkg
        assert len(imports) >= 3

        # Check standard library imports
        fmt_import = next((i for i in imports if i.module == "fmt"), None)
        assert fmt_import is not None

        os_import = next((i for i in imports if i.module == "os"), None)
        assert os_import is not None

        # Check external package import
        pkg_import = next((i for i in imports if "github.com" in i.module), None)
        assert pkg_import is not None

    def test_extract_imports_unsupported_language(self):
        """Should handle unsupported languages gracefully."""
        graph = DependencyGraph()
        imports = graph.extract_imports("some code", "rust")

        # Should return empty list for unsupported languages
        assert imports == []


class TestCallGraph:
    """Test call graph construction."""

    def test_build_python_call_graph(self):
        """Build call graph for Python code."""
        graph = DependencyGraph()
        call_edges = graph.build_call_graph(SAMPLE_PYTHON, "python")

        # Should find: main->helper, main->Path, main->print
        assert len(call_edges) >= 3

        # Find edges from 'main' function
        main_calls = [e for e in call_edges if e.caller == "main"]
        callees = {e.callee for e in main_calls}

        assert "helper" in callees
        assert "Path" in callees
        assert "print" in callees

        # Helper function should have no calls (returns constant)
        helper_calls = [e for e in call_edges if e.caller == "helper"]
        assert len(helper_calls) == 0

    def test_build_javascript_call_graph(self):
        """Build call graph for JavaScript code."""
        graph = DependencyGraph()
        call_edges = graph.build_call_graph(SAMPLE_JAVASCRIPT, "javascript")

        # Should find: fetchData->axios.get, App->useState, App->useEffect, App->fetchData
        assert len(call_edges) >= 4

        # Check fetchData calls axios methods
        fetchdata_calls = [e for e in call_edges if "fetchData" in e.caller]
        assert any("axios" in e.callee or "get" in e.callee for e in fetchdata_calls)

        # Check App uses hooks
        app_calls = [e for e in call_edges if "App" in e.caller]
        callees = {e.callee for e in app_calls}
        assert "useState" in callees or "useEffect" in callees

    def test_build_go_call_graph(self):
        """Build call graph for Go code."""
        graph = DependencyGraph()
        call_edges = graph.build_call_graph(SAMPLE_GO, "go")

        # Should find: main->helper, main->fmt.Println, main->os.Exit
        assert len(call_edges) >= 3

        # Find edges from 'main' function
        main_calls = [e for e in call_edges if e.caller == "main"]
        callees = {e.callee for e in main_calls}

        assert "helper" in callees
        assert any("Println" in c for c in callees)
        assert any("Exit" in c for c in callees)


class TestCallGraphQueries:
    """Test querying the call graph."""

    def test_find_callers(self):
        """Find all functions that call a target function."""
        graph = DependencyGraph()

        # Find who calls 'helper' in Python code
        callers = graph.find_callers(SAMPLE_PYTHON, "helper", "python")

        assert "main" in callers
        assert len(callers) >= 1

    def test_find_callees(self):
        """Find all functions called by a target function."""
        graph = DependencyGraph()

        # Find what 'main' calls in Python code
        callees = graph.find_callees(SAMPLE_PYTHON, "main", "python")

        assert "helper" in callees
        assert "Path" in callees
        assert "print" in callees
        assert len(callees) >= 3

    def test_find_callers_no_matches(self):
        """Should return empty list when function has no callers."""
        graph = DependencyGraph()

        # 'helper' is never called in SAMPLE_GO
        callers = graph.find_callers(SAMPLE_GO, "nonexistent_func", "go")

        assert callers == []

    def test_find_callees_no_matches(self):
        """Should return empty list when function makes no calls."""
        graph = DependencyGraph()

        # 'helper' in Python makes no calls (returns constant)
        callees = graph.find_callees(SAMPLE_PYTHON, "helper", "python")

        assert callees == []


class TestFileAnalysis:
    """Test full file analysis combining imports and call graph."""

    def test_analyze_python_file(self, tmp_path):
        """Analyze a complete Python file."""
        # Create temporary Python file
        test_file = tmp_path / "test_module.py"
        test_file.write_text(SAMPLE_PYTHON)

        graph = DependencyGraph()
        analysis = graph.analyze_file(str(test_file))

        # Should contain all key sections
        assert "imports" in analysis
        assert "symbols" in analysis
        assert "call_graph" in analysis

        # Check imports were extracted
        assert len(analysis["imports"]) >= 4

        # Check call graph was built
        assert len(analysis["call_graph"]) >= 3

    def test_analyze_javascript_file(self, tmp_path):
        """Analyze a complete JavaScript file."""
        test_file = tmp_path / "test_module.js"
        test_file.write_text(SAMPLE_JAVASCRIPT)

        graph = DependencyGraph()
        analysis = graph.analyze_file(str(test_file))

        assert "imports" in analysis
        assert "call_graph" in analysis
        assert len(analysis["imports"]) >= 3

    def test_analyze_nonexistent_file(self):
        """Should handle missing files gracefully."""
        graph = DependencyGraph()

        with pytest.raises(FileNotFoundError):
            graph.analyze_file("/nonexistent/path/file.py")

    def test_analyze_unsupported_extension(self, tmp_path):
        """Should handle unsupported file types."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("not code")

        graph = DependencyGraph()
        analysis = graph.analyze_file(str(test_file))

        # Should return empty analysis for unsupported types
        assert analysis["imports"] == []
        assert analysis["call_graph"] == []


class TestLazyCodeAnalyzer:
    """Test lazy initialization of CodeAnalyzer."""

    def test_analyzer_lazy_init(self):
        """CodeAnalyzer should be lazily initialized."""
        graph = DependencyGraph()

        # Analyzer should not be initialized yet
        assert graph._analyzer is None

        # First use should trigger initialization
        graph.extract_imports(SAMPLE_PYTHON, "python")

        # Now analyzer should exist
        assert graph._analyzer is not None

    def test_analyzer_reuse(self):
        """Should reuse same analyzer instance across calls."""
        graph = DependencyGraph()

        graph.extract_imports(SAMPLE_PYTHON, "python")
        analyzer1 = graph._analyzer

        graph.build_call_graph(SAMPLE_PYTHON, "python")
        analyzer2 = graph._analyzer

        # Should be the same instance
        assert analyzer1 is analyzer2


class TestDataClasses:
    """Test the dataclass structures."""

    def test_import_dataclass(self):
        """Test Import dataclass structure."""
        imp = Import(
            module="os",
            names=["path", "environ"],
            alias="operating_system",
            line_number=5
        )

        assert imp.module == "os"
        assert imp.names == ["path", "environ"]
        assert imp.alias == "operating_system"
        assert imp.line_number == 5

    def test_call_edge_dataclass(self):
        """Test CallEdge dataclass structure."""
        edge = CallEdge(
            caller="main",
            callee="helper",
            line_number=42
        )

        assert edge.caller == "main"
        assert edge.callee == "helper"
        assert edge.line_number == 42
