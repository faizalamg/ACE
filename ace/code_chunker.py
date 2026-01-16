"""AST-based semantic code chunking module.

This module provides intelligent code chunking that respects language syntax
boundaries (functions, classes, methods) rather than arbitrary line counts.

Supports multiple languages via tree-sitter:
- Python (via built-in ast module or tree-sitter)
- JavaScript/TypeScript (via tree-sitter)
- Go (via tree-sitter)

Configuration:
    ACE_ENABLE_AST_CHUNKING: Enable/disable AST chunking (default: false)
    ACE_AST_MAX_LINES: Maximum lines per chunk (default: 120)
    ACE_AST_OVERLAP_LINES: Overlap between chunks (default: 20)

When disabled, falls back to passthrough mode (returns original content as single chunk).
"""

from __future__ import annotations

import os
import ast
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Optional tree-sitter imports
TREE_SITTER_AVAILABLE = False
TREE_SITTER_LANGUAGES: Dict[str, Any] = {}

try:
    import tree_sitter
    TREE_SITTER_AVAILABLE = True
    
    # Try to import language grammars
    try:
        import tree_sitter_python
        TREE_SITTER_LANGUAGES["python"] = tree_sitter.Language(tree_sitter_python.language())
    except ImportError:
        pass
    
    try:
        import tree_sitter_javascript
        TREE_SITTER_LANGUAGES["javascript"] = tree_sitter.Language(tree_sitter_javascript.language())
        TREE_SITTER_LANGUAGES["js"] = TREE_SITTER_LANGUAGES["javascript"]
    except ImportError:
        pass
    
    try:
        import tree_sitter_typescript
        TREE_SITTER_LANGUAGES["typescript"] = tree_sitter.Language(tree_sitter_typescript.language_typescript())
        TREE_SITTER_LANGUAGES["ts"] = TREE_SITTER_LANGUAGES["typescript"]
        TREE_SITTER_LANGUAGES["tsx"] = tree_sitter.Language(tree_sitter_typescript.language_tsx())
    except ImportError:
        pass
    
    try:
        import tree_sitter_go
        TREE_SITTER_LANGUAGES["go"] = tree_sitter.Language(tree_sitter_go.language())
    except ImportError:
        pass
        
except ImportError:
    pass


@dataclass
class CodeChunk:
    """A chunk of code with metadata."""
    content: str
    start_line: int
    end_line: int
    symbols: List[str] = field(default_factory=list)
    language: str = "unknown"
    is_semantic_unit: bool = False


@dataclass
class CodeSymbol:
    """Rich metadata for a code symbol extracted via AST parsing.
    
    Modeled after m1rl0k/Context-Engine's symbol representation.
    """
    name: str
    kind: str  # function, class, method, interface, struct, type, variable
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    signature: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    parent: Optional[str] = None  # For nested symbols (methods in classes)
    calls: List[str] = field(default_factory=list)  # Functions called by this symbol
    imports: List[str] = field(default_factory=list)  # Imports used
    
    @property
    def qualified_name(self) -> str:
        """Return fully qualified name including parent."""
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name


class ASTChunker:
    """AST-based semantic code chunker.
    
    Chunks code by language syntax boundaries (functions, classes) rather than
    arbitrary line counts. Uses tree-sitter for multi-language support with
    fallback to line-based chunking for unsupported languages.
    
    Configuration via environment variables:
        ACE_ENABLE_AST_CHUNKING: "true" to enable, anything else to disable
        ACE_AST_MAX_LINES: Maximum lines per chunk (default: 120)
        ACE_AST_OVERLAP_LINES: Overlap between chunks (default: 20)
    """
    
    # Languages with AST support (tree-sitter or built-in)
    _SUPPORTED_LANGUAGES = {
        "python", "javascript", "typescript", "java", "go", "rust", "c", "cpp",
        "js", "ts", "tsx"  # Aliases
    }
    
    def __init__(self):
        """Initialize chunker with config from environment."""
        self._enabled = os.environ.get("ACE_ENABLE_AST_CHUNKING", "false").lower() == "true"
        self._max_lines = int(os.environ.get("ACE_AST_MAX_LINES", "120"))
        self._overlap_lines = int(os.environ.get("ACE_AST_OVERLAP_LINES", "20"))
        self._parsers: Dict[str, Any] = {}
    
    def is_enabled(self) -> bool:
        """Check if AST chunking is enabled."""
        return self._enabled
    
    @property
    def max_lines(self) -> int:
        """Maximum lines per chunk."""
        return self._max_lines
    
    @property
    def overlap_lines(self) -> int:
        """Overlap lines between chunks."""
        return self._overlap_lines
    
    def supported_languages(self) -> Set[str]:
        """Return set of languages with AST support."""
        return self._SUPPORTED_LANGUAGES.copy()
    
    def _get_parser(self, language: str) -> Optional[Any]:
        """Get or create a tree-sitter parser for the language."""
        if not TREE_SITTER_AVAILABLE:
            return None
        
        lang_lower = language.lower()
        if lang_lower in self._parsers:
            return self._parsers[lang_lower]
        
        if lang_lower in TREE_SITTER_LANGUAGES:
            import tree_sitter
            parser = tree_sitter.Parser(TREE_SITTER_LANGUAGES[lang_lower])
            self._parsers[lang_lower] = parser
            return parser
        
        return None
    
    def chunk(self, content: str, language: str = "python") -> List[CodeChunk]:
        """Chunk code content by semantic boundaries.
        
        Args:
            content: Source code to chunk
            language: Programming language (default: python)
        
        Returns:
            List of CodeChunk objects
        """
        # Passthrough mode when disabled
        if not self._enabled:
            return self._passthrough_chunk(content, language)
        
        # Check language support
        lang_lower = language.lower()
        if lang_lower not in self._SUPPORTED_LANGUAGES:
            logger.debug(f"Language '{language}' not supported, using line-based fallback")
            return self._line_based_chunk(content, language)
        
        # Use AST-based chunking for supported languages
        if lang_lower == "python":
            return self._chunk_python(content)
        elif lang_lower in ("javascript", "js"):
            return self._chunk_javascript(content)
        elif lang_lower in ("typescript", "ts", "tsx"):
            return self._chunk_typescript(content, lang_lower)
        elif lang_lower == "go":
            return self._chunk_go(content)
        else:
            # For now, other languages use line-based fallback
            return self._line_based_chunk(content, language)
    
    def _passthrough_chunk(self, content: str, language: str) -> List[CodeChunk]:
        """Return content as a single chunk (disabled mode)."""
        lines = content.split('\n')
        return [CodeChunk(
            content=content,
            start_line=1,
            end_line=len(lines),
            symbols=[],
            language=language,
            is_semantic_unit=False
        )]
    
    def _line_based_chunk(self, content: str, language: str) -> List[CodeChunk]:
        """Fallback line-based chunking for unsupported languages."""
        lines = content.split('\n')
        chunks = []
        
        i = 0
        while i < len(lines):
            end = min(i + self._max_lines, len(lines))
            chunk_lines = lines[i:end]
            
            chunks.append(CodeChunk(
                content='\n'.join(chunk_lines),
                start_line=i + 1,  # 1-indexed
                end_line=end,
                symbols=[],
                language=language,
                is_semantic_unit=False
            ))
            
            # Move forward with overlap
            i = end - self._overlap_lines if end < len(lines) else end
        
        return chunks if chunks else [self._passthrough_chunk(content, language)[0]]
    
    def _chunk_python(self, content: str) -> List[CodeChunk]:
        """AST-based chunking for Python code."""
        lines = content.split('\n')
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.debug(f"Python syntax error, falling back to line-based: {e}")
            return self._line_based_chunk(content, "python")
        
        # Extract top-level definitions
        definitions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Only top-level definitions
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    definitions.append({
                        'name': node.name,
                        'start': node.lineno,
                        'end': node.end_lineno,
                        'type': 'class' if isinstance(node, ast.ClassDef) else 'function'
                    })
        
        # Sort by start line
        definitions.sort(key=lambda x: x['start'])
        
        if not definitions:
            # No definitions found, use line-based
            return self._line_based_chunk(content, "python")
        
        chunks = []
        current_symbols = []
        current_start = 1
        current_end = 0
        
        for defn in definitions:
            # Check if adding this definition exceeds max_lines
            potential_end = defn['end']
            potential_lines = potential_end - current_start + 1
            
            if potential_lines > self._max_lines and current_symbols:
                # Flush current chunk
                chunk_content = '\n'.join(lines[current_start-1:current_end])
                chunks.append(CodeChunk(
                    content=chunk_content,
                    start_line=current_start,
                    end_line=current_end,
                    symbols=current_symbols.copy(),
                    language="python",
                    is_semantic_unit=True
                ))
                current_symbols = []
                current_start = defn['start']
            
            current_symbols.append(defn['name'])
            current_end = defn['end']
        
        # Flush remaining
        if current_symbols:
            chunk_content = '\n'.join(lines[current_start-1:current_end])
            chunks.append(CodeChunk(
                content=chunk_content,
                start_line=current_start,
                end_line=current_end,
                symbols=current_symbols,
                language="python",
                is_semantic_unit=True
            ))
        
        # Handle module-level code before first definition
        if definitions and definitions[0]['start'] > 1:
            preamble = '\n'.join(lines[:definitions[0]['start']-1])
            if preamble.strip():
                chunks.insert(0, CodeChunk(
                    content=preamble,
                    start_line=1,
                    end_line=definitions[0]['start']-1,
                    symbols=[],
                    language="python",
                    is_semantic_unit=False
                ))
        
        return chunks if chunks else self._passthrough_chunk(content, "python")
    
    def _chunk_javascript(self, content: str) -> List[CodeChunk]:
        """AST-based chunking for JavaScript code using tree-sitter."""
        parser = self._get_parser("javascript")
        if not parser:
            logger.debug("tree-sitter JavaScript not available, using line-based fallback")
            return self._line_based_chunk(content, "javascript")
        
        return self._chunk_with_treesitter(content, parser, "javascript")
    
    def _chunk_typescript(self, content: str, variant: str = "typescript") -> List[CodeChunk]:
        """AST-based chunking for TypeScript code using tree-sitter."""
        # Use tsx parser for tsx files
        lang_key = "tsx" if variant == "tsx" else "typescript"
        parser = self._get_parser(lang_key)
        if not parser:
            logger.debug(f"tree-sitter {lang_key} not available, using line-based fallback")
            return self._line_based_chunk(content, "typescript")
        
        return self._chunk_with_treesitter(content, parser, "typescript")
    
    def _chunk_go(self, content: str) -> List[CodeChunk]:
        """AST-based chunking for Go code using tree-sitter."""
        parser = self._get_parser("go")
        if not parser:
            logger.debug("tree-sitter Go not available, using line-based fallback")
            return self._line_based_chunk(content, "go")
        
        return self._chunk_with_treesitter(content, parser, "go")
    
    def _chunk_with_treesitter(self, content: str, parser: Any, language: str) -> List[CodeChunk]:
        """Generic tree-sitter based chunking.
        
        Extracts functions, classes, methods, interfaces, structs, and type definitions.
        """
        lines = content.split('\n')
        
        try:
            tree = parser.parse(bytes(content, "utf8"))
        except Exception as e:
            logger.debug(f"tree-sitter parse error for {language}: {e}")
            return self._line_based_chunk(content, language)
        
        # Extract definitions based on language
        definitions = self._extract_definitions_treesitter(tree.root_node, language)
        
        if not definitions:
            return self._line_based_chunk(content, language)
        
        # Sort by start line
        definitions.sort(key=lambda x: x['start'])
        
        chunks = []
        current_symbols = []
        current_start = 1
        current_end = 0
        
        for defn in definitions:
            potential_end = defn['end']
            potential_lines = potential_end - current_start + 1
            
            if potential_lines > self._max_lines and current_symbols:
                chunk_content = '\n'.join(lines[current_start-1:current_end])
                chunks.append(CodeChunk(
                    content=chunk_content,
                    start_line=current_start,
                    end_line=current_end,
                    symbols=current_symbols.copy(),
                    language=language,
                    is_semantic_unit=True
                ))
                current_symbols = []
                current_start = defn['start']
            
            current_symbols.append(defn['name'])
            current_end = defn['end']
        
        # Flush remaining
        if current_symbols:
            chunk_content = '\n'.join(lines[current_start-1:current_end])
            chunks.append(CodeChunk(
                content=chunk_content,
                start_line=current_start,
                end_line=current_end,
                symbols=current_symbols,
                language=language,
                is_semantic_unit=True
            ))
        
        # Handle preamble (imports, etc.) before first definition
        if definitions and definitions[0]['start'] > 1:
            preamble = '\n'.join(lines[:definitions[0]['start']-1])
            if preamble.strip():
                chunks.insert(0, CodeChunk(
                    content=preamble,
                    start_line=1,
                    end_line=definitions[0]['start']-1,
                    symbols=[],
                    language=language,
                    is_semantic_unit=False
                ))
        
        return chunks if chunks else self._passthrough_chunk(content, language)
    
    def _extract_definitions_treesitter(self, root_node: Any, language: str) -> List[Dict[str, Any]]:
        """Extract top-level definitions from a tree-sitter parse tree.
        
        Args:
            root_node: The root node of the tree-sitter parse tree
            language: The programming language
        
        Returns:
            List of definition dictionaries with name, start, end, type keys
        """
        definitions = []
        
        # Node types to extract per language
        if language in ("javascript", "typescript"):
            target_types = {
                'function_declaration',
                'function_expression',
                'arrow_function',
                'class_declaration',
                'method_definition',
                'interface_declaration',  # TypeScript
                'type_alias_declaration',  # TypeScript
                'lexical_declaration',  # const/let with function
                'variable_declaration',  # var with function
            }
        elif language == "go":
            target_types = {
                'function_declaration',
                'method_declaration',
                'type_declaration',  # structs, interfaces, type aliases
            }
        else:
            target_types = set()
        
        self._walk_tree(root_node, target_types, definitions, language)
        return definitions
    
    def _walk_tree(self, node: Any, target_types: Set[str], definitions: List[Dict], language: str, depth: int = 0) -> None:
        """Walk the tree-sitter tree and extract definitions."""
        # Only process top-level or class-level definitions (depth <= 1)
        if node.type in target_types and depth <= 1:
            defn = self._extract_definition(node, language)
            if defn:
                definitions.append(defn)
        
        # Recurse into children for nested structures
        for child in node.children:
            # Go deeper into class bodies but not too deep
            if node.type in ('class_body', 'class_declaration', 'program', 'source_file'):
                self._walk_tree(child, target_types, definitions, language, depth + 1)
            elif depth == 0:  # Only recurse at top level
                self._walk_tree(child, target_types, definitions, language, depth)
    
    def _extract_definition(self, node: Any, language: str) -> Optional[Dict[str, Any]]:
        """Extract definition info from a tree-sitter node."""
        name = None
        kind = node.type
        
        if language in ("javascript", "typescript"):
            name = self._extract_js_ts_name(node)
        elif language == "go":
            name = self._extract_go_name(node)
        
        if not name:
            return None
        
        return {
            'name': name,
            'start': node.start_point[0] + 1,  # tree-sitter is 0-indexed
            'end': node.end_point[0] + 1,
            'type': kind
        }
    
    def _extract_js_ts_name(self, node: Any) -> Optional[str]:
        """Extract function/class name from JS/TS AST node."""
        # Function/class declarations have an 'identifier' child
        for child in node.children:
            if child.type == 'identifier':
                return child.text.decode('utf8')
            # For TypeScript interfaces
            if child.type == 'type_identifier':
                return child.text.decode('utf8')
        
        # For lexical_declaration (const foo = () => {}), look deeper
        if node.type in ('lexical_declaration', 'variable_declaration'):
            for child in node.children:
                if child.type == 'variable_declarator':
                    for subchild in child.children:
                        if subchild.type == 'identifier':
                            # Check if it's a function assignment
                            for sibling in child.children:
                                if sibling.type in ('arrow_function', 'function_expression', 'function'):
                                    return subchild.text.decode('utf8')
        
        return None
    
    def _extract_go_name(self, node: Any) -> Optional[str]:
        """Extract function/struct name from Go AST node."""
        if node.type == 'function_declaration':
            for child in node.children:
                if child.type == 'identifier':
                    return child.text.decode('utf8')
        
        elif node.type == 'method_declaration':
            # Methods have (receiver) then name
            for child in node.children:
                if child.type == 'field_identifier':
                    return child.text.decode('utf8')
        
        elif node.type == 'type_declaration':
            # type Foo struct {...} or type Bar interface {...}
            for child in node.children:
                if child.type == 'type_spec':
                    for subchild in child.children:
                        if subchild.type == 'type_identifier':
                            return subchild.text.decode('utf8')
        
        return None


# Convenience function for direct use
def chunk_code(content: str, language: str = "python") -> List[CodeChunk]:
    """Chunk code using the default ASTChunker instance.
    
    Args:
        content: Source code to chunk
        language: Programming language
    
    Returns:
        List of CodeChunk objects
    """
    return ASTChunker().chunk(content, language)
