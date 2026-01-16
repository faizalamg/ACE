"""Adaptive chunking module for file-type-specific chunking strategies.

This module provides intelligent file-type detection and selects optimal
chunking strategies for different content types:

- **Code files**: AST-based semantic chunking (functions, classes)
- **Markdown/docs**: Section-based chunking (headers, paragraphs)
- **Config files**: Key-value or structure-based chunking
- **Plain text**: Paragraph-based chunking

The goal is to preserve semantic boundaries appropriate to each file type,
improving retrieval accuracy over one-size-fits-all line-based chunking.

Configuration:
    ACE_ADAPTIVE_CHUNKING: Enable/disable adaptive chunking (default: true)
    ACE_CHUNK_MAX_LINES: Maximum lines per chunk (default: 120)
    ACE_CHUNK_OVERLAP_LINES: Overlap between chunks (default: 20)
"""

from __future__ import annotations

import os
import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Any, Protocol, Type
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Chunk:
    """A chunk of content with metadata."""
    content: str
    start_line: int
    end_line: int
    chunk_type: str  # "code", "section", "config", "paragraph"
    symbols: List[str] = field(default_factory=list)  # For code: function/class names
    heading: Optional[str] = None  # For docs: section heading
    language: str = "unknown"
    is_semantic_unit: bool = False


# =============================================================================
# CHUNKING STRATEGIES
# =============================================================================

class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, content: str, **kwargs) -> List[Chunk]:
        """Chunk content into semantic units.
        
        Args:
            content: The text content to chunk
            **kwargs: Strategy-specific options
            
        Returns:
            List of Chunk objects
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging/debugging."""
        pass


class MarkdownChunker(ChunkingStrategy):
    """Chunk Markdown files by section headers.
    
    Splits on ## and ### headers, keeping each section as a unit.
    Respects code blocks and maintains context.
    """
    
    # Header patterns
    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```', re.MULTILINE)
    
    def __init__(self, max_lines: int = 120, min_section_lines: int = 5):
        self._max_lines = max_lines
        self._min_section_lines = min_section_lines
    
    @property
    def name(self) -> str:
        return "markdown"
    
    def chunk(self, content: str, **kwargs) -> List[Chunk]:
        """Chunk markdown by sections."""
        lines = content.split('\n')
        if not lines:
            return []
        
        # Find all header positions
        headers = []
        for i, line in enumerate(lines):
            match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                headers.append({
                    'line': i,
                    'level': level,
                    'text': text
                })
        
        if not headers:
            # No headers - use paragraph-based chunking
            return self._chunk_by_paragraphs(content, lines)
        
        chunks = []
        
        # Handle content before first header
        if headers[0]['line'] > 0:
            preamble_lines = lines[:headers[0]['line']]
            preamble_content = '\n'.join(preamble_lines)
            if preamble_content.strip():
                chunks.append(Chunk(
                    content=preamble_content,
                    start_line=1,
                    end_line=headers[0]['line'],
                    chunk_type="section",
                    heading="(preamble)",
                    language="markdown",
                    is_semantic_unit=True
                ))
        
        # Chunk by sections
        for i, header in enumerate(headers):
            start_line = header['line']
            end_line = headers[i + 1]['line'] if i + 1 < len(headers) else len(lines)
            
            section_lines = lines[start_line:end_line]
            section_content = '\n'.join(section_lines)
            
            # If section is too long, split it further
            if len(section_lines) > self._max_lines:
                sub_chunks = self._split_long_section(
                    section_content, section_lines, start_line, header['text']
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(Chunk(
                    content=section_content,
                    start_line=start_line + 1,
                    end_line=end_line,
                    chunk_type="section",
                    heading=header['text'],
                    language="markdown",
                    is_semantic_unit=True
                ))
        
        return chunks
    
    def _chunk_by_paragraphs(self, content: str, lines: List[str]) -> List[Chunk]:
        """Fallback: chunk by paragraph boundaries."""
        chunks = []
        current_lines = []
        current_start = 0
        
        for i, line in enumerate(lines):
            if not line.strip() and current_lines:
                # Empty line - end of paragraph
                if len(current_lines) >= self._min_section_lines or not chunks:
                    chunks.append(Chunk(
                        content='\n'.join(current_lines),
                        start_line=current_start + 1,
                        end_line=i,
                        chunk_type="paragraph",
                        language="markdown",
                        is_semantic_unit=False
                    ))
                elif chunks:
                    # Merge small paragraph with previous
                    prev = chunks[-1]
                    chunks[-1] = Chunk(
                        content=prev.content + '\n\n' + '\n'.join(current_lines),
                        start_line=prev.start_line,
                        end_line=i,
                        chunk_type="paragraph",
                        language="markdown",
                        is_semantic_unit=False
                    )
                current_lines = []
                current_start = i + 1
            else:
                current_lines.append(line)
        
        # Handle remaining content
        if current_lines:
            chunks.append(Chunk(
                content='\n'.join(current_lines),
                start_line=current_start + 1,
                end_line=len(lines),
                chunk_type="paragraph",
                language="markdown",
                is_semantic_unit=False
            ))
        
        return chunks if chunks else [Chunk(
            content=content,
            start_line=1,
            end_line=len(lines),
            chunk_type="paragraph",
            language="markdown",
            is_semantic_unit=False
        )]
    
    def _split_long_section(
        self, content: str, lines: List[str], base_line: int, heading: str
    ) -> List[Chunk]:
        """Split a section that exceeds max_lines."""
        chunks = []
        i = 0
        part = 0
        
        while i < len(lines):
            end = min(i + self._max_lines, len(lines))
            chunk_lines = lines[i:end]
            
            part += 1
            chunks.append(Chunk(
                content='\n'.join(chunk_lines),
                start_line=base_line + i + 1,
                end_line=base_line + end,
                chunk_type="section",
                heading=f"{heading} (part {part})" if part > 1 else heading,
                language="markdown",
                is_semantic_unit=True
            ))
            
            i = end
        
        return chunks


class ConfigChunker(ChunkingStrategy):
    """Chunk configuration files by logical sections.
    
    Handles YAML, JSON, TOML, INI files by preserving structural units.
    """
    
    def __init__(self, max_lines: int = 80):
        self._max_lines = max_lines
    
    @property
    def name(self) -> str:
        return "config"
    
    def chunk(self, content: str, file_ext: str = "", **kwargs) -> List[Chunk]:
        """Chunk config files by structure."""
        lines = content.split('\n')
        
        if file_ext.lower() in ('.yaml', '.yml'):
            return self._chunk_yaml(content, lines)
        elif file_ext.lower() == '.json':
            return self._chunk_json(content, lines)
        elif file_ext.lower() == '.toml':
            return self._chunk_toml(content, lines)
        elif file_ext.lower() in ('.ini', '.cfg', '.conf'):
            return self._chunk_ini(content, lines)
        else:
            # Generic config: chunk by top-level keys or sections
            return self._chunk_generic(content, lines)
    
    def _chunk_yaml(self, content: str, lines: List[str]) -> List[Chunk]:
        """Chunk YAML by top-level keys."""
        chunks = []
        current_lines = []
        current_start = 0
        current_key = None
        
        for i, line in enumerate(lines):
            # Top-level key (no indentation, ends with :)
            if line and not line[0].isspace() and ':' in line:
                if current_lines:
                    chunks.append(Chunk(
                        content='\n'.join(current_lines),
                        start_line=current_start + 1,
                        end_line=i,
                        chunk_type="config",
                        heading=current_key,
                        language="yaml",
                        is_semantic_unit=True
                    ))
                current_lines = [line]
                current_start = i
                current_key = line.split(':')[0].strip()
            else:
                current_lines.append(line)
        
        if current_lines:
            chunks.append(Chunk(
                content='\n'.join(current_lines),
                start_line=current_start + 1,
                end_line=len(lines),
                chunk_type="config",
                heading=current_key,
                language="yaml",
                is_semantic_unit=True
            ))
        
        return chunks if chunks else [self._single_chunk(content, lines, "yaml")]
    
    def _chunk_json(self, content: str, lines: List[str]) -> List[Chunk]:
        """Chunk JSON - for large JSON, keep as single chunk or split by top-level keys."""
        # JSON is harder to split without parsing - keep as single chunk if reasonable
        if len(lines) <= self._max_lines:
            return [self._single_chunk(content, lines, "json")]
        
        # For large JSON, try to split by top-level keys (heuristic)
        # Look for lines that look like top-level keys: "key": 
        chunks = []
        current_lines = []
        current_start = 0
        current_key = None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Heuristic: top-level key starts at indent 2 (after opening {)
            if line.startswith('  "') and '": ' in line and not line.startswith('    '):
                if current_lines and len(current_lines) > 2:
                    chunks.append(Chunk(
                        content='\n'.join(current_lines),
                        start_line=current_start + 1,
                        end_line=i,
                        chunk_type="config",
                        heading=current_key,
                        language="json",
                        is_semantic_unit=True
                    ))
                    current_lines = []
                    current_start = i
                match = re.match(r'\s*"([^"]+)"', line)
                current_key = match.group(1) if match else None
            current_lines.append(line)
        
        if current_lines:
            chunks.append(Chunk(
                content='\n'.join(current_lines),
                start_line=current_start + 1,
                end_line=len(lines),
                chunk_type="config",
                heading=current_key,
                language="json",
                is_semantic_unit=True
            ))
        
        return chunks if chunks else [self._single_chunk(content, lines, "json")]
    
    def _chunk_toml(self, content: str, lines: List[str]) -> List[Chunk]:
        """Chunk TOML by [section] headers."""
        chunks = []
        current_lines = []
        current_start = 0
        current_section = None
        
        for i, line in enumerate(lines):
            # Section header: [section] or [[array]]
            if re.match(r'^\[+[^\]]+\]+\s*$', line.strip()):
                if current_lines:
                    chunks.append(Chunk(
                        content='\n'.join(current_lines),
                        start_line=current_start + 1,
                        end_line=i,
                        chunk_type="config",
                        heading=current_section,
                        language="toml",
                        is_semantic_unit=True
                    ))
                current_lines = [line]
                current_start = i
                current_section = line.strip().strip('[]')
            else:
                current_lines.append(line)
        
        if current_lines:
            chunks.append(Chunk(
                content='\n'.join(current_lines),
                start_line=current_start + 1,
                end_line=len(lines),
                chunk_type="config",
                heading=current_section,
                language="toml",
                is_semantic_unit=True
            ))
        
        return chunks if chunks else [self._single_chunk(content, lines, "toml")]
    
    def _chunk_ini(self, content: str, lines: List[str]) -> List[Chunk]:
        """Chunk INI by [section] headers."""
        # Same logic as TOML - section headers
        return self._chunk_toml(content, lines)
    
    def _chunk_generic(self, content: str, lines: List[str]) -> List[Chunk]:
        """Generic config chunking - keep as single chunk if small."""
        if len(lines) <= self._max_lines:
            return [self._single_chunk(content, lines, "config")]
        
        # Fall back to line-based
        return self._line_based_chunk(content, lines, "config")
    
    def _single_chunk(self, content: str, lines: List[str], language: str) -> Chunk:
        """Create a single chunk for entire content."""
        return Chunk(
            content=content,
            start_line=1,
            end_line=len(lines),
            chunk_type="config",
            language=language,
            is_semantic_unit=True
        )
    
    def _line_based_chunk(
        self, content: str, lines: List[str], language: str
    ) -> List[Chunk]:
        """Fall back to line-based chunking."""
        chunks = []
        i = 0
        overlap = 10
        
        while i < len(lines):
            end = min(i + self._max_lines, len(lines))
            chunk_lines = lines[i:end]
            
            chunks.append(Chunk(
                content='\n'.join(chunk_lines),
                start_line=i + 1,
                end_line=end,
                chunk_type="config",
                language=language,
                is_semantic_unit=False
            ))
            
            i = end - overlap if end < len(lines) else end
        
        return chunks


class CodeChunker(ChunkingStrategy):
    """Wrapper around existing ASTChunker for code files."""
    
    def __init__(self):
        from ace.code_chunker import ASTChunker, CodeChunk
        self._ast_chunker = ASTChunker()
        self._CodeChunk = CodeChunk
    
    @property
    def name(self) -> str:
        return "code"
    
    def chunk(self, content: str, language: str = "python", **kwargs) -> List[Chunk]:
        """Delegate to ASTChunker and convert results."""
        ast_chunks = self._ast_chunker.chunk(content, language)
        
        return [
            Chunk(
                content=c.content,
                start_line=c.start_line,
                end_line=c.end_line,
                chunk_type="code",
                symbols=c.symbols,
                language=c.language,
                is_semantic_unit=c.is_semantic_unit
            )
            for c in ast_chunks
        ]


class PlainTextChunker(ChunkingStrategy):
    """Chunk plain text files by paragraphs or line groups."""
    
    def __init__(self, max_lines: int = 100, overlap_lines: int = 10):
        self._max_lines = max_lines
        self._overlap_lines = overlap_lines
    
    @property
    def name(self) -> str:
        return "plaintext"
    
    def chunk(self, content: str, **kwargs) -> List[Chunk]:
        """Chunk by paragraphs (double newline) or line batches."""
        # Try paragraph-based first
        paragraphs = re.split(r'\n\s*\n', content)
        
        if len(paragraphs) > 1:
            return self._chunk_by_paragraphs(paragraphs)
        
        # Fall back to line-based
        lines = content.split('\n')
        return self._line_based_chunk(lines)
    
    def _chunk_by_paragraphs(self, paragraphs: List[str]) -> List[Chunk]:
        """Chunk by paragraph boundaries."""
        chunks = []
        current_line = 1
        
        for para in paragraphs:
            if not para.strip():
                continue
            
            para_lines = para.count('\n') + 1
            chunks.append(Chunk(
                content=para,
                start_line=current_line,
                end_line=current_line + para_lines - 1,
                chunk_type="paragraph",
                language="text",
                is_semantic_unit=True
            ))
            current_line += para_lines + 1  # +1 for separator
        
        return chunks if chunks else [Chunk(
            content="",
            start_line=1,
            end_line=1,
            chunk_type="paragraph",
            language="text",
            is_semantic_unit=False
        )]
    
    def _line_based_chunk(self, lines: List[str]) -> List[Chunk]:
        """Fall back to line-based chunking."""
        chunks = []
        i = 0
        
        while i < len(lines):
            end = min(i + self._max_lines, len(lines))
            chunk_lines = lines[i:end]
            
            chunks.append(Chunk(
                content='\n'.join(chunk_lines),
                start_line=i + 1,
                end_line=end,
                chunk_type="paragraph",
                language="text",
                is_semantic_unit=False
            ))
            
            i = end - self._overlap_lines if end < len(lines) else end
        
        return chunks


class HTMLChunker(ChunkingStrategy):
    """Chunk HTML/XML files by semantic tags.
    
    Splits content by major HTML elements like <head>, <body>, <section>,
    <article>, <div>, <script>, <style>, etc. Also handles XML with similar logic.
    """
    
    # Major HTML tags that should be kept as units
    BLOCK_TAGS = {
        'html', 'head', 'body', 'header', 'footer', 'main', 'nav', 'aside',
        'section', 'article', 'div', 'form', 'table', 'ul', 'ol', 'dl',
        'script', 'style', 'template', 'iframe',
    }
    
    def __init__(self, max_lines: int = 120):
        self._max_lines = max_lines
    
    @property
    def name(self) -> str:
        return "html"
    
    def chunk(self, content: str, **kwargs) -> List[Chunk]:
        """Chunk HTML/XML by major semantic elements."""
        lines = content.split('\n')
        chunks = []
        current_lines: List[str] = []
        current_start = 0
        current_tag = None
        tag_stack: List[str] = []
        
        # Regex to match opening and closing tags
        open_tag_pattern = re.compile(r'<(\w+)(?:\s|>|/>)')
        close_tag_pattern = re.compile(r'</(\w+)>')
        
        for i, line in enumerate(lines):
            # Check for opening tags
            for match in open_tag_pattern.finditer(line):
                tag = match.group(1).lower()
                if tag in self.BLOCK_TAGS:
                    # Start a new chunk if we have accumulated lines and at root level
                    if current_lines and len(tag_stack) == 0:
                        chunks.append(Chunk(
                            content='\n'.join(current_lines),
                            start_line=current_start + 1,
                            end_line=i,
                            chunk_type="html_element",
                            heading=current_tag,
                            language="html",
                            is_semantic_unit=True
                        ))
                        current_lines = []
                        current_start = i
                    tag_stack.append(tag)
                    if current_tag is None:
                        current_tag = tag
            
            current_lines.append(line)
            
            # Check for closing tags
            for match in close_tag_pattern.finditer(line):
                tag = match.group(1).lower()
                if tag in self.BLOCK_TAGS and tag_stack and tag_stack[-1] == tag:
                    tag_stack.pop()
                    # If we closed a root-level tag, finalize chunk
                    if len(tag_stack) == 0 and current_lines:
                        chunks.append(Chunk(
                            content='\n'.join(current_lines),
                            start_line=current_start + 1,
                            end_line=i + 1,
                            chunk_type="html_element",
                            heading=current_tag,
                            language="html",
                            is_semantic_unit=True
                        ))
                        current_lines = []
                        current_start = i + 1
                        current_tag = None
        
        # Handle remaining content
        if current_lines:
            chunks.append(Chunk(
                content='\n'.join(current_lines),
                start_line=current_start + 1,
                end_line=len(lines),
                chunk_type="html_element",
                heading=current_tag,
                language="html",
                is_semantic_unit=len(tag_stack) == 0
            ))
        
        return chunks if chunks else [Chunk(
            content=content,
            start_line=1,
            end_line=len(lines),
            chunk_type="html",
            language="html",
            is_semantic_unit=True
        )]


class ShellChunker(ChunkingStrategy):
    """Chunk shell scripts by functions and comment blocks.
    
    Splits shell scripts into semantic units based on:
    - Function definitions
    - Major comment blocks (sections)
    - Logical groupings
    """
    
    # Function pattern for bash/sh
    FUNCTION_PATTERN = re.compile(
        r'^(?:function\s+)?(\w+)\s*\(\s*\)\s*\{?$'
    )
    # Major comment section (multiple # or description)
    SECTION_COMMENT = re.compile(r'^#{2,}\s*(.+)\s*#{0,}$|^#\s*[-=]{3,}')
    
    def __init__(self, max_lines: int = 80):
        self._max_lines = max_lines
    
    @property
    def name(self) -> str:
        return "shell"
    
    def chunk(self, content: str, **kwargs) -> List[Chunk]:
        """Chunk shell script by functions and sections."""
        lines = content.split('\n')
        chunks = []
        current_lines: List[str] = []
        current_start = 0
        current_heading = None
        in_function = False
        brace_count = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check for function start
            func_match = self.FUNCTION_PATTERN.match(stripped)
            if func_match and not in_function:
                # Save previous chunk
                if current_lines:
                    chunks.append(Chunk(
                        content='\n'.join(current_lines),
                        start_line=current_start + 1,
                        end_line=i,
                        chunk_type="shell",
                        heading=current_heading,
                        symbols=[current_heading] if current_heading else [],
                        language="shell",
                        is_semantic_unit=True
                    ))
                current_lines = [line]
                current_start = i
                current_heading = func_match.group(1)
                in_function = True
                if '{' in line:
                    brace_count = 1
                continue
            
            # Check for section comment (start new chunk)
            section_match = self.SECTION_COMMENT.match(stripped)
            if section_match and not in_function:
                if current_lines:
                    chunks.append(Chunk(
                        content='\n'.join(current_lines),
                        start_line=current_start + 1,
                        end_line=i,
                        chunk_type="shell",
                        heading=current_heading,
                        language="shell",
                        is_semantic_unit=True
                    ))
                current_lines = [line]
                current_start = i
                current_heading = section_match.group(1) if section_match.group(1) else "section"
                continue
            
            current_lines.append(line)
            
            # Track braces for function end
            if in_function:
                brace_count += stripped.count('{') - stripped.count('}')
                if brace_count <= 0:
                    in_function = False
                    # Function ended - save chunk
                    chunks.append(Chunk(
                        content='\n'.join(current_lines),
                        start_line=current_start + 1,
                        end_line=i + 1,
                        chunk_type="function",
                        heading=current_heading,
                        symbols=[current_heading] if current_heading else [],
                        language="shell",
                        is_semantic_unit=True
                    ))
                    current_lines = []
                    current_start = i + 1
                    current_heading = None
        
        # Handle remaining content
        if current_lines:
            chunks.append(Chunk(
                content='\n'.join(current_lines),
                start_line=current_start + 1,
                end_line=len(lines),
                chunk_type="shell",
                heading=current_heading,
                language="shell",
                is_semantic_unit=not in_function
            ))
        
        return chunks if chunks else [Chunk(
            content=content,
            start_line=1,
            end_line=len(lines),
            chunk_type="shell",
            language="shell",
            is_semantic_unit=True
        )]


# =============================================================================
# STRATEGY SELECTOR
# =============================================================================

class AdaptiveChunker:
    """Automatically selects and applies optimal chunking strategy based on file type.
    
    Usage:
        chunker = AdaptiveChunker()
        chunks = chunker.chunk(content, file_path="/path/to/file.md")
        
        # Or with explicit file type
        chunks = chunker.chunk(content, file_type="markdown")
    
    Configuration via environment variables:
        ACE_ADAPTIVE_CHUNKING: "true" to enable (default), "false" to disable
        ACE_CHUNK_MAX_LINES: Maximum lines per chunk (default: 120)
    """
    
    # File extension to strategy mapping
    _EXTENSION_STRATEGIES: Dict[str, str] = {
        # Documentation
        '.md': 'markdown',
        '.mdx': 'markdown',
        '.rst': 'markdown',  # reStructuredText similar to markdown
        '.txt': 'plaintext',
        '.text': 'plaintext',
        
        # Config files
        '.yaml': 'config',
        '.yml': 'config',
        '.json': 'config',
        '.toml': 'config',
        '.ini': 'config',
        '.cfg': 'config',
        '.conf': 'config',
        '.env': 'config',
        '.properties': 'config',
        
        # HTML/XML files
        '.html': 'html',
        '.htm': 'html',
        '.xhtml': 'html',
        '.xml': 'html',
        '.svg': 'html',
        '.xsl': 'html',
        '.xslt': 'html',
        
        # Shell scripts
        '.sh': 'shell',
        '.bash': 'shell',
        '.zsh': 'shell',
        '.fish': 'shell',
        '.ksh': 'shell',
        
        # Code files (handled by ASTChunker)
        '.py': 'code',
        '.pyi': 'code',
        '.js': 'code',
        '.jsx': 'code',
        '.ts': 'code',
        '.tsx': 'code',
        '.go': 'code',
        '.java': 'code',
        '.c': 'code',
        '.cpp': 'code',
        '.h': 'code',
        '.hpp': 'code',
        '.rs': 'code',
        '.rb': 'code',
        '.php': 'code',
        '.swift': 'code',
        '.kt': 'code',
        '.scala': 'code',
        '.cs': 'code',
        '.ps1': 'code',
        '.sql': 'code',
    }
    
    def __init__(
        self,
        enabled: Optional[bool] = None,
        max_lines: Optional[int] = None,
        overlap_lines: Optional[int] = None,
    ):
        """Initialize adaptive chunker.
        
        Args:
            enabled: Override ACE_ADAPTIVE_CHUNKING env var
            max_lines: Override ACE_CHUNK_MAX_LINES env var
            overlap_lines: Override ACE_CHUNK_OVERLAP_LINES env var
        """
        self._enabled = enabled if enabled is not None else (
            os.environ.get("ACE_ADAPTIVE_CHUNKING", "true").lower() == "true"
        )
        self._max_lines = max_lines or int(os.environ.get("ACE_CHUNK_MAX_LINES", "120"))
        self._overlap_lines = overlap_lines or int(os.environ.get("ACE_CHUNK_OVERLAP_LINES", "20"))
        
        # Initialize strategies lazily
        self._strategies: Dict[str, ChunkingStrategy] = {}
    
    def _get_strategy(self, strategy_name: str) -> ChunkingStrategy:
        """Get or create a chunking strategy."""
        if strategy_name not in self._strategies:
            if strategy_name == 'markdown':
                self._strategies[strategy_name] = MarkdownChunker(
                    max_lines=self._max_lines
                )
            elif strategy_name == 'config':
                self._strategies[strategy_name] = ConfigChunker(
                    max_lines=self._max_lines
                )
            elif strategy_name == 'code':
                self._strategies[strategy_name] = CodeChunker()
            elif strategy_name == 'html':
                self._strategies[strategy_name] = HTMLChunker(
                    max_lines=self._max_lines
                )
            elif strategy_name == 'shell':
                self._strategies[strategy_name] = ShellChunker(
                    max_lines=self._max_lines
                )
            elif strategy_name == 'plaintext':
                self._strategies[strategy_name] = PlainTextChunker(
                    max_lines=self._max_lines,
                    overlap_lines=self._overlap_lines
                )
            else:
                # Default to plaintext
                self._strategies[strategy_name] = PlainTextChunker(
                    max_lines=self._max_lines,
                    overlap_lines=self._overlap_lines
                )
        
        return self._strategies[strategy_name]
    
    def detect_strategy(
        self,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        content: Optional[str] = None,
    ) -> str:
        """Detect the optimal chunking strategy for content.
        
        Args:
            file_path: Path to file (used for extension detection)
            file_type: Explicit file type override
            content: Content for heuristic detection
            
        Returns:
            Strategy name: 'code', 'markdown', 'config', 'html', 'shell', or 'plaintext'
        """
        # Explicit override takes priority
        if file_type:
            return file_type.lower()
        
        # Try extension detection
        if file_path:
            ext = Path(file_path).suffix.lower()
            if ext in self._EXTENSION_STRATEGIES:
                return self._EXTENSION_STRATEGIES[ext]
        
        # Try content-based heuristics
        if content:
            return self._detect_from_content(content)
        
        return 'plaintext'
    
    def _detect_from_content(self, content: str) -> str:
        """Detect file type from content heuristics."""
        first_lines = content[:2000]  # Check first 2KB
        
        # HTML/XML indicators
        if re.search(r'^\s*<(!DOCTYPE|html|xml|head|body)\b', first_lines, re.IGNORECASE):
            return 'html'
        if re.search(r'<\?xml\s+version', first_lines):
            return 'html'
        
        # Shell script indicators
        if first_lines.strip().startswith('#!') and re.search(r'(bash|sh|zsh|fish)', first_lines[:100]):
            return 'shell'
        if re.search(r'^function\s+\w+\s*\(\s*\)\s*\{', first_lines, re.MULTILINE):
            return 'shell'
        if re.search(r'^\w+\s*\(\s*\)\s*\{', first_lines, re.MULTILINE) and 'echo' in first_lines:
            return 'shell'
        
        # Markdown indicators
        if re.search(r'^#{1,6}\s+\w', first_lines, re.MULTILINE):
            return 'markdown'
        if re.search(r'^\s*[-*]\s+\w', first_lines, re.MULTILINE):  # Lists
            return 'markdown'
        
        # JSON indicator
        if first_lines.strip().startswith('{') or first_lines.strip().startswith('['):
            return 'config'
        
        # YAML indicators
        if re.search(r'^\w+:\s*$', first_lines, re.MULTILINE):
            return 'config'
        
        # Code indicators
        code_patterns = [
            r'\bdef\s+\w+\s*\(',  # Python
            r'\bfunction\s+\w+\s*\(',  # JavaScript
            r'\bclass\s+\w+',  # Multiple languages
            r'\bimport\s+',  # Multiple languages
            r'\bfrom\s+\w+\s+import',  # Python
        ]
        for pattern in code_patterns:
            if re.search(pattern, first_lines):
                return 'code'
        
        return 'plaintext'
    
    def chunk(
        self,
        content: str,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[Chunk]:
        """Chunk content using the optimal strategy.
        
        Args:
            content: The content to chunk
            file_path: Path to file (for strategy detection)
            file_type: Explicit file type override
            language: Programming language (for code files)
            
        Returns:
            List of Chunk objects
        """
        if not self._enabled:
            # Fall back to simple line-based chunking
            return self._fallback_chunk(content, language or "unknown")
        
        strategy_name = self.detect_strategy(file_path, file_type, content)
        strategy = self._get_strategy(strategy_name)
        
        # Build kwargs for strategy
        kwargs: Dict[str, Any] = {}
        if strategy_name == 'code' and language:
            kwargs['language'] = language
        elif strategy_name == 'config' and file_path:
            kwargs['file_ext'] = Path(file_path).suffix
        
        logger.debug(f"Using {strategy.name} strategy for {file_path or 'content'}")
        
        return strategy.chunk(content, **kwargs)
    
    def _fallback_chunk(self, content: str, language: str) -> List[Chunk]:
        """Simple line-based fallback chunking."""
        lines = content.split('\n')
        chunks = []
        i = 0
        
        while i < len(lines):
            end = min(i + self._max_lines, len(lines))
            chunk_lines = lines[i:end]
            
            chunks.append(Chunk(
                content='\n'.join(chunk_lines),
                start_line=i + 1,
                end_line=end,
                chunk_type="line",
                language=language,
                is_semantic_unit=False
            ))
            
            i = end - self._overlap_lines if end < len(lines) else end
        
        return chunks if chunks else [Chunk(
            content=content,
            start_line=1,
            end_line=len(lines) or 1,
            chunk_type="line",
            language=language,
            is_semantic_unit=False
        )]
    
    @property
    def enabled(self) -> bool:
        """Check if adaptive chunking is enabled."""
        return self._enabled
    
    @property
    def supported_strategies(self) -> List[str]:
        """Return list of supported strategy names."""
        return ['code', 'markdown', 'config', 'plaintext']


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def chunk_adaptive(
    content: str,
    file_path: Optional[str] = None,
    file_type: Optional[str] = None,
    language: Optional[str] = None,
) -> List[Chunk]:
    """Convenience function for adaptive chunking.
    
    Args:
        content: Content to chunk
        file_path: Path to file
        file_type: Explicit file type
        language: Programming language
        
    Returns:
        List of Chunk objects
    """
    return AdaptiveChunker().chunk(content, file_path, file_type, language)
