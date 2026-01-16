"""Code-aware bullet enrichment for ACE Phase 2B.

This module provides CodeAwareEnricher which enriches bullets with code-specific
metadata extracted from code context:

- Symbol extraction (functions, classes, methods)
- Auto-generation of trigger patterns from code symbols
- Programming language detection
- Code snippet extraction from markdown-style content
- Optimized embedding text for code search

Integration with code_analysis.py (when available) is done via lazy import
to avoid circular dependencies.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .code_analysis import CodeAnalyzer

from .playbook import Bullet, EnrichedBullet


class CodeAwareEnricher:
    """Enrich bullets with code-specific metadata for improved code search.

    This class analyzes code context to extract:
    - Programming language
    - Function/class/method names as trigger patterns
    - Key programming terms
    - Optimized embedding text combining content + code symbols

    Example:
        >>> enricher = CodeAwareEnricher()
        >>> bullet = Bullet(id="b1", section="debug", content="Check null pointers")
        >>> code = "public void findUser(Long id) { if (id == null) throw... }"
        >>> enriched = enricher.enrich_bullet_with_code(bullet, code)
        >>> print(enriched.domains)
        ['java']
        >>> print(enriched.trigger_patterns)
        ['findUser', 'null', 'pointer', ...]
    """

    def __init__(self, code_analyzer: Optional["CodeAnalyzer"] = None) -> None:
        """Initialize CodeAwareEnricher.

        Args:
            code_analyzer: Optional CodeAnalyzer instance. If not provided,
                          will be lazy-loaded when needed to avoid circular deps.
        """
        self._analyzer = code_analyzer

    def _get_analyzer(self) -> Optional["CodeAnalyzer"]:
        """Lazy-load CodeAnalyzer to avoid circular dependency.

        Returns:
            CodeAnalyzer instance or None if not available.
        """
        if self._analyzer is None:
            try:
                from .code_analysis import CodeAnalyzer
                self._analyzer = CodeAnalyzer()
            except ImportError:
                # CodeAnalyzer not available yet - fallback to heuristics
                pass
        return self._analyzer

    def enrich_bullet_with_code(
        self,
        bullet: Bullet,
        code_context: str
    ) -> EnrichedBullet:
        """Enrich a bullet with code-specific metadata from code context.

        Extracts:
        - Programming language → added to domains
        - Code symbols (functions/classes) → added to trigger_patterns
        - Optimized embedding_text for semantic search

        Args:
            bullet: Bullet to enrich (Bullet or EnrichedBullet)
            code_context: Code snippet or markdown with code blocks

        Returns:
            EnrichedBullet with code-aware metadata.
        """
        # If already enriched, preserve existing metadata
        if isinstance(bullet, EnrichedBullet):
            trigger_patterns = list(bullet.trigger_patterns)
            domains = list(bullet.domains)
            task_types = list(bullet.task_types)
        else:
            trigger_patterns = []
            domains = []
            task_types = []

        # Extract code blocks from markdown-style content
        code_blocks = self.extract_code_blocks(code_context)

        # If no code blocks found, treat entire context as code
        if not code_blocks:
            if code_context.strip():
                # Detect language from context
                detected_lang = self.detect_code_language(code_context)
                if detected_lang:
                    if detected_lang not in domains:
                        domains.append(detected_lang)

                    # Generate triggers from code
                    code_triggers = self.generate_code_triggers(code_context, detected_lang)
                    for trigger in code_triggers:
                        if trigger not in trigger_patterns:
                            trigger_patterns.append(trigger)
        else:
            # Process each code block
            for lang, code in code_blocks:
                if lang and lang not in domains:
                    domains.append(lang)

                # Generate triggers from this block
                if lang:
                    code_triggers = self.generate_code_triggers(code, lang)
                else:
                    # Try to detect language
                    detected = self.detect_code_language(code)
                    if detected:
                        if detected not in domains:
                            domains.append(detected)
                        code_triggers = self.generate_code_triggers(code, detected)
                    else:
                        code_triggers = []

                for trigger in code_triggers:
                    if trigger not in trigger_patterns:
                        trigger_patterns.append(trigger)

        # Add code-related task type if not present
        if "code" not in task_types and domains:
            task_types.append("code")

        # Generate optimized embedding text
        embedding_text = self._generate_embedding_text(
            bullet.content,
            trigger_patterns,
            domains
        )

        # Create EnrichedBullet
        if isinstance(bullet, EnrichedBullet):
            # Update existing enriched bullet
            enriched = EnrichedBullet(
                id=bullet.id,
                section=bullet.section,
                content=bullet.content,
                helpful=bullet.helpful,
                harmful=bullet.harmful,
                neutral=bullet.neutral,
                created_at=bullet.created_at,
                updated_at=bullet.updated_at,
                confidence=bullet.confidence,
                task_types=task_types,
                domains=domains,
                complexity_level=bullet.complexity_level,
                preconditions=bullet.preconditions,
                trigger_patterns=trigger_patterns,
                anti_patterns=bullet.anti_patterns,
                related_bullets=bullet.related_bullets,
                supersedes=bullet.supersedes,
                derived_from=bullet.derived_from,
                successful_contexts=bullet.successful_contexts,
                failure_contexts=bullet.failure_contexts,
                retrieval_type=bullet.retrieval_type,
                embedding_text=embedding_text,
            )
        else:
            # Create new enriched bullet
            enriched = EnrichedBullet(
                id=bullet.id,
                section=bullet.section,
                content=bullet.content,
                helpful=bullet.helpful,
                harmful=bullet.harmful,
                neutral=bullet.neutral,
                created_at=bullet.created_at,
                updated_at=bullet.updated_at,
                task_types=task_types or ["code"],
                domains=domains or ["general"],
                trigger_patterns=trigger_patterns,
                embedding_text=embedding_text,
            )

        return enriched

    def generate_code_triggers(self, code: str, language: str) -> List[str]:
        """Generate trigger patterns from code symbols.

        Extracts:
        - Function/method names
        - Class names
        - Interface names
        - Key programming terms

        Args:
            code: Source code text
            language: Programming language

        Returns:
            List of trigger pattern strings (deduplicated).
        """
        triggers = []

        # Language-specific patterns
        if language == "python":
            # def function_name(
            triggers.extend(re.findall(r'\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)', code))
            # class ClassName
            triggers.extend(re.findall(r'\bclass\s+([a-zA-Z_][a-zA-Z0-9_]*)', code))
            # Extract snake_case identifiers
            triggers.extend(re.findall(r'\b([a-z_][a-z0-9_]{2,})\b', code))

        elif language in ["javascript", "typescript"]:
            # function functionName(
            triggers.extend(re.findall(r'\bfunction\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', code))
            # const/let/var functionName =
            triggers.extend(re.findall(r'\b(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=', code))
            # async function
            triggers.extend(re.findall(r'\basync\s+function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', code))
            # class ClassName
            triggers.extend(re.findall(r'\bclass\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', code))
            # interface InterfaceName (TypeScript)
            triggers.extend(re.findall(r'\binterface\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', code))
            # Arrow functions: const name = () =>
            triggers.extend(re.findall(r'\b(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(?:async\s+)?\(', code))

        elif language == "java":
            # public/private/protected class ClassName
            triggers.extend(re.findall(r'\b(?:public|private|protected)?\s*class\s+([A-Z][a-zA-Z0-9_]*)', code))
            # public/private/protected Type methodName(
            triggers.extend(re.findall(r'\b(?:public|private|protected)\s+(?:\w+\s+)?(\w+)\s*\(', code))

        elif language == "go":
            # func functionName(
            triggers.extend(re.findall(r'\bfunc\s+([a-zA-Z_][a-zA-Z0-9_]*)', code))
            # type TypeName struct
            triggers.extend(re.findall(r'\btype\s+([A-Z][a-zA-Z0-9_]*)\s+struct', code))

        elif language == "rust":
            # fn function_name(
            triggers.extend(re.findall(r'\bfn\s+([a-z_][a-z0-9_]*)', code))
            # struct StructName
            triggers.extend(re.findall(r'\bstruct\s+([A-Z][a-zA-Z0-9_]*)', code))
            # impl TraitName
            triggers.extend(re.findall(r'\bimpl\s+([A-Z][a-zA-Z0-9_]*)', code))

        elif language == "csharp":
            # class ClassName
            triggers.extend(re.findall(r'\bclass\s+([A-Z][a-zA-Z0-9_]*)', code))
            # namespace NamespaceName
            triggers.extend(re.findall(r'\bnamespace\s+([A-Z][a-zA-Z0-9_.]*)', code))
            # public/private Type MethodName(
            triggers.extend(re.findall(r'\b(?:public|private|protected)\s+(?:\w+\s+)?(\w+)\s*\(', code))

        # Extract CamelCase words (common in many languages)
        triggers.extend(re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', code))

        # Extract key programming terms (language-agnostic)
        key_terms = [
            'async', 'await', 'promise', 'callback',
            'error', 'exception', 'try', 'catch', 'throw',
            'null', 'undefined', 'none', 'nil',
            'fetch', 'request', 'response', 'api',
            'database', 'query', 'select', 'insert', 'update',
            'validate', 'validation', 'check',
            'parse', 'serialize', 'deserialize',
            'cache', 'session', 'cookie',
            'auth', 'login', 'logout', 'token',
        ]

        code_lower = code.lower()
        for term in key_terms:
            if term in code_lower and term not in triggers:
                triggers.append(term)

        # Deduplicate while preserving order
        seen = set()
        deduplicated = []
        for trigger in triggers:
            if trigger and trigger not in seen and len(trigger) > 1:
                seen.add(trigger)
                deduplicated.append(trigger)

        return deduplicated

    def detect_code_language(self, text: str) -> Optional[str]:
        """Detect programming language from code patterns.

        Uses heuristic matching of language-specific markers.

        Args:
            text: Code or text to analyze

        Returns:
            Language name (e.g., "python", "javascript") or None if not detected.
        """
        if not text or not text.strip():
            return None

        text_lower = text.lower()

        # Python markers
        python_markers = [
            r'\bdef\s+\w+\s*\(',
            r'\bimport\s+\w+',
            r'\bfrom\s+\w+\s+import',
            r':\s*\w+\s*=',  # Type hints
            r'@\w+',  # Decorators
        ]
        if any(re.search(marker, text) for marker in python_markers):
            return "python"

        # TypeScript markers (check before JavaScript)
        typescript_markers = [
            r'\binterface\s+\w+',
            r':\s*(?:string|number|boolean|any)',
            r'\btype\s+\w+\s*=',
            r'<[A-Z]\w*>',  # Generic types
        ]
        if any(re.search(marker, text) for marker in typescript_markers):
            return "typescript"

        # JavaScript markers
        javascript_markers = [
            r'\bconst\s+\w+\s*=',
            r'\blet\s+\w+\s*=',
            r'=>',  # Arrow functions
            r'\basync\s+function',
            r'\bawait\s+',
            r'\.then\s*\(',
        ]
        if any(re.search(marker, text) for marker in javascript_markers):
            return "javascript"

        # Java markers
        java_markers = [
            r'\bpublic\s+class\s+\w+',
            r'\bprivate\s+\w+\s+\w+\s*;',
            r'\bvoid\s+\w+\s*\(',
            r'\bNullPointerException',
            r'@Override',
        ]
        if any(re.search(marker, text) for marker in java_markers):
            return "java"

        # Go markers
        go_markers = [
            r'\bfunc\s+\w+\s*\(',
            r'\bpackage\s+\w+',
            r'\bimport\s+\(',
            r':=',
            r'\bfmt\.Println',
        ]
        if any(re.search(marker, text) for marker in go_markers):
            return "go"

        # Rust markers
        rust_markers = [
            r'\bfn\s+\w+\s*\(',
            r'\blet\s+mut\s+',
            r'\bimpl\s+\w+',
            r'\buse\s+\w+::',
            r':\s*i32|i64|u32|u64|f32|f64',
        ]
        if any(re.search(marker, text) for marker in rust_markers):
            return "rust"

        # C# markers
        csharp_markers = [
            r'\bnamespace\s+\w+',
            r'\bpublic\s+class\s+\w+',
            r'\busing\s+System',
            r'\bvar\s+\w+\s*=',
        ]
        if any(re.search(marker, text) for marker in csharp_markers):
            return "csharp"

        # SQL markers
        sql_markers = [
            r'\bSELECT\s+',
            r'\bFROM\s+\w+',
            r'\bWHERE\s+',
            r'\bINSERT\s+INTO',
            r'\bUPDATE\s+\w+\s+SET',
        ]
        if any(re.search(marker, text, re.IGNORECASE) for marker in sql_markers):
            return "sql"

        # No language detected
        return None

    def extract_code_blocks(self, text: str) -> List[Tuple[str, str]]:
        """Extract code blocks from markdown-style text.

        Extracts fenced code blocks (```) with optional language hints.

        Args:
            text: Text containing markdown code blocks

        Returns:
            List of (language, code) tuples. Language may be empty string if not specified.
        """
        # Pattern for fenced code blocks: ```language\ncode\n```
        pattern = r'```(\w*)\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)

        blocks = []
        for lang, code in matches:
            # If no language specified, try to detect
            if not lang.strip():
                detected = self.detect_code_language(code)
                lang = detected if detected else ""
            blocks.append((lang, code.strip()))

        return blocks

    def _generate_embedding_text(
        self,
        content: str,
        trigger_patterns: List[str],
        domains: List[str]
    ) -> str:
        """Generate optimized embedding text for code search.

        Combines bullet content with code-derived metadata for better
        semantic matching on code-related queries.

        Args:
            content: Original bullet content
            trigger_patterns: Code symbols and triggers
            domains: Programming languages

        Returns:
            Optimized text for embedding.
        """
        parts = [content]

        # Add domains (languages) for language-specific queries
        if domains:
            parts.append(" ".join(domains))

        # Add top trigger patterns (limit to avoid bloat)
        if trigger_patterns:
            # Take top 10 most distinctive triggers
            top_triggers = trigger_patterns[:10]
            parts.append(" ".join(top_triggers))

        return " ".join(parts)
