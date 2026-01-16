"""
Query preprocessing module for ACE framework.

Provides text normalization, non-query detection, conversational wrapper removal,
and typo correction.
"""

import re
from dataclasses import dataclass, field
from typing import List

from .typo_correction import TypoCorrector


@dataclass
class PreprocessResult:
    """Result of query preprocessing."""
    cleaned_query: str
    is_valid_query: bool
    original_query: str
    transformations_applied: List[str] = field(default_factory=list)


class QueryPreprocessor:
    """Preprocesses queries for semantic search and typo correction."""

    def __init__(self):
        """Initialize query preprocessor."""
        self._typo_corrector = TypoCorrector()

    # Conversational wrappers to remove
    CONVERSATIONAL_PREFIXES = [
        r"^yes!?\s+",
        r"^okay,?\s+",
        r"^alright,?\s+",
        r"^sure,?\s+",
        r"^so,?\s+",
        r"^and\s+",
        r"^well,?\s+",
        r"^now,?\s+",
    ]
    
    # Non-query patterns
    TABLE_PATTERN = r"\|.*\|"  # Simplified: any pipe-separated content
    VERDICT_PATTERN = r"(?i)^verdict:"
    COMMAND_PATTERN = r"^\$\s+"
    JSON_PATTERN = r"^\s*\{.*:.*\}"
    CODE_BLOCK_PATTERN = r"```"
    # Pattern for queries that start with "THIS:" followed by non-query content
    THIS_PATTERN = r"(?i)^this:\s*[\n|]"
    
    def preprocess(self, query: str) -> PreprocessResult:
        """
        Preprocess query text.
        
        Args:
            query: Raw query string
            
        Returns:
            PreprocessResult with cleaned query and metadata
        """
        transformations = []
        original = query
        
        # Strip whitespace first
        if query != query.strip():
            transformations.append("stripped_whitespace")
            query = query.strip()
        
        # Check for non-queries early
        if self._is_non_query(query):
            non_query_type = self._detect_non_query_type(query)
            transformations.append(non_query_type)
            return PreprocessResult(
                cleaned_query="",
                is_valid_query=False,
                original_query=original,
                transformations_applied=transformations
            )
        
        # Check for empty/whitespace-only
        if not query:
            return PreprocessResult(
                cleaned_query="",
                is_valid_query=False,
                original_query=original,
                transformations_applied=transformations
            )
        
        # Normalize case
        if query != query.lower():
            transformations.append("normalized_case")
            query = query.lower()
        
        # Collapse multiple spaces
        if "  " in query:
            transformations.append("collapsed_spaces")
            query = re.sub(r"\s+", " ", query)
        
        # Normalize punctuation
        original_punct = query
        query = re.sub(r"\?{2,}", "?", query)
        query = re.sub(r"!{2,}", "!", query)
        if query != original_punct:
            transformations.append("normalized_punctuation")
        
        # Remove conversational wrappers (iteratively until no more matches)
        original_conv = query
        max_iterations = 10  # Prevent infinite loops
        for _ in range(max_iterations):
            prev_query = query
            for pattern in self.CONVERSATIONAL_PREFIXES:
                query = re.sub(pattern, "", query, flags=re.IGNORECASE)
            query = query.strip()  # Strip after each iteration
            if query == prev_query:  # No more changes
                break
        if query != original_conv:
            transformations.append("removed_conversational_wrapper")
        
        # Final cleanup
        query = query.strip()
        
        # Determine if valid query
        is_valid = bool(query) and not self._is_non_query(query)
        
        return PreprocessResult(
            cleaned_query=query,
            is_valid_query=is_valid,
            original_query=original,
            transformations_applied=transformations
        )
    
    def _is_non_query(self, text: str) -> bool:
        """Check if text is a non-query (table, verdict, etc.)."""
        if not text:
            return True

        # Check for "THIS:" pattern (common copy-paste indicator)
        if re.search(self.THIS_PATTERN, text):
            return True

        # Check for tables (pipe-separated content)
        if re.search(self.TABLE_PATTERN, text, re.MULTILINE):
            return True

        # Check for verdicts
        if re.search(self.VERDICT_PATTERN, text):
            return True

        # Check for command outputs
        if re.search(self.COMMAND_PATTERN, text, re.MULTILINE):
            return True

        # Check for JSON
        if re.search(self.JSON_PATTERN, text, re.DOTALL):
            return True

        # Check for code blocks
        if re.search(self.CODE_BLOCK_PATTERN, text):
            return True

        return False
    
    def _detect_non_query_type(self, text: str) -> str:
        """Detect specific type of non-query."""
        if re.search(self.TABLE_PATTERN, text, re.MULTILINE):
            return "detected_table"
        if re.search(self.VERDICT_PATTERN, text):
            return "detected_verdict"
        if re.search(self.COMMAND_PATTERN, text, re.MULTILINE):
            return "detected_command_output"
        if re.search(self.JSON_PATTERN, text, re.DOTALL):
            return "detected_json"
        if re.search(self.CODE_BLOCK_PATTERN, text):
            return "detected_code_block"
        return "detected_non_query"

    def correct_typos(self, query: str, similarity_threshold: float = 0.80) -> str:
        """
        Correct typos in query using fuzzy matching.

        Args:
            query: Input query string
            similarity_threshold: Minimum similarity ratio (0-1)

        Returns:
            Query with typos corrected
        """
        return self._typo_corrector.correct_typos(query, similarity_threshold)
