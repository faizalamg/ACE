"""
Code Retrieval - Semantic search for code with ThatOtherContextEngine-style output formatting.

This module provides:
1. Semantic search over indexed code chunks
2. ThatOtherContextEngine MCP-compatible output formatting
3. Blended results (code + memory)
4. Result deduplication and ranking

Example usage:
    retriever = CodeRetrieval()
    results = retriever.search("unified memory index")
    formatted = retriever.format_ThatOtherContextEngine_style(results)
"""

import os
import logging
from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CodeResult:
    """A code search result."""
    file_path: str
    content: str
    start_line: int
    end_line: int
    language: str
    score: float
    symbols: List[str] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = []


class CodeRetrieval:
    """Semantic code search with ThatOtherContextEngine-style output formatting.
    
    Queries indexed code chunks and formats results in a way compatible
    with ThatOtherContextEngine MCP output, supporting blended code + memory results.
    
    Uses Voyage-code-3 embeddings (1024d) for optimal code semantic 
    understanding compared to general-purpose embeddings.
    
    Configuration via environment variables:
        QDRANT_URL: Qdrant server URL (default: http://localhost:6333)
        ACE_CODE_COLLECTION: Collection name (default: ace_code_context)
        ACE_CODE_EMBEDDING_URL: Code embedding server URL
        ACE_CODE_EMBEDDING_MODEL: Code embedding model name
    """
    
    def __init__(
        self,
        qdrant_url: str = None,
        collection_name: str = None,
        embed_fn: Callable[[str], List[float]] = None,
    ):
        """
        Initialize code retrieval.
        
        Args:
            qdrant_url: Qdrant server URL (default: from env or localhost:6333)
            collection_name: Qdrant collection name (default: from env or ace_code_context)
            embed_fn: Custom embedding function (default: None, uses Voyage code embedder)
        """
        # Load Voyage code embedding config (required)
        from ace.config import VoyageCodeEmbeddingConfig
        self._voyage_config = VoyageCodeEmbeddingConfig()
        
        if not self._voyage_config.is_configured():
            raise RuntimeError(
                "VOYAGE_API_KEY environment variable is required for code embeddings. "
                "Set VOYAGE_API_KEY to use voyage-code-3."
            )
        
        self.qdrant_url = qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")
        self.collection_name = collection_name or os.environ.get("ACE_CODE_COLLECTION", "ace_code_context")
        self._embed_fn = embed_fn
        self._client = None
        
        self._init_qdrant()
    
    def _init_qdrant(self) -> None:
        """Initialize Qdrant client."""
        try:
            from qdrant_client import QdrantClient
            
            self._client = QdrantClient(url=self.qdrant_url)
            logger.debug(f"Connected to Qdrant at {self.qdrant_url}")
        except ImportError:
            logger.warning("qdrant-client not installed")
            self._client = None
        except Exception as e:
            logger.warning(f"Failed to connect to Qdrant: {e}")
            self._client = None
    
    def _get_embedder(self) -> Optional[Callable[[str], List[float]]]:
        """Get or create code-specific embedding function.
        
        REQUIRES Voyage API (voyage-code-3, 1024d).
        Voyage-code-3 is specifically trained for code retrieval.
        
        Raises:
            RuntimeError: If VOYAGE_API_KEY is not configured
        """
        if self._embed_fn:
            return self._embed_fn
        
        # Voyage API is REQUIRED for code embeddings
        from ace.config import VoyageCodeEmbeddingConfig
        voyage_config = VoyageCodeEmbeddingConfig()
        
        if not voyage_config.is_configured():
            raise RuntimeError(
                "VOYAGE_API_KEY environment variable is required for code embeddings. "
                "Voyage-code-3 is the only supported code embedding model. "
                "Get your API key from https://www.voyageai.com/"
            )
        
        try:
            import voyageai
        except ImportError:
            raise RuntimeError(
                "voyageai package is required for code embeddings. "
                "Install with: pip install voyageai"
            )
        
        # Create Voyage client
        vo_client = voyageai.Client(api_key=voyage_config.api_key)
        logger.info(f"Using Voyage {voyage_config.model} for code embeddings ({voyage_config.dimension}d)")
        
        def voyage_embed(text: str) -> List[float]:
            """Embed text using Voyage code model."""
            try:
                result = vo_client.embed(
                    [text],
                    model=voyage_config.model,
                    input_type=voyage_config.query_input_type
                )
                return result.embeddings[0]
            except Exception as e:
                logger.error(f"Voyage embedding error: {e}")
                return [0.0] * voyage_config.dimension
        
        return voyage_embed
    
    def _expand_query(self, query: str) -> str:
        """
        Expand query with code-specific terms for better retrieval.
        
        Adds function signatures, common patterns, and code keywords
        to improve embedding similarity with actual code.
        """
        import re
        
        # Detect code entity patterns
        expansions = []
        query_lower = query.lower()
        
        # =================================================================
        # CONCEPTUAL METHOD SYNONYMS
        # Maps common conceptual terms to their actual implementation names.
        # E.g., "store" conceptually means "index_bullet" in memory systems
        # =================================================================
        METHOD_SYNONYMS = {
            'store': ['index', 'index_bullet', 'upsert', 'insert', 'save', 'add'],
            'save': ['index', 'index_bullet', 'upsert', 'insert', 'store', 'add'],
            'add': ['index', 'index_bullet', 'insert', 'append', 'push'],
            'get': ['retrieve', 'fetch', 'query', 'search', 'find', 'load'],
            'retrieve': ['get', 'fetch', 'query', 'search', 'find', 'load'],
            'fetch': ['get', 'retrieve', 'query', 'search', 'find', 'load'],
            'find': ['search', 'query', 'get', 'retrieve', 'lookup'],
            'search': ['query', 'find', 'retrieve', 'get', 'lookup'],
            'delete': ['remove', 'drop', 'clear', 'purge', 'erase'],
            'remove': ['delete', 'drop', 'clear', 'purge', 'erase'],
            'update': ['modify', 'edit', 'change', 'set', 'patch'],
            'create': ['initialize', 'init', 'setup', 'make', 'new'],
        }
        
        # Pattern: "ClassName class methodName method" (e.g., "CodeRetrieval class search method")
        # Extract: class name comes before "class", method name comes before "method"
        class_method = re.search(r'(\w+)\s+class\s+(\w+)\s+method', query, re.I)
        if class_method:
            cls_name, method_name = class_method.groups()
            # Add method definition pattern (most important for finding the method)
            expansions.append(f"def {method_name}(")
            # Add class definition
            expansions.append(f"class {cls_name}:")
            # Add synonyms for conceptual method names
            if method_name.lower() in METHOD_SYNONYMS:
                for syn in METHOD_SYNONYMS[method_name.lower()][:3]:  # Top 3 synonyms
                    expansions.append(f"def {syn}(")
        else:
            # Try simpler patterns
            # "methodName method" or "methodName function"
            method_match = re.search(r'(\w+)\s+(method|function)\b', query, re.I)
            if method_match:
                method_name = method_match.group(1)
                # Skip common words
                if method_name.lower() not in ('the', 'a', 'this', 'that', 'my', 'async'):
                    expansions.append(f"def {method_name}(")
                    # Add synonyms for conceptual method names
                    if method_name.lower() in METHOD_SYNONYMS:
                        for syn in METHOD_SYNONYMS[method_name.lower()][:3]:
                            expansions.append(f"def {syn}(")
            
            # "class ClassName" pattern
            class_match = re.search(r'\bclass\s+(\w+)\b', query, re.I)
            if class_match:
                cls_name = class_match.group(1)
                expansions.append(f"class {cls_name}:")
        
        # Also expand standalone conceptual terms (without "method" keyword)
        # E.g., "UnifiedMemoryIndex store implementation" -> add "index_bullet"
        for concept, synonyms in METHOD_SYNONYMS.items():
            if concept in query_lower and f"def {concept}" not in ' '.join(expansions):
                for syn in synonyms[:2]:  # Top 2 synonyms
                    expansions.append(f"def {syn}(")
        
        if expansions:
            return f"{query} {' '.join(expansions)}"
        return query
    
    def _apply_filename_boost(self, query: str, file_path: str, score: float, content: str = "") -> float:
        """
        Apply filename and content boost when query terms match file path or definitions.
        
        This mimics ThatOtherContextEngine MCP's behavior where files with names matching
        query terms OR containing class/function definitions get prioritized.
        
        Args:
            query: Original search query
            file_path: File path being scored
            score: Original embedding similarity score
            content: Code content for definition matching
            
        Returns:
            Boosted score
        """
        import re
        import os
        
        # Extract meaningful terms from query (3+ chars, not common words)
        stop_words = {'the', 'and', 'for', 'with', 'this', 'that', 'from', 'how', 'what',
                      'where', 'when', 'why', 'can', 'will', 'method', 'function', 'class',
                      'code', 'file', 'def', 'implementation', 'search', 'find', 'get', 'set',
                      'pattern', 'error', 'handling', 'import', 'logging', 'logger', 'setup',
                      'exception', 'try', 'except', 'connection', 'settings', 'configuration'}
        query_terms = set()
        for term in re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', query.lower()):
            if len(term) >= 3 and term not in stop_words:
                query_terms.add(term)
        
        # Also extract CamelCase/PascalCase terms as they might be class names
        camel_terms = set()
        for match in re.finditer(r'[A-Z][a-z]+(?:[A-Z][a-z]+)*', query):
            term = match.group().lower()
            if len(term) >= 3:
                camel_terms.add(term)
        
        # Extract path components for matching
        path_lower = file_path.lower()
        filename = os.path.basename(path_lower)
        filename_no_ext = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1].lower()
        dir_parts = os.path.dirname(path_lower).replace('\\', '/').split('/')
        
        boost = 0.0
        
        # ============================================================
        # DOCUMENTATION FILE HANDLING (must be done BEFORE early return)
        # For pattern/concept queries (no specific code entities), docs
        # with code examples should rank higher (matching ThatOtherContextEngine behavior)
        # 
        # CRITICAL FIX: ThatOtherContextEngine strongly prefers CODE over DOCS for most
        # code queries. Even for pattern queries like "error handling patterns",
        # ThatOtherContextEngine returns MOSTLY CODE with some docs mixed in.
        # We should PENALIZE docs (not boost) to match ThatOtherContextEngine behavior.
        #
        # EXCEPTION: Documentation queries (guide, tutorial, setup, quickstart)
        # should BOOST docs because user is explicitly asking for documentation!
        # ============================================================
        if ext in ('.md', '.rst', '.txt'):
            query_lower_for_doc = query.lower()
            
            # FIRST: Check if this is a DOCUMENTATION query
            # These queries EXPLICITLY ask for docs (guide, tutorial, quickstart)
            # User wants documentation, not code!
            doc_query_indicators = {
                'quick start', 'quickstart', 'getting started', 'setup guide',
                'installation guide', 'installation tutorial', 'setup tutorial',
                'readme', 'documentation', 'docs', 'help', 'manual', 'handbook',
                'changelog', 'release notes', 'release history', 'what\'s new',
                'version history', 'contributing', 'license', 'api docs', 'api reference'
            }
            # Strong doc indicators - these almost always want docs
            strong_doc_terms = {'guide', 'tutorial', 'quickstart', 'setup', 'install', 'installation',
                               'changelog', 'release', 'contributing', 'license'}
            
            # Count how many doc terms are present
            doc_term_count = sum(1 for term in strong_doc_terms if term in query_lower_for_doc)
            has_doc_phrase = any(phrase in query_lower_for_doc for phrase in doc_query_indicators)
            
            # If query has 2+ doc terms OR a doc phrase, it's a DOCUMENTATION query
            is_doc_query = doc_term_count >= 2 or has_doc_phrase
            
            # SECOND: Check if this is a PATTERN/CONCEPTUAL query (not doc query)
            # Pattern queries allow docs but don't BOOST them
            pattern_indicators = {'pattern', 'patterns', 'best practice', 'best practices', 
                                  'how to', 'example', 'examples'}
            # Note: 'guide' and 'tutorial' removed from pattern_indicators - handled above
            is_pattern_query = not is_doc_query and any(indicator in query_lower_for_doc for indicator in pattern_indicators)
            
            # Check if query is about a specific code entity (not a general pattern)
            has_code_entity = bool(camel_terms) or any(
                term in query_lower_for_doc for term in ['class', 'dataclass', 'method', 'function', 'def', 'implementation']
            )
            # Check if query contains specific symbol names (PascalCase or snake_case with significant terms)
            has_specific_symbol = bool(re.search(r'[A-Z][a-z]+[A-Z]', query)) or \
                                  bool(re.search(r'\b\w+_\w+\b', query) and not all(
                                      w in stop_words for w in query.lower().split()))
            
            # Check for implementation-specific keywords (ONLY if NOT a pattern query)
            # These indicate user wants ACTUAL CODE, not documentation
            impl_keywords = {
                'threshold', 'similarity', 'overlapping', 'duplicate', 'deduplicate',
                'merge', 'filter', 'score', 'ranking', 'vector', 'embedding',
                'algorithm', 'calculate', 'compute', 'process', 'parse', 'validate',
                'encode', 'decode', 'serialize', 'deserialize', 'transform',
                'chunk', 'split', 'batch', 'limit', 'offset', 'cursor', 'pagination',
                'buffer', 'queue', 'stack', 'graph', 'node'
            }
            has_impl_terms = not is_pattern_query and sum(1 for kw in impl_keywords if kw in query_lower_for_doc) >= 2
            
            # Count non-stop-word terms - if query has 3+ specific terms AND NOT a pattern query
            non_stop_terms = [t for t in query_terms if t not in stop_words and len(t) >= 4]
            is_specific_query = not is_pattern_query and len(non_stop_terms) >= 3
            
            if is_doc_query:
                # User is asking for DOCUMENTATION - BOOST docs!
                # Skip internal memory/ace files - they're not user documentation
                is_internal_doc = '.ace/' in path_lower or 'memories/' in path_lower or 'checkpoints/' in path_lower
                
                if is_internal_doc:
                    # Internal memory files should be treated neutrally
                    boost += 0.0
                else:
                    # Check if doc file matches query terms (strong_doc_terms + key doc_query words)
                    doc_path_terms = strong_doc_terms | {'readme', 'documentation', 'docs', 'quick_start', 'quickstart', 
                                                          'changelog', 'contributing', 'license', 'release', 'history'}
                    has_matching_terms = any(term in path_lower for term in doc_path_terms if term in query_lower_for_doc)
                    
                    if has_matching_terms:
                        boost += 0.35  # Strong boost for matching doc files (e.g., README.md for "readme")
                    else:
                        boost += 0.15  # Moderate boost for other docs
            elif has_code_entity or has_specific_symbol or has_impl_terms or is_specific_query:
                # User is looking for ACTUAL CODE - strong penalty for docs
                if has_impl_terms:
                    boost -= 0.40  # Strong penalty for impl term queries
                else:
                    boost -= 0.30  # Standard penalty for code entity queries
            elif is_pattern_query:
                # For PATTERN queries, docs are OK but should NOT outrank code
                # Apply small penalty to let code naturally rank higher
                # ONLY give tiny boost if doc has actual code examples AND is a guide
                has_code_examples = content and ('```' in content or 'try:' in content.lower() or 'except' in content.lower())
                is_guide_or_ref = 'docs/' in file_path.lower() or 'guide' in file_path.lower() or 'reference' in file_path.lower()
                
                if has_code_examples and is_guide_or_ref:
                    boost -= 0.05  # Small penalty (not boost!) - docs still rank lower than code
                else:
                    boost -= 0.15  # Medium penalty for docs without code examples
            else:
                # Generic queries - medium penalty for docs, prefer code
                boost -= 0.20
            
            # Cap boost and return early for docs (no filename matching needed)
            boost = min(boost, 0.50)
            return score + boost
        
        # ============================================================
        # NON-PRODUCTION DIRECTORY PENALTY
        # Files in training, examples, scripts, benchmarks, tests directories
        # should be deprioritized vs core module files. This is a general
        # code organization pattern, not project-specific.
        # ============================================================
        non_prod_patterns = {
            'training', 'examples', 'scripts', 'benchmarks', 'experiments',
            'sandbox', 'playground', 'demo', 'sample', 'tutorial', 'test_data',
            'finetuning', 'finetuned', 'optimizations', 'prototype', 'scratch',
            'tests', 'test', 'spec', 'specs', '__tests__'  # Added test directories
        }
        path_parts_for_check = file_path.lower().replace('\\', '/').split('/')
        is_non_prod = any(
            any(pattern in part for pattern in non_prod_patterns)
            for part in path_parts_for_check[:-1]  # Check directories, not filename
        )
        
        non_prod_penalty = 0.0
        if is_non_prod:
            non_prod_penalty = -0.25  # Increased penalty for non-production directories
        
        # ============================================================
        # DEMO/EXAMPLE/TEST FILENAME PENALTY
        # Files with demo_, example_, sample_, test_ in filename
        # are demonstration/test files, not implementations. Penalize them
        # so the actual implementation file ranks higher.
        # Also penalize conftest.py (pytest fixtures) and _test.py suffix
        # ============================================================
        demo_filename_prefixes = ('demo_', 'example_', 'sample_', 'test_')
        demo_filename_suffixes = ('_test.py', '_spec.py', '_tests.py')
        demo_filename_exact = ('conftest.py', 'test.py', 'tests.py')
        
        if filename_no_ext.startswith(demo_filename_prefixes):
            non_prod_penalty -= 0.35  # Strong penalty for demo/test prefix files
            logger.debug(f"DEMO FILE PENALTY: {file_path} gets -{0.35}")
        elif filename.endswith(demo_filename_suffixes) or filename in demo_filename_exact:
            non_prod_penalty -= 0.30  # Penalty for test suffix files
            logger.debug(f"TEST FILE PENALTY: {file_path} gets -{0.30}")
        
        # Early return for non-code-entity queries (no filename boosting to do)
        if not query_terms and not camel_terms:
            return score + non_prod_penalty
        
        boost = non_prod_penalty
        matched_terms = 0
        
        # ============================================================
        # CONTENT-FIRST BOOSTING STRATEGY
        # Only apply filename boost when content ACTUALLY CONTAINS the
        # queried symbol/term. This prevents files with matching filenames
        # from outranking files with actual implementations.
        # ============================================================
        
        # Pre-check: does content contain ANY of the specific query terms?
        # If looking for "_expand_query" and file doesn't contain it, NO BOOST
        content_lower = content.lower() if content else ""
        
        # Extract specific symbol terms (snake_case or CamelCase identifiers)
        specific_symbols = set()
        # snake_case like _expand_query, get_config
        for match in re.finditer(r'_?[a-z][a-z0-9]*(?:_[a-z0-9]+)+', query.lower()):
            if len(match.group()) >= 3:
                specific_symbols.add(match.group())
        # CamelCase/PascalCase like UnifiedMemoryIndex
        for match in re.finditer(r'[A-Z][a-z]+(?:[A-Z][a-z]+)+', query):
            specific_symbols.add(match.group().lower())
        
        # If query has specific symbols, file MUST contain them to get any boost
        has_required_symbols = True
        if specific_symbols:
            has_required_symbols = any(sym in content_lower for sym in specific_symbols)
        else:
            # No specific symbols - this is a phrase-based query.
            # For phrase queries like "Qdrant REST error", only boost files
            # that contain the key terms together (not just in filename).
            # Extract non-stop-word terms from query
            phrase_terms = [t for t in query_terms if t not in stop_words and len(t) >= 4]
            if len(phrase_terms) >= 2:
                # Check if file contains all key terms
                terms_in_content = sum(1 for t in phrase_terms if t in content_lower)
                # Require at least 2 key terms present to get filename boost
                has_required_symbols = terms_in_content >= 2
        
        # ============================================================
        # EXACT PHRASE MATCH BOOST
        # If the query appears as an exact phrase in the content, give
        # a significant boost. This helps when chunking is coarse and
        # the embedding doesn't capture the specific phrase well.
        # 
        # CRITICAL: Skip boost if phrase appears inside a string literal
        # (quotes). This prevents scripts that use the query as a variable
        # from outranking files with actual implementations.
        # ============================================================
        query_lower = query.lower().strip()
        # Remove common prefix words for phrase matching
        phrase_to_match = query_lower
        for prefix in ['how to ', 'what is ', 'where is ', 'find ', 'search ', 'show ']:
            if phrase_to_match.startswith(prefix):
                phrase_to_match = phrase_to_match[len(prefix):]
                break
        
        # Check for exact phrase match (at least 3 words, 15 chars)
        if len(phrase_to_match) >= 15 and ' ' in phrase_to_match:
            if phrase_to_match in content_lower:
                # CRITICAL: Check if phrase appears inside a string literal
                # Look for patterns like: "phrase" or 'phrase' or """phrase"""
                # If found inside quotes, this is likely a test/example file, not implementation
                phrase_in_string_literal = False
                for quote_char in ['"', "'"]:
                    # Check for phrase surrounded by quotes (within ~50 chars)
                    idx = content_lower.find(phrase_to_match)
                    if idx != -1:
                        # Look backwards for opening quote
                        start_check = max(0, idx - 50)
                        before_phrase = content_lower[start_check:idx]
                        after_phrase = content_lower[idx + len(phrase_to_match):idx + len(phrase_to_match) + 50]
                        
                        # Count quotes before and after - odd count before means inside string
                        quotes_before = before_phrase.count(quote_char)
                        if quotes_before % 2 == 1:  # Odd = inside string
                            phrase_in_string_literal = True
                            break
                        
                        # Also check for assignment pattern: query = "phrase" or similar
                        if re.search(rf'(?:query|q|search|text)\s*=\s*["\']', before_phrase[-30:] if len(before_phrase) >= 30 else before_phrase):
                            phrase_in_string_literal = True
                            break
                
                if phrase_in_string_literal:
                    # PENALIZE files that contain query as string literal
                    # This is almost certainly a test/example file, not implementation
                    # The embedding score is artificially inflated by exact text match
                    boost -= 0.25  # Strong penalty for string literal false positive
                else:
                    boost += 0.30  # Strong boost for exact phrase match in actual code
        
        # ============================================================
        # FILENAME PRIORITY BOOST
        # When a query topic is clearly reflected in the filename,
        # that file should rank higher. This is INDEPENDENT of symbol
        # requirements - we're matching concept to filename.
        # E.g., "email validation" -> email_validator.py should rank above
        # demo_email_validation.py because the filename matches the concept
        # ============================================================
        # Extract underscore-joinable query terms
        query_words = [w for w in query_lower.split() if w not in stop_words and len(w) >= 3]
        if len(query_words) >= 2:
            # Create potential filename patterns
            for i in range(len(query_words)):
                for j in range(i + 1, min(i + 3, len(query_words) + 1)):
                    pattern_words = query_words[i:j]
                    snake_pattern = '_'.join(pattern_words)
                    # Check if filename IS the pattern (primary concept file)
                    if filename_no_ext == snake_pattern or filename_no_ext == snake_pattern + 's':
                        boost += 0.35  # Strong boost for PRIMARY concept file
                    elif snake_pattern in filename_no_ext:
                        # Check if pattern is prefix vs suffix vs embedded
                        if filename_no_ext.startswith(snake_pattern):
                            boost += 0.15  # Moderate boost for prefix match
                        elif filename_no_ext.endswith(snake_pattern):
                            boost += 0.20  # Good boost for suffix match (e.g., email_validator)
                        else:
                            boost += 0.08  # Small boost for embedded match
        
        # ============================================================
        # CORE MODULE BOOST
        # Files in the primary source directory (ace/) without test prefix/suffix
        # should rank higher than test files with the same name pattern.
        # E.g., ace/code_chunker.py > tests/test_code_chunker.py
        # ============================================================
        is_core_module = (
            not is_non_prod and
            ext == '.py' and
            not filename_no_ext.startswith(demo_filename_prefixes) and
            not filename.endswith(demo_filename_suffixes) and
            filename not in demo_filename_exact
        )
        
        if is_core_module:
            # Additional boost for core source files
            boost += 0.20
            
            # Extra boost if filename directly matches a query term
            for term in query_terms | camel_terms:
                if term in filename_no_ext:
                    boost += 0.10
                    break
        
        # Check each query term against path
        for term in query_terms | camel_terms:
            # Direct filename match is strongest - BUT ONLY IF content has required symbols
            if term in filename_no_ext and has_required_symbols:
                # Exact match in filename (e.g., "unified_memory" in "unified_memory.py")
                if term == filename_no_ext or f"_{term}" in filename_no_ext or f"{term}_" in filename_no_ext:
                    boost += 0.20  # Stronger boost for exact term in filename
                else:
                    boost += 0.10  # Partial match
                matched_terms += 1
            # Directory match (e.g., "ace" in "ace/unified_memory.py") - less strict
            elif term in dir_parts:
                boost += 0.06
                matched_terms += 1
        
        # Bonus for multiple term matches (only if content verified)
        if matched_terms >= 2 and has_required_symbols:
            boost += 0.06 * (matched_terms - 1)
        
        # Special handling for snake_case compound terms - ONLY IF content verified
        # e.g., "unified memory" should match "unified_memory"
        if has_required_symbols:
            words = re.findall(r'[a-zA-Z]+', query.lower())
            for i in range(len(words) - 1):
                if words[i] not in stop_words and words[i+1] not in stop_words:
                    compound = f"{words[i]}_{words[i+1]}"
                    if compound in filename_no_ext:
                        boost += 0.15
        
        # PENALTY: If file lacks required symbols but got filename match, PENALIZE
        # This prevents "retrieval_optimized.py" from outranking "code_retrieval.py"
        # when query is "_expand_query function code retrieval" and retrieval_optimized
        # doesn't contain _expand_query
        if specific_symbols and not has_required_symbols and matched_terms > 0:
            boost -= 0.15  # Penalize misleading filename matches
        
        # CONTENT-BASED BOOST: Boost when content contains class/dataclass definitions
        # that match query terms (this helps config.py rank higher for "QdrantConfig" queries)
        if content:
            # content_lower already set above, but ensure query_lower is set
            query_lower = query.lower()
            
            # ============================================================
            # PATTERN/CONCEPT QUERY DETECTION
            # For queries like "error handling patterns" or "logging patterns",
            # we want files that are PRIMARILY ABOUT the topic, not just files
            # that happen to mention the term. This prevents false positives
            # like fibonacci.py ranking high for "error handling" just because
            # its docstring mentions "comprehensive error handling".
            # ============================================================
            pattern_terms = {'pattern', 'patterns', 'example', 'examples', 'best practice',
                            'how to', 'guide', 'tutorial', 'handling'}
            is_pattern_query = any(term in query_lower for term in pattern_terms)
            
            # Boost for exact multi-word query phrase matches in content
            # BUT only if the file has MULTIPLE occurrences (indicates it's about the topic)
            # OR if this is NOT a pattern query (specific symbol lookups are different)
            query_phrases = []
            query_words = query_lower.split()
            for i in range(len(query_words)):
                for j in range(i + 2, min(i + 5, len(query_words) + 1)):  # 2-4 word phrases
                    phrase = ' '.join(query_words[i:j])
                    if len(phrase) >= 10:  # Only meaningful phrases
                        query_phrases.append(phrase)
            
            for phrase in query_phrases:
                if phrase in content_lower:
                    if is_pattern_query:
                        # For pattern queries, require MULTIPLE occurrences to boost
                        # This prevents single mention in docstring from getting boost
                        occurrence_count = content_lower.count(phrase)
                        if occurrence_count >= 2:
                            boost += 0.15  # Reduced boost for repeated mentions
                        # Single mention gets no boost for pattern queries
                    else:
                        # For specific queries, single phrase match is still valuable
                        boost += 0.18
                    break  # Only apply once
            
            # ============================================================
            # CONFIG-SPECIFIC BOOSTING
            # When query mentions "config/configuration/settings", strongly
            # prefer files that DEFINE Config classes that MATCH query terms.
            # E.g., "embedding model configuration" -> boost EmbeddingConfig
            # but NOT TrainingConfig (which is about training, not embeddings)
            # ============================================================
            config_query_terms = {'config', 'configuration', 'settings', 'options'}
            query_mentions_config = any(term in query_lower for term in config_query_terms)
            
            if query_mentions_config:
                # Find all Config class definitions in content
                config_matches = re.findall(
                    r'class\s+(\w*config\w*)\s*[:\(]',
                    content_lower,
                    re.IGNORECASE
                )
                
                if config_matches:
                    # Check if any config class name contains OTHER query terms
                    # (not just "config" - we want EmbeddingConfig for "embedding config")
                    other_query_terms = query_terms - config_query_terms
                    
                    for config_name in config_matches:
                        config_name_lower = config_name.lower()
                        for term in other_query_terms:
                            if term in config_name_lower:
                                # Config class name matches a query term!
                                # E.g., "EmbeddingConfig" contains "embedding"
                                boost += 0.25
                                break
                        else:
                            continue
                        break  # Only apply boost once
            
            # ============================================================
            # DEFINITION vs USAGE DISCRIMINATION
            # Strongly prefer files that DEFINE a symbol over files that
            # just IMPORT or USE it. This is crucial for disambiguation.
            # ============================================================

            # ONLY check SPECIFIC symbols (class names, function names, methods)
            # NOT general query terms like "method", "function", "retrieve"
            # Use specific_symbols (extracted above) instead of all query_terms
            symbols_to_check = specific_symbols if specific_symbols else camel_terms
            
            # Track if we found ANY definition or just imports/usages
            found_definition = False
            found_import_only = False
            found_usage_only = False
            
            for term in symbols_to_check:
                term_lower = term.lower()
                # Check if this file DEFINES the term (exact class/function name, not substring)
                # The term must be followed by : or ( to be an actual definition
                has_definition = (
                    f'class {term_lower}:' in content_lower or
                    f'class {term_lower}(' in content_lower or
                    f'def {term_lower}(' in content_lower
                )
                # Also check for dataclass definition (must be exact class name)
                if '@dataclass' in content_lower:
                    # Check for class TermName: or class TermName( after @dataclass
                    if f'class {term_lower}:' in content_lower or f'class {term_lower}(' in content_lower:
                        has_definition = True

                # Check if file only IMPORTS the term (no definition)
                has_import = (
                    f'from ' in content_lower and f'import {term_lower}' in content_lower or
                    f'import {term_lower}' in content_lower or
                    f'from ' in content_lower and f' {term_lower}' in content_lower and 'import' in content_lower
                )

                # Check if file just USES the term (variable assignment, function call)
                has_usage = term_lower in content_lower

                if has_definition:
                    found_definition = True
                    # Strong boost for DEFINITION - use exact checks
                    is_dataclass = '@dataclass' in content_lower and (
                        f'class {term_lower}:' in content_lower or f'class {term_lower}(' in content_lower
                    )
                    is_class = f'class {term_lower}:' in content_lower or f'class {term_lower}(' in content_lower
                    is_function = f'def {term_lower}(' in content_lower
                    
                    if is_dataclass:
                        boost += 0.25
                    elif is_class:
                        boost += 0.20
                    elif is_function:
                        boost += 0.15
                    # Don't break - check all symbols
                elif has_import and not has_definition:
                    found_import_only = True
                elif has_usage and not has_definition:
                    found_usage_only = True
            
            # Apply penalties ONLY if NO definition was found for ANY queried symbol
            if not found_definition:
                if found_import_only:
                    # PENALTY for import-only files (they just use, don't define)
                    boost -= 0.15  # Stronger penalty
                elif found_usage_only:
                    # Small penalty for usage-only (could be example/script)
                    boost -= 0.08
            
            # ============================================================
            # CLASS METHOD QUERY BOOST
            # For queries like "ClassName class methodName method", we need
            # to STRONGLY boost chunks that contain the actual method definition
            # because embeddings often rank the class docstring higher than
            # the method implementation (which may be hundreds of lines away).
            # ============================================================
            class_method_match = re.search(r'(\w+)\s+class\s+(\w+)\s+method', query, re.I)
            if class_method_match:
                target_cls = class_method_match.group(1).lower()
                target_method = class_method_match.group(2).lower()
                
                # Check if this chunk contains the target method definition
                has_method_def = f'def {target_method}(' in content_lower
                has_class = f'class {target_cls}' in content_lower
                
                if has_method_def:
                    # STRONG boost for chunks containing the actual method definition
                    # This overcomes the embedding bias toward docstrings/class headers
                    boost += 0.30
                    logger.debug(f"Method definition boost: {file_path} has def {target_method}(")
                
                # Don't penalize class-only chunks - they provide context too
                # But method definition chunks should rank HIGHER
            
            # ============================================================
            # CONCEPTUAL METHOD SYNONYM BOOST
            # For queries like "UnifiedMemoryIndex store method", the user
            # is asking about the CONCEPTUAL "store" operation which maps to
            # "index_bullet" in the actual implementation. Boost chunks that
            # contain method definitions for synonymous operations.
            # ============================================================
            METHOD_SYNONYMS = {
                'store': ['index', 'index_bullet', 'upsert', 'insert', 'save', 'add'],
                'save': ['index', 'index_bullet', 'upsert', 'insert', 'store', 'add'],
                'add': ['index', 'index_bullet', 'insert', 'append', 'push'],
                'get': ['retrieve', 'fetch', 'query', 'search', 'find', 'load'],
                'retrieve': ['get', 'fetch', 'query', 'search', 'find', 'load'],
                'fetch': ['get', 'retrieve', 'query', 'search', 'find', 'load'],
                'find': ['search', 'query', 'get', 'retrieve', 'lookup'],
                'search': ['query', 'find', 'retrieve', 'get', 'lookup'],
                'delete': ['remove', 'drop', 'clear', 'purge', 'erase'],
                'remove': ['delete', 'drop', 'clear', 'purge', 'erase'],
                'update': ['modify', 'edit', 'change', 'set', 'patch'],
                'create': ['initialize', 'init', 'setup', 'make', 'new'],
            }
            
            # Check for conceptual method terms in query
            for concept, synonyms in METHOD_SYNONYMS.items():
                if concept in query_lower and 'method' in query_lower:
                    # User is asking about a conceptual operation
                    # Check if this chunk contains a synonym method definition
                    for syn in synonyms:
                        if f'def {syn}(' in content_lower:
                            # Found a synonym method definition!
                            boost += 0.25
                            logger.debug(f"Conceptual synonym boost: {file_path} has def {syn}( for concept '{concept}'")
                            break
                    break  # Only process one conceptual match
            
            # ============================================================
            # TECHNICAL IDENTIFIER BOOST
            # For queries with hyphenated technical terms like "voyage-code-3",
            # "bge-m3", "gpt-4", "claude-3", etc., strongly boost files that
            # actually contain these specific identifiers. This overcomes
            # semantic similarity issues where generic files about the same
            # topic (e.g., "embeddings") rank higher than files specifically
            # using the mentioned technology.
            # ============================================================
            # Extract hyphenated technical identifiers (model names, versions)
            tech_identifiers = set()
            for match in re.finditer(r'[a-zA-Z]+(?:-[a-zA-Z0-9]+)+', query):
                tech_id = match.group().lower()
                if len(tech_id) >= 5:  # Meaningful identifiers
                    tech_identifiers.add(tech_id)
            
            if tech_identifiers and content:
                # Count how many technical identifiers are present in content
                tech_matches = sum(1 for tid in tech_identifiers if tid in content_lower)
                tech_mentions = sum(content_lower.count(tid) for tid in tech_identifiers)
                
                if tech_matches > 0:
                    # Strong boost for files containing the specific technical term
                    # Base boost + extra for multiple mentions (indicates this file is ABOUT the tech)
                    boost += 0.25 + min(0.15, tech_mentions * 0.02)
                    logger.debug(f"Technical identifier boost: {file_path} has {tech_matches} identifiers, {tech_mentions} mentions")
                else:
                    # Small penalty for files missing mentioned technical identifiers
                    # This helps demote generic files that don't actually discuss the specific tech
                    boost -= 0.10
            
            # ============================================================
            # CLASS NAME BOOST
            # For queries mentioning specific class names (PascalCase),
            # strongly boost files that actually define that class.
            # This is critical for queries like "PlaybookManager class initialization"
            # where semantic similarity ranks generic files higher than the
            # file actually containing the PlaybookManager class.
            # ============================================================
            # Extract class names from query (supports multiple patterns):
            # - Standard PascalCase: CodeRetrieval, UnifiedMemoryIndex
            # - Acronym PascalCase: HyDEGenerator, BM25Config, APIClient
            # - Mixed case: JSON, HTTP, etc.
            class_names = set()
            # Pattern 1: Standard PascalCase (CodeRetrieval)
            for match in re.finditer(r'(?<![a-z])([A-Z][a-z]+(?:[A-Z][a-z]+)+)', query):
                class_name = match.group(1)
                if len(class_name) >= 4:  # Meaningful class names
                    class_names.add(class_name.lower())
            # Pattern 2: Acronym PascalCase (HyDEGenerator, BM25Config)
            for match in re.finditer(r'([A-Z][A-Z]*[a-z]+[A-Za-z0-9]*)(?=\s|$)', query):
                class_name = match.group(1)
                if len(class_name) >= 4 and not class_name.isupper():
                    class_names.add(class_name.lower())
            
            if class_names and content:
                # Check if file contains class definition for any mentioned class
                for class_name in class_names:
                    # Look for class definition patterns
                    class_def_pattern = f'class {class_name}[^a-z]'  # class ClassName( or class ClassName:
                    if re.search(class_def_pattern, content_lower, re.I):
                        # VERY STRONG boost for files that actually DEFINE the class
                        # This overcomes embedding bias toward method names
                        boost += 0.45
                        logger.debug(f"Class name boost: {file_path} defines class {class_name}")
                        break
                else:
                    # Check if any class name is mentioned (usage, not definition)
                    class_mentioned = any(cn in content_lower for cn in class_names)
                    if class_mentioned:
                        # Smaller boost for files that mention the class (could be imports)
                        boost += 0.05
                    else:
                        # STRONG penalty for files that don't have the mentioned class
                        # If user asks about "HyDEGenerator class", files without HyDEGenerator
                        # should rank lower than files that define it
                        boost -= 0.20
        
        # Cap boost to avoid excessive inflation
        # Non-production directories should NOT benefit from filename matches
        # They can only lose score (caps at 0, not gain)
        if is_non_prod:
            boost = min(boost, 0.0)  # No positive boost for non-prod
        else:
            boost = min(boost, 0.60)  # Increased cap for class name boost
        
        return score + boost
    
    def search(
        self,
        query: str,
        limit: int = 10,
        deduplicate: bool = True,
        min_score: float = 0.0,
        use_reranker: bool = False,  # Disabled by default - text rerankers hurt code retrieval
        exclude_tests: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant code chunks using dense vector search.
        
        Uses Voyage-code-3 embeddings (1024d) for semantic similarity.
        BM25 hybrid search is disabled because it hurts code retrieval quality -
        exact term matching prefers imports/usages over definitions.
        
        Args:
            query: Natural language search query
            limit: Maximum number of results
            deduplicate: Whether to deduplicate overlapping results
            min_score: Minimum relevance score (0.0-1.0)
            use_reranker: Whether to use cross-encoder reranking (disabled by default
                         because text-based rerankers prefer docstrings over implementations)
            exclude_tests: Whether to exclude test files from results (default: True)
        
        Returns:
            List of code results with metadata
        """
        if not self._client:
            logger.warning("Qdrant client not available")
            return []
        
        embedder = self._get_embedder()
        if not embedder:
            logger.warning("No embedding function available")
            return []
        
        try:
            # Expand query with code-specific terms for better retrieval
            expanded_query = self._expand_query(query)
            logger.debug(f"Expanded query: {expanded_query}")
            
            # Embed expanded query using code-specific model
            query_vector = embedder(expanded_query)
            
            # Determine if this is a pattern/concept query (no specific symbols)
            # Pattern queries need more results because docs rank lower with code embeddings
            import re
            has_camel_case = bool(re.search(r'[A-Z][a-z]+(?:[A-Z][a-z]+)+', query))
            has_snake_case_symbol = bool(re.search(r'_[a-z]+_|^[a-z]+_[a-z]+$', query))
            is_pattern_query = not has_camel_case and not has_snake_case_symbol and any(
                term in query.lower() for term in ['pattern', 'example', 'how to', 'error handling', 'logging', 'exception']
            )
            
            # For phrase queries (multi-word, no specific symbols), embedding similarity
            # may not surface the exact phrase match - need to fetch more and rely on boosting
            is_phrase_query = (
                ' ' in query and  # Multi-word
                len(query) >= 15 and  # Meaningful phrase
                not has_camel_case and  # No class names
                not has_snake_case_symbol  # No function/method names
            )
            
            # Detect "ClassName class methodName method" pattern - need to fetch MORE
            # because the method chunk may rank low (embedding prefers docstrings)
            is_class_method_query = bool(re.search(r'\w+\s+class\s+\w+\s+method', query, re.I))
            
            # Fetch more results when excluding tests to compensate for filtering
            fetch_limit = limit * 3 if (use_reranker or deduplicate) else limit
            if exclude_tests:
                fetch_limit = fetch_limit * 2  # Double fetch for filtering
            if is_pattern_query:
                # Pattern/concept queries need to fetch more to get docs (ranked lower by code embeddings)
                fetch_limit = max(fetch_limit, 100)
            if is_phrase_query:
                # Phrase queries need larger fetch to find exact phrase matches that
                # embeddings may rank low (coarse chunks dilute phrase signal)
                fetch_limit = max(fetch_limit, 200)
            if is_class_method_query:
                # "ClassName class methodName method" queries need large fetch because
                # the method implementation chunk often ranks low (embeddings prefer
                # docstrings/class headers over method bodies far down in the file)
                fetch_limit = max(fetch_limit, 200)
            
            # ============================================================
            # EXACT PHRASE TEXT SEARCH FALLBACK
            # For phrase queries, embeddings may not surface chunks with
            # the exact phrase (due to coarse chunking diluting signal).
            # Do a text filter search to find exact matches and merge.
            # NOTE: Qdrant text search is case-sensitive, so we try original case
            # ============================================================
            exact_phrase_matches = []
            if is_phrase_query:
                # Remove common prefix words for phrase matching (keep original case)
                phrase_to_search = query.strip()
                for prefix in ['How to ', 'how to ', 'What is ', 'what is ', 'Where is ', 
                               'where is ', 'Find ', 'find ', 'Search ', 'search ', 'Show ', 'show ']:
                    if phrase_to_search.startswith(prefix):
                        phrase_to_search = phrase_to_search[len(prefix):]
                        break
                
                # Text filter search for exact phrase in content (case-sensitive)
                import httpx
                text_search_payload = {
                    "filter": {
                        "must": [{"key": "content", "match": {"text": phrase_to_search}}]
                    },
                    "limit": 20,
                    "with_payload": True,
                }
                try:
                    text_response = httpx.post(
                        f"{self.qdrant_url}/collections/{self.collection_name}/points/scroll",
                        json=text_search_payload,
                        timeout=10.0,
                    )
                    if text_response.status_code == 200:
                        text_result = text_response.json()
                        exact_phrase_matches = text_result.get("result", {}).get("points", [])
                        logger.debug(f"Found {len(exact_phrase_matches)} exact phrase matches for '{phrase_to_search}'")
                except Exception as e:
                    logger.warning(f"Text search failed: {e}")
            
            # Use dense-only search - Voyage-code-3 embeddings are
            # specifically trained for semantic code similarity, and BM25/sparse
            # actually hurts results by matching imports/usages over definitions
            import httpx
            query_payload = {
                "query": query_vector,
                "using": "dense",
                "limit": fetch_limit,
                "with_payload": True,
            }
            
            rest_response = httpx.post(
                f"{self.qdrant_url}/collections/{self.collection_name}/points/query",
                json=query_payload,
                timeout=30.0,
            )
            
            if rest_response.status_code != 200:
                raise Exception(f"Qdrant REST error: {rest_response.text}")
            
            rest_result = rest_response.json()
            results = rest_result.get("result", {}).get("points", [])
            
            # Convert to result dicts (REST API returns dicts, not objects)
            code_results = []
            for point in results:
                payload = point.get("payload", {})
                score = point.get("score", 0.0)
                
                # Validate required fields
                if not payload.get("file_path") or not payload.get("content"):
                    continue
                
                if score < min_score:
                    continue
                
                file_path = payload.get("file_path", "")

                # Filter out test files if requested
                if exclude_tests:
                    # Skip files that are tests
                    if self._is_test_file(file_path):
                        continue

                # Filter out noisy config files
                if self._should_exclude_result(file_path):
                    continue

                content = payload.get("content", "")
                
                # Apply filename and content boost - boost score when query terms match
                # filename/path OR when content contains matching class/function definitions
                boosted_score = self._apply_filename_boost(query, file_path, score, content)
                
                code_results.append({
                    "file_path": file_path,
                    "content": payload.get("content", ""),
                    "start_line": payload.get("start_line", 1),
                    "end_line": payload.get("end_line", 1),
                    "language": payload.get("language", ""),
                    "symbols": payload.get("symbols", []),
                    "score": boosted_score,
                    "original_score": score,  # Keep original for debugging
                })
            
            # ============================================================
            # MERGE EXACT PHRASE MATCHES
            # Add results from text search that weren't in vector results
            # These get high boost since they contain the exact phrase
            # ============================================================
            if exact_phrase_matches:
                # Get file paths already in results
                existing_files = {r["file_path"] + str(r.get("start_line", 0)) for r in code_results}
                
                for point in exact_phrase_matches:
                    payload = point.get("payload", {})
                    file_path = payload.get("file_path", "")
                    start_line = payload.get("start_line", 1)
                    content = payload.get("content", "")

                    # Skip if already in results
                    key = file_path + str(start_line)
                    if key in existing_files:
                        continue

                    # Skip test files if requested
                    if exclude_tests and self._is_test_file(file_path):
                        continue

                    # Filter out noisy config files
                    if self._should_exclude_result(file_path):
                        continue

                    # Give exact phrase matches a synthetic high score
                    # Since they don't have embedding scores, use a baseline + boost
                    baseline_score = 0.5  # Reasonable baseline
                    boosted_score = self._apply_filename_boost(query, file_path, baseline_score, content)
                    
                    code_results.append({
                        "file_path": file_path,
                        "content": content,
                        "start_line": start_line,
                        "end_line": payload.get("end_line", 1),
                        "language": payload.get("language", ""),
                        "symbols": payload.get("symbols", []),
                        "score": boosted_score,
                        "original_score": baseline_score,
                        "exact_phrase_match": True,  # Flag for debugging
                    })
                    logger.debug(f"Added exact phrase match: {file_path} with boosted score {boosted_score:.4f}")
            
            # Re-sort by boosted score
            code_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Cross-encoder reranking for precision boost (disabled by default for code)
            if use_reranker and code_results and len(code_results) > 1:
                code_results = self._rerank_results(query, code_results)
            
            # Deduplicate overlapping chunks
            if deduplicate:
                code_results = self._deduplicate_results(code_results)
            
            return code_results[:limit]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    


    def _is_test_file(self, file_path: str) -> bool:
        """Check if a file path is a test file.

        Args:
            file_path: Relative file path

        Returns:
            True if file is a test file
        """
        path_lower = file_path.lower()

        # Test directory patterns
        if '/tests/' in path_lower or '\\tests\\' in path_lower:
            return True
        if '/test/' in path_lower or '\\test\\' in path_lower:
            return True

        # Test file name patterns
        import os
        filename = os.path.basename(path_lower)
        if filename.startswith('test_'):
            return True
        if filename.endswith('_test.py'):
            return True
        if filename == 'conftest.py':
            return True

        return False

    def _should_exclude_result(self, file_path: str) -> bool:
        """Check if a result should be excluded from results.

        Args:
            file_path: Relative file path

        Returns:
            True if file should be excluded (noisy config files, etc.)
        """
        path_lower = file_path.lower()

        # Exclude noisy IDE/config files that match generic queries
        noisy_patterns = [
            '.claude/settings.json',
            '.vscode/settings.json',
            '.idea/',
            'node_modules/',
            '.venv/',
            'venv/',
            '__pycache__/',
        ]

        for pattern in noisy_patterns:
            if pattern.replace('/', '\\') in path_lower or pattern.replace('\\', '/') in path_lower:
                return True

        return False

    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using cross-encoder for precision boost.
        
        Uses ms-marco-MiniLM-L-6-v2 model (same as memory retrieval).
        """
        if not results or len(results) <= 1:
            return results
        
        try:
            from ace.reranker import get_reranker
            
            reranker = get_reranker()
            
            # Prepare documents (truncate content for efficiency)
            documents = [r["content"][:500] for r in results]
            
            # Get cross-encoder scores
            ce_scores = reranker.predict(query, documents)
            
            # Sort by cross-encoder score (higher is better)
            scored = list(zip(results, ce_scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            
            # Update scores with CE scores for transparency
            reranked = []
            for r, ce_score in scored:
                r = r.copy()
                r["score"] = ce_score  # Use CE score as final score
                reranked.append(r)
            
            return reranked
            
        except ImportError:
            logger.warning("Reranker not available, skipping")
            return results
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return results

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate and merge overlapping/adjacent code chunks from the same file.
        
        When multiple chunks from the same file are retrieved, this merges adjacent
        or overlapping chunks to provide fuller context. This helps when a query
        like "class X method Y" retrieves both the class definition and the method
        as separate chunks - we merge them into one combined result.
        """
        if not results:
            return results
        
        # Group by file
        by_file: Dict[str, List[Dict[str, Any]]] = {}
        for r in results:
            path = r["file_path"]
            if path not in by_file:
                by_file[path] = []
            by_file[path].append(r)
        
        # Merge within each file
        merged_results = []
        for file_path, file_results in by_file.items():
            if len(file_results) == 1:
                merged_results.append(file_results[0])
                continue
            
            # Sort by start line
            file_results.sort(key=lambda x: x["start_line"])
            
            # Merge overlapping and adjacent chunks (within 10 lines gap)
            merged = []
            for r in file_results:
                if not merged:
                    merged.append(dict(r))  # Copy to avoid mutation
                    continue
                
                last = merged[-1]
                gap = r["start_line"] - last["end_line"]
                
                # Merge if overlapping or adjacent (within 10 lines)
                if gap <= 10:
                    # Merge: combine content and expand line range
                    # Read the full content from start to end
                    new_end = max(last["end_line"], r["end_line"])
                    
                    # Combine content by reading lines from file
                    # For now, concatenate with separator if not overlapping
                    if r["start_line"] > last["end_line"]:
                        # There's a gap - need to read intermediate lines
                        # For simplicity, just concatenate with marker
                        last["content"] = last["content"] + "\n...\n" + r["content"]
                    else:
                        # Overlapping - just use the larger chunk
                        if len(r["content"]) > len(last["content"]):
                            last["content"] = r["content"]
                            last["start_line"] = min(last["start_line"], r["start_line"])
                    
                    last["end_line"] = new_end
                    # Take the higher score
                    last["score"] = max(last["score"], r["score"])
                    # Combine symbols
                    if "symbols" in r and r["symbols"]:
                        last["symbols"] = list(set(last.get("symbols", []) + r["symbols"]))
                else:
                    # Not adjacent - keep separate
                    merged.append(dict(r))
            
            merged_results.extend(merged)
        
        # Re-sort by score
        merged_results.sort(key=lambda x: x["score"], reverse=True)
        return merged_results

    def expand_context(
        self,
        results: List[Dict[str, Any]],
        context_lines_before: int = 20,
        context_lines_after: int = 20,
        max_lines: int = 300,
    ) -> List[Dict[str, Any]]:
        """
        Expand results with surrounding context from the actual source files.
        
        This provides ThatOtherContextEngine-like behavior where results include surrounding code
        context beyond just the indexed chunk boundaries.
        
        Args:
            results: Code search results from search()
            context_lines_before: Lines to include before the matched chunk
            context_lines_after: Lines to include after the matched chunk
            max_lines: Maximum total lines per result (prevents huge contexts)
        
        Returns:
            Results with expanded content including surrounding context
        """
        if not results:
            return results
        
        expanded = []
        for result in results:
            file_path = result["file_path"]
            start_line = result.get("start_line", 1)
            end_line = result.get("end_line", start_line)
            
            try:
                # Read the actual file
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        file_lines = f.readlines()
                else:
                    # If file doesn't exist (maybe indexed from different location), use original
                    expanded.append(result)
                    continue
                
                total_file_lines = len(file_lines)
                
                # Calculate expanded range
                new_start = max(1, start_line - context_lines_before)
                new_end = min(total_file_lines, end_line + context_lines_after)
                
                # Ensure we don't exceed max_lines
                if new_end - new_start + 1 > max_lines:
                    # Center around the original chunk
                    center = (start_line + end_line) // 2
                    half = max_lines // 2
                    new_start = max(1, center - half)
                    new_end = min(total_file_lines, new_start + max_lines - 1)
                
                # Extract lines (convert to 0-indexed)
                expanded_content = "".join(file_lines[new_start - 1 : new_end])
                
                # Create expanded result
                expanded_result = dict(result)
                expanded_result["content"] = expanded_content.rstrip()
                expanded_result["start_line"] = new_start
                expanded_result["end_line"] = new_end
                expanded_result["original_start_line"] = start_line
                expanded_result["original_end_line"] = end_line
                expanded_result["expanded"] = True
                
                expanded.append(expanded_result)
                
            except Exception as e:
                logger.warning(f"Could not expand context for {file_path}: {e}")
                expanded.append(result)
        
        return expanded
    
    def format_ThatOtherContextEngine_style(
        self,
        results: List[Dict[str, Any]],
        max_lines_per_file: int = 200,
        expand_context: bool = True,
        context_lines_before: int = 20,
        context_lines_after: int = 20,
    ) -> str:
        """
        Format results in ThatOtherContextEngine MCP-compatible style.
        
        ThatOtherContextEngine format:
            The following code sections were retrieved:
            Path: file/path.py
                 1	line content
                 2	line content
            ...
            Path: another/file.py
                10	more content
        
        Args:
            results: Code search results
            max_lines_per_file: Maximum lines to show per file (truncate if more)
            expand_context: Whether to expand results with surrounding context
            context_lines_before: Lines to include before the matched chunk
            context_lines_after: Lines to include after the matched chunk
        
        Returns:
            Formatted string in ThatOtherContextEngine style
        """
        if not results:
            return "No relevant code sections found."
        
        # Optionally expand context to provide more surrounding code
        if expand_context:
            results = self.expand_context(
                results,
                context_lines_before=context_lines_before,
                context_lines_after=context_lines_after,
                max_lines=max_lines_per_file,
            )
        
        lines = ["The following code sections were retrieved:"]
        
        for result in results:
            file_path = result["file_path"]
            content = result["content"]
            start_line = result.get("start_line", 1)
            
            lines.append(f"Path: {file_path}")
            
            # Split content into lines
            content_lines = content.split("\n")
            
            # Truncate if needed
            truncated = False
            if len(content_lines) > max_lines_per_file:
                content_lines = content_lines[:max_lines_per_file]
                truncated = True
            
            # Format with line numbers (right-aligned, 6 chars wide)
            for i, line_content in enumerate(content_lines):
                line_num = start_line + i
                # ThatOtherContextEngine uses right-aligned line numbers with tab separator
                formatted_line = f"{line_num:>6}\t{line_content}"
                lines.append(formatted_line)
            
            if truncated:
                lines.append("...")
            
            # Add blank line between files
            lines.append("")
        
        return "\n".join(lines).strip()
    
    def retrieve_blended(
        self,
        query: str,
        code_limit: int = 5,
        memory_limit: int = 5,
    ) -> Dict[str, Any]:
        """
        Retrieve both code and memory results.
        
        Args:
            query: Search query
            code_limit: Max code results
            memory_limit: Max memory results
        
        Returns:
            Dict with 'code_results' and 'memory_results' keys
        """
        # Get code results
        code_results = self.search(query, limit=code_limit)
        
        # Get memory results
        memory_results = []
        try:
            from ace.unified_memory import UnifiedMemoryIndex
            
            memory_index = UnifiedMemoryIndex()
            # Use retrieve() not search() - UnifiedMemoryIndex.retrieve() returns UnifiedBullets
            memories = memory_index.retrieve(query, limit=memory_limit)
            
            for mem in memories:
                # UnifiedBullet has attributes, not dict keys
                memory_results.append({
                    "content": getattr(mem, "content", ""),
                    "category": getattr(mem, "category", ""),
                    "namespace": getattr(mem, "namespace", "").value if hasattr(getattr(mem, "namespace", ""), "value") else str(getattr(mem, "namespace", "")),
                    "section": getattr(mem, "section", ""),
                    "severity": getattr(mem, "severity", 5),
                })
        except Exception as e:
            logger.warning(f"Failed to retrieve memories: {e}")
        
        return {
            "code_results": code_results,
            "memory_results": memory_results,
        }
    
    def format_blended(self, results: Dict[str, Any]) -> str:
        """
        Format blended code + memory results.
        
        Args:
            results: Dict from retrieve_blended()
        
        Returns:
            Formatted string with both code and memory sections
        """
        parts = []
        
        # Code section first
        code_results = results.get("code_results", [])
        if code_results:
            code_formatted = self.format_ThatOtherContextEngine_style(code_results)
            parts.append(code_formatted)
        
        # Memory section
        memory_results = results.get("memory_results", [])
        if memory_results:
            if parts:
                parts.append("")  # Blank line separator
            
            parts.append("Relevant memories:")
            for mem in memory_results:
                content = mem.get("content", "")
                category = mem.get("category", "")
                severity = mem.get("severity", 5)
                
                parts.append(f"- [{category}] (Severity: {severity}) {content}")
        
        if not parts:
            return "No relevant code or memories found."
        
        return "\n".join(parts)
    
    def to_mcp_response(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format results as MCP tool response.
        
        Args:
            results: Code search results
        
        Returns:
            MCP-compatible response dict
        """
        formatted = self.format_ThatOtherContextEngine_style(results)
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": formatted,
                }
            ]
        }


def search_code(query: str, limit: int = 10) -> str:
    """
    Convenience function for searching code.
    
    Args:
        query: Search query
        limit: Max results
    
    Returns:
        ThatOtherContextEngine-style formatted results
    """
    retriever = CodeRetrieval()
    results = retriever.search(query, limit=limit)
    return retriever.format_ThatOtherContextEngine_style(results)


def search_blended(query: str, code_limit: int = 5, memory_limit: int = 5) -> str:
    """
    Convenience function for blended search.
    
    Args:
        query: Search query
        code_limit: Max code results
        memory_limit: Max memory results
    
    Returns:
        Formatted blended results
    """
    retriever = CodeRetrieval()
    results = retriever.retrieve_blended(query, code_limit, memory_limit)
    return retriever.format_blended(results)
