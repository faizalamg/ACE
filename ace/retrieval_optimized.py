"""
Optimized Retrieval Module for ACE

This module implements state-of-the-art retrieval techniques based on
extensive RAG optimization research and testing:

Best Configuration (V2 - 62.52% R@5):
- Hybrid search (dense + BM25 sparse + RRF)
- Query expansion (4 variations)
- Multi-query retrieval with RRF fusion
- Cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2)

Performance:
- Recall@1: 41.71% (vs 22.06% baseline)
- Recall@5: 62.52% (vs 53.28% baseline)
- MRR: 0.5077 (vs 0.3583 baseline)

Usage:
    retriever = OptimizedRetriever()
    results = retriever.search("how to validate user input")
"""

import hashlib
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
import logging

from .config import EmbeddingConfig, QdrantConfig, BM25Config, RetrievalConfig, LLMConfig, ELFConfig

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    from .gpu_reranker import GPUCrossEncoder
    CROSS_ENCODER_AVAILABLE = True
    GPU_RERANKER_AVAILABLE = True
except ImportError:
    GPU_RERANKER_AVAILABLE = False
    try:
        from sentence_transformers import CrossEncoder
        CROSS_ENCODER_AVAILABLE = True
    except ImportError:
        CROSS_ENCODER_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION (loads from centralized config with overridable defaults)
# ============================================================================

def _get_default_config() -> Dict[str, Any]:
    """Get default configuration from centralized config."""
    _embedding_config = EmbeddingConfig()
    _qdrant_config = QdrantConfig()
    _bm25_config = BM25Config()
    _retrieval_config = RetrievalConfig()

    return {
        # Qdrant connection
        "qdrant_url": _qdrant_config.url,
        "collection_name": _qdrant_config.memories_collection,

        # Embedding
        "embedding_url": _embedding_config.url,
        "embedding_model": _embedding_config.model,

        # Retrieval parameters from centralized config
        "num_expanded_queries": _retrieval_config.num_expanded_queries,
        "candidates_per_query": _retrieval_config.candidates_per_query,
        "first_stage_k": _retrieval_config.first_stage_k,
        "final_k": _retrieval_config.final_k,

        # Re-ranking from centralized config
        "cross_encoder_model": _retrieval_config.cross_encoder_model,
        "enable_reranking": _retrieval_config.enable_reranking,

        # BM25 parameters from centralized config
        "bm25_k1": _bm25_config.k1,
        "bm25_b": _bm25_config.b,
        "avg_doc_length": _bm25_config.avg_doc_length,

        # RRF parameter
        "rrf_k": 60,
    }

# Stopwords for BM25
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
    'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have',
    'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
    'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'it', 'its',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'what',
    'which', 'who', 'whom', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
    'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now'
}

# Technical synonyms for query expansion
SYNONYMS = {
    'api': ['endpoint', 'interface', 'service'],
    'config': ['configuration', 'settings', 'options'],
    'db': ['database', 'datastore', 'storage'],
    'auth': ['authentication', 'authorization', 'login', 'security'],
    'ui': ['interface', 'frontend', 'view', 'component'],
    'validate': ['verify', 'check', 'ensure', 'confirm'],
    'create': ['generate', 'build', 'construct', 'initialize'],
    'delete': ['remove', 'destroy', 'cleanup', 'clear'],
    'update': ['modify', 'change', 'edit', 'patch'],
    'fetch': ['get', 'retrieve', 'load', 'query'],
    'store': ['save', 'persist', 'cache', 'write'],
    'handle': ['process', 'manage', 'respond to'],
    'parse': ['process', 'interpret', 'decode', 'extract'],
    'test': ['spec', 'unit test', 'verification'],
    'error': ['exception', 'failure', 'issue', 'problem', 'bug'],
    'fix': ['resolve', 'repair', 'correct', 'patch'],
    'debug': ['troubleshoot', 'diagnose', 'investigate'],
    'function': ['method', 'procedure', 'routine', 'handler'],
    'class': ['type', 'model', 'entity', 'object'],
    'module': ['package', 'library', 'component'],
    'input': ['data', 'parameter', 'argument'],
    'output': ['result', 'response', 'return value'],
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class RetrievalResult:
    """A single retrieval result with metadata."""
    id: int
    score: float
    payload: Dict[str, Any]
    content: str
    category: Optional[str] = None
    reranked: bool = False


@dataclass
class SearchMetrics:
    """Metrics for a search operation."""
    total_latency_ms: float
    expansion_latency_ms: float
    retrieval_latency_ms: float
    rerank_latency_ms: float
    num_candidates: int
    num_results: int
    expanded_queries: List[str]


# ============================================================================
# QUERY COMPLEXITY CLASSIFIER
# ============================================================================

class QueryComplexityClassifier:
    """
    Classify query complexity to determine if LLM rewriting is needed.

    Technical queries with clear intent can skip expensive LLM rewriting
    in favor of efficient keyword expansion.
    """

    # Technical terms that indicate clear intent (skip LLM)
    TECHNICAL_TERMS = {
        'api', 'endpoint', 'config', 'configuration', 'settings',
        'error', 'exception', 'bug', 'fix', 'debug',
        'async', 'await', 'promise', 'callback',
        'validate', 'sanitize', 'auth', 'authentication',
        'database', 'query', 'sql', 'cache',
        'test', 'mock', 'spec', 'unittest',
        'import', 'export', 'module', 'package',
        'class', 'function', 'method', 'variable',
        'git', 'commit', 'branch', 'merge',
        'docker', 'container', 'deploy', 'ci', 'cd',
        'install', 'build', 'compile', 'run',
        'route', 'middleware', 'handler', 'controller',
        'model', 'view', 'schema', 'migration',
        'log', 'logging', 'monitor', 'trace',
        'parse', 'serialize', 'deserialize', 'encode',
        'request', 'response', 'http', 'https',
        'token', 'session', 'cookie', 'jwt',
        'permission', 'role', 'access', 'security',
        'encrypt', 'decrypt', 'hash', 'sign',
        'file', 'path', 'directory', 'folder',
    }

    def __init__(self, config: ELFConfig = None):
        self.config = config or ELFConfig()

    def needs_llm_rewrite(self, query: str) -> bool:
        """
        Determine if query needs expensive LLM rewriting.

        Returns True if LLM rewriting would add value.
        Returns False if simple keyword expansion is sufficient.
        """
        if not self.config.enable_query_classifier:
            return True  # Classifier disabled, always use LLM

        words = query.lower().split()

        # If query has technical terms, keyword expansion is sufficient
        if self.config.technical_terms_bypass_llm:
            if any(w in self.TECHNICAL_TERMS for w in words):
                logger.debug(f"Query '{query}' contains technical terms, skipping LLM rewrite")
                return False

        # Short queries without technical terms need LLM for semantic expansion
        if len(words) <= 3:
            logger.debug(f"Query '{query}' is short without technical terms, needs LLM")
            return True

        # Longer technical queries don't need LLM
        logger.debug(f"Query '{query}' is longer technical query, skipping LLM")
        return False


# ============================================================================
# QUERY EXPANSION
# ============================================================================

class QueryExpander:
    """Rule-based query expansion for improved retrieval coverage."""

    def __init__(self, synonyms: Dict[str, List[str]] = None):
        self.synonyms = synonyms or SYNONYMS

    def expand(self, query: str, num_expansions: int = 3) -> List[str]:
        """
        Expand a query into multiple variations.

        Returns list starting with original query.
        """
        expansions = [query]
        word_count = len(query.split())

        # AGGRESSIVE expansion for very short queries (3 words or less)
        if word_count <= 3:
            short_expansions = self._expand_short_query(query)
            expansions.extend(short_expansions)
        else:
            # Standard expansion for longer queries
            # Synonym expansion
            synonym_version = self._expand_synonyms(query)
            if synonym_version != query:
                expansions.append(synonym_version)

            # Question reformulation
            question_version = self._reformulate_question(query)
            if question_version and question_version != query:
                expansions.append(question_version)

            # Technical term expansion
            tech_version = self._expand_technical_terms(query)
            if tech_version != query:
                expansions.append(tech_version)

            # Context addition for short queries (4-5 words)
            if word_count <= 5:
                context_version = self._add_context(query)
                if context_version != query:
                    expansions.append(context_version)

        # Deduplicate
        seen = set()
        unique = []
        for exp in expansions:
            if exp.lower() not in seen:
                seen.add(exp.lower())
                unique.append(exp)

        return unique[:num_expansions + 1]

    def _expand_short_query(self, query: str) -> List[str]:
        """
        Aggressive expansion for very short queries (<=3 words).

        Short queries lack context, so we generate multiple rich variations.
        """
        expansions = []
        words = query.lower().split()

        # 1. Multiple synonym combinations
        syn_version = self._expand_synonyms(query)
        if syn_version != query:
            expansions.append(syn_version)

        # 2. All synonyms for each word (creates richer query)
        all_syns = []
        for word in words:
            clean = re.sub(r'[^\w]', '', word)
            if clean in self.synonyms:
                all_syns.extend(self.synonyms[clean][:2])  # Top 2 synonyms
            all_syns.append(word)
        if all_syns:
            expansions.append(' '.join(all_syns))

        # 3. Question formulations
        expansions.append(f"how to {query}")
        expansions.append(f"what is {query}")

        # 4. Context-rich expansion
        context_version = self._add_context(query)
        if context_version != query:
            expansions.append(context_version)

        # 5. Technical term expansion
        tech_version = self._expand_technical_terms(query)
        if tech_version != query:
            expansions.append(tech_version)

        # 6. Phrase completion with common patterns
        phrase_completions = self._complete_short_phrase(query)
        expansions.extend(phrase_completions)

        return expansions

    def _complete_short_phrase(self, query: str) -> List[str]:
        """Add common technical phrase completions for short queries."""
        completions = []
        query_lower = query.lower()

        # Common technical phrase patterns (expanded for better coverage)
        patterns = [
            (('error', 'bug', 'issue', 'problem'), ['handling', 'fix', 'resolution', 'debugging', 'recovery']),
            (('test', 'testing'), ['unit', 'integration', 'coverage', 'automation', 'verification']),
            (('api', 'endpoint'), ['design', 'integration', 'call', 'response', 'request']),
            (('data', 'input'), ['validation', 'processing', 'transformation', 'sanitization']),
            (('config', 'setting'), ['management', 'file', 'environment', 'options']),
            (('auth', 'login', 'user'), ['authentication', 'authorization', 'session', 'permission']),
            (('cache', 'memory'), ['management', 'optimization', 'storage', 'eviction']),
            (('file', 'path'), ['handling', 'system', 'operations', 'access']),
            (('log', 'logging'), ['system', 'monitoring', 'debug', 'trace']),
            (('deploy', 'build'), ['pipeline', 'automation', 'process', 'release']),
            # Additional patterns for common failures
            (('validate', 'sanitize', 'check'), ['input', 'data', 'before processing', 'parameters']),
            (('directive', 'always', 'never'), ['follow', 'enforce', 'apply', 'rule']),
            (('frustration', 'correction'), ['lesson learned', 'feedback', 'improvement']),
            (('when', 'before', 'after'), ['processing', 'execution', 'operation']),
            (('isolate', 'extract', 'filter'), ['logic', 'component', 'module', 'function']),
        ]

        for triggers, suffixes in patterns:
            if any(t in query_lower for t in triggers):
                for suffix in suffixes[:3]:  # Add top 3 completions
                    completions.append(f"{query} {suffix}")
                break

        # Generic completions if no pattern matched
        if not completions:
            completions.append(f"{query} implementation")
            completions.append(f"{query} best practices")
            completions.append(f"{query} guidelines")

        return completions

    def _expand_synonyms(self, query: str) -> str:
        words = query.lower().split()
        expanded = []
        for word in words:
            clean = re.sub(r'[^\w]', '', word)
            if clean in self.synonyms:
                expanded.append(self.synonyms[clean][0])
            else:
                expanded.append(word)
        return ' '.join(expanded)

    def _reformulate_question(self, query: str) -> Optional[str]:
        query_lower = query.lower().strip()
        patterns = {
            'how': 'what is the approach to',
            'what': 'which',
            'why': 'what is the reason for',
            'where': 'in which location',
        }
        for qword, replacement in patterns.items():
            if query_lower.startswith(qword + ' '):
                rest = query[len(qword) + 1:].strip()
                return f"{replacement} {rest}"
        if not query_lower.endswith('?') and not any(
            query_lower.startswith(q) for q in ['how', 'what', 'why', 'where']
        ):
            return f"how to {query}"
        return None

    def _expand_technical_terms(self, query: str) -> str:
        expansions = {
            'api': 'API application programming interface',
            'ui': 'UI user interface',
            'db': 'database',
            'config': 'configuration',
            'auth': 'authentication',
            'env': 'environment',
        }
        words = query.split()
        expanded = []
        for word in words:
            clean = word.lower().strip('.,!?')
            if clean in expansions:
                expanded.append(expansions[clean])
            else:
                expanded.append(word)
        return ' '.join(expanded)

    def _add_context(self, query: str) -> str:
        query_lower = query.lower()
        context_map = {
            ('validate', 'input', 'check', 'sanitize'): 'data validation security',
            ('error', 'exception', 'handle'): 'error handling recovery',
            ('test', 'mock', 'assert'): 'testing quality assurance',
            ('api', 'endpoint', 'request'): 'API design integration',
            ('config', 'setting', 'option'): 'configuration management',
        }
        for keywords, context in context_map.items():
            if any(w in query_lower for w in keywords):
                return f"{query} {context}"
        return f"{query} best practices"


# ============================================================================
# QUERY SPECIFICITY SCORING & ADAPTIVE EXPANSION
# ============================================================================

@dataclass
class QuerySpecificityScore:
    """Query specificity analysis result."""
    word_count: int
    specificity_score: float  # 0.0 (vague) to 1.0 (highly specific)
    expansion_level: str  # 'maximum', 'moderate', 'minimal', 'none'
    use_llm_expansion: bool
    use_structured_expansion: bool
    expansion_terms_limit: int  # Max terms to add per domain
    rationale: str


class QuerySpecificityScorer:
    """
    Scores query specificity to determine optimal expansion strategy.
    
    Based on research: Short queries benefit from aggressive expansion,
    while long/specific queries can be degraded by over-expansion.
    
    Scoring factors:
    - Word count (primary factor)
    - Technical term density
    - Named entity presence
    - Question completeness
    - Domain specificity indicators
    """
    
    # Technical terms that indicate specificity
    TECHNICAL_TERMS = {
        'api', 'auth', 'oauth', 'jwt', 'cors', 'csrf', 'xss', 'sql', 'nosql',
        'rest', 'graphql', 'grpc', 'http', 'https', 'websocket', 'tcp', 'udp',
        'docker', 'kubernetes', 'k8s', 'nginx', 'redis', 'postgres', 'mongodb',
        'aws', 'azure', 'gcp', 's3', 'ec2', 'lambda', 'cloudfront', 'dynamodb',
        'react', 'vue', 'angular', 'svelte', 'nextjs', 'nuxt', 'remix',
        'python', 'javascript', 'typescript', 'java', 'golang', 'rust', 'cpp',
        'async', 'await', 'promise', 'callback', 'middleware', 'decorator',
        'unittest', 'pytest', 'jest', 'mocha', 'cypress', 'playwright',
        'git', 'github', 'gitlab', 'cicd', 'jenkins', 'terraform', 'ansible',
        'exception', 'error', 'traceback', 'stacktrace', 'timeout', 'deadlock',
        'cache', 'memoize', 'lazy', 'eager', 'singleton', 'factory', 'observer',
    }
    
    # Version patterns (highly specific)
    VERSION_PATTERN = re.compile(r'\bv?\d+\.\d+(?:\.\d+)?(?:-\w+)?\b', re.IGNORECASE)
    
    # File path patterns (highly specific)
    PATH_PATTERN = re.compile(r'[\w/\\]+\.[a-zA-Z]{1,5}\b|/[\w/\\]+')
    
    # Error code patterns (highly specific)
    ERROR_CODE_PATTERN = re.compile(r'\b[A-Z]{2,}[-_]?\d+\b|\b\d{3,}\b')
    
    # Vague/ambiguous terms
    VAGUE_TERMS = {
        'issue', 'problem', 'bug', 'error', 'broke', 'broken', 'wrong',
        'help', 'need', 'want', 'something', 'stuff', 'thing', 'things',
        'not working', 'doesn\'t work', 'failed', 'failing',
    }
    
    def __init__(self):
        self.technical_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(t) for t in self.TECHNICAL_TERMS) + r')\b',
            re.IGNORECASE
        )
        self.vague_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(t) for t in self.VAGUE_TERMS) + r')\b',
            re.IGNORECASE
        )
    
    def score(self, query: str) -> QuerySpecificityScore:
        """
        Score query specificity and determine expansion strategy.
        
        Returns:
            QuerySpecificityScore with all expansion parameters
        """
        words = query.split()
        word_count = len(words)
        
        # Calculate specificity factors (0.0 to 1.0 each)
        length_score = self._score_length(word_count)
        technical_score = self._score_technical_terms(query)
        entity_score = self._score_entities(query)
        vagueness_penalty = self._score_vagueness(query)
        
        # IMPORTANT: If query has technical terms, reduce vagueness penalty
        # e.g., "authentication error" - "authentication" is specific even though "error" is vague
        if technical_score > 0.2:
            vagueness_penalty *= 0.5  # Halve the penalty when technical terms present
        
        # Weighted combination
        specificity_score = (
            length_score * 0.4 +
            technical_score * 0.3 +
            entity_score * 0.2 -
            vagueness_penalty * 0.3
        )
        specificity_score = max(0.0, min(1.0, specificity_score))
        
        # Determine expansion level based on score
        expansion_config = self._determine_expansion_level(word_count, specificity_score)
        
        return QuerySpecificityScore(
            word_count=word_count,
            specificity_score=round(specificity_score, 3),
            **expansion_config
        )
    
    def _score_length(self, word_count: int) -> float:
        """Score based on word count. Longer = more specific."""
        if word_count <= 2:
            return 0.1
        elif word_count <= 3:
            return 0.25
        elif word_count <= 5:
            return 0.5
        elif word_count <= 8:
            return 0.75
        else:
            return 0.95
    
    def _score_technical_terms(self, query: str) -> float:
        """Score based on technical term density."""
        matches = self.technical_pattern.findall(query)
        word_count = len(query.split())
        if word_count == 0:
            return 0.0
        density = len(matches) / word_count
        # Cap at 0.5 density for max score
        return min(1.0, density * 2)
    
    def _score_entities(self, query: str) -> float:
        """Score based on presence of specific entities."""
        score = 0.0
        
        # Version numbers are highly specific
        if self.VERSION_PATTERN.search(query):
            score += 0.4
        
        # File paths are highly specific
        if self.PATH_PATTERN.search(query):
            score += 0.3
        
        # Error codes are highly specific
        if self.ERROR_CODE_PATTERN.search(query):
            score += 0.3
        
        return min(1.0, score)
    
    def _score_vagueness(self, query: str) -> float:
        """Penalize vague/ambiguous terms."""
        matches = self.vague_pattern.findall(query)
        word_count = len(query.split())
        if word_count == 0:
            return 0.0
        # Higher penalty for higher density of vague terms
        return min(1.0, len(matches) / max(1, word_count - 1))
    
    def _determine_expansion_level(
        self, word_count: int, specificity_score: float
    ) -> dict:
        """
        Determine expansion configuration based on word count and specificity.
        
        Strategy:
        - Ultra-vague queries (specificity < 0.15): SKIP expansion - let semantic search handle
        - Short queries (≤3 words): Maximum expansion (unless ultra-vague)
        - Medium queries (4-8 words): Moderate expansion
        - Long queries (≥9 words): Minimal/no expansion
        
        Specificity score adjusts within each tier.
        
        IMPORTANT: Ultra-vague queries like "something is broken" perform WORSE
        with expansion because generic terms like "stack trace", "logging" etc.
        pollute the embedding space and pull in technical docs instead of the
        user's actual intent (which is often about communication/frustration).
        """
        # CRITICAL FIX: Ultra-vague queries should SKIP expansion entirely
        # These queries rely purely on semantic similarity to find relevant content
        # Expansion terms like "stack trace", "debugging" etc. actively HARM results
        if specificity_score < 0.15:
            return {
                'expansion_level': 'semantic_only',
                'use_llm_expansion': False,
                'use_structured_expansion': False,
                'expansion_terms_limit': 0,
                'rationale': f'Ultra-vague query (specificity {specificity_score:.2f}): skipping expansion - semantic search only'
            }
        
        # Short queries (≤3 words): Maximum expansion
        if word_count <= 3:
            if specificity_score >= 0.7:
                # Short but specific (e.g., "pytest fixtures async")
                return {
                    'expansion_level': 'moderate',
                    'use_llm_expansion': False,  # Specific enough
                    'use_structured_expansion': True,
                    'expansion_terms_limit': 3,
                    'rationale': f'Short query ({word_count} words) with high specificity ({specificity_score:.2f}): moderate expansion'
                }
            elif specificity_score >= 0.3:
                # Short with some specificity - use expansion but conservatively
                return {
                    'expansion_level': 'moderate',
                    'use_llm_expansion': False,
                    'use_structured_expansion': True,
                    'expansion_terms_limit': 3,
                    'rationale': f'Short query ({word_count} words) with moderate specificity ({specificity_score:.2f}): moderate expansion'
                }
            else:
                # Short and vague but not ultra-vague - still expand but limit LLM
                return {
                    'expansion_level': 'conservative',
                    'use_llm_expansion': False,  # LLM can over-expand vague queries
                    'use_structured_expansion': True,
                    'expansion_terms_limit': 2,  # Reduced from 5
                    'rationale': f'Short vague query ({word_count} words, specificity {specificity_score:.2f}): conservative expansion'
                }
        
        # Medium queries (4-8 words): Moderate expansion
        elif word_count <= 8:
            if specificity_score >= 0.6:
                # Medium, fairly specific
                return {
                    'expansion_level': 'minimal',
                    'use_llm_expansion': False,
                    'use_structured_expansion': True,
                    'expansion_terms_limit': 2,
                    'rationale': f'Medium query ({word_count} words) with good specificity ({specificity_score:.2f}): minimal expansion'
                }
            else:
                # Medium, somewhat vague
                return {
                    'expansion_level': 'moderate',
                    'use_llm_expansion': False,
                    'use_structured_expansion': True,
                    'expansion_terms_limit': 3,
                    'rationale': f'Medium query ({word_count} words) with moderate specificity ({specificity_score:.2f}): moderate expansion'
                }
        
        # Long queries (≥9 words): Minimal/no expansion
        else:
            if specificity_score >= 0.5:
                # Long and specific enough
                return {
                    'expansion_level': 'none',
                    'use_llm_expansion': False,
                    'use_structured_expansion': False,
                    'expansion_terms_limit': 0,
                    'rationale': f'Long query ({word_count} words) with sufficient specificity ({specificity_score:.2f}): no expansion needed'
                }
            else:
                # Long but still vague (unusual)
                return {
                    'expansion_level': 'minimal',
                    'use_llm_expansion': False,
                    'use_structured_expansion': True,
                    'expansion_terms_limit': 2,
                    'rationale': f'Long query ({word_count} words) but low specificity ({specificity_score:.2f}): minimal expansion'
                }


class AdaptiveExpansionController:
    """
    Controls query expansion based on QuerySpecificityScorer.
    
    Integrates with both:
    - StructuredQueryEnhancer (rule-based domain expansion)
    - LLMQueryRewriter (GLM-powered expansion)
    """
    
    def __init__(self):
        self.scorer = QuerySpecificityScorer()
        self._structured_enhancer = None
        self._llm_rewriter = None
    
    @property
    def structured_enhancer(self):
        """Lazy-load structured enhancer."""
        if self._structured_enhancer is None:
            try:
                from ace.structured_enhancer import StructuredQueryEnhancer
                self._structured_enhancer = StructuredQueryEnhancer()
            except ImportError:
                logger.warning("StructuredQueryEnhancer not available")
        return self._structured_enhancer
    
    def analyze(self, query: str) -> QuerySpecificityScore:
        """Analyze query and return specificity score."""
        return self.scorer.score(query)
    
    def expand(
        self,
        query: str,
        llm_rewriter: Optional['LLMQueryRewriter'] = None,
        force_level: Optional[str] = None,  # Override: 'maximum', 'moderate', 'minimal', 'none'
    ) -> Tuple[str, QuerySpecificityScore, List[str]]:
        """
        Adaptively expand query based on specificity.
        
        Args:
            query: Original query
            llm_rewriter: Optional LLMQueryRewriter instance
            force_level: Force specific expansion level (overrides auto-detection)
        
        Returns:
            Tuple of (enhanced_query, score, expansion_terms)
        """
        score = self.scorer.score(query)
        
        # Apply force_level override if specified
        if force_level:
            score = self._apply_force_level(score, force_level)
        
        expansion_terms = []
        enhanced_query = query
        
        # Apply structured expansion if enabled
        if score.use_structured_expansion and self.structured_enhancer:
            try:
                from ace.structured_enhancer import StructuredQueryEnhancer
                enhancer = StructuredQueryEnhancer()
                result = enhancer.enhance(query)
                
                # Limit expansion terms based on score
                limited_terms = result.expansion_terms[:score.expansion_terms_limit]
                if limited_terms:
                    enhanced_query = f"{query} {' '.join(limited_terms)}"
                    expansion_terms.extend(limited_terms)
            except Exception as e:
                logger.debug(f"Structured expansion failed: {e}")
        
        # Apply LLM expansion if enabled and available
        if score.use_llm_expansion and llm_rewriter:
            try:
                # LLM rewriter returns multiple variations - we take the enhanced terms
                # This is typically handled at a higher level in the retriever
                pass  # LLM expansion handled separately in multi-query search
            except Exception as e:
                logger.debug(f"LLM expansion failed: {e}")
        
        return enhanced_query, score, expansion_terms
    
    def _apply_force_level(
        self, score: QuerySpecificityScore, level: str
    ) -> QuerySpecificityScore:
        """Apply forced expansion level to score."""
        level_configs = {
            'maximum': {
                'expansion_level': 'maximum',
                'use_llm_expansion': True,
                'use_structured_expansion': True,
                'expansion_terms_limit': 5,
            },
            'moderate': {
                'expansion_level': 'moderate',
                'use_llm_expansion': False,
                'use_structured_expansion': True,
                'expansion_terms_limit': 3,
            },
            'minimal': {
                'expansion_level': 'minimal',
                'use_llm_expansion': False,
                'use_structured_expansion': True,
                'expansion_terms_limit': 2,
            },
            'none': {
                'expansion_level': 'none',
                'use_llm_expansion': False,
                'use_structured_expansion': False,
                'expansion_terms_limit': 0,
            },
        }
        
        config = level_configs.get(level, level_configs['moderate'])
        return QuerySpecificityScore(
            word_count=score.word_count,
            specificity_score=score.specificity_score,
            rationale=f"Forced expansion level: {level}",
            **config
        )


# ============================================================================
# LLM QUERY REWRITER (GLM-powered)
# ============================================================================

class LLMQueryRewriter:
    """
    LLM-based query rewriting for short, ambiguous queries.
    Uses Z.ai GLM to expand short queries into richer semantic variations.
    """

    # Domain context for the ACE memory knowledge base
    DOMAIN_CONTEXT = """The knowledge base contains:
- User preferences (coding style, tool choices, workflow patterns)
- Task strategies (debugging approaches, optimization techniques)
- Error patterns and fixes (common bugs, solutions, root causes)
- Configuration best practices (env vars, settings, defaults)
- Security guidelines (validation, sanitization, auth patterns)
- Code patterns (async/await, error handling, testing)"""

    SYSTEM_PROMPT = f"""You are a search query expansion tool for a developer knowledge base.

{DOMAIN_CONTEXT}

IMPORTANT: Respond in English only. Output ONLY 5 search queries (max 8 words each), one per line. Make queries specific to software development. No explanations."""

    REWRITE_PROMPT = """Expand this developer query into 5 specific variations (max 8 words each).

Query: "{query}"

Consider: What coding concept, pattern, error, or preference might the user be searching for?

Respond in English. Output 5 queries, one per line:"""

    BATCH_SYSTEM_PROMPT = f"""You are a search query expansion tool for a developer knowledge base.

{DOMAIN_CONTEXT}

IMPORTANT: Respond in English only. For EACH numbered query, output 3 expanded variations (max 8 words each).
Format your response as:
1: expanded1 | expanded2 | expanded3
2: expanded1 | expanded2 | expanded3
...

No explanations. Just the numbered expansions."""

    BATCH_REWRITE_PROMPT = """Expand each of these developer queries into 3 specific variations (max 8 words each):

{queries}

Respond in English. Format: NUMBER: expansion1 | expansion2 | expansion3"""

    # Class-level semaphore for GLM rate limiting (max 5 concurrent requests)
    _semaphore = None
    _semaphore_lock = None

    @classmethod
    def _get_semaphore(cls):
        """Lazy init semaphore (GLM allows max 5 concurrent requests)."""
        if cls._semaphore is None:
            import threading
            cls._semaphore_lock = threading.Lock()
            cls._semaphore = threading.Semaphore(4)  # Leave 1 slot buffer
        return cls._semaphore

    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self._cache: Dict[str, List[str]] = {}  # Query -> expanded queries cache
        self._client = None

        if HTTPX_AVAILABLE and self.config.api_key:
            self._client = httpx.Client(timeout=60.0)  # 60s for GLM reasoning

    def rewrite(self, query: str) -> List[str]:
        """
        Rewrite a short query into multiple semantic variations using LLM.

        Returns original query + LLM-generated variations.
        Returns just [query] if LLM unavailable or query is cached empty.
        """
        if not self._client or not self.config.enable_query_rewrite:
            return [query]

        # Check cache
        cache_key = query.lower().strip()
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return [query] + cached if cached else [query]

        try:
            # Rate limit: GLM allows max 5 concurrent requests
            semaphore = self._get_semaphore()
            with semaphore:
                # Call GLM API
                response = self._client.post(
                    f"{self.config.api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.config.model,
                        "messages": [
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": self.REWRITE_PROMPT.format(query=query)}
                        ],
                        "max_tokens": self.config.rewrite_max_tokens,
                        "temperature": self.config.rewrite_temperature,
                    }
                )

            if response.status_code == 200:
                result = response.json()
                message = result.get("choices", [{}])[0].get("message", {})
                # GLM returns reasoning in 'reasoning_content', final answer in 'content'
                content = message.get("content", "")
                reasoning = message.get("reasoning_content", "")

                expanded = []

                def extract_queries(text: str) -> List[str]:
                    """Extract query-like lines from GLM response."""
                    queries = []
                    for line in text.strip().split("\n"):
                        line = line.strip()
                        if not line or len(line) > 100:
                            continue
                        # Skip XML/HTML tags
                        if line.startswith("<") or line.startswith("**"):
                            continue
                        # Skip obvious meta-text (Chinese or English analysis)
                        if any(word in line.lower() for word in ["analyze", "brainstorm", "consider", "the user is"]):
                            continue
                        # Remove common prefixes
                        if line.startswith(("- ", "* ", ">> ")):
                            line = line[2:].strip()
                        elif len(line) > 2 and line[0].isdigit():
                            # Handle "1. query" or "1: query" or "1) query"
                            match = re.match(r'^\d+[.:\)]\s*(.+)$', line)
                            if match:
                                line = match.group(1).strip()
                        # Valid query: 2-12 words, no backticks at start
                        words = line.split()
                        if 2 <= len(words) <= 12 and line not in queries:
                            queries.append(line)
                    return queries

                # Try content first (GLM puts final answer here)
                if content:
                    expanded = extract_queries(content)

                # Try reasoning_content if content is empty/insufficient
                if reasoning and len(expanded) < 3:
                    # Strip <think> tags
                    think_text = reasoning
                    if "<think>" in reasoning:
                        match = re.search(r'<think>(.*?)(?:</think>|$)', reasoning, re.DOTALL)
                        if match:
                            think_text = match.group(1)
                    expanded.extend(extract_queries(think_text))

                    # Fallback: backtick-quoted queries
                    for match in re.findall(r'`([^`]+)`', reasoning):
                        if 8 < len(match) < 80 and ' ' in match and match not in expanded:
                            expanded.append(match)

                if not expanded:
                    logger.warning("LLM returned no usable queries")
                    self._cache[cache_key] = []
                    return [query]

                # Cache and return
                self._cache[cache_key] = expanded[:5]  # Max 5 expansions
                logger.debug(f"LLM rewrote '{query}' -> {len(expanded)} variations")
                return [query] + expanded[:5]

            else:
                logger.warning(f"LLM rewrite failed: {response.status_code}")
                self._cache[cache_key] = []
                return [query]

        except Exception as e:
            logger.warning(f"LLM rewrite error: {e}")
            self._cache[cache_key] = []
            return [query]

    def close(self):
        """Close HTTP client."""
        if self._client:
            self._client.close()

    def batch_rewrite(self, queries: List[str], batch_size: int = 10) -> Dict[str, List[str]]:
        """
        Batch rewrite multiple queries in a single GLM call.

        Args:
            queries: List of short queries to expand
            batch_size: Number of queries per GLM call (default 10)

        Returns:
            Dict mapping original query -> list of expanded queries
        """
        if not self._client or not self.config.enable_query_rewrite:
            return {q: [q] for q in queries}

        results = {}
        uncached = []

        # Check cache first
        for q in queries:
            cache_key = q.lower().strip()
            if cache_key in self._cache:
                results[q] = [q] + self._cache[cache_key]
            else:
                uncached.append(q)

        if not uncached:
            return results

        # Process uncached queries in batches
        for i in range(0, len(uncached), batch_size):
            batch = uncached[i:i + batch_size]
            batch_results = self._call_batch(batch)
            results.update(batch_results)

        return results

    def _call_batch(self, queries: List[str]) -> Dict[str, List[str]]:
        """Make a single GLM call for multiple queries."""
        # Format queries as numbered list
        numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))

        try:
            semaphore = self._get_semaphore()
            with semaphore:
                response = self._client.post(
                    f"{self.config.api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.config.model,
                        "messages": [
                            {"role": "system", "content": self.BATCH_SYSTEM_PROMPT},
                            {"role": "user", "content": self.BATCH_REWRITE_PROMPT.format(queries=numbered)}
                        ],
                        "max_tokens": self.config.rewrite_max_tokens * 2,  # More tokens for batch
                        "temperature": self.config.rewrite_temperature,
                    }
                )

            results = {q: [q] for q in queries}  # Default: just original query

            if response.status_code == 200:
                result = response.json()
                message = result.get("choices", [{}])[0].get("message", {})
                content = message.get("content", "")
                reasoning = message.get("reasoning_content", "")

                # Parse response - look for "N: expansion1 | expansion2 | expansion3" format
                text_to_parse = content or reasoning

                for line in text_to_parse.split("\n"):
                    line = line.strip()
                    if not line:
                        continue

                    # Match "1: ..." or "1. ..." format
                    match = re.match(r'^(\d+)[:\.]?\s*(.+)$', line)
                    if match:
                        idx = int(match.group(1)) - 1
                        expansions_str = match.group(2)

                        if 0 <= idx < len(queries):
                            # Split by | or common separators
                            expansions = [e.strip() for e in re.split(r'\s*\|\s*', expansions_str)]
                            expansions = [e for e in expansions if e and 2 <= len(e.split()) <= 10]

                            if expansions:
                                q = queries[idx]
                                cache_key = q.lower().strip()
                                self._cache[cache_key] = expansions[:3]
                                results[q] = [q] + expansions[:3]

                logger.debug(f"Batch rewrote {len(queries)} queries")
            else:
                logger.warning(f"Batch rewrite failed: {response.status_code}")

            return results

        except Exception as e:
            logger.warning(f"Batch rewrite error: {e}")
            return {q: [q] for q in queries}


# ============================================================================
# BM25 UTILITIES
# ============================================================================

def tokenize_bm25(text: str) -> List[str]:
    """
    Tokenize text for BM25, preserving technical terms.
    """
    # Split CamelCase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Split snake_case
    text = text.replace('_', ' ')
    # Extract tokens
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    # Filter
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def compute_bm25_sparse(
    text: str,
    k1: float = 1.5,
    b: float = 0.75,
    avg_doc_length: float = 50
) -> Dict[str, Any]:
    """
    Compute BM25-style sparse vector for Qdrant.
    """
    tokens = tokenize_bm25(text)
    if not tokens:
        return {"indices": [], "values": []}

    tf = Counter(tokens)
    doc_length = len(tokens)

    indices = []
    values = []

    for term, freq in tf.items():
        # Consistent hash for term -> index
        term_hash = int(hashlib.sha256(term.encode()).hexdigest()[:8], 16)
        indices.append(term_hash)

        # BM25 term weight
        tf_weight = (freq * (k1 + 1)) / (
            freq + k1 * (1 - b + b * doc_length / avg_doc_length)
        )
        values.append(float(tf_weight))

    return {"indices": indices, "values": values}


# ============================================================================
# RRF FUSION
# ============================================================================

def reciprocal_rank_fusion(
    result_sets: List[List[Dict]],
    k: int = 60
) -> List[Dict]:
    """
    Merge multiple ranked result sets using Reciprocal Rank Fusion.
    """
    scores: Dict[int, float] = {}
    doc_data: Dict[int, Dict] = {}

    for results in result_sets:
        for rank, doc in enumerate(results, start=1):
            doc_id = doc["id"]
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank)
            if doc_id not in doc_data:
                doc_data[doc_id] = doc

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    fused = []
    for doc_id, score in sorted_items:
        doc = doc_data[doc_id].copy()
        doc["rrf_score"] = score
        doc["score"] = score
        fused.append(doc)

    return fused


# ============================================================================
# OPTIMIZED RETRIEVER
# ============================================================================

class OptimizedRetriever:
    """
    State-of-the-art retriever with hybrid search, query expansion,
    multi-query RRF fusion, and cross-encoder re-ranking.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize retriever with configuration.

        Args:
            config: Override default configuration values
        """
        self.config = {**_get_default_config(), **(config or {})}

        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required. Install with: pip install httpx")

        self.client = httpx.Client(timeout=60.0)
        self.expander = QueryExpander()

        # Initialize query complexity classifier
        elf_config = ELFConfig()
        self.classifier = QueryComplexityClassifier(elf_config)
        logger.info(f"Query complexity classifier enabled: {elf_config.enable_query_classifier}")

        # Initialize LLM query rewriter for short queries
        self.llm_rewriter = None
        llm_config = LLMConfig()
        if llm_config.api_key and llm_config.enable_query_rewrite:
            self.llm_rewriter = LLMQueryRewriter(llm_config)
            logger.info(f"LLM query rewriter enabled (model: {llm_config.model})")

        # Initialize cross-encoder if enabled (prefer GPU-accelerated version)
        self.cross_encoder = None
        if self.config["enable_reranking"] and CROSS_ENCODER_AVAILABLE:
            try:
                if GPU_RERANKER_AVAILABLE:
                    logger.info(f"Loading GPU-accelerated cross-encoder: {self.config['cross_encoder_model']}")
                    self.cross_encoder = GPUCrossEncoder(
                        model_name=self.config["cross_encoder_model"],
                        use_gpu=True,
                        max_length=512
                    )
                    logger.info(f"Cross-encoder loaded (GPU: {self.cross_encoder.use_gpu})")
                else:
                    logger.info(f"Loading cross-encoder (CPU): {self.config['cross_encoder_model']}")
                    self.cross_encoder = CrossEncoder(
                        self.config["cross_encoder_model"],
                        max_length=512
                    )
                    logger.info("Cross-encoder loaded (CPU mode)")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {e}")
        elif self.config["enable_reranking"]:
            logger.warning(
                "Cross-encoder re-ranking enabled but no backend available. "
                "Install with: pip install sentence-transformers"
            )

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get dense embedding vector with automatic EOS token handling."""
        try:
            # Add EOS token for Qwen models to fix GGUF tokenizer warning
            # This ensures proper sentence boundary detection in embeddings
            if "qwen" in self.config["embedding_model"].lower() and not text.endswith("</s>"):
                text = f"{text}</s>"
            
            resp = self.client.post(
                f"{self.config['embedding_url']}/v1/embeddings",
                json={
                    "model": self.config["embedding_model"],
                    "input": text[:8000]
                }
            )
            if resp.status_code == 200:
                return resp.json()["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Embedding error: {e}")
        return None

    def _hybrid_search_single(self, query: str, limit: int, bm25_boost: float = 1.0) -> List[Dict]:
        """Execute hybrid search for a single query.

        Args:
            query: Search query
            limit: Max results
            bm25_boost: Multiplier for BM25 sparse values (higher = more weight to exact matches)
        """
        dense_embedding = self.get_embedding(query)
        if not dense_embedding:
            return []

        sparse = compute_bm25_sparse(
            query,
            k1=self.config["bm25_k1"],
            b=self.config["bm25_b"],
            avg_doc_length=self.config["avg_doc_length"]
        )

        hybrid_query = {
            "prefetch": [
                {
                    "query": dense_embedding,
                    "using": "dense",
                    "limit": limit * 2
                }
            ],
            "query": {"fusion": "rrf"},
            "limit": limit,
            "with_payload": True
        }

        if sparse.get("indices"):
            # Apply BM25 boost for short queries (exact keyword matching helps)
            boosted_values = [v * bm25_boost for v in sparse["values"]]
            hybrid_query["prefetch"].append({
                "query": {
                    "indices": sparse["indices"],
                    "values": boosted_values
                },
                "using": "sparse",
                "limit": limit * 2
            })

        try:
            resp = self.client.post(
                f"{self.config['qdrant_url']}/collections/{self.config['collection_name']}/points/query",
                json=hybrid_query
            )
            if resp.status_code == 200:
                return resp.json().get("result", {}).get("points", [])
        except Exception as e:
            logger.error(f"Search error: {e}")

        return []

    def _multi_query_search(
        self,
        queries: List[str],
        limit: int,
        bm25_boost: float = 1.0
    ) -> Tuple[List[Dict], float]:
        """Execute multi-query search with RRF fusion."""
        start = time.perf_counter()

        all_results = []
        for query in queries:
            results = self._hybrid_search_single(
                query,
                self.config["candidates_per_query"],
                bm25_boost=bm25_boost
            )
            if results:
                all_results.append(results)

        latency = (time.perf_counter() - start) * 1000

        if not all_results:
            return [], latency

        fused = reciprocal_rank_fusion(all_results, k=self.config["rrf_k"])
        return fused[:limit], latency

    def _rerank(
        self,
        query: str,
        candidates: List[Dict],
        limit: int
    ) -> Tuple[List[Dict], float]:
        """Re-rank candidates using cross-encoder."""
        if not self.cross_encoder or not candidates:
            return candidates[:limit], 0.0

        start = time.perf_counter()

        pairs = []
        for c in candidates:
            payload = c.get("payload", {})
            doc_text = payload.get("lesson", "") or payload.get("content", "") or str(payload)
            pairs.append([query, doc_text[:1000]])

        scores = self.cross_encoder.predict(pairs)

        scored = list(zip(scores, candidates))
        scored.sort(key=lambda x: x[0], reverse=True)

        reranked = []
        for score, candidate in scored[:limit]:
            candidate["rerank_score"] = float(score)
            candidate["score"] = float(score)
            candidate["reranked"] = True
            reranked.append(candidate)

        latency = (time.perf_counter() - start) * 1000
        return reranked, latency

    def search(
        self,
        query: str,
        limit: int = None,
        return_metrics: bool = False
    ) -> List[RetrievalResult] | Tuple[List[RetrievalResult], SearchMetrics]:
        """
        Search for relevant memories.

        Args:
            query: Search query
            limit: Maximum results (default from config)
            return_metrics: Whether to return search metrics

        Returns:
            List of RetrievalResult, optionally with SearchMetrics
        """
        limit = limit or self.config["final_k"]
        word_count = len(query.split())
        is_short_query = word_count <= 3

        # 1. Query expansion (more aggressive for short queries)
        exp_start = time.perf_counter()

        # Classify query complexity to determine if LLM rewriting is needed
        needs_llm = self.classifier.needs_llm_rewrite(query) if self.classifier else True

        # For short queries needing semantic expansion: Use LLM rewriting first, then standard expansion
        if is_short_query and needs_llm and self.llm_rewriter:
            # LLM generates semantic variations
            llm_queries = self.llm_rewriter.rewrite(query)
            # Apply standard expansion to each LLM variation
            expanded_queries = []
            for q in llm_queries[:3]:  # Use top 3 LLM variations
                expanded_queries.extend(self.expander.expand(q, 2))
            # Deduplicate
            seen = set()
            expanded_queries = [q for q in expanded_queries if not (q.lower() in seen or seen.add(q.lower()))]
            logger.debug(f"LLM rewriting used for query: '{query}' -> {len(expanded_queries)} variations")
        else:
            # Standard expansion for longer queries or technical queries
            num_exp = self.config["num_expanded_queries"] * 2 if is_short_query else self.config["num_expanded_queries"]
            expanded_queries = self.expander.expand(query, num_exp - 1)
            if not needs_llm:
                logger.debug(f"Keyword expansion used for technical query: '{query}' -> {len(expanded_queries)} variations")

        expansion_latency = (time.perf_counter() - exp_start) * 1000

        # 2. Multi-query retrieval with RRF fusion
        # Short queries get 2x candidates to improve recall
        first_stage = self.config["first_stage_k"] * 2 if is_short_query else self.config["first_stage_k"]
        # Short queries benefit from 1.5x boosted BM25 (exact keyword matching)
        bm25_boost = 1.5 if is_short_query else 1.0
        candidates, retrieval_latency = self._multi_query_search(
            expanded_queries,
            first_stage,
            bm25_boost=bm25_boost
        )

        # 3. Cross-encoder re-ranking
        reranked, rerank_latency = self._rerank(query, candidates, limit)

        # Convert to RetrievalResult
        results = []
        for r in reranked:
            payload = r.get("payload", {})
            results.append(RetrievalResult(
                id=r["id"],
                score=r.get("score", 0),
                payload=payload,
                content=payload.get("lesson", "") or payload.get("content", ""),
                category=payload.get("category"),
                reranked=r.get("reranked", False)
            ))

        if return_metrics:
            metrics = SearchMetrics(
                total_latency_ms=expansion_latency + retrieval_latency + rerank_latency,
                expansion_latency_ms=expansion_latency,
                retrieval_latency_ms=retrieval_latency,
                rerank_latency_ms=rerank_latency,
                num_candidates=len(candidates),
                num_results=len(results),
                expanded_queries=expanded_queries
            )
            return results, metrics

        return results

    def search_simple(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Simple search returning raw dictionaries (for compatibility).
        """
        results = self.search(query, limit)
        return [
            {
                "id": r.id,
                "score": r.score,
                "content": r.content,
                "category": r.category,
                **r.payload
            }
            for r in results
        ]

    def close(self):
        """Close HTTP client and LLM rewriter."""
        self.client.close()
        if self.llm_rewriter:
            self.llm_rewriter.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_retriever(config: Dict[str, Any] = None) -> OptimizedRetriever:
    """Create an optimized retriever instance."""
    return OptimizedRetriever(config)


def search_memories(
    query: str,
    limit: int = 10,
    config: Dict[str, Any] = None
) -> List[Dict]:
    """
    One-shot memory search (creates and closes retriever).

    For repeated searches, use OptimizedRetriever directly.
    """
    with OptimizedRetriever(config) as retriever:
        return retriever.search_simple(query, limit)
