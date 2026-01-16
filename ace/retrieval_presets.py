"""
Retrieval Presets - Optimized configurations for different query types.

Based on empirical testing:
- Baseline precision: 75.6%
- Architecture queries: 33.3% (worst)
- Target: 95%+ precision across all categories

Winning optimizations:
1. BM25-heavy weighting (dense=0.3, sparse=0.7) -> +50% P@3
2. Post-retrieval deduplication (0.90 threshold) -> +2.7%
3. Query expansion with domain synonyms -> +3% (conditional)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math


class RetrievalPreset(Enum):
    """Pre-configured retrieval strategies."""

    # Default balanced approach (baseline)
    BALANCED = "balanced"

    # BM25-heavy for technical/keyword queries (RECOMMENDED)
    BM25_HEAVY = "bm25_heavy"

    # Dense-heavy for semantic/conceptual queries
    SEMANTIC = "semantic"

    # Maximum precision mode (all optimizations)
    MAX_PRECISION = "max_precision"


@dataclass
class RetrievalConfig:
    """Configuration for a retrieval preset."""

    # Prefetch multipliers
    dense_prefetch_multiplier: int = 3
    sparse_prefetch_multiplier: int = 3

    # BM25 boost factor (multiplies sparse vector values)
    bm25_boost: float = 1.0

    # Post-retrieval deduplication threshold (0 = disabled)
    dedup_threshold: float = 0.0

    # Query expansion settings
    expand_queries: bool = False
    expansion_synonyms: Dict[str, List[str]] = field(default_factory=dict)

    # Score weighting (for custom fusion, not RRF)
    # Note: RRF doesn't use weights directly, but we can influence via prefetch limits
    dense_weight: float = 0.5
    sparse_weight: float = 0.5


# Pre-configured presets based on empirical testing
PRESET_CONFIGS: Dict[RetrievalPreset, RetrievalConfig] = {
    RetrievalPreset.BALANCED: RetrievalConfig(
        dense_prefetch_multiplier=3,
        sparse_prefetch_multiplier=3,
        bm25_boost=1.0,
        dedup_threshold=0.0,
        expand_queries=False,
        dense_weight=0.5,
        sparse_weight=0.5,
    ),

    # RECOMMENDED for technical queries - tested +50% P@3 improvement
    RetrievalPreset.BM25_HEAVY: RetrievalConfig(
        dense_prefetch_multiplier=2,
        sparse_prefetch_multiplier=5,
        bm25_boost=2.0,  # Boost BM25 scores
        dedup_threshold=0.85,  # Aggressive dedup for higher P@3
        expand_queries=True,  # Enable query expansion for better recall
        expansion_synonyms={
            # Core technical terms
            "playbook": ["bullet", "strategy", "learned", "guidance"],
            "memory": ["qdrant", "store", "knowledge", "context"],
            "storage": ["persistence", "save", "file", "backend"],
            "config": ["configuration", "settings", "setup"],
            "search": ["retrieval", "query", "lookup", "find"],
            "validate": ["check", "verify", "ensure", "enforce"],
            "hook": ["pretool", "guard", "intercept", "handler"],
            "inject": ["insert", "add", "augment", "context"],
            "style": ["format", "convention", "approach", "pattern"],
            "preference": ["prefer", "want", "like", "approach"],
            "performance": ["speed", "latency", "optimize", "benchmark"],
            "strategy": ["approach", "method", "technique", "pattern"],
            "namespace": ["scope", "isolation", "partition", "category"],
        },
        dense_weight=0.3,
        sparse_weight=0.7,
    ),

    RetrievalPreset.SEMANTIC: RetrievalConfig(
        dense_prefetch_multiplier=5,
        sparse_prefetch_multiplier=2,
        bm25_boost=0.5,
        dedup_threshold=0.0,
        expand_queries=False,
        dense_weight=0.7,
        sparse_weight=0.3,
    ),

    # Maximum precision - all optimizations enabled
    RetrievalPreset.MAX_PRECISION: RetrievalConfig(
        dense_prefetch_multiplier=3,
        sparse_prefetch_multiplier=5,
        bm25_boost=2.5,  # Strong BM25 boost
        dedup_threshold=0.88,  # Aggressive dedup
        expand_queries=True,
        expansion_synonyms={
            # Architecture terms
            "wired": ["configured", "setup", "architecture", "connected"],
            "system": ["infrastructure", "architecture", "platform", "framework"],
            "qdrant": ["vector database", "vector store", "collection"],
            "playbook": ["bullet", "strategy", "learned", "lesson", "guidance"],
            "memory": ["memory system", "unified memory", "qdrant", "store", "knowledge"],
            "storage": ["persistence", "data store", "backend", "save", "file"],
            "config": ["configuration", "settings", "setup", "environment"],
            # Technical terms
            "hybrid": ["dense sparse", "bm25 embedding", "rrf fusion", "combine"],
            "search": ["retrieval", "query", "lookup", "find"],
            "json": ["json file", "json storage", "serialize"],
            # Namespace/structure terms
            "namespace": ["scope", "separation", "isolation", "partition", "category"],
            "separate": ["isolate", "decouple", "modular", "extract"],
            # Validation terms
            "validation": ["check", "verify", "enforce", "guard", "ensure"],
            "validate": ["check", "verify", "enforce", "ensure", "confirm"],
            "pre-tool": ["pretool", "before tool", "hook", "guard"],
            # Context/injection terms
            "inject": ["insert", "add", "include", "augment", "context"],
            "prompt": ["context", "input", "message", "request"],
            # Style/preference terms
            "style": ["format", "convention", "approach", "pattern", "preference"],
            "preference": ["prefer", "want", "like", "style", "approach"],
            "coding": ["code", "programming", "development", "implement"],
            "communication": ["respond", "interact", "reply", "output", "message"],
            # Performance terms
            "performance": ["speed", "latency", "fast", "efficient", "benchmark", "optimize"],
            "optimization": ["optimize", "improve", "enhance", "tune", "performance"],
            "strategy": ["approach", "method", "technique", "pattern", "practice"],
            # Debug terms
            "leak": ["resource", "cleanup", "dispose", "release", "free"],
            "detection": ["detect", "find", "identify", "discover", "monitor"],
            "troubleshoot": ["debug", "diagnose", "investigate", "analyze", "fix"],
        },
        dense_weight=0.3,
        sparse_weight=0.7,
    ),
}

# Common synonyms applied to all presets that have expand_queries enabled
COMMON_SYNONYMS: Dict[str, List[str]] = {
    # Core semantic mappings
    "how": ["what way", "method", "approach"],
    "store": ["save", "persist", "keep", "write"],
    "retrieve": ["get", "fetch", "load", "read", "find"],
    "workflow": ["process", "flow", "procedure", "steps"],
    "approach": ["method", "strategy", "technique", "way"],
}


def get_preset_config(preset: RetrievalPreset) -> RetrievalConfig:
    """Get configuration for a preset."""
    return PRESET_CONFIGS.get(preset, PRESET_CONFIGS[RetrievalPreset.BALANCED])


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def compute_adaptive_threshold(
    scores: List[float],
    percentile: int = 70,
    use_gap_detection: bool = True,
    min_gap_ratio: float = 0.15,
) -> float:
    """
    Compute adaptive score threshold for Stage 2 filtering in multi-stage retrieval.

    Uses two strategies:
    1. Percentile-based: Keep scores above the Nth percentile
    2. Gap detection: Find significant drops in score distribution

    This ensures we filter based on the actual score distribution rather than
    a fixed threshold, adapting to both high-confidence and low-confidence results.

    Args:
        scores: List of retrieval scores (higher = more relevant)
        percentile: Percentile threshold (0-100), default 70 = keep top 30%
        use_gap_detection: If True, also look for score gaps
        min_gap_ratio: Minimum relative gap size to detect (0.15 = 15% drop)

    Returns:
        Adaptive threshold score. Results with score >= threshold should be kept.
        Returns 0.0 for empty scores (keep all).

    Examples:
        >>> compute_adaptive_threshold([0.9, 0.85, 0.8, 0.3, 0.2])
        0.8  # Gap detected between 0.8 and 0.3

        >>> compute_adaptive_threshold([0.6, 0.58, 0.55, 0.52, 0.50])
        0.55  # 70th percentile (no significant gaps)
    """
    if not scores:
        return 0.0

    # Sort scores descending for analysis
    sorted_scores = sorted(scores, reverse=True)
    n = len(sorted_scores)

    # Strategy 1: Percentile-based threshold
    # percentile=70 means keep top 30% (index at 30% from top)
    percentile_idx = max(0, int(n * (100 - percentile) / 100) - 1)
    percentile_threshold = sorted_scores[min(percentile_idx, n - 1)]

    # Strategy 2: Gap detection
    gap_threshold = 0.0
    if use_gap_detection and n > 2:
        # Look for significant drops between consecutive scores
        for i in range(n - 1):
            current = sorted_scores[i]
            next_score = sorted_scores[i + 1]

            if current > 0:
                # Calculate relative gap
                gap_ratio = (current - next_score) / current

                if gap_ratio >= min_gap_ratio:
                    # Found a significant gap - threshold is the higher score
                    gap_threshold = current
                    break

    # Return the higher threshold (more aggressive filtering)
    # But if gap detection found something, prefer it as it's more semantic
    if gap_threshold > 0:
        return gap_threshold
    return percentile_threshold


def filter_by_adaptive_threshold(
    results_with_scores: List[Tuple[any, float]],
    percentile: int = 70,
    use_gap_detection: bool = True,
    min_keep: int = 3,
) -> List[Tuple[any, float]]:
    """
    Filter results using adaptive threshold computed from score distribution.

    Args:
        results_with_scores: List of (result, score) tuples
        percentile: Percentile threshold for filtering
        use_gap_detection: Enable score gap detection
        min_keep: Minimum results to keep regardless of threshold

    Returns:
        Filtered list of (result, score) tuples
    """
    if not results_with_scores:
        return []

    scores = [score for _, score in results_with_scores]
    threshold = compute_adaptive_threshold(scores, percentile, use_gap_detection)

    # Filter by threshold
    filtered = [(r, s) for r, s in results_with_scores if s >= threshold]

    # Ensure we keep at least min_keep results
    if len(filtered) < min_keep and len(results_with_scores) >= min_keep:
        # Sort by score and keep top min_keep
        sorted_results = sorted(results_with_scores, key=lambda x: x[1], reverse=True)
        return sorted_results[:min_keep]

    return filtered


def deduplicate_results(
    results: List[Tuple[any, List[float]]],  # List of (result, embedding) tuples
    threshold: float = 0.90
) -> List[any]:
    """
    Remove near-duplicate results based on embedding similarity.

    Keeps the first occurrence (highest ranked) of each semantic cluster.

    Args:
        results: List of (result_object, embedding_vector) tuples
        threshold: Cosine similarity threshold for deduplication

    Returns:
        Deduplicated list of result objects
    """
    if threshold <= 0 or not results:
        return [r for r, _ in results]

    deduplicated = []
    seen_embeddings = []

    for result, embedding in results:
        is_duplicate = False

        if embedding:
            for seen_emb in seen_embeddings:
                if cosine_similarity(embedding, seen_emb) >= threshold:
                    is_duplicate = True
                    break

        if not is_duplicate:
            deduplicated.append(result)
            if embedding:
                seen_embeddings.append(embedding)

    return deduplicated


def expand_query(query: str, synonyms: Dict[str, List[str]]) -> List[str]:
    """
    Expand query with synonyms for better recall.

    Args:
        query: Original query string
        synonyms: Dict mapping terms to list of synonyms

    Returns:
        List of query variations (original + expanded)
    """
    queries = [query]
    query_lower = query.lower()

    # Add synonym expansions
    for term, syn_list in synonyms.items():
        if term.lower() in query_lower:
            for syn in syn_list[:2]:  # Limit to 2 synonyms per term
                expanded = query_lower.replace(term.lower(), syn)
                if expanded not in queries:
                    queries.append(expanded)

    # Add context-enriched version
    if len(queries) == 1:  # No synonyms matched
        # Add generic technical context
        context_query = f"{query} architecture configuration system"
        queries.append(context_query)

    return queries[:5]  # Limit total expansions


# LLM-powered query expansion cache (avoid repeated API calls)
_llm_expansion_cache: Dict[str, List[str]] = {}

# LLM filtering cache - key: hash(query + content_ids), value: filtered indices
_llm_filter_cache: Dict[str, List[int]] = {}
_llm_filter_cache_ttl: Dict[str, float] = {}
LLM_FILTER_CACHE_TTL = 300  # 5 minutes


def expand_query_with_llm(
    query: str,
    llm_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
) -> List[str]:
    """
    Use LLM (GLM 4.6) to semantically expand query for better retrieval.

    This generates related terms and rephrased queries that capture
    the semantic intent beyond simple synonym matching.

    All settings default to values from ace.config.LLMConfig.

    Args:
        query: Original query string
        llm_url: Z.ai API URL (default: from config)
        model: Model to use for expansion (default: from config)
        timeout: Request timeout in seconds (default: from config)

    Returns:
        List of expanded queries including original
    """
    from ace.config import get_llm_config

    llm_config = get_llm_config()

    # Use config defaults if not specified
    llm_url = llm_url or llm_config.api_base
    model = model or llm_config.model
    timeout = timeout or llm_config.expansion_timeout

    # Check if feature is disabled
    if not llm_config.enable_llm_expansion:
        return [query]

    # Check cache first
    cache_key = query.lower().strip()
    if cache_key in _llm_expansion_cache:
        return _llm_expansion_cache[cache_key]

    expansions = [query]

    try:
        import httpx

        prompt = f"""Given this search query, generate 3 alternative phrasings that capture the same intent but use different keywords. Return ONLY the alternatives, one per line, no numbering or explanation.

Query: {query}

Alternatives:"""

        # Z.ai GLM 4.6 API requires Bearer auth
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {llm_config.api_key}",
        }

        response = httpx.post(
            f"{llm_url}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": llm_config.expansion_max_tokens,
                "temperature": 0.3,
            },
            timeout=timeout,
        )

        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Parse alternatives (one per line)
            for line in content.strip().split("\n"):
                line = line.strip().lstrip("0123456789.-) ")
                if line and len(line) > 5 and line.lower() != query.lower():
                    expansions.append(line)

        # Cache the result
        _llm_expansion_cache[cache_key] = expansions[:4]  # Original + 3 expansions

    except Exception:
        # Silently fall back to original query only
        pass

    return expansions[:4]


def clear_llm_expansion_cache() -> int:
    """Clear the LLM expansion cache. Returns number of entries cleared."""
    count = len(_llm_expansion_cache)
    _llm_expansion_cache.clear()
    return count


def clear_llm_filter_cache() -> int:
    """Clear the LLM filter cache. Returns number of entries cleared."""
    count = len(_llm_filter_cache)
    _llm_filter_cache.clear()
    _llm_filter_cache_ttl.clear()
    return count


def get_llm_cache_stats() -> dict:
    """Get LLM cache statistics."""
    import time
    now = time.time()
    valid_filter = sum(1 for k, t in _llm_filter_cache_ttl.items() if now - t < LLM_FILTER_CACHE_TTL)
    return {
        "expansion_cache_size": len(_llm_expansion_cache),
        "filter_cache_size": len(_llm_filter_cache),
        "filter_cache_valid": valid_filter,
        "filter_cache_ttl": LLM_FILTER_CACHE_TTL,
    }


def llm_filter_and_rerank(
    query: str,
    results: List[Tuple[any, float]],  # List of (result, score) tuples
    llm_url: Optional[str] = None,
    model: Optional[str] = None,
    top_k: Optional[int] = None,
    timeout: Optional[float] = None,
    use_cache: bool = True,
) -> List[any]:
    """
    Use LLM natural language to filter AND rerank results for precision.

    Asks the LLM to directly judge relevance using natural language,
    returning only results it deems relevant and properly ordered.

    All settings default to values from ace.config.LLMConfig.

    PERFORMANCE: Caches results for 5 minutes to avoid redundant API calls.
    Identical query + content combinations return cached results instantly.

    Args:
        query: Original search query
        results: List of (result_object, vector_score) tuples
        llm_url: Z.ai API URL (default: from config)
        model: Model to use (default: from config)
        top_k: Number of results to evaluate (default: from config)
        timeout: Request timeout in seconds (default: from config)
        use_cache: Enable response caching (default: True)

    Returns:
        Filtered and reranked list of result objects (only relevant ones)
    """
    import time
    import hashlib
    from ace.config import get_llm_config

    llm_config = get_llm_config()

    # Use local LLM for speed if configured, otherwise Z.ai
    if llm_config.use_local_llm:
        llm_url = llm_url or llm_config.local_llm_url
        model = model or llm_config.local_llm_model
        max_tokens = llm_config.local_llm_max_tokens
        timeout = timeout or llm_config.local_llm_timeout
        api_key = ""  # Local LLM doesn't need auth
    else:
        llm_url = llm_url or llm_config.api_base
        model = model or llm_config.model
        max_tokens = llm_config.filtering_max_tokens
        timeout = timeout or llm_config.filtering_timeout
        api_key = llm_config.api_key

    top_k = top_k or llm_config.filtering_top_k

    # Check if feature is disabled
    if not llm_config.enable_llm_filtering:
        return [r for r, _ in results]

    if not results or len(results) <= 1:
        return [r for r, _ in results]

    candidates = results[:top_k]
    remaining = results[top_k:]

    # Generate cache key from query + model (5-min TTL is short enough for query-only key)
    if use_cache:
        cache_key = hashlib.sha256(f"{query}:{model}".encode()).hexdigest()

        # Check cache
        now = time.time()
        if cache_key in _llm_filter_cache:
            cache_time = _llm_filter_cache_ttl.get(cache_key, 0)
            if now - cache_time < LLM_FILTER_CACHE_TTL:
                # Cache hit - return cached bullet IDs in relevance order
                cached_ids = _llm_filter_cache[cache_key]
                # Build ID->bullet map from current candidates
                id_to_bullet = {getattr(r, 'id', None): r for r, _ in candidates}
                # Return bullets in cached order (skip missing IDs)
                filtered = [id_to_bullet[bid] for bid in cached_ids if bid in id_to_bullet]
                # Add remaining (non-evaluated) results but NOT candidates deemed irrelevant
                filtered.extend([r for r, _ in remaining])
                return filtered

    try:
        import httpx
        import json

        # Build candidate descriptions
        candidate_texts = []
        for i, (result, score) in enumerate(candidates):
            content = getattr(result, 'content', str(result))[:200]
            candidate_texts.append(f"{i+1}. {content}")

        # Natural language prompt for direct relevance judgment
        prompt = f"""You are a relevance filter. Given a search query and candidate results,
identify ONLY the results that are DIRECTLY RELEVANT to answering the query.

A result is relevant if it:
- Directly answers or addresses the query topic
- Contains information specifically about what the query is asking
- Would be useful to someone searching for this query

A result is NOT relevant if it:
- Is about a different topic even if it shares some keywords
- Is too generic or vague to be useful
- Does not actually answer or address the query

Query: "{query}"

Candidates:
{chr(10).join(candidate_texts)}

Return a JSON object with:
- "relevant": array of result numbers that ARE relevant, ordered by relevance (most relevant first)
- "irrelevant": array of result numbers that are NOT relevant

Example: {{"relevant": [3, 1, 5], "irrelevant": [2, 4]}}

JSON response:"""

        # Build headers (auth only needed for Z.ai, not local LLM)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Determine endpoint path (local LLM uses /v1/, Z.ai uses direct path)
        endpoint = f"{llm_url}/v1/chat/completions" if llm_config.use_local_llm else f"{llm_url}/chat/completions"

        # Retry with exponential backoff for rate limits
        max_retries = 3
        for retry in range(max_retries):
            response = httpx.post(
                endpoint,
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                },
                timeout=timeout,
            )

            if response.status_code == 200:
                break
            elif response.status_code == 429:
                # Rate limited - wait and retry
                wait_time = (2 ** retry) * 2  # 2s, 4s, 8s
                time.sleep(wait_time)
            else:
                # Other error - don't retry
                break

        if response.status_code == 200:
            result_json = response.json()
            content = result_json.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Handle markdown code blocks
            if "```" in content:
                content = content.split("```")[1].replace("json", "").strip()

            judgment = json.loads(content)
            relevant_indices = judgment.get("relevant", [])

            # Build filtered list in LLM-determined order
            filtered = []
            valid_ids = []  # Store bullet IDs for caching
            for idx in relevant_indices:
                i = int(idx) - 1  # Convert to 0-indexed
                if 0 <= i < len(candidates):
                    bullet = candidates[i][0]
                    filtered.append(bullet)
                    valid_ids.append(getattr(bullet, 'id', None))

            # Store bullet IDs in cache for future identical queries
            if use_cache:
                _llm_filter_cache[cache_key] = valid_ids
                _llm_filter_cache_ttl[cache_key] = time.time()

            # Add remaining results (not evaluated by LLM)
            filtered.extend([r for r, _ in remaining])
            return filtered

    except Exception:
        pass

    return [r for r, _ in results]


# Keep old function name as alias for backwards compatibility
def llm_rerank_results(
    query: str,
    results: List[Tuple[any, float]],
    llm_url: str = "http://localhost:1234",
    model: str = "glm-4-9b-chat",
    top_k: int = 5,
    timeout: float = 8.0,
    filter_threshold: float = 5.0,
) -> List[any]:
    """Alias for llm_filter_and_rerank (backwards compatible)."""
    return llm_filter_and_rerank(query, results, llm_url, model, top_k, timeout)


def boost_sparse_vector(
    sparse: Dict[str, List],
    boost_factor: float = 1.0
) -> Dict[str, List]:
    """
    Boost BM25 sparse vector values.

    Args:
        sparse: Dict with 'indices' and 'values' lists
        boost_factor: Multiplier for values (>1 = more weight)

    Returns:
        Boosted sparse vector dict
    """
    if boost_factor == 1.0 or not sparse.get("values"):
        return sparse

    return {
        "indices": sparse["indices"],
        "values": [v * boost_factor for v in sparse["values"]]
    }


def detect_query_type(query: str) -> RetrievalPreset:
    """
    Auto-detect optimal preset based on query characteristics.

    Args:
        query: Query string

    Returns:
        Recommended RetrievalPreset
    """
    query_lower = query.lower()

    # Technical/architecture queries -> BM25 heavy
    technical_indicators = [
        "how is", "wired", "configured", "setup", "architecture",
        "qdrant", "memory", "storage", "database", "json",
        "playbook", "config", "system", "hook", "mcp"
    ]

    technical_score = sum(1 for ind in technical_indicators if ind in query_lower)

    # Semantic/conceptual queries -> Dense heavy
    semantic_indicators = [
        "why", "explain", "understand", "concept", "meaning",
        "best practice", "strategy", "approach", "philosophy"
    ]

    semantic_score = sum(1 for ind in semantic_indicators if ind in query_lower)

    # Decision
    if technical_score >= 3:
        return RetrievalPreset.MAX_PRECISION
    elif technical_score >= 2:
        return RetrievalPreset.BM25_HEAVY
    elif semantic_score >= 2:
        return RetrievalPreset.SEMANTIC
    else:
        return RetrievalPreset.BM25_HEAVY  # Default to BM25 heavy for technical content
