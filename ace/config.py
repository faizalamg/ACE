"""Centralized ACE configuration.

All embedding and retrieval settings in one place.
Override via environment variables or .env file.

=== CONFIGURATION HIERARCHY ===

1. Feature Flags (ACE_FEATURE_*)
   - Master switches for ACE components
   - ACE_FEATURE_MEMORIES: Enable/disable memory storage/retrieval
   - ACE_FEATURE_CODE_CONTEXT: Enable/disable code workspace indexing
   - ACE_FEATURE_MCP_SERVER: Enable/disable MCP server endpoints

2. Embedding Providers (ACE_*_PROVIDER)
   - Select between "local" (LM Studio) or "external" (cloud APIs)
   - ACE_TEXT_EMBEDDING_PROVIDER: For memories (local=LM Studio, external=future)
   - ACE_CODE_EMBEDDING_PROVIDER: For code (local=LM Studio, external=Voyage)

3. Provider-Specific Settings
   - Local: ACE_LOCAL_EMBEDDING_URL, ACE_LOCAL_EMBEDDING_MODEL, ACE_LOCAL_EMBEDDING_DIM
   - Voyage: VOYAGE_API_KEY, ACE_VOYAGE_MODEL, ACE_VOYAGE_DIMENSION, ACE_VOYAGE_BATCH_*

4. Qdrant Settings (ACE_QDRANT_*)
   - ACE_QDRANT_URL: Qdrant server URL (default: http://localhost:6333)
   - ACE_QDRANT_PORT: Alternative port specification
   - ACE_*_COLLECTION: Collection names for each feature
   - ACE_*_DIMENSION: Vector dimensions for each collection
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


def _get_env(key: str, default: str) -> str:
    """Get environment variable with default."""
    return os.getenv(key, default)


def _get_env_int(key: str, default: int) -> int:
    """Get integer environment variable with default."""
    return int(os.getenv(key, str(default)))


def _get_env_float(key: str, default: float) -> float:
    """Get float environment variable with default."""
    return float(os.getenv(key, str(default)))


def _get_env_bool(key: str, default: bool) -> bool:
    """Get boolean environment variable with default."""
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


# ============================================================================
# CONFIG FILE LOADING
# ============================================================================

def _get_workspace_root() -> Path:
    """
    Find the workspace root by looking for .ace/.ace.json.

    Searches upward from current directory until .ace/.ace.json is found.

    Returns:
        Path to workspace root directory.

    Raises:
        FileNotFoundError: If no .ace/.ace.json found.
    """
    current = Path.cwd()

    # Search upward until .ace directory found or filesystem root
    while current != current.parent:
        ace_config = current / ".ace" / ".ace.json"
        if ace_config.exists():
            return current

        # Move up one level
        current = current.parent

    # Not found - check if .ace exists in cwd
    cwd_config = Path.cwd() / ".ace" / ".ace.json"
    if cwd_config.exists():
        return Path.cwd()

    raise FileNotFoundError(
        "No .ace/.ace.json found. "
        "Run ace_onboard to initialize workspace configuration."
    )


def _load_ace_config() -> dict:
    """
    Load .ace/.ace.json config file if it exists.

    Returns:
        Dict with config values, or empty dict if file not found.

    Valid config fields:
        - code_embedding_model: str ("voyage" | "jina" | "nomic")
        - workspace_name: str (used by ace_onboard)
        - workspace_path: str (used by ace_onboard)
        - collection_name: str (for Qdrant collection naming)
        - onboarded_at: str (timestamp for ace_onboard)
    """
    try:
        workspace_root = _get_workspace_root()
        config_path = workspace_root / ".ace" / ".ace.json"

        if not config_path.exists():
            return {}

        import json
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except FileNotFoundError:
        # No workspace config found - return empty
        return {}
    except Exception as e:
        logger.warning(f"Failed to load .ace/.ace.json: {e}")
        return {}


# Module-level cache for config file
_ace_config_file_cache: dict = {}


def _get_ace_config_field(field: str, default: str = None) -> str | None:
    """
    Get a config field from .ace/.ace.json with caching.

    Priority: Config file > env var > default

    Args:
        field: Config field name (e.g., "code_embedding_model")
        default: Default value if not found

    Returns:
        Field value from config, env var, or default
    """
    global _ace_config_file_cache

    # Load config file once
    if not _ace_config_file_cache:
        _ace_config_file_cache = _load_ace_config()

    # Check config file first
    if field in _ace_config_file_cache:
        return _ace_config_file_cache[field]

    # Fall back to env var
    env_var = f"ACE_{field.upper()}"
    return os.getenv(env_var, default)


def update_ace_config_field(field: str, value: str) -> None:
    """
    Update a field in .ace/.ace.json.

    Creates or updates .ace/.ace.json file with new field value.

    Args:
        field: Config field name
        value: New value to set

    Raises:
        FileNotFoundError: If no .ace/.ace.json exists
    """
    workspace_root = _get_workspace_root()
    config_path = workspace_root / ".ace" / ".ace.json"

    import json

    # Read existing config
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Update field
    config[field] = value

    # Write back
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Clear cache so next read picks up change
    global _ace_config_file_cache
    _ace_config_file_cache = {}


# ============================================================================
# FEATURE FLAGS - Master switches for ACE components
# ============================================================================

@dataclass
class FeatureFlags:
    """Master feature flags for enabling/disabling ACE components.
    
    Environment Variables:
        ACE_FEATURE_MEMORIES: Enable memory storage/retrieval (default: true)
        ACE_FEATURE_CODE_CONTEXT: Enable code workspace indexing (default: true)
        ACE_FEATURE_MCP_SERVER: Enable MCP server endpoints (default: true)
    """
    
    # Core features
    enable_memories: bool = field(default_factory=lambda: _get_env_bool("ACE_FEATURE_MEMORIES", True))
    enable_code_context: bool = field(default_factory=lambda: _get_env_bool("ACE_FEATURE_CODE_CONTEXT", True))
    enable_mcp_server: bool = field(default_factory=lambda: _get_env_bool("ACE_FEATURE_MCP_SERVER", True))
    
    # Sub-features (only active if parent feature is enabled)
    enable_llm_features: bool = field(default_factory=lambda: _get_env_bool("ACE_FEATURE_LLM", True))
    enable_reranking: bool = field(default_factory=lambda: _get_env_bool("ACE_FEATURE_RERANKING", True))
    enable_hybrid_search: bool = field(default_factory=lambda: _get_env_bool("ACE_FEATURE_HYBRID", True))


def get_feature_flags() -> FeatureFlags:
    """Get feature flags configuration."""
    return FeatureFlags()


# ============================================================================
# EMBEDDING PROVIDER SELECTION
# ============================================================================

@dataclass
class EmbeddingProviderConfig:
    """Provider selection for embedding models.

    Allows choosing between local (LM Studio) and external (cloud API) providers
    for both text and code embeddings.

    Environment Variables:
        ACE_TEXT_EMBEDDING_PROVIDER: "local" or "external" (default: local)
        ACE_CODE_EMBEDDING_PROVIDER: "local", "nomic", or "voyage" (default: voyage)
            - "local": jina-embeddings-v2-base-code (768d)
            - "nomic": nomic-embed-code (3584d) - SOTA, outperforms Voyage on CodeSearchNet
            - "voyage": voyage-code-3 (1024d) - cloud API
        ACE_CODE_EMBEDDING_MODEL: Override model selection (same as ACE_CODE_EMBEDDING_PROVIDER)
            - Reads from ACEConfig.code_embedding_model first
            - Falls back to ACE_CODE_EMBEDDING_PROVIDER env var
    """

    # Provider selection: "local" (LM Studio) or "external" (cloud APIs)
    text_provider: str = field(default_factory=lambda: _get_env("ACE_TEXT_EMBEDDING_PROVIDER", "local"))
    code_provider: str = field(default_factory=lambda: _get_env("ACE_CODE_EMBEDDING_PROVIDER", "voyage"))

    def __post_init__(self):
        """Post-initialization: Check ACEConfig for code_embedding_model override."""
        # Import here to avoid circular dependency
        try:
            # Import get_config from this module (may already be in namespace)
            from ace.config import get_config
            ace_config = get_config()
            # If ACEConfig has code_embedding_model set (from config file), use it
            if hasattr(ace_config, 'code_embedding_model') and ace_config.code_embedding_model:
                self.code_provider = ace_config.code_embedding_model
        except Exception:
            # If ACEConfig not initialized yet, use default from field
            pass

    def is_text_local(self) -> bool:
        """Check if text embeddings use local provider."""
        return self.text_provider.lower() == "local"
    
    def is_code_local(self) -> bool:
        """Check if code embeddings use local provider (Jina)."""
        return self.code_provider.lower() in ("local", "jina")
    
    def is_code_nomic(self) -> bool:
        """Check if code embeddings use nomic-embed-code (local, 3584d)."""
        return self.code_provider.lower() == "nomic"
    
    def is_code_voyage(self) -> bool:
        """Check if code embeddings use Voyage API."""
        return self.code_provider.lower() in ("voyage", "external")


def get_embedding_provider_config() -> EmbeddingProviderConfig:
    """Get embedding provider configuration."""
    return EmbeddingProviderConfig()


# ============================================================================
# LOCAL EMBEDDING CONFIGURATION (LM Studio)
# ============================================================================

@dataclass
class LocalEmbeddingConfig:
    """Local embedding model configuration (LM Studio/Ollama).
    
    Used for both text and code embeddings when provider is set to "local".
    
    Environment Variables:
        ACE_LOCAL_EMBEDDING_URL: LM Studio server URL (default: http://localhost:1234)
        ACE_LOCAL_TEXT_MODEL: Model for text embeddings (default: text-embedding-qwen3-embedding-8b)
        ACE_LOCAL_TEXT_DIM: Text embedding dimension (default: 4096)
        ACE_LOCAL_CODE_MODEL: Model for code embeddings (default: jina-embeddings-v2-base-code)
        ACE_LOCAL_CODE_DIM: Code embedding dimension (default: 768)
    """
    
    # Server URL (shared between text and code)
    url: str = field(default_factory=lambda: _get_env("ACE_LOCAL_EMBEDDING_URL", "http://localhost:1234"))
    
    # Text embedding model (for memories/lessons)
    text_model: str = field(default_factory=lambda: _get_env("ACE_LOCAL_TEXT_MODEL", "text-embedding-qwen3-embedding-8b"))
    text_dimension: int = field(default_factory=lambda: _get_env_int("ACE_LOCAL_TEXT_DIM", 4096))
    text_max_length: int = field(default_factory=lambda: _get_env_int("ACE_LOCAL_TEXT_MAX_LENGTH", 8000))
    
    # Code embedding model (for code context)
    code_model: str = field(default_factory=lambda: _get_env("ACE_LOCAL_CODE_MODEL", "jina-embeddings-v2-base-code"))
    code_dimension: int = field(default_factory=lambda: _get_env_int("ACE_LOCAL_CODE_DIM", 768))
    code_max_length: int = field(default_factory=lambda: _get_env_int("ACE_LOCAL_CODE_MAX_LENGTH", 8000))


def get_local_embedding_config() -> LocalEmbeddingConfig:
    """Get local embedding configuration."""
    return LocalEmbeddingConfig()


# ============================================================================
# TEXT EMBEDDING CONFIGURATION (General-purpose for memories)
# ============================================================================

@dataclass
class EmbeddingConfig:
    """Embedding model configuration for memory/lessons (general-purpose).
    
    NOTE: This is a legacy config. For new deployments, use:
    - ACE_TEXT_EMBEDDING_PROVIDER to select provider
    - LocalEmbeddingConfig or external provider config
    
    Environment Variables:
        ACE_EMBEDDING_URL: LM Studio server URL (default: from LocalEmbeddingConfig)
        ACE_EMBEDDING_MODEL: Model name (default: from LocalEmbeddingConfig)
        ACE_EMBEDDING_DIM: Embedding dimension (default: from LocalEmbeddingConfig)
    """

    # LM Studio server
    url: str = field(default_factory=lambda: _get_env("ACE_EMBEDDING_URL", "http://localhost:1234"))

    # Model name (Qwen3-Embedding-8B - proper embedding model, 4096 dims)
    model: str = field(default_factory=lambda: _get_env("ACE_EMBEDDING_MODEL", "text-embedding-qwen3-embedding-8b"))

    # Embedding dimension
    dimension: int = field(default_factory=lambda: _get_env_int("ACE_EMBEDDING_DIM", 4096))

    # Max input length (chars)
    max_input_length: int = field(default_factory=lambda: _get_env_int("ACE_EMBEDDING_MAX_LENGTH", 8000))


@dataclass
class CodeEmbeddingConfig:
    """DEPRECATED - Use VoyageCodeEmbeddingConfig instead.
    
    This configuration was for the old Jina-v2-base-code model (768d).
    Code indexing now uses Voyage-code-3 (1024d) exclusively.
    
    Environment Variables:
        ACE_CODE_EMBEDDING_URL: LM Studio server URL (DEPRECATED)
        ACE_CODE_EMBEDDING_MODEL: Model name (DEPRECATED)
        ACE_CODE_EMBEDDING_DIM: Embedding dimension (DEPRECATED)
    """

    # LM Studio server (defaults to same as general embedding)
    url: str = field(default_factory=lambda: _get_env("ACE_CODE_EMBEDDING_URL", _get_env("ACE_EMBEDDING_URL", "http://localhost:1234")))

    # Model name (DEPRECATED - was Jina-v2-base-code, now use Voyage-code-3)
    model: str = field(default_factory=lambda: _get_env("ACE_CODE_EMBEDDING_MODEL", "voyage-code-3"))

    # Embedding dimension (1024d for Voyage-code-3)
    dimension: int = field(default_factory=lambda: _get_env_int("ACE_CODE_EMBEDDING_DIM", 1024))

    # Max input length (chars)
    max_input_length: int = field(default_factory=lambda: _get_env_int("ACE_CODE_EMBEDDING_MAX_LENGTH", 8000))


@dataclass
class NomicCodeEmbeddingConfig:
    """Nomic AI code embedding configuration (nomic-embed-code).
    
    Uses nomic-embed-code which is a SOTA 7B code embedding model that
    outperforms Voyage Code 3 on CodeSearchNet benchmarks.
    Runs locally via LM Studio.
    
    Key Features:
        - 7B parameters, 32K context window
        - Trained on CoRNStack dataset (Python, Java, Ruby, PHP, JS, Go)
        - Based on Qwen2.5-Coder-7B
        - Output dimension: 3584
    
    Environment Variables:
        ACE_NOMIC_URL: LM Studio server URL (default: http://localhost:1234)
        ACE_NOMIC_MODEL: Model name (default: nomic-embed-code)
        ACE_NOMIC_DIMENSION: Embedding dimension (default: 3584)
    """

    # LM Studio server URL
    url: str = field(default_factory=lambda: _get_env("ACE_NOMIC_URL", "http://localhost:1234"))

    # Model name in LM Studio (q8_0 is faster, f16 is higher precision)
    model: str = field(default_factory=lambda: _get_env("ACE_NOMIC_MODEL", "text-embedding-nomic-embed-code@q8_0"))

    # Embedding dimension (3584d for nomic-embed-code based on Qwen2.5-7B)
    dimension: int = field(default_factory=lambda: _get_env_int("ACE_NOMIC_DIMENSION", 3584))

    # Max input length (tokens) - 32K for nomic-embed-code
    max_input_tokens: int = field(default_factory=lambda: _get_env_int("ACE_NOMIC_MAX_TOKENS", 32000))


def get_nomic_code_embedding_config() -> NomicCodeEmbeddingConfig:
    """Get Nomic code embedding configuration."""
    return NomicCodeEmbeddingConfig()


@dataclass
class VoyageCodeEmbeddingConfig:
    """Voyage AI code embedding configuration (voyage-code-3).
    
    Uses voyage-code-3 which is optimized for code retrieval tasks,
    with 32K context window and code-specific training.
    
    Environment Variables:
        VOYAGE_API_KEY: Voyage AI API key (required)
        ACE_VOYAGE_MODEL: Model name (default: voyage-code-3)
        ACE_VOYAGE_DIMENSION: Embedding dimension (default: 1024)
        ACE_VOYAGE_BATCH_SIZE: Max texts per batch (default: 300, max: 1000)
        ACE_VOYAGE_BATCH_TOKENS: Max tokens per batch (default: 80000, max: 120000)
        ACE_VOYAGE_PARALLEL: Parallel batch requests (default: 4)
    """

    # API key (required)
    api_key: str = field(default_factory=lambda: _get_env("VOYAGE_API_KEY", ""))

    # Model name (voyage-code-3 - code-optimized)
    model: str = field(default_factory=lambda: _get_env("ACE_VOYAGE_MODEL", "voyage-code-3"))

    # Embedding dimension (1024d default, options: 256, 512, 1024, 2048)
    dimension: int = field(default_factory=lambda: _get_env_int("ACE_VOYAGE_DIMENSION", 1024))

    # Max input length (tokens) - 32K for voyage-code-3
    max_input_tokens: int = field(default_factory=lambda: _get_env_int("ACE_VOYAGE_MAX_TOKENS", 32000))
    
    # Batch processing settings (for workspace indexing speed)
    # Official limits: 1000 texts, 120K tokens per request
    # Using conservative limits: 300 texts, 80K tokens (code has ~2-3 chars/token)
    batch_size: int = field(default_factory=lambda: _get_env_int("ACE_VOYAGE_BATCH_SIZE", 300))
    batch_max_tokens: int = field(default_factory=lambda: _get_env_int("ACE_VOYAGE_BATCH_TOKENS", 80000))
    parallel_batches: int = field(default_factory=lambda: _get_env_int("ACE_VOYAGE_PARALLEL", 4))

    # Input type for queries vs documents
    query_input_type: str = "query"
    document_input_type: str = "document"
    
    def is_configured(self) -> bool:
        """Check if Voyage API is configured."""
        return bool(self.api_key)


@dataclass
class QdrantConfig:
    """Qdrant vector database configuration.
    
    Centralized configuration for all Qdrant connections and collections.
    
    Environment Variables:
        # Server connection
        ACE_QDRANT_URL: Full URL (default: http://localhost:6333)
        ACE_QDRANT_HOST: Host only (default: localhost) - used if URL not set
        ACE_QDRANT_PORT: Port only (default: 6333) - used if URL not set
        ACE_QDRANT_API_KEY: API key for Qdrant Cloud (optional)
        ACE_QDRANT_GRPC: Use gRPC instead of HTTP (default: false)
        
        # Collection names
        ACE_MEMORIES_COLLECTION: Memory storage collection (default: ace_memories_hybrid)
        ACE_UNIFIED_COLLECTION: Unified memory collection (default: ace_memories_hybrid)
        ACE_BULLETS_COLLECTION: Bullet points collection (default: ace_bullets)
        ACE_CODE_COLLECTION: Code context collection (default: ace_code_context)
        
        # Collection dimensions (must match embedding model output)
        ACE_MEMORIES_DIMENSION: Memory collection vector size (default: 4096)
        ACE_CODE_DIMENSION: Code collection vector size (default: 1024)
    """

    # Server URL (takes precedence over host:port)
    url: str = field(default_factory=lambda: _get_env("ACE_QDRANT_URL", "http://localhost:6333"))
    
    # Alternative: host and port separately
    host: str = field(default_factory=lambda: _get_env("ACE_QDRANT_HOST", "localhost"))
    port: int = field(default_factory=lambda: _get_env_int("ACE_QDRANT_PORT", 6333))
    
    # Authentication (for Qdrant Cloud)
    api_key: Optional[str] = field(default_factory=lambda: _get_env("ACE_QDRANT_API_KEY", "") or None)
    
    # Use gRPC for better performance
    use_grpc: bool = field(default_factory=lambda: _get_env_bool("ACE_QDRANT_GRPC", False))
    grpc_port: int = field(default_factory=lambda: _get_env_int("ACE_QDRANT_GRPC_PORT", 6334))

    # Collection names
    memories_collection: str = field(default_factory=lambda: _get_env("ACE_MEMORIES_COLLECTION", "ace_memories_hybrid"))
    unified_collection: str = field(default_factory=lambda: _get_env("ACE_UNIFIED_COLLECTION", "ace_memories_hybrid"))
    bullets_collection: str = field(default_factory=lambda: _get_env("ACE_BULLETS_COLLECTION", "ace_bullets"))
    code_collection: str = field(default_factory=lambda: _get_env("ACE_CODE_COLLECTION", "ace_code_context"))
    
    # Collection dimensions (must match embedding model)
    memories_dimension: int = field(default_factory=lambda: _get_env_int("ACE_MEMORIES_DIMENSION", 4096))
    code_dimension: int = field(default_factory=lambda: _get_env_int("ACE_CODE_DIMENSION", 1024))

    def get_code_collection_name(self) -> str:
        """
        Get the correct code collection name based on embedding provider.

        Returns:
            "ace_code_context_voyage" for Voyage (1024d)
            "ace_code_context_jina" for Jina (768d)
            "ace_code_context_nomic" for Nomic (3584d)
            "ace_code_context" as fallback (Voyage default)
        """
        # Import provider config to determine suffix
        from ace.config import get_embedding_provider_config
        provider_config = get_embedding_provider_config()

        if provider_config.is_code_nomic():
            return "ace_code_context_nomic"
        elif provider_config.is_code_local():
            return "ace_code_context_jina"
        elif provider_config.is_code_voyage():
            return "ace_code_context_voyage"
        else:
            # Fallback - check if env var is explicitly set
            env_collection = os.environ.get("ACE_CODE_COLLECTION", "")
            if env_collection:
                return env_collection
            return "ace_code_context"
    
    def get_connection_url(self) -> str:
        """Get the effective connection URL."""
        # If ACE_QDRANT_URL was explicitly set (not default), use it
        if os.getenv("ACE_QDRANT_URL"):
            return self.url
        # Otherwise construct from host:port
        protocol = "grpc" if self.use_grpc else "http"
        port = self.grpc_port if self.use_grpc else self.port
        return f"{protocol}://{self.host}:{port}"


@dataclass
class BM25Config:
    """BM25 sparse vector configuration."""

    k1: float = field(default_factory=lambda: _get_env_float("ACE_BM25_K1", 1.5))
    b: float = field(default_factory=lambda: _get_env_float("ACE_BM25_B", 0.75))
    avg_doc_length: int = field(default_factory=lambda: _get_env_int("ACE_BM25_AVG_DOC_LENGTH", 50))


@dataclass
class LLMConfig:
    """LLM configuration for query rewriting and retrieval enhancements."""

    # Z.ai GLM API (default)
    # Note: GLM 4.7 has 2 concurrency limit but better quality than 4.6
    api_key: str = field(default_factory=lambda: _get_env("ZAI_API_KEY", ""))
    api_base: str = field(default_factory=lambda: _get_env("ZAI_API_BASE", "https://api.z.ai/api/coding/paas/v4"))
    model: str = field(default_factory=lambda: _get_env("ZAI_MODEL", "glm-4.7"))

    # Query rewriting settings
    enable_query_rewrite: bool = field(default_factory=lambda: _get_env_bool("ACE_QUERY_REWRITE", True))
    rewrite_max_tokens: int = field(default_factory=lambda: _get_env_int("ACE_REWRITE_MAX_TOKENS", 1000))
    rewrite_temperature: float = field(default_factory=lambda: _get_env_float("ACE_REWRITE_TEMPERATURE", 0.3))

    # Structured Query Enhancement (rule-based, no LLM - fast!)
    # Uses .enhancedprompt.md methodology: intent classification + domain expansion
    # Improves keyword precision by ~8% with zero latency cost
    enable_structured_enhancement: bool = field(default_factory=lambda: _get_env_bool("ACE_STRUCTURED_ENHANCEMENT", True))

    # LLM Query Expansion (generates semantic alternatives)
    # Note: GLM 4.6 uses reasoning mode which needs high token limit
    enable_llm_expansion: bool = field(default_factory=lambda: _get_env_bool("ACE_LLM_EXPANSION", True))
    expansion_max_tokens: int = field(default_factory=lambda: _get_env_int("ACE_EXPANSION_MAX_TOKENS", 1500))
    expansion_timeout: float = field(default_factory=lambda: _get_env_float("ACE_EXPANSION_TIMEOUT", 60.0))

    # LLM Relevance Filtering (filters noise from retrieval results)
    # Note: Disabled by default - use retrieval tuning instead for efficiency
    enable_llm_filtering: bool = field(default_factory=lambda: _get_env_bool("ACE_LLM_FILTERING", False))
    filtering_max_tokens: int = field(default_factory=lambda: _get_env_int("ACE_FILTERING_MAX_TOKENS", 2000))
    filtering_timeout: float = field(default_factory=lambda: _get_env_float("ACE_FILTERING_TIMEOUT", 120.0))
    filtering_top_k: int = field(default_factory=lambda: _get_env_int("ACE_FILTERING_TOP_K", 10))

    # Fast local LLM fallback (for speed-critical operations)
    # Set ACE_USE_LOCAL_LLM=true to use local LM Studio instead of Z.ai
    use_local_llm: bool = field(default_factory=lambda: _get_env_bool("ACE_USE_LOCAL_LLM", False))
    local_llm_url: str = field(default_factory=lambda: _get_env("ACE_LOCAL_LLM_URL", "http://localhost:1234"))
    local_llm_model: str = field(default_factory=lambda: _get_env("ACE_LOCAL_LLM_MODEL", "gpt-oss-20b"))
    # Max tokens for local LLM (high for reasoning models that output reasoning + content)
    local_llm_max_tokens: int = field(default_factory=lambda: _get_env_int("ACE_LOCAL_LLM_MAX_TOKENS", 800))
    # Timeout for local LLM (high default for JIT model loading on first request)
    local_llm_timeout: float = field(default_factory=lambda: _get_env_float("ACE_LOCAL_LLM_TIMEOUT", 120.0))


@dataclass
class RetrievalConfig:
    """Retrieval pipeline configuration."""

    # Number of candidates per query
    candidates_per_query: int = field(default_factory=lambda: _get_env_int("ACE_CANDIDATES_PER_QUERY", 20))

    # First stage retrieval limit (initial_k - before reranking/filtering)
    first_stage_k: int = field(default_factory=lambda: _get_env_int("ACE_FIRST_STAGE_K", 40))
    initial_k: int = field(default_factory=lambda: _get_env_int("ACE_INITIAL_K", 100))  # Alias for first_stage_k

    # Final results limit (updated default to 64 for P7.1 balanced preset)
    final_k: int = field(default_factory=lambda: _get_env_int("ACE_FINAL_K", 64))

    # Query expansion count
    num_expanded_queries: int = field(default_factory=lambda: _get_env_int("ACE_NUM_EXPANDED_QUERIES", 4))

    # HyDE (Hypothetical Document Embeddings)
    # Can be bool or "auto" (auto decides based on query complexity)
    use_hyde: bool | str = field(default_factory=lambda: os.getenv("ACE_USE_HYDE", "auto"))

    # Hybrid search alpha (0.0 = pure semantic, 1.0 = pure keyword/BM25)
    hybrid_alpha: float = field(default_factory=lambda: _get_env_float("ACE_HYBRID_ALPHA", 0.5))

    # Re-ranking
    enable_reranking: bool = field(default_factory=lambda: _get_env_bool("ACE_ENABLE_RERANKING", True))
    cross_encoder_model: str = field(default_factory=lambda: _get_env("ACE_CROSS_ENCODER", "cross-encoder/ms-marco-MiniLM-L-6-v2"))

    # Cross-encoder relevance threshold
    # Lower threshold = more permissive (higher recall, lower precision)
    # Higher threshold = more strict (lower recall, higher precision)
    # MS-MARCO MiniLM-L-6-v2 scores range from -12 to -6
    # -11.5 targets 95%+ P@3 with high recall
    # -10.5 achieves ~93% P@3 (previous default)
    # -10.0 achieves ~91% P@3 (very permissive)
    cross_encoder_threshold: float = field(default_factory=lambda: _get_env_float("ACE_CROSS_ENCODER_THRESHOLD", -11.5))


@dataclass
class ELFConfig:
    """
    ELF-inspired feature configuration.

    Inspired by the Emergent Learning Framework (ELF):
    https://github.com/Spacehunterz/Emergent-Learning-Framework_ELF

    ELF is MIT licensed. These features are adaptations of ELF concepts
    for the ACE retrieval pipeline, not direct code copies.
    """

    # Confidence Decay
    enable_confidence_decay: bool = field(default_factory=lambda: _get_env_bool("ACE_CONFIDENCE_DECAY", True))
    decay_rate_per_week: float = field(default_factory=lambda: _get_env_float("ACE_DECAY_RATE", 0.95))
    min_confidence_threshold: float = field(default_factory=lambda: _get_env_float("ACE_MIN_CONFIDENCE", 0.1))

    # Query Complexity Classifier
    enable_query_classifier: bool = field(default_factory=lambda: _get_env_bool("ACE_QUERY_CLASSIFIER", True))
    technical_terms_bypass_llm: bool = field(default_factory=lambda: _get_env_bool("ACE_TECHNICAL_BYPASS", True))

    # Golden Rules Auto-Promotion
    enable_golden_rules: bool = field(default_factory=lambda: _get_env_bool("ACE_GOLDEN_RULES", True))
    golden_rule_helpful_threshold: int = field(default_factory=lambda: _get_env_int("ACE_GOLDEN_THRESHOLD", 10))
    golden_rule_max_harmful: int = field(default_factory=lambda: _get_env_int("ACE_GOLDEN_MAX_HARMFUL", 0))
    golden_rule_demotion_harmful_threshold: int = field(default_factory=lambda: _get_env_int("ACE_GOLDEN_DEMOTION_HARMFUL", 3))

    # Tiered Model Selection (4 tiers: most capable → most economical)
    enable_tiered_models: bool = field(default_factory=lambda: _get_env_bool("ACE_TIERED_MODELS", True))
    tier1_model: str = field(default_factory=lambda: _get_env("ACE_TIER1_MODEL", "claude-opus-4-5-20251101"))  # Most capable
    tier2_model: str = field(default_factory=lambda: _get_env("ACE_TIER2_MODEL", "claude-sonnet-4-5-20241022"))  # Balanced
    tier3_model: str = field(default_factory=lambda: _get_env("ACE_TIER3_MODEL", "glm-4.6"))  # Cost-effective
    tier4_model: str = field(default_factory=lambda: _get_env("ACE_TIER4_MODEL", "claude-3-haiku-20240307"))  # Most economical


@dataclass(frozen=True)
class PresetConfig:
    """
    Retrieval preset configuration (P7.1 Multi-Preset System).

    Immutable preset definitions for common retrieval scenarios.
    Presets control: final_k, use_hyde, enable_reranking, num_expanded_queries.
    """
    final_k: int
    use_hyde: bool | str
    enable_reranking: bool
    num_expanded_queries: int


# P7.1 Multi-Preset System - Predefined Configurations
PRESETS: dict[str, PresetConfig] = {
    # Precision mode: Auggie-like focused results, minimal context usage
    # Use when you want fewer, highly relevant results
    "precision": PresetConfig(
        final_k=10,           # Return few, focused results
        use_hyde=False,       # No hypothetical docs - direct matching only
        enable_reranking=True,  # Keep reranker for quality filtering
        num_expanded_queries=1  # No query expansion - single focused query
    ),
    "fast": PresetConfig(
        final_k=40,
        use_hyde=False,
        enable_reranking=False,
        num_expanded_queries=1
    ),
    "balanced": PresetConfig(
        final_k=64,
        use_hyde="auto",
        enable_reranking=True,
        num_expanded_queries=4
    ),
    "deep": PresetConfig(
        final_k=96,
        use_hyde=True,
        enable_reranking=True,
        num_expanded_queries=6
    ),
    "diverse": PresetConfig(
        final_k=80,
        use_hyde=False,
        enable_reranking=True,
        num_expanded_queries=4
    ),
}


def get_preset(name: str) -> PresetConfig:
    """
    Get a preset configuration by name (case-insensitive).

    Args:
        name: Preset name (fast, balanced, deep, diverse)

    Returns:
        New PresetConfig instance with preset values

    Raises:
        ValueError: If preset name is invalid

    Performance: <1ms latency
    """
    name_lower = name.lower()

    if name_lower not in PRESETS:
        valid_presets = ", ".join(PRESETS.keys())
        raise ValueError(f"Invalid preset name: '{name}'. Valid presets: {valid_presets}")

    # Return new instance (frozen dataclass creates new object on construction)
    preset = PRESETS[name_lower]
    return PresetConfig(
        final_k=preset.final_k,
        use_hyde=preset.use_hyde,
        enable_reranking=preset.enable_reranking,
        num_expanded_queries=preset.num_expanded_queries
    )


def get_active_preset_name() -> str | None:
    """
    Get the active preset name from ACE_PRESET environment variable.
    
    Returns:
        Preset name if set and valid, None otherwise.
        
    Valid presets: precision, fast, balanced, deep, diverse
    
    Example:
        Set ACE_PRESET=precision for Auggie-like focused results.
    """
    preset_name = os.getenv("ACE_PRESET", "").lower()
    if preset_name and preset_name in PRESETS:
        return preset_name
    return None


def apply_preset_to_retrieval_config(config: RetrievalConfig, preset: str) -> RetrievalConfig:
    """
    Apply a preset to a RetrievalConfig, returning a new config.

    Modifies only preset-controlled fields:
    - final_k
    - use_hyde
    - enable_reranking
    - num_expanded_queries

    All other fields are preserved from the original config.

    Args:
        config: Original RetrievalConfig instance
        preset: Preset name (fast, balanced, deep, diverse)

    Returns:
        New RetrievalConfig with preset values applied

    Performance: <1ms latency
    """
    preset_config = get_preset(preset)

    # Create new config with preset values, preserving ALL other fields
    return RetrievalConfig(
        # Preset-controlled fields (overridden)
        final_k=preset_config.final_k,
        use_hyde=preset_config.use_hyde,
        enable_reranking=preset_config.enable_reranking,
        num_expanded_queries=preset_config.num_expanded_queries,

        # Preserved fields (from original config)
        candidates_per_query=config.candidates_per_query,
        first_stage_k=config.first_stage_k,
        initial_k=config.initial_k,
        hybrid_alpha=config.hybrid_alpha,
        cross_encoder_model=config.cross_encoder_model,
    )


@dataclass
class ARIAConfig:
    """
    ARIA (Adaptive Retrieval Intelligence Architecture) configuration.

    P7 Feature: LinUCB contextual bandit for adaptive retrieval strategy selection.
    The bandit learns which retrieval preset (FAST/BALANCED/DEEP/DIVERSE) works
    best for different query types based on feedback.

    Algorithm Reference:
        Li, Lihong, et al. "A Contextual-Bandit Approach to Personalized News
        Article Recommendation." WWW 2010. arXiv:1003.0146
    """

    # Enable ARIA adaptive retrieval
    enable_aria: bool = field(default_factory=lambda: _get_env_bool("ACE_ENABLE_ARIA", True))

    # LinUCB bandit parameters
    bandit_alpha: float = field(default_factory=lambda: _get_env_float("ACE_BANDIT_ALPHA", 1.0))
    bandit_dimensions: int = field(default_factory=lambda: _get_env_int("ACE_BANDIT_DIMS", 10))

    # State persistence
    enable_bandit_persistence: bool = field(default_factory=lambda: _get_env_bool("ACE_BANDIT_PERSIST", True))
    bandit_state_file: str = field(default_factory=lambda: _get_env("ACE_BANDIT_STATE", ".ace_bandit_state.json"))

    # Quality feedback boost
    enable_quality_boost: bool = field(default_factory=lambda: _get_env_bool("ACE_QUALITY_BOOST", True))
    quality_boost_scale: float = field(default_factory=lambda: _get_env_float("ACE_QUALITY_SCALE", 0.1))


@dataclass
class MemoryArchitectureConfig:
    """
    Memory architecture configuration (Reddit r/Rag inspired).

    Features from: "Should 'User Memory' be architecturally distinct from Vector Store?"
    https://www.reddit.com/r/Rag/s/H1RdYfF390

    These features implement mutable memory management with version history,
    conflict detection, and deterministic lookup capabilities.
    """

    # Version History - soft-delete with audit trail
    enable_version_history: bool = field(default_factory=lambda: _get_env_bool("ACE_VERSION_HISTORY", True))
    max_versions_per_bullet: int = field(default_factory=lambda: _get_env_int("ACE_MAX_VERSIONS", 10))

    # Entity-Key O(1) Lookup - deterministic retrieval by key
    enable_entity_key_lookup: bool = field(default_factory=lambda: _get_env_bool("ACE_ENTITY_KEY_LOOKUP", True))

    # Conflict Detection - detect contradictory bullets
    enable_conflict_detection: bool = field(default_factory=lambda: _get_env_bool("ACE_CONFLICT_DETECTION", True))
    conflict_similarity_threshold: float = field(default_factory=lambda: _get_env_float("ACE_CONFLICT_THRESHOLD", 0.85))
    use_llm_for_conflict_analysis: bool = field(default_factory=lambda: _get_env_bool("ACE_CONFLICT_LLM", False))

    # Temporal Filtering - filter by creation/update timestamps
    enable_temporal_filtering: bool = field(default_factory=lambda: _get_env_bool("ACE_TEMPORAL_FILTERING", True))

    # Default filtering behavior
    exclude_superseded_by_default: bool = field(default_factory=lambda: _get_env_bool("ACE_EXCLUDE_SUPERSEDED", False))


@dataclass
class MultiStageConfig:
    """
    Multi-stage retrieval configuration (coarse-to-fine optimization).

    Implements a 4-stage retrieval pipeline:
    1. Stage 1 (Coarse): High-recall candidate retrieval (10x limit)
    2. Stage 2 (Filter): Score-based filtering (DISABLED by default - RRF scores unreliable)
    3. Stage 3 (Rerank): Cross-encoder reranking on all Stage 1 candidates
    4. Stage 4 (Final): Deduplication and final selection

    Benefits:
    - Higher recall by fetching more candidates in Stage 1
    - Lower latency by filtering before expensive cross-encoder
    - Better precision through multi-stage refinement

    Environment Variables:
        ACE_ENABLE_MULTISTAGE: Enable/disable multi-stage (default: True)
        ACE_MULTISTAGE_STAGE1_MULT: Stage 1 candidate multiplier (default: 10)
        ACE_MULTISTAGE_STAGE2_RATIO: Stage 2 keep ratio (default: 1.0, keep all)
        ACE_MULTISTAGE_STAGE2_PERCENTILE: Adaptive threshold percentile (default: 0, disabled)
        ACE_MULTISTAGE_GAP_DETECT: Enable gap detection (default: False)
        ACE_MULTISTAGE_STAGE3_ENABLED: Enable cross-encoder in stage 3 (default: True)
        ACE_MULTISTAGE_STAGE4_DEDUP: Stage 4 deduplication threshold (default: 0.90)
    """

    # Master enable flag
    enable_multistage: bool = field(default_factory=lambda: _get_env_bool("ACE_ENABLE_MULTISTAGE", True))

    # Stage 1: Coarse retrieval (high recall)
    # Fetch stage1_multiplier * limit candidates for maximum recall
    stage1_multiplier: int = field(default_factory=lambda: _get_env_int("ACE_MULTISTAGE_STAGE1_MULT", 10))

    # Stage 2: Score-based filtering
    # NOTE: Disabled by default (percentile=0) because RRF scores poorly predict true relevance.
    # Cross-encoder in Stage 3 handles ranking. Only enable Stage 2 for latency optimization.
    stage2_keep_ratio: float = field(default_factory=lambda: _get_env_float("ACE_MULTISTAGE_STAGE2_RATIO", 1.0))
    stage2_percentile: int = field(default_factory=lambda: _get_env_int("ACE_MULTISTAGE_STAGE2_PERCENTILE", 0))
    stage2_use_gap_detection: bool = field(default_factory=lambda: _get_env_bool("ACE_MULTISTAGE_GAP_DETECT", False))

    # Stage 3: Cross-encoder reranking
    stage3_enabled: bool = field(default_factory=lambda: _get_env_bool("ACE_MULTISTAGE_STAGE3_ENABLED", True))
    stage3_max_candidates: int = field(default_factory=lambda: _get_env_int("ACE_MULTISTAGE_STAGE3_MAX", 50))

    # Stage 4: Deduplication and final selection
    stage4_dedup_threshold: float = field(default_factory=lambda: _get_env_float("ACE_MULTISTAGE_STAGE4_DEDUP", 0.90))


def get_multistage_config() -> MultiStageConfig:
    """Get multi-stage retrieval configuration."""
    return MultiStageConfig()


@dataclass
class TypoCorrectionConfig:
    """Typo correction configuration with auto-learning support."""

    # Enable auto-learning of typos (persists corrections for instant future lookups)
    enable_auto_learning: bool = field(default_factory=lambda: _get_env_bool("ACE_TYPO_AUTO_LEARN", True))

    # Path to persist learned typos (default: tenant_data/learned_typos.json)
    learned_typos_path: str = field(default_factory=lambda: _get_env(
        "ACE_LEARNED_TYPOS_PATH",
        str(Path(__file__).parent.parent / "tenant_data" / "learned_typos.json")
    ))

    # Enable async GLM validation of learned typos
    # When enabled, corrections are validated by GLM in the background before being persisted
    enable_glm_validation: bool = field(default_factory=lambda: _get_env_bool("ACE_TYPO_GLM_VALIDATION", True))

    # Fuzzy matching similarity threshold (0.0-1.0)
    similarity_threshold: float = field(default_factory=lambda: _get_env_float("ACE_TYPO_THRESHOLD", 0.80))

    # Maximum learned typos to keep (prevents unbounded growth)
    max_learned_typos: int = field(default_factory=lambda: _get_env_int("ACE_TYPO_MAX_LEARNED", 1000))

    # LLM-based typo correction (for common English words, not just technical terms)
    # When enabled, uses LLM to correct typos that fuzzy matching can't handle
    # e.g., "updste" -> "update", "plsn" -> "plan"
    enable_llm_correction: bool = field(default_factory=lambda: _get_env_bool("ACE_TYPO_LLM_CORRECTION", True))

    # Provider for LLM typo correction: "zai" (z.ai GLM) or "local" (LM Studio)
    llm_correction_provider: str = field(default_factory=lambda: _get_env("ACE_TYPO_LLM_PROVIDER", "zai"))

    # Model for LLM typo correction (only used if provider is "local")
    # z.ai uses ZAI_MODEL from LLMConfig, local uses this model
    llm_correction_model: str = field(default_factory=lambda: _get_env("ACE_TYPO_LLM_MODEL", "gpt-oss-20b"))

    # LM Studio URL (only used if provider is "local")
    llm_correction_url: str = field(default_factory=lambda: _get_env("ACE_TYPO_LLM_URL", "http://localhost:1234/v1"))

    # Timeout for LLM typo correction (ms)
    llm_correction_timeout: float = field(default_factory=lambda: _get_env_float("ACE_TYPO_LLM_TIMEOUT", 5.0))


def get_typo_config() -> TypoCorrectionConfig:
    """Get typo correction configuration."""
    return TypoCorrectionConfig()


@dataclass
class ACEConfig:
    """Master ACE configuration."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    code_embedding: CodeEmbeddingConfig = field(default_factory=CodeEmbeddingConfig)

    # Load code_embedding_model from config file first, then env var
    code_embedding_model: str = field(
        default_factory=lambda: _get_ace_config_field(
            "code_embedding_model",
            _get_env("ACE_CODE_EMBEDDING_PROVIDER", "voyage")
        )
    )
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    elf: ELFConfig = field(default_factory=ELFConfig)
    aria: ARIAConfig = field(default_factory=ARIAConfig)
    memory: MemoryArchitectureConfig = field(default_factory=MemoryArchitectureConfig)
    typo: TypoCorrectionConfig = field(default_factory=TypoCorrectionConfig)


# Global singleton
_config: Optional[ACEConfig] = None


def get_config() -> ACEConfig:
    """Get the global ACE configuration singleton."""
    global _config
    if _config is None:
        _config = ACEConfig()
    return _config


def reset_config() -> None:
    """Reset configuration (for testing)."""
    global _config
    _config = None


# Convenience accessors
def get_embedding_config() -> EmbeddingConfig:
    """Get embedding configuration (general-purpose for memory/lessons)."""
    return get_config().embedding


def get_code_embedding_config() -> CodeEmbeddingConfig:
    """Get code-specific embedding configuration (optimized for code search)."""
    return get_config().code_embedding


def get_qdrant_config() -> QdrantConfig:
    """Get Qdrant configuration."""
    return get_config().qdrant


def get_bm25_config() -> BM25Config:
    """Get BM25 configuration."""
    return get_config().bm25


def get_retrieval_config() -> RetrievalConfig:
    """Get retrieval configuration."""
    return get_config().retrieval


def get_llm_config() -> LLMConfig:
    """Get LLM configuration for query rewriting."""
    return get_config().llm


def get_elf_config() -> ELFConfig:
    """Get ELF-inspired features configuration."""
    return get_config().elf


def get_memory_config() -> MemoryArchitectureConfig:
    """Get memory architecture configuration (Reddit r/Rag inspired)."""
    return get_config().memory


def get_aria_config() -> ARIAConfig:
    """Get ARIA (Adaptive Retrieval Intelligence Architecture) configuration."""
    return get_config().aria


# Legacy compatibility constants (deprecated, use get_config() instead)
DEFAULT_EMBEDDING_MODEL = get_embedding_config().model
DEFAULT_EMBEDDING_URL = get_embedding_config().url
DEFAULT_EMBEDDING_DIM = get_embedding_config().dimension
DEFAULT_QDRANT_URL = get_qdrant_config().url
DEFAULT_COLLECTION = get_qdrant_config().memories_collection
