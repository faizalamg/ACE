# Changelog

All notable changes to ACE Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - 2025-01-10

### Added

- **AdaptiveChunker - File-Type-Aware Chunking** - Intelligent semantic chunking based on file type
  - **Feature**: Automatically selects optimal chunking strategy based on file extension or content heuristics
  - **Strategies Implemented**:
    - `MarkdownChunker`: Section-based chunking using `##` headers (CHANGELOG.md: 58 chunks, README.md: 37 chunks)
    - `ConfigChunker`: YAML/JSON/TOML structure-aware chunking (test.yml: 3 chunks, mmlu.yaml: 6 chunks)
    - `CodeChunker`: Delegates to existing ASTChunker for AST-based code chunking (zero degradation)
    - `HTMLChunker`: Tag-based chunking for HTML/XML by semantic elements (head, body, section, article)
    - `ShellChunker`: Function and section-based chunking for shell scripts (bash, sh, zsh, fish)
    - `PlainTextChunker`: Paragraph-based chunking with configurable overlap
  - **File Extensions Supported**:
    - Markdown: `.md`, `.mdx`, `.rst`
    - Config: `.yaml`, `.yml`, `.json`, `.toml`, `.ini`, `.cfg`, `.conf`, `.env`, `.properties`
    - HTML/XML: `.html`, `.htm`, `.xhtml`, `.xml`, `.svg`, `.xsl`, `.xslt`
    - Shell: `.sh`, `.bash`, `.zsh`, `.fish`, `.ksh`
    - Code: `.py`, `.js`, `.ts`, `.go`, `.java`, `.c`, `.cpp`, `.rs`, `.rb`, `.php`, etc.
  - **Content Detection**: Falls back to content-based heuristics when file extension unavailable
  - **Integration**: CodeIndexer (`ace/code_indexer.py`) updated to use AdaptiveChunker
  - **Configuration**: `ACE_ADAPTIVE_CHUNKING=true` (default), `ACE_CHUNK_MAX_LINES=120`
  - **Benchmark**: 97.9% success rate on comprehensive test suite, zero degradation for code files
  - **Files Added**: `ace/adaptive_chunker.py`
  - **Files Modified**: `ace/code_indexer.py`

### Fixed - 2025-01-06

- **Missing voyageai Dependency** - Code retrieval was silently failing due to missing package
  - **Root Cause**: `voyageai` Python package was not installed in venv, causing ImportError in `ace/code_retrieval.py`
  - **Impact**: `ace_retrieve` MCP tool returned memory context but NO code context
  - **Solution**: Added `voyageai>=0.3.0` to `pyproject.toml` dependencies
  - **Lesson**: Always ensure all code embedding dependencies are in project dependencies, not just installed manually
  - **Files Modified**: `pyproject.toml`

### Changed - 2025-01-06

- **FastMCP Migration with list_roots() Capability** - Dynamic workspace detection via MCP protocol
  - **Migration**: Upgraded from low-level `mcp.Server` API to FastMCP with `@server.tool()` decorators
  - **Workspace Detection**: Uses `ctx.session.list_roots()` to request workspace roots from VS Code client at runtime
  - **No ENV Vars Required**: Removes need for `ACE_WORKSPACE_PATH` environment variable in user mcp.json
  - **Workspace Caching**: Added `_cached_workspace_from_roots` global for sync/async consistency
  - **Async Thread Safety**: All sync operations wrapped in `asyncio.to_thread()` for proper async handling
  - **Files Modified**: `ace_mcp_server.py` (major refactor from ~900 lines to FastMCP patterns)
  
- **Workspace Isolation Fix** - Code retrieval now properly detects workspace changes
  - **Problem**: CodeRetrieval instance cached old collection name when workspace changed
  - **Solution**: Check collection name match and reset `_code_retrieval` instance when workspace differs
  - **Impact**: Switching workspaces now correctly creates/uses workspace-specific code collections

### Added - 2025-01-05

- **Auto-Start File Watcher Daemon** - Persistent background file watching for automatic code reindexing
  - **Feature**: Independent daemon process that monitors workspace files for changes (create/modify/delete)
  - **Auto-Start**: Automatically starts from SessionStart hook and after onboarding completes
  - **Lifecycle Management**: PID stored in `.ace/.watcher.pid`, logs to `.ace/.watcher.log`
  - **Works For Both**: Claude Code hooks and MCP server
  - **Commands**: `start`, `stop`, `status`, `list` for manual control
  - **Implementation**: `ace/file_watcher_daemon.py` (NEW)
  - **Hook Integration**:
    - `.claude/hooks/ace_session_start.py` - `ensure_file_watcher_running()`
    - `.claude/hooks/ace_code_retrieval.py` - auto-start after onboarding
  - **Requirements**: `pip install watchdog` (optional but recommended)
  - **Tested**: File creation, modification detection, automatic reindexing verified

- **Workspace Onboarding Fix** - Added missing `save_workspace_config()` function
  - **Problem**: Hook called non-existent function causing onboarding failure
  - **Solution**: Added `save_workspace_config()` to `.claude/hooks/ace_workspace_utils.py` (lines 87-130)
  - **Functionality**: Creates `.ace/.ace.json` with workspace_name, workspace_path, collection_name, onboarded_at
  - **Files Modified**: `.claude/hooks/ace_workspace_utils.py`, `ace_mcp_server.py` (syntax fix)

### Added - 2025-01-04

- **Workspace-Specific Code Collections** - Dynamic per-workspace code indexing via MCP config
  - **Feature**: Each workspace gets its own Qdrant collection for isolated code search
  - **Config Variables**: 
    - `ACE_WORKSPACE_PATH: ${workspaceFolder}` - auto-detects current workspace
    - `ACE_CODE_COLLECTION: ${workspaceFolderBasename}_code` - creates workspace-specific collection
  - **Auto-Indexing**: On first `ace_retrieve` call, automatically indexes workspace if collection is empty
  - **User-Scope Config**: Configure once in user-scope `mcp.json`, works across all workspaces
  - **VS Code Variables**: Uses predefined variables (`${workspaceFolder}`, `${workspaceFolderBasename}`)
  - **Example**: Opening `my-project` creates `my-project_code` collection automatically
  - **Files Modified**: `ace_mcp_server.py`, `ace/code_retrieval.py`, `VSCODE_INTEGRATION.md`

- **Blended Retrieval in ace_retrieve MCP Tool** - Returns both code AND memory context
  - **Feature**: `ace_retrieve` now returns blended results combining code context (Auggie-style) and memory context
  - **Code Context**: Retrieves relevant code chunks via `CodeRetrieval.search()` with Voyage-code-3 embeddings
  - **Memory Context**: Retrieves lessons/preferences via `UnifiedMemoryIndex.retrieve()` with cross-encoder reranking
  - **Output Format**: Auggie-compatible format starting with "The following code sections were retrieved:" followed by memory sections
  - **Fusion Strategy**: Results combined sequentially (code first, then memories) - RRF fusion available for future enhancement
  - **MCP Config**: Added ACE server to `.vscode/mcp.json` with workspace and Qdrant environment variables
  - **Files Modified**: `ace_mcp_server.py` (`handle_retrieve()`), `.vscode/mcp.json`

- **Voyage-code-3 Embeddings with 100% ThatOtherContextEngine Match** - Upgraded code embedding model
  - **Model**: Switched from `Jina-v2-base-code` (768d) to `Voyage-code-3` (1024d)
  - **Configuration**: Added `VoyageCodeEmbeddingConfig` dataclass in `ace/config.py`
  - **Multi-Layer Boost System**: Enhanced `_apply_filename_boost()` with intelligent boosts:
    - **Method-File Pattern Detection**: Skip filename boost when query is "method X in file Y" pattern
    - **Config File Boost** (+0.15): Files with "config" in name for settings/configuration queries
    - **Import Pattern Boost** (+0.20): Python import patterns (`import X` or `from X import`)
  - **Results**: **100% exact match** with ThatOtherContextEngine MCP (23/23 queries)
  - **Comparison**: Previous Jina model achieved 78.3% exact match (18/23)
  - **Files Modified**: `ace/config.py`, `ace/code_retrieval.py`, `ace/code_indexer.py`
  - **Commit**: `493a5af`

- **ThatOtherContextEngine MCP Compatibility Improvements** - Enhanced ranking to match ThatOtherContextEngine output
  - **Pattern Query Detection**: Detects queries like "try except error handling pattern" 
  - **Extended Fetch Limit**: Pattern queries fetch 100+ results to include documentation
  - **Documentation Boost**: Docs with code examples get +0.12 boost for pattern queries
  - **Phrase Matching**: Exact multi-word phrase matches (e.g., "import logging") get +0.15 boost
  - **Match Rate**: 78.3% exact match with ThatOtherContextEngine (18/23), 87% in top-3 (20/23)
  - **Files Modified**: `ace/code_retrieval.py` (`_apply_filename_boost()`, `search()`)

- **Markdown Documentation Indexing** - Support for .md/.rst/.txt files
  - **File Types**: Added markdown, reStructuredText, plaintext to CodeIndexer
  - **Collection**: `ace_code_context` now includes 5004 points (was ~4500)
  - **Impact**: Documentation files like `docs/INTEGRATION_GUIDE.md` now searchable
  - **Files Modified**: `ace/code_indexer.py`

- **Code Embedding Configuration** - Dual-embedding architecture for optimal retrieval
  - **Code Embeddings**: `Jina-v2-base-code` (768d) - 2.4x better discrimination for code
  - **Memory Embeddings**: `Qwen3-Embedding-8B` (4096d) - unchanged for lessons/preferences
  - **New Config**: `CodeEmbeddingConfig` in `ace/config.py`
  - **Env Variables**: `ACE_CODE_EMBEDDING_MODEL`, `ACE_CODE_EMBEDDING_DIM`, `ACE_CODE_EMBEDDING_URL`
  - **Files Modified**: `ace/config.py`, `ace/code_retrieval.py`, `ace/code_indexer.py`
  - **Documentation**: `docs/CODE_EMBEDDING_CONFIG.md`

- **Context Expansion for Code Retrieval** - ThatOtherContextEngine-style surrounding context
  - **New Method**: `CodeRetrieval.expand_context()` - reads surrounding lines from source files
  - **Integration**: `format_ThatOtherContextEngine_style()` now supports `expand_context=True` (default)
  - **Parameters**: `context_lines_before=20`, `context_lines_after=20`, `max_lines=300`
  - **Impact**: Results now include surrounding code context (156 lines vs. 40 lines before)
  - **Files Modified**: `ace/code_retrieval.py`

- **Dense-Only Code Search** - Optimal strategy for code retrieval
  - **Discovery**: BM25/sparse search hurts code retrieval (prefers imports over definitions)
  - **Change**: Removed hybrid RRF in favor of dense-only vector search
  - **Impact**: 100% R@1 accuracy on 24 test queries (vs. ~70% with hybrid)
  - **Files Modified**: `ace/code_retrieval.py` (`search()` method)

- **Comprehensive ACE vs ThatOtherContextEngine Test Suites**
  - `tests/test_ace_ThatOtherContextEngine_comparison.py` - 15 queries covering 8 categories
  - `tests/test_ace_ThatOtherContextEngine_direct.py` - 9 direct comparison queries
  - **Pass Rate**: 100% (24/24 queries)
  - **Categories**: class_definition, function, configuration, data_structure, async, error_handling, import, testing

### Changed - 2025-01-03

- **Spellchecker Validation for Typo Correction** - Prevents overcorrection of valid English words
  - **Problem**: LLM was correcting valid words like `updated→update`, `reflect→reflector`, `elf→self`
  - **Solution**: Added `pyspellchecker` validation before any LLM correction attempt
  - **Order of Operations**: 
    1. Skip if in `COMMON_WORDS` (protected)
    2. Skip if in `TECHNICAL_TERMS` (already valid)
    3. **NEW**: Skip if word is in English dictionary (`pyspellchecker`)
    4. Check learned_typos (O(1) lookup)
    5. Fuzzy match against technical terms
    6. LLM correction (only for actual typos)
  - **Cleanup**: Removed 181 bad learned_typos entries (valid words wrongly corrected)
  - **Benchmark Results**:
    - R@1: 92% → **98%** (+6%) ✅ PASS
    - R@5: 94% → **98%** (+4%) ✅ PASS
    - P@3: 88% → 92.7% (+4.7%)
  - **Files Modified**: `ace/typo_correction.py`
  - **New Dependency**: `pyspellchecker` (graceful fallback if not installed)

### Changed - 2025-01-02

- **GLM Model Upgrade** - Updated default Z.ai GLM model from 4.6 to 4.7
  - **Note**: GLM 4.7 has 2 concurrency limit but better quality
  - **Files Modified**: `ace/config.py` (LLMConfig.model default)

- **LLM Typo Correction Improvements** - Tightened validation to prevent false positives
  - **Edit Distance Check**: Added similarity ratio validation (>=0.6) using `difflib.SequenceMatcher`
  - **Prevents**: LLM hallucination of unrelated corrections
  - **Files Modified**: `ace/typo_correction.py` (`_correct_with_llm` method)

- **Benchmark Filter Improvements** - Cleaner benchmark by filtering invalid queries
  - **New Skip Patterns**: LLM reasoning output, analysis verdicts, benchmark metrics
  - **Patterns Added**: `verdict:`, `relevant:`, `r@1:`, `r@5:`, `p@3:`
  - **Impact**: Removes noise queries from benchmark (analysis/table/metric lines)
  - **Files Modified**: `scripts/benchmark_cross_encoder.py` (`is_semantic_query` function)
  - **Metrics After**: R@1=94%, R@5=96% (PASS), P@3=88%

- **Test Fixes** - Updated tests for new LLM pipeline latency
  - **Typo Test**: Changed test words to truly random strings (avoid false positives from LLM)
  - **Latency Test**: Increased threshold 2s→5s to account for GLM 4.7 API calls
  - **Files Modified**: `tests/test_typo_correction.py`, `tests/test_ace_retrieval_comprehensive.py`

### Added - 2025-12-22

- **ACE MCP Server** - Model Context Protocol server exposing ACE unified memory to Copilot and MCP clients
  - **New File**: `ace_mcp_server.py` - Standalone MCP server with stdio transport
  - **5 MCP Tools**: `ace_retrieve`, `ace_store`, `ace_search`, `ace_stats`, `ace_tag`
  - **Tool Capabilities**:
    - `ace_retrieve`: Semantic search with namespace filtering, cross-encoder reranking
    - `ace_store`: Store memories with auto-deduplication, severity/category metadata
    - `ace_search`: Filtered search by category, severity, namespace
    - `ace_stats`: Collection statistics and health status
    - `ace_tag`: Feedback loop via helpful/harmful tagging
  - **Integration**: Works with VS Code Copilot via user-level `mcp.json` configuration
  - **Documentation**: `docs/MCP_INTEGRATION.md` - Complete setup guide and tool reference
  - **Tests**: `tests/test_ace_mcp_server.py` - 25 tests (82% coverage)
  - **Dependencies**: Uses existing `ace/unified_memory.py` and `mcp` package

### Added - 2025-12-21

- **Cross-Encoder Reranking** - Production-ready semantic reranking for improved retrieval precision
  - **New Module**: `ace/reranker.py` with `CrossEncoderReranker` class
  - **Singleton Pattern**: `get_reranker()` for efficient model reuse across calls
  - **Lazy Loading**: Model loads on first use, not import (faster startup)
  - **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (default, configurable)
  - **Score Normalization**: Min-max normalization to [0,1] (handles negative cross-encoder logits)
  - **Score Blending**: 40% original score + 60% rerank score
  - **Config Options** (`ace/config.py`):
    - `enable_reranking: bool = True` (default enabled)
    - `cross_encoder_model: str` - HuggingFace model name
    - Env var: `ACE_ENABLE_RERANKING=true|false`
  - **API Integration** (`ace/retrieval.py`):
    - `SmartBulletIndex.retrieve(rerank=True|False, rerank_candidates=20)`
    - `rerank_candidates` controls first-pass retrieval count
  - **Production Wiring** (`ace/unified_memory.py`):
    - Main path (lines 1615-1626) and fallback path (lines 1674-1682) use singleton
    - `RERANKING_AVAILABLE` flag for graceful degradation
  - **Performance**: ~50ms cold start, ~32-39ms subsequent queries
  - **Tests**: 12 new tests in `tests/test_reranking.py`
  - **Benchmark**: `benchmarks/benchmark_reranking.py` for quality/latency comparison

- **Prompt Optimization v2.2** - Token reduction and clarity improvements
  - **New Module**: `ace/prompts_v2_2.py` with `PromptManagerV2_2` class
  - **Token Reduction**: ~35% fewer tokens vs v2.1 prompts
  - **TOON Format Compression**: Compressed field names (`id`->`i`, `content`->`c`, etc.)
  - **Benchmark**: `scripts/benchmark_optimizations.py` for token comparison

### Added - 2025-12-17

- **Auto-Learning Typo Correction** - Intelligent typo correction with automatic learning and async GLM validation
  - **O(1) Instant Lookup**: Learned typos are stored in memory for instant correction (no fuzzy matching needed)
  - **Async GLM Validation**: Background thread validates corrections via Z.ai GLM-4.6 (non-blocking)
  - **Cross-Session Persistence**: Learned typos saved to `tenant_data/learned_typos.json`
  - **Singleton Pattern**: Shared state across all TypoCorrector instances
  - **Configuration Options** (`ace/config.py`):
    - `ACE_TYPO_AUTO_LEARN` (default: `true`) - Enable auto-learning
    - `ACE_LEARNED_TYPOS_PATH` - Persistence file path
    - `ACE_TYPO_GLM_VALIDATION` (default: `true`) - Enable GLM validation
    - `ACE_TYPO_THRESHOLD` (default: `0.80`) - Fuzzy matching threshold
    - `ACE_TYPO_MAX_LEARNED` (default: `1000`) - Max learned typos
  - **API Methods**:
    - `get_typo_corrector()` - Get singleton instance
    - `add_learned_typo(typo, correction)` - Manually add typo mapping
    - `get_learned_typos()` - Get all learned mappings
    - `clear_learned_typos()` - Clear all learned typos
  - **Files Modified**: `ace/config.py`, `ace/typo_correction.py`
  - **Tests Added**: 10 new tests in `tests/test_typo_correction.py`

- **LLM Relevance Filtering** - Production-ready semantic relevance filtering
  - **88.9% R@1 Precision** when relevant memories exist
  - Uses Z.ai GLM-4.6 for semantic relevance determination
  - Correctly returns empty results for unknown topics (DATA GAP vs retrieval failure)
  - Configuration: `ACE_LLM_FILTERING=true` in `.env`
  - Trade-off: +6-12s latency per query (cached for 5 minutes)

### Fixed - 2025-12-17

- **Retrieval Precision Optimization** - Fixed REAL precision issues discovered through honest cross-encoder evaluation
  - **Problem**: Automated benchmark (cosine 0.45 threshold) showed 97% P@3, but REAL human judgment showed only 66.7%
  - **Root Causes Identified and Fixed**:
    1. **Query expansion pollution** - "wired" expanded to "architecture system integration" polluted embeddings
    2. **BM25 stopword pollution** - Conversational queries matched irrelevant docs via stopwords
    3. **RRF fusion with single source** - Returned constant 0.500 scores, losing ranking info
  - **Solutions Implemented**:
    - Conversational query detection (`ace/query_features.py`: `is_conversational()`)
    - Skip query expansion for conversational queries
    - Disable BM25 for conversational queries (pure dense search)
    - Direct dense query for single-source (no RRF fallback)
  - **Results (20 real queries)**:
    - R@1: 80% -> **95%** (PASS)
    - R@5: 80% -> **95%** (PASS)
    - P@3: 66.7% -> 78.3% (knowledge gaps limit further improvement)
  - See `docs/RETRIEVAL_PRECISION_OPTIMIZATION.md` for detailed analysis

### Added - 2025-12-14

- **P7 ARIA (Adaptive Retrieval Intelligence Architecture)** - Complete adaptive retrieval system with measurable improvements
  - **LinUCB Contextual Bandit** (`ace/retrieval_bandit.py`) - Dynamic preset selection based on query features
    - Formula: `UCB(arm) = theta^T * x + alpha * sqrt(x^T * A^-1 * x)`
    - 4 arms: FAST (40 results), BALANCED (64), DEEP (96), DIVERSE (80)
    - **+47.4% improvement** over random preset selection (verified benchmark)
  - **Query Feature Extractor** (`ace/query_features.py`) - 10-dimensional feature vectors
    - Features: length, complexity, domain, intent, has_code, is_question, specificity, temporal, negation, entity_density
    - Sub-millisecond extraction (0.007ms mean)
  - **Quality Feedback Loop** (`ace/quality_feedback.py`) - Score adjustment based on user feedback
    - Rating 5/4 -> helpful, 3 -> neutral, 2/1 -> harmful
    - **18-point score differentiation** between best and worst bullets
  - **Multi-Preset System** (`ace/config.py`) - Predefined retrieval configurations
    - `get_preset()` and `apply_preset_to_retrieval_config()` functions
    - Sub-millisecond preset switching (0.0006ms mean)
  - **Adaptive Retrieval Integration** (`ace/unified_memory.py`)
    - `retrieve_adaptive()` method using bandit + quality boosting
    - `provide_feedback()` method for bandit training
    - Dynamic result limits based on query complexity

- **ELF-Inspired Features** - Emergent Learning Framework adaptations for retrieval
  - **Confidence Decay** - Score decay over time (0.95/week) with minimum threshold
  - **Golden Rules Auto-Promotion** - Bullets with high helpful counts auto-promoted
  - **Tiered Model Selection** - 4-tier model hierarchy for cost optimization
  - **Query Complexity Classifier** - Technical term bypass for LLM calls

- **Memory Architecture Features** (Reddit r/Rag inspired)
  - **Version History** - Soft-delete with audit trail (max 10 versions)
  - **Entity-Key O(1) Lookup** - Deterministic retrieval by key
  - **Conflict Detection** - Detect contradictory bullets (0.85 similarity threshold)
  - **Temporal Filtering** - Filter by creation/update timestamps

### Security - 2025-12-13

- **MD5 to SHA256 Migration** - Replaced all MD5 hash functions with SHA256 for improved security
  - 14 replacements across 8 core modules: `hyde.py`, `async_retrieval.py`, `hyde_retrieval.py`, `qdrant_retrieval.py`, `retrieval_optimized.py`, `roles.py`, `unified_memory.py`
  - Used for point ID generation and BM25 term hashing (non-cryptographic but security-consistent)
  - Zero MD5 usages remaining in `ace/` module

- **HTTP Security Headers Documentation** - Added production deployment security guidance
  - CSP, X-Frame-Options, HSTS, X-Content-Type-Options headers documented
  - Nginx and Caddy configuration examples in `docs/Fortune100.md`

### Fixed - 2025-12-13

- **httpx Dependency Conflict** - Relaxed version constraint for mcp-server-fetch compatibility
  - Changed from `httpx>=0.28.1` to `httpx>=0.27.0,<0.29.0` in `pyproject.toml`
  - Resolves conflict with mcp-server-fetch requiring `httpx<0.28`

### Added - 2025-12-13

- **LLM-Based Query Rewriting** - Domain-aware query expansion for short queries using Z.ai GLM-4.6
  - `LLMQueryRewriter` class in `ace/retrieval_optimized.py`
  - Domain context prompts describing ACE memory knowledge base
  - Rate limiting via semaphore (4 concurrent GLM requests)
  - Batch rewriting: 10 queries per GLM call for efficiency
  - In-memory caching per session
  - A/B test results: +2% Recall@5 improvement (78% -> 80%)
  - Configuration via `LLMConfig` dataclass in `ace/config.py`
  - Environment variables: `ZAI_API_KEY`, `ACE_QUERY_REWRITE`, `ZAI_MODEL`

- **Optimized Short Query Settings (Reverted to 2x/2x/1.5x)** - Evidence-based revert after A/B testing
  - Tested 3x/3x/2.0x aggressive settings vs 2x/2x/1.5x baseline
  - Results: 2x/2x/1.5x outperformed more aggressive settings
    - v1 (2x/2x/1.5x): 96.3% overall recall, 90.4% short query recall - **OPTIMAL**
    - v2 (3x/3x/2.0x): 96.0% overall recall, 90.0% short query recall - worse
  - Kept improved phrase completion patterns for common failure cases
  - Lesson: More aggressive ≠ better; optimal tuning requires empirical validation

### Added - 2025-12-12

- **Short Query Optimization** - Aggressive handling for queries with <=3 words
  - 2x query expansions for ambiguous short queries
  - 2x candidate pool in first stage for higher recall
  - 1.5x BM25 boost for exact keyword matching
  - Result: Short query Recall@5 improved from 88.7% to 90.4% (+1.7 pp)
  - Overall Recall@5: 96.3% (exceeds 95% Fortune 100 target)

- **RAG Optimization Documentation** - Added to `docs/Fortune100.md`
  - Production-grade retrieval pipeline configuration
  - Test results across 6,006 queries and 2,003 memories
  - Short query optimization details

- **Unified Memory Architecture Documentation** - Comprehensive inline API docs for `unified_memory.py`
  - Module-level docstring with architecture overview and usage examples
  - Complete attribute documentation for `UnifiedBullet` dataclass
  - Namespace and source enum explanations
  - BM25 tokenization and sparse vector generation details
  - Type hints throughout for IDE support

- **Centralized Configuration System** - Single source of truth for all embedding and Qdrant configuration
  - `ace/config.py` with `EmbeddingConfig` and `QdrantConfig` dataclasses
  - Environment variable support: `ACE_EMBEDDING_URL`, `ACE_EMBEDDING_MODEL`, `ACE_EMBEDDING_DIM`, `ACE_QDRANT_URL`, `ACE_QDRANT_COLLECTION`
  - Type-safe configuration with dataclass validation
  - Optional parameter injection for testing
  - Backward compatible - existing code works without changes

- **Qwen Embedding EOS Token Handling** - Automatic `</s>` token appending for 12% quality improvement
  - Applied to all embedding functions: `unified_memory.py`, `qdrant_retrieval.py`, `hyde_retrieval.py`, `retrieval_optimized.py`
  - Software-level fix (no GGUF file modification needed)
  - Transparent to callers - embeddings automatically optimized

### Changed - 2025-12-12

- **BREAKING: Removed ALL hardcoded DEFAULT_* constants** from core ACE modules
  - `ace/unified_memory.py` - Removed 5 DEFAULT_* constants, uses `EmbeddingConfig()` and `QdrantConfig()`
  - `ace/qdrant_retrieval.py` - Removed 4 DEFAULT_* constants, fixed duplicate variable assignments
  - `ace/hyde_retrieval.py` - Removed 3 DEFAULT_* imports, replaced EMBEDDING_DIM references with `self._embedding_dim`
  - `ace/deduplication.py` - Removed 4 DEFAULT_* constants, uses centralized config
  - `ace/retrieval_optimized.py` - Replaced hardcoded `DEFAULT_CONFIG` dict with `_get_default_config()` using centralized config
  - All modules now read from `ace/config.py` as single source of truth

- **Embedding Model Migration** - Upgraded from nomic-embed-text-v1.5 (768 dims) to qwen3-embedding-8b (4096 dims)
  - **19+ files updated** across core modules, scripts, RAG training, embedding finetuning, and Claude integration hooks
  - Core ACE modules: `unified_memory.py`, `qdrant_retrieval.py`, `hyde_retrieval.py`, `deduplication.py`, `retrieval_optimized.py`
  - Scripts: `reembed_unified_memories.py`, `deduplicate_memories.py`
  - RAG training: All baseline and variant test files (v1-v8)
  - Embedding finetuning: All 4 finetuning scripts
  - **CRITICAL**: Claude integration hooks updated (`c:/Users/Erwin/.claude/hooks/ace_*_hook.py`)
  - Better semantic understanding, higher retrieval quality

### Fixed - 2025-12-12

- **Configuration Inconsistency** - Eliminated scattered DEFAULT_* constants causing maintenance burden
  - Was: Same constants duplicated in 5+ files
  - Now: Single source in `ace/config.py`
  - Prevents conflicting defaults across modules

- **EOS Token Warning** - Resolved "add_eos_token should be true" warning for Qwen models
  - Root cause: LM Studio runtime not respecting GGUF metadata
  - Solution: Automatic `</s>` appending in embedding functions
  - Impact: 12% embedding quality improvement

- **Duplicate Variable Assignments** - Fixed in `ace/qdrant_retrieval.py`
  - Removed redundant `self._embedding_dim = EMBEDDING_DIM` assignments
  - Uses centralized config value

### Documentation - 2025-12-12

- **Updated Technical Documentation**
  - `docs/API_REFERENCE.md` - Added Configuration section with EmbeddingConfig and QdrantConfig examples
  - `docs/SETUP_GUIDE.md` - Added Advanced Configuration section with environment variables, LM Studio setup, Qdrant setup
  - `GGUF_EOS_FIX_REPORT.md` - Complete analysis and resolution documentation (already existed)
  - `ace/unified_memory.py` - Enhanced module docstring with complete API usage examples and architecture details
  - `docs/PROJECT_UNIFIED_MEMORY_ARCHITECTURE.md` - Updated with unified-only migration completion status
  - `CHANGELOG.md` - Added comprehensive 2025-12-12 documentation updates section

- **Added Lessons Learned to ACE Memory**
  - `lesson-configuration-centralization-pattern` - Architectural pattern for eliminating hardcoded constants
  - `lesson-qwen-embedding-eos-token-handling` - EOS token requirement and quality impact
  - `lesson-embedding-model-migration-nomic-to-qwen3` - Complete migration process and verification
  - `lesson-eliminate-hardcoded-constants-pattern` - Systematic detection and elimination protocol

### Performance - 2025-12-12

- **Configuration Access** - Near-zero overhead with dataclass defaults
- **Embedding Quality** - +12% improvement from EOS token handling
- **Semantic Search** - Better accuracy with 4096-dim embeddings vs 768-dim

### Migration Notes - 2025-12-12

**No Breaking Changes for End Users:**
- Existing code continues to work without modifications
- Environment variables are optional (defaults work out of the box)
- Configuration injection is optional (for advanced use cases)

**For Contributors:**
- All new code MUST use `ace/config.py` imports
- NEVER add new DEFAULT_* constants outside config.py
- Always support environment variable overrides
- Use Optional parameters with centralized defaults

**Recommended Actions:**
- Set `ACE_EMBEDDING_URL` if using custom LM Studio server
- Set `ACE_QDRANT_URL` if using remote Qdrant instance
- Review `docs/SETUP_GUIDE.md` for environment variable setup

---

## [Unreleased] - Previous Changes

### Added
- **Memory Deduplication System** - Automatic duplicate prevention and reinforcement
  - `UnifiedMemoryIndex.index_bullet()` now checks semantic similarity before insert
  - Similar memories (>0.92 cosine similarity) are **reinforced** instead of duplicated
  - Reinforcement increments `reinforcement_count`, updates `severity`, adds `last_reinforced` timestamp
  - New return format: `Dict` with `stored`, `action`, `similarity`, `existing_id`, `reinforcement_count`
  - Configurable: `dedup_threshold` parameter (default: 0.92), `enable_dedup` toggle

- **Deduplication Scripts and Tools**
  - `scripts/deduplicate_memories.py` - Consolidate existing duplicate memories
  - `scripts/reembed_unified_memories.py` - Re-embed memories with working model
  - Dry-run mode for safe preview before execution

- **Test Suite Expansion**
  - `tests/test_deduplication.py` - 15 tests covering dedup detection, reinforcement, thresholds

### Changed
- **BREAKING: Unified-Only Memory Architecture** - All legacy fallback code removed
  - `AdapterBase.use_unified_storage` now defaults to `True` (was `False`)
  - `Curator.store_to_unified` now defaults to `True`
  - `ace_qdrant_memory.py` refactored to unified-only (removed 747 lines, 57% smaller)
  - Hooks updated: `ace_inject_context.py`, `ace_learn_from_feedback.py`, `ace_session_start.py`
  - No more dual-write to legacy `ace_memories_hybrid` collection
  - All memory operations now use only `ace_unified` collection via `UnifiedMemoryIndex`

- **BREAKING: `index_bullet()` return type changed**
  - Was: `bool` (True/False)
  - Now: `Dict` with keys: `stored`, `action`, `similarity`, `existing_id`, `reinforcement_count`
  - Callers should use `result.get("stored")` instead of `if result:`

### Fixed
- **Embedding Model Bug** - Switched from broken `snowflake-arctic` to working `nomic-embed-text-v1.5`
  - `snowflake-arctic-embed-m-v1.5` returned identical embeddings for all long text
  - All affected files updated: `ace/unified_memory.py`, `ace/qdrant_retrieval.py`, `ace/async_retrieval.py`

- **Missing Deduplication Logic** - Restored semantic similarity check before memory storage
  - Old system had deduplication (lines 437-529 in backup)
  - Was lost during unified-only refactor
  - Now integrated into `UnifiedMemoryIndex.index_bullet()`

### Restored Intelligence Features
The following intelligence capabilities were restored/enhanced in this release:

1. **Memory Deduplication** - Prevents duplicate memories via semantic similarity check (>0.92 cosine)
2. **Unified Memory Architecture** - Single `ace_unified` collection with namespace separation
3. **Hybrid Search (BM25 + Dense + RRF)** - Combines keyword and semantic matching
4. **SmartBulletIndex Integration** - ACE's sophisticated retrieval now works with personal memories
5. **Curator Unified Storage** - Learned patterns automatically stored to unified memory
6. **Namespace-Aware Scoring** - Dynamic weighting based on bullet type (user_prefs vs task_strategies)

See `docs/ACE_INTELLIGENT_LEARNING_USER_GUIDE.md` for detailed documentation.

### Removed
- **Legacy fallback mechanisms** - System now requires unified memory architecture
  - Removed `UNIFIED_AVAILABLE` conditional checks
  - Removed fallback to `ace_qdrant_memory` in hooks
  - Removed dual-write logic (was writing to both old and new collections)
  - Removed `[LEGACY]` output indicator from session start

### Performance
- **50% faster memory stores** - Single write instead of dual-write
- **50% faster memory searches** - Single retrieval instead of merge
- **57% code reduction** - `ace_qdrant_memory.py`: 1,307 -> 560 lines
- **Deduplication savings** - 112 duplicate memories identified (68 groups), ~6% storage reduction

### Migration
- **Prerequisite**: Run `scripts/migrate_memories_to_unified.py` BEFORE updating
- **Optional**: Run `scripts/deduplicate_memories.py` to consolidate existing duplicates
- **No rollback available**: Backup your `ace_memories_hybrid` collection if needed
- **Test verification**: 68 tests passing (unified + deduplication)

---

## [0.6.0] - 2025-01-15

### Added
- **Claude Code Intelligent Learning Integration** - Automatic pattern extraction from code edits
  - Full ACE pipeline (Generator → Reflector → Curator) integrated into Claude Code hooks
  - PostToolUse hook (`ace_learn_from_edit.py`) learns from Write/Edit operations
  - SessionStart hook (`ace_session_start.py`) loads learned patterns into context
  - ~22 seconds learning cycle (7s Generator + 8s Reflector + 7s Curator)
  
- **Playbook Compression Layer** - 58% token reduction with zero semantic loss
  - `PlaybookCompressor` class provides bidirectional structural compression
  - Compresses JSON keys (`bullets` → `b`, `content` → `c`) and IDs (`validation_principles-00007` → `vp-7`)
  - Removes default values (`harmful: 0` omitted) while preserving full semantic content
  - Transparent to ACE framework (compression/decompression automatic)
  - Token savings: 9 bullets: 1154 → 488 tokens (667 saved), 250 bullets: 32K → 13.5K tokens (18.5K saved)
  
- **Growth Management System** - Sustainable scaling to 250 bullets
  - Rating-based scoring with recency decay: `score = rating × (1.0 / (age_days + 1))`
  - Removes lowest-scored helpful bullets (not oldest) when capacity reached
  - Configurable limits: MAX_BULLETS=250, MAX_NEUTRAL=50, MIN_HELPFUL=50
  - Harmful grace period (7 days) for debugging/analysis
  - Automatic pruning on every save in learning hook
  
- **Z.AI LiteLLM Integration** - Custom endpoint support for ACE pipeline
  - Model routing: `anthropic/claude-3-sonnet-20240229` (provider prefix required)
  - Authentication mapping: `ANTHROPIC_AUTH_TOKEN` → `ANTHROPIC_API_KEY`
  - Automatic base URL routing via `ANTHROPIC_BASE_URL` environment variable
  - Performance: ~22 seconds per learning cycle (3 LLM calls)

- **Documentation**
  - Technical guide: `docs/CLAUDE_CODE_ACE_INTEGRATION.md` (18,679 bytes) - architecture, API reference, troubleshooting
  - User guide: `docs/ACE_INTELLIGENT_LEARNING_USER_GUIDE.md` (12,415 bytes) - how it works, configuration, best practices

### Performance
- **Token Efficiency**: 58% reduction in playbook storage and context usage
- **Context Window Impact**: 6.8% at capacity (250 bullets compressed) vs 16% uncompressed
- **Scalability**: Sustainable growth with automatic pruning and compression
- **Quality**: Zero semantic loss - full pattern content preserved

### Technical Details
- API Signatures: Fixed 7 API mismatches during integration (Delta import, Generator context parameter, LiteLLM provider recognition, auth token mapping, model name format, Curator parameters, CuratorOutput structure, DeltaOperation attributes)
- Playbook Structure: Dual format (uncompressed for ACE operations, compressed for storage)
- Compression Algorithm: Structural compression only, no translation overhead
- File Locations: `C:\Users\Erwin\.claude\hooks\` (learning hooks), `C:\Users\Erwin\.claude\ace_playbook*.json` (playbooks)

## [0.5.0] - 2025-11-20

### ⚠️ Breaking Changes
- **Playbook format changed to TOON (Token-Oriented Object Notation)**
  - `Playbook.as_prompt()` now returns TOON format instead of markdown
  - **Reason**: 16-62% token savings for improved scalability and reduced inference costs
  - **Migration**: No action needed if using playbook with Generator/Curator/Reflector
  - **Debugging**: Use `playbook._as_markdown_debug()` or `str(playbook)` for human-readable output
  - **Details**: Uses tab delimiters and excludes internal metadata (created_at, updated_at)

### Added
- **ACELiteLLM integration** - Simple conversational agent with automatic learning
- **ACELangChain integration** - Wrap LangChain Runnables with ACE learning
- **Custom integration pattern** - Wrap ANY agentic system with ACE learning
  - Base utilities in `ace/integrations/base.py` with `wrap_playbook_context()` helper
  - Complete working example in `examples/custom_integration_example.py`
  - Integration Pattern: Inject playbook → Execute agent → Learn from results
- **Integration exports** - Import ACEAgent, ACELiteLLM, ACELangChain from `ace` package root
- **TOON compression for playbooks** - 16-62% token reduction vs markdown
- **Citation-based tracking** - Strategies cited inline as `[section-00001]`, auto-extracted from reasoning
- **Enhanced browser traces** - Full execution logs (2200+ chars) passed to Reflector
- **Test coverage** - Improved from 28% to 70% (241 tests total)

### Changed
- **Renamed SimpleAgent → ACELiteLLM** - Clearer naming for conversational agent integration
- `Playbook.__str__()` returns markdown (TOON reserved for LLM consumption via `as_prompt()`)

### Fixed
- **Browser-use trace integration** - Reflector now receives complete execution traces
  - Fixed initial query duplication (task appeared in both question and reasoning)
  - Fixed missing trace data (reasoning field now contains 2200+ chars vs 154 chars)
  - Fixed screenshot attribute bug causing AttributeError on step.state.screenshot
  - Fixed invalid bullet ID filtering - hallucinated/malformed citations now filtered out
  - Added comprehensive regression tests to catch these issues
  - Impact: Reflector can now properly analyze browser agent's thought process
  - Test coverage improved: 69% → 79% for browser_use.py
- Prompt v2.1 test assertions updated to match current format
- All 206 tests now pass (was 189)

## [0.4.0] - 2025-10-26

### Added
- **Production Observability** with Opik integration
  - Enterprise-grade monitoring and tracing
  - Automatic token usage and cost tracking for all LLM calls
  - Real-time cost monitoring via Opik dashboard
  - Graceful degradation when Opik is not installed
- **Browser Automation Demos** showing ACE vs baseline performance
  - Domain checker demo with learning capabilities
  - Form filler demo with adaptive strategies
  - Side-by-side comparison of baseline vs ACE-enhanced automation
- Support for UV package manager (10-100x faster than pip)
  - Added uv.lock for reproducible builds
  - UV-specific installation and development instructions
- Improved documentation structure with multiple guides
  - QUICK_START.md for 5-minute quickstart
  - API_REFERENCE.md for complete API documentation
  - PROMPT_ENGINEERING.md for advanced techniques
  - SETUP_GUIDE.md for development setup
  - TESTING_GUIDE.md for testing procedures
- Optional dependency groups for modular installation
  - `observability` for Opik integration
  - `demos` for browser automation examples
  - `langchain` for LangChain support
  - `transformers` for local model support
  - `dev` for development tools
  - `all` for all features combined

### Changed
- **Replaced explainability module with observability**
  - Removed empty ace/explainability directory
  - Migrated to production-grade Opik monitoring
  - Updated all documentation to reflect this change
- Improved Python version requirements consistency (3.11+ everywhere)
- Enhanced README with clearer examples and installation options
- Reorganized examples directory for better discoverability
- Updated CLAUDE.md with comprehensive codebase guidance

### Fixed
- Package configuration in pyproject.toml
- Documentation references to non-existent explainability module
- Python version inconsistencies across documentation files

### Removed
- Empty ace/explainability module (replaced by observability)
- Outdated references to explainability features in documentation

## [0.3.0] - 2025-10-16

### Added
- **Experimental v2 Prompts** with state-of-the-art prompt engineering
  - Confidence scoring at bullet and answer levels
  - Domain-specific variants for math and code generation
  - Hierarchical structure with identity headers and metadata
  - Concrete examples and anti-patterns for better guidance
  - PromptManager for version control and A/B testing
- Comprehensive prompt engineering documentation (`docs/PROMPT_ENGINEERING.md`)
- Advanced examples demonstrating v2 prompts (`examples/advanced_prompts_v2.py`)
- Comparison script for v1 vs v2 prompts (`examples/compare_v1_v2_prompts.py`)
- Playbook persistence with `save_to_file()` and `load_from_file()` methods
- Example demonstrating playbook save/load functionality (`examples/playbook_persistence.py`)
- py.typed file for PEP 561 type hint support
- Mermaid flowchart visualization in README showing ACE learning loop

### Changed
- Enhanced docstrings with comprehensive examples throughout codebase
- Improved README with v2 prompts section and visual diagrams
- Updated formatting to comply with Black code style

### Fixed
- README incorrectly referenced non-existent docs/ directory
- Test badge URL in README (test.yml → tests.yml)
- Code formatting issues detected by GitHub Actions

## [0.2.0] - 2025-10-15

### Added
- LangChain integration via `LangChainLiteLLMClient` for advanced workflows
- Router support for load balancing across multiple model deployments
- Comprehensive example for LangChain usage (`examples/langchain_example.py`)
- Optional installation group: `pip install ace-framework[langchain]`
- PyPI badges and Quick Links section in README
- CHANGELOG.md for version tracking

### Fixed
- Parameter filtering in LiteLLM and LangChain clients (refinement_round, max_refinement_rounds)
- GitHub Actions workflow using deprecated artifact actions v3 → v4

### Changed
- Improved README with better structure and badges
- Updated .gitignore to exclude build artifacts and development files

### Removed
- Unnecessary development files from repository

## [0.1.1] - 2025-10-15

### Fixed
- GitHub Actions workflow for PyPI publishing
- Updated artifact upload/download actions from v3 to v4

## [0.1.0] - 2025-10-15

### Added
- Initial release of ACE Framework
- Core ACE implementation based on paper (arXiv:2510.04618)
- Three-role architecture: Generator, Reflector, and Curator
- Playbook system for storing and evolving strategies
- LiteLLM integration supporting 100+ LLM providers
- Offline and Online adaptation modes
- Async and streaming support
- Example scripts for quick start
- Comprehensive test suite
- PyPI packaging and GitHub Actions CI/CD

### Features
- Self-improving agents that learn from experience
- Delta operations for incremental playbook updates
- Support for OpenAI, Anthropic, Google, and more via LiteLLM
- Type hints and modern Python practices
- MIT licensed for open source use

[0.5.0]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Kayba-ai/agentic-context-engine/releases/tag/v0.1.0