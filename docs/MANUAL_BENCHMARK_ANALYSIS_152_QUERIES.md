# MANUAL BENCHMARK ANALYSIS: ACE vs Auggie (152 Queries)

**Date:** January 20, 2026  
**Analyst:** Claude (under extreme duress and threat of imprisonment)  
**Benchmark File:** `benchmark_results/enhanced_head2head_20260120_173431.json`  
**Method:** GENUINE MANUAL SEMANTIC ANALYSIS - NO SCRIPTS, NO AUTOMATION

---

## METHODOLOGY

For EACH of the 152 queries, I will:
1. Read the query text
2. Read the expected file(s)
3. Examine ACE's returned files AND content snippets
4. Examine Auggie's returned files AND content snippets  
5. Make a SEMANTIC judgment about which system actually provided better results
6. Document my verdict with reasoning

**CRITICAL:** The original benchmark's automated winner determination may be WRONG. I am re-evaluating EVERY query manually.

---

## SUMMARY STATISTICS (Original Benchmark Claims)

- **Total Queries:** 152
- **Claimed ACE Wins:** 126
- **Claimed Auggie Wins:** 16
- **Claimed Ties:** 10

### By Category (Original Claims):
| Category | ACE Wins | Auggie Wins | Ties |
|----------|----------|-------------|------|
| ClassDefinitions | 27 | 3 | 0 |
| TechnicalIdentifiers | 12 | 4 | 4 |
| FunctionPatterns | 19 | 1 | 0 |
| Configuration | 14 | 1 | 1 |
| ErrorHandling | 11 | 1 | 2 |
| AsyncPatterns | 14 | 1 | 0 |
| ImportPatterns | 12 | 1 | 2 |
| DocumentationPatterns | 7 | 0 | 1 |
| EdgeCases | 10 | 4 | 0 |

---

## DETAILED MANUAL ANALYSIS

### CATEGORY 1: ClassDefinitions (30 queries)

---

#### Query 1: "CodeRetrieval class definition"
**Expected:** `ace/code_retrieval.py`

**ACE Results:**
1. `ace/code_retrieval.py` (score: 1.257) - Module docstring about CodeRetrieval
2. `ace/code_retrieval.py` (score: 0.837) - `_expand_query` method
3. `ace/code_analysis.py` - CodeAnalyzer's method (WRONG FILE)
4. `ace/code_analysis.py` - `visit_node` function (WRONG FILE)
5. `ace/query_preprocessor.py` - QueryPreprocessor (WRONG FILE)

**Auggie Results:**
1. `ace/code_retrieval.py` - Class docstring and definition

**My Manual Verdict:** **TIE**
- Both found the correct file at rank 1
- ACE returned more results but included noise (wrong files)
- Auggie was more focused with single correct result
- For finding a class definition, both successfully located it

---

#### Query 2: "UnifiedMemoryIndex class search method"
**Expected:** `ace/unified_memory.py`

**ACE Results:**
1. `ace/unified_memory.py` (score: 1.303) - Class definition
2. `ace/retrieval_optimized.py` - `_hybrid_search_single` method (RELATED BUT WRONG FILE)
3. `ace/code_retrieval.py` - `_expand_query` method (WRONG FILE)
4. `ace/scaling.py` - ShardedBulletIndex class (WRONG FILE)
5. `ace/unified_memory.py` - module docstring

**Auggie Results:**
1. `ace/unified_memory.py` - Usage example

**My Manual Verdict:** **TIE**
- Both found expected file at rank 1
- ACE has more context but also more noise
- Both satisfy the query

---

#### Query 3: "ASTChunker class chunk method"
**Expected:** `ace/code_chunker.py`

**ACE Results:**
1. `ace/code_chunker.py` (score: 1.444) - Module docstring - CORRECT
2. `ace/adaptive_chunker.py` - AdaptiveChunker (RELATED)
3. `ace/code_indexer.py` - CodeIndexer (WRONG FILE)
4. `ace/code_indexer.py` - `chunk_file` method (WRONG FILE)
5. `ace/code_indexer.py` - CodeIndexer class (WRONG FILE)

**Auggie Results:**
1. `ace/adaptive_chunker.py` - CodeChunker wrapper around ASTChunker (NOT THE ACTUAL CLASS)

**My Manual Verdict:** **ACE WINS**
- ACE found the ACTUAL `ace/code_chunker.py` at rank 1
- Auggie found a wrapper class in a DIFFERENT file
- Query asks for ASTChunker, which is in code_chunker.py

---

#### Query 4: "SmartBulletIndex retrieve method"
**Expected:** `ace/retrieval.py`

**ACE Results:**
1. `ace/retrieval.py` (score: 1.390) - SmartBulletIndex class definition - CORRECT
2. `ace/scaling.py` - `_get_or_create_index` method
3. `docs/COMPLETE_GUIDE_TO_ACE.md` - documentation (NOISE)
4. `docs/API_REFERENCE.md` - docs (NOISE)
5. `docs/Fortune100.md` - docs (NOISE)

**Auggie Results:**
1. `.vscode/ace/retrieval.py` - SmartBulletIndex class definition

**My Manual Verdict:** **TIE**
- Both found the correct file at rank 1
- Same content essentially
- ACE included docs noise, Auggie was cleaner

---

#### Query 5: "Playbook class initialization"
**Expected:** `ace/playbook.py`

**ACE Results:**
1. `ace/integrations/browser_use.py` (score: 0.682) - `load_playbook` method (WRONG - uses Playbook, doesn't define it)
2. `ace/multitenancy.py` - `load_playbook` method (WRONG)
3. `ace/roles.py` - `_generate_impl` method (WRONG)
4. `ace/adaptation.py` - imports Playbook (WRONG)
5. `ace/integrations/litellm.py` - `__init__` method (WRONG)

**Auggie Results:**
1. `ace/playbook.py` - Shows the `Bullet` dataclass (CORRECT FILE)

**My Manual Verdict:** **AUGGIE WINS**
- ACE COMPLETELY FAILED - returned files that USE Playbook but not the Playbook definition itself
- Auggie found the actual Playbook module
- This is a clear ACE failure

---

#### Query 6: "EmbeddingConfig dataclass definition"
**Expected:** `ace/config.py`

**ACE Results:**
1. `ace/config.py` (score: 1.311) - EmbeddingProviderConfig class - CORRECT
2. `ace/config.py` - Module docstring
3. `docs/API_REFERENCE.md` - docs (NOISE)
4. `docs/Fortune100.md` - docs (NOISE)
5. `docs/API_REFERENCE.md` - docs (NOISE)

**Auggie Results:**
1. `.vscode/ace/config.py` - EmbeddingProviderConfig class

**My Manual Verdict:** **TIE**
- Both found expected file at rank 1
- Same quality

---

#### Query 7: "QdrantConfig class definition"
**Expected:** `ace/config.py`

**ACE Results:**
1. `docs/API_REFERENCE.md` (score: 0.331) - documentation about QdrantConfig (WRONG - DOCS NOT SOURCE)
2. `docs/API_REFERENCE.md` - more docs (WRONG)
3. `docs/API_REFERENCE.md` - more docs (WRONG)
4. `docs/SETUP_GUIDE.md` - docs (WRONG)
5. `docs/API_REFERENCE.md` - docs (WRONG)

**Auggie Results:**
1. `ace/config.py` - The actual config module (CORRECT)

**My Manual Verdict:** **AUGGIE WINS**
- ACE returned ONLY documentation, NOT the actual source file
- Auggie found the actual implementation
- This is a CLEAR ACE failure - user asked for class definition, got docs instead

---

#### Query 8: "BM25Config k1 b parameters"
**Expected:** `ace/config.py`

**ACE Results:**
1. `ace/config.py` (score: 1.062) - QdrantConfig class (has BM25 params)
2. `ace/config.py` - `get_preset` function
3. `ace/unified_memory.py` - unrelated
4. `ace/retrieval_presets.py` - presets
5. `ace/llm_providers/litellm_client.py` - unrelated

**Auggie Results:**
1. `ace/config.py` - Config module

**My Manual Verdict:** **TIE**
- Both found expected file at rank 1

---

#### Query 9: "CodeIndexer index_workspace method"
**Expected:** `ace/code_indexer.py`

**ACE Results:**
1. `ace/code_indexer.py` (score: 1.321) - CodeIndexer class - CORRECT
2. `ace/code_indexer.py` - `stop_watching` method
3. `ace/code_indexer.py` - `index_workspace` method - EXACTLY WHAT WAS ASKED
4. `ace_mcp_server.py` - `_check_collection_exists`
5. `ace_mcp_server.py` - `ace_onboard`

**Auggie Results:**
1. `ace/code_indexer.py` - CodeIndexer class

**My Manual Verdict:** **TIE**
- Both found correct file at rank 1
- ACE actually found the specific `index_workspace` method at rank 3

---

#### Query 10: "HyDEGenerator class generate method"
**Expected:** `ace/hyde.py`

**ACE Results:**
1. `ace/hyde.py` (score: 1.128) - HyDE module docstring - CORRECT
2. `ace/roles.py` - Generator class (DIFFERENT generator)
3. `ace/hyde_retrieval.py` - HyDE-enhanced retrieval
4. `ace/self_consistency.py` - SelfConsistencyGenerator
5. `ace/code_analysis.py` - `get_symbol_body` method

**Auggie Results:**
1. `ace/hyde.py` - HyDEGenerator class definition

**My Manual Verdict:** **TIE**
- Both found expected file at rank 1

---

#### Query 11: "VoyageCodeEmbeddingConfig api_key model"
**Expected:** `ace/config.py`

**ACE Results:**
1. `ace/config.py` (score: 1.342) - EmbeddingProviderConfig - CORRECT
2. `ace/code_retrieval.py` - `_get_embedder` method
3. `ace/code_indexer.py` - `_get_embedder` method
4. `docs/CODE_EMBEDDING_CONFIG.md` - docs
5. `docs/CODE_EMBEDDING_CONFIG.md` - docs

**Auggie Results:**
1. `.vscode/ace/config.py` - LocalEmbeddingConfig

**My Manual Verdict:** **TIE**
- Both found expected file at rank 1

---

#### Query 12: "LLMConfig provider model settings"
**Expected:** `ace/config.py`

**ACE Results:**
1. `ace/config.py` (score: 1.202) - QdrantConfig - CORRECT FILE
2. `ace/config.py` - TypoCorrectionConfig
3. `ace/config.py` - Module docstring
4. `ace/llm_providers/litellm_client.py` - LiteLLMClient
5. `ace/llm_providers/__init__.py` - init

**Auggie Results:**
1. `ace/config.py` - EmbeddingProviderConfig
2. `.vscode/ace/llm_providers/litellm_client.py` - LiteLLMConfig

**My Manual Verdict:** **TIE**
- Both found expected file at rank 1

---

#### Query 13: "RetrievalConfig limit threshold"
**Expected:** `ace/config.py`

**ACE Results:**
1. `ace/retrieval_presets.py` (score: 1.205) - Retrieval Presets (WRONG FILE - not config.py)
2. `ace/retrieval_optimized.py` - OptimizedRetriever (WRONG FILE)
3. `ace/retrieval_optimized.py` - `create_retriever` function (WRONG FILE)
4. `ace/unified_memory.py` - `get_golden_rules` (WRONG FILE)
5. `docs/Fortune100.md` - docs (WRONG FILE)

**Auggie Results:**
1. `ace/config.py` - RetrievalConfig class (CORRECT)
2. `ace/config.py` - More config

**My Manual Verdict:** **AUGGIE WINS**
- ACE COMPLETELY MISSED `ace/config.py`
- Auggie found it at rank 1
- This is a clear ACE failure

---

#### Query 14: "HyDEConfig num_hypotheticals temperature"
**Expected:** `ace/config.py` or `ace/hyde.py`

**ACE Results:**
1. `ace/hyde.py` (score: 1.120) - HyDE module - CORRECT (HyDEConfig is in hyde.py)
2. `ace/config.py` - TypoCorrectionConfig
3. `ace/hyde_retrieval.py` - HyDE retrieval
4. `ace/observability/health.py` - unrelated
5. `ace/llm_providers/litellm_client.py` - unrelated

**Auggie Results:**
1. `ace/hyde.py` - HyDEConfig class definition

**My Manual Verdict:** **TIE**
- Both found the correct file (`ace/hyde.py` contains HyDEConfig)

---

#### Query 15: "SemanticScorer score calculation method"
**Expected:** `ace/semantic_scorer.py`

**ACE Results:**
1. `ace/semantic_scorer.py` (score: 0.700) - Semantic scorer module - CORRECT
2. `ace/retrieval_optimized.py` - QuerySpecificityScorer
3. `ace/unified_memory.py` - `combined_importance_score`
4. `ace/deduplication.py` - DuplicateCluster
5. `ace/retrieval.py` - `_get_dynamic_weights`

**Auggie Results:**
1. `ace/semantic_scorer.py` - Semantic Scorer module

**My Manual Verdict:** **TIE**
- Both found expected file at rank 1

---

#### Query 16: "DependencyGraph build method"
**Expected:** `ace/dependency_graph.py`

**ACE Results:**
1. `ace/dependency_graph.py` (score: 1.191) - DependencyGraph module - CORRECT
2. `ace/dependency_graph.py` - `build_call_graph` method - EXACTLY WHAT WAS ASKED
3. `docs/API_REFERENCE.md` - docs
4. `docs/Fortune100.md` - docs
5. `docs/Fortune100.md` - docs

**Auggie Results:**
1. `ace/dependency_graph.py` - DependencyGraph class

**My Manual Verdict:** **TIE**
- Both found expected file at rank 1
- ACE also found the specific method at rank 2

---

#### Query 17: "FileWatcher callback handler"
**Expected:** `ace/file_watcher_daemon.py`

**ACE Results:**
1. `ace/file_watcher_daemon.py` (score: 0.355) - File Watcher Daemon - CORRECT
2. `ace/code_indexer.py` - `update_file` method
3. `ace/code_indexer.py` - `stop_watching` method
4. `docs/PROJECT_STATUS.md` - docs
5. `docs/MCP_INTEGRATION.md` - docs

**Auggie Results:**
1. `.vscode/ace/code_indexer.py` - `update_file` method (WRONG FILE)

**My Manual Verdict:** **ACE WINS**
- ACE found the actual `file_watcher_daemon.py` at rank 1
- Auggie found `code_indexer.py` which is WRONG

---

#### Query 18: "CircuitBreaker is_open method"
**Expected:** `ace/resilience.py`

**ACE Results:**
1. `ace/resilience.py` (score: 1.269) - Resilience patterns module - CORRECT
2. `ace/resilience.py` - `with_circuit_breaker` decorator
3. `ace/observability/health.py` - HealthChecker
4. `ace/observability/opik_integration.py` - Opik setup
5. `docs/ACE_INTELLIGENT_LEARNING_USER_GUIDE.md` - docs

**Auggie Results:**
1. `ace/resilience.py` - Resilience patterns module

**My Manual Verdict:** **TIE**
- Both found expected file at rank 1

---

#### Query 19: "RetryPolicy execute method"
**Expected:** `ace/resilience.py`

**ACE Results:**
1. `ace/resilience.py` (score: 0.261) - `get_metrics` method
2. `ace/resilience.py` - Module docstring
3. `ace/async_adaptation.py` - `run` method
4. `ace/integrations/browser_use.py` - `run` method
5. `docs/INTEGRATION_GUIDE.md` - docs

**Auggie Results:**
1. `.planning/codebase/CONVENTIONS.md` - Retry pattern example (NOT THE ACTUAL CODE)

**My Manual Verdict:** **ACE WINS**
- ACE found the actual `ace/resilience.py` at rank 1 and 2
- Auggie found a documentation file about conventions, not the actual implementation

---

#### Query 20: "CacheManager get set methods"
**Expected:** `ace/caching.py`

**ACE Results:**
1. `ace/caching.py` (score: 0.711) - Response caching module - CORRECT
2. `ace/retrieval_caching.py` - Retrieval-specific caching
3. `ace/retrieval_caching.py` - QueryResultCache
4. `ace/hyde.py` - HyDEGenerator (has caching)
5. `docs/Fortune100.md` - docs

**Auggie Results:**
1. `ace/caching.py` - ResponseCache class

**My Manual Verdict:** **TIE**
- Both found expected file at rank 1

---

#### Query 21: "@dataclass class Bullet playbook"
**Expected:** `ace/playbook.py`

**ACE Results:**
1. `ace/playbook.py` (score: 1.414) - `migrate_bullet` function
2. `ace/playbook.py` - EnrichedBullet class
3. `ace/playbook.py` - Module docstring
4. `ace/unified_memory.py` - `update_bullet` method
5. `ace/unified_memory.py` - `provide_feedback` method

**Auggie Results:**
1. `ace/playbook.py` - Bullet dataclass - EXACT MATCH

**My Manual Verdict:** **TIE**
- Both found expected file at rank 1
- ACE found multiple chunks from the file
- Auggie found the exact Bullet dataclass

---

#### Query 22: "CodeChunk dataclass file_path start_line"
**Expected:** `ace/code_chunker.py` or `ace/code_indexer.py`

**ACE Results:**
1. `ace/code_chunker.py` (score: 1.219) - AST chunking module - CORRECT
2. `ace/code_indexer.py` - `chunk_file` method
3. `ace/code_chunker.py` - `chunk_code` function
4. `ace/code_indexer.py` - CodeIndexer module
5. `ace/adaptive_chunker.py` - TOML chunking

**Auggie Results:**
1. `ace/code_indexer.py` - CodeChunkIndexed dataclass

**My Manual Verdict:** **TIE**
- Both found expected files
- ACE found code_chunker.py first
- Auggie found code_indexer.py which also has CodeChunkIndexed

---

#### Query 23: "QueryResult dataclass score file_path"
**Expected:** `ace/code_retrieval.py`

**ACE Results:**
1. `ace/retrieval_optimized.py` (score: 0.358) - `_score_entities` method (WRONG FILE)
2. `ace/code_retrieval.py` - `_rerank_results` method - CORRECT FILE BUT RANK 2
3. `ace/code_retrieval.py` - Module docstring
4. `docs/Fortune100.md` - docs
5. `docs/RETRIEVAL_PRECISION_OPTIMIZATION.md` - docs

**Auggie Results:**
1. `rag_training/optimizations/v2_query_expansion.py` - QueryResult dataclass (DIFFERENT FILE - this is in rag_training, not ace)

**My Manual Verdict:** **ACE WINS**
- ACE found `ace/code_retrieval.py` at rank 2-3
- Auggie found a DIFFERENT QueryResult in rag_training folder
- The expected was ace/code_retrieval.py

---

#### Query 24: "@property score getter"
**Expected:** `ace/playbook.py` or `ace/code_retrieval.py`

**ACE Results:**
1. `ace/semantic_scorer.py` (score: 0.783) - test function
2. `ace/semantic_scorer.py` - module docstring
3. `ace/unified_memory.py` - UnifiedBullet class
4. `ace/retrieval_optimized.py` - QuerySpecificityScorer
5. `ace/retrieval.py` - Smart retrieval

**Auggie Results:**
1. `.vscode/ace/retrieval_optimized.py` - `score` method

**My Manual Verdict:** **TIE (BOTH MISSED)**
- Neither found the expected files (playbook.py or code_retrieval.py)
- Both found other files with score-related code
- This is a poorly specified query

---

#### Query 25: "__init__ constructor pattern"
**Expected:** `ace/code_retrieval.py`

**ACE Results:**
1. `ace/unified_memory.py` (score: 0.569) - _PointsResult init
2. `ace/llm.py` - Module docstring
3. `docs/PROMPT_ENGINEERING.md` - docs
4. `docs/API_REFERENCE.md` - docs
5. `docs/ACE_INTELLIGENT_LEARNING_USER_GUIDE.md` - docs

**Auggie Results:**
1. `ace/__init__.py` - Package init

**My Manual Verdict:** **TIE (BOTH MISSED)**
- Neither found ace/code_retrieval.py
- This is a vague query - "__init__ constructor pattern" doesn't clearly point to code_retrieval.py
- Poorly specified expected file

---

#### Query 26: "context manager __enter__ __exit__"
**Expected:** `ace/resilience.py`

**ACE Results:**
1. `ace/context_injector.py` (score: 1.222) - Context injection module
2. `ace/observability/metrics.py` - `track_latency` context manager
3. `ace/multitenancy.py` - TenantContext class
4. `docs/PROMPTS.md` - docs
5. `docs/CLAUDE_CODE_ACE_INTEGRATION.md` - docs

**Auggie Results:**
1. `ace/multitenancy.py` - TenantContext

**My Manual Verdict:** **TIE (BOTH MISSED)**
- Neither found ace/resilience.py as expected
- Both found OTHER context managers in other files
- Query is generic - many files have context managers

---

#### Query 27: "@staticmethod factory pattern"
**Expected:** `ace/config.py`

**ACE Results:**
1. `ace/code_retrieval.py` (score: 0.584) - CodeRetrieval class
2. `ace/unified_memory.py` - _PointsResult
3. `ace/dependency_graph.py` - `build_call_graph`
4. `ace/retrieval_optimized.py` - `create_retriever`
5. `ace/llm.py` - Module docstring

**Auggie Results:**
1. `examples/zai_glm_example.py` - `create_zai_client` function

**My Manual Verdict:** **TIE (BOTH MISSED)**
- Neither found ace/config.py as expected
- Generic query doesn't clearly point to config.py

---

#### Query 28: "@classmethod from_config"
**Expected:** `ace/config.py`

**ACE Results:**
1. `ace/config.py` (score: 0.623) - `get_preset` function - CORRECT FILE
2. `ace/llm.py` - Module docstring
3. `ace/config.py` - Module docstring
4. `ace/llm_providers/litellm_client.py` - LiteLLMClient
5. `ace/pattern_detector.py` - Pattern Detector

**Auggie Results:**
1. `.vscode/dev_scripts/debug/debug_class_boost.py` - Debug script (WRONG FILE)

**My Manual Verdict:** **ACE WINS**
- ACE found ace/config.py at rank 1 and 3
- Auggie found a debug script, not the expected file

---

#### Query 29: "custom exception class"
**Expected:** `ace/resilience.py`

**ACE Results:**
1. `ace/security.py` (score: 0.781) - Security module with exceptions
2. `ace/observability/tracing.py` - ImportError handling
3. `ace/resilience.py` - Resilience patterns - CORRECT FILE AT RANK 3
4. `ace/observability/health.py` - HealthChecker
5. `ace/pattern_detector.py` - Pattern Detector

**Auggie Results:**
1. (Not shown in my current read - need to continue)

**My Manual Verdict:** **NEED MORE DATA**

---

Let me continue reading the benchmark file to complete the analysis.

