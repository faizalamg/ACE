# ACE Code Embedding Configuration

## Overview

ACE uses a **dual-embedding architecture** for optimal retrieval quality:

1. **Memory Embeddings**: `Qwen3-Embedding-8B` (4096d) for lessons, preferences, corrections
2. **Code Embeddings**: `Jina-v2-base-code` (768d) for code context retrieval (local via LM Studio)

This separation allows each domain to use embeddings specifically trained for its content type.

## Why Jina-v2-base-code for Code?

After migrating from Voyage-code-3 (cloud) to local embeddings for privacy and cost:

| Model | Dimension | API | Context | Code Quality |
|-------|-----------|-----|---------|--------------|
| Qwen3-Embedding-8B | 4096 | Local | 8K | Good general |
| Voyage-code-3 | 1024 | Cloud | 16K | Best (but cloud) |
| **Jina-v2-base-code** | **768** | **Local** | **8192** | **Excellent local** |

**Key findings:**
- Jina-v2-base-code runs 100% locally via LM Studio (no API keys, no cloud)
- Specifically trained on code corpora with 8192 token context
- 768d dimension provides good balance of quality and performance
- **AST-based chunking** ensures code fits within context limits


## Current Routing Status (Verified 2026-05-03)

ACE is currently configured to use the local code-embedding route unless `ACE_CODE_EMBEDDING_PROVIDER` is explicitly set.

- `ACE_CODE_EMBEDDING_PROVIDER` default: `local`
- Local code route: Jina-compatible LM Studio embedding model, 768 dimensions
- Nomic route: `ACE_CODE_EMBEDDING_PROVIDER=nomic`, 3584 dimensions
- Code collection routing:
  - Local/Jina: `ace_code_context_jina`
  - Nomic: `ace_code_context_nomic`

Do not configure `VOYAGE_API_KEY` for the default runtime. ACE code retrieval uses
local LM Studio embeddings and fails fast if the local embedding server is not
available.

### ACE Context Engine vs. SigMap

ACE context and SigMap serve different context roles:

| Tool | Primary Role | Best Use |
|------|--------------|----------|
| ACE context engine | Semantic code retrieval over indexed code chunks | Natural-language questions, vague implementation intent, blended code + memory evidence |
| SigMap | Lexical/signature ranking and generated context artifacts | Fast deterministic file/module prefiltering, static AI instruction/context outputs |
| Serena | Source-of-truth code navigation and edits | Symbol reads, references, precise edits |

Recommended flow:

1. Use SigMap to rank likely files/modules quickly.
2. Use ACE context engine when semantic code chunks are needed.
3. Use Serena to verify symbols, references, and exact source before edits.
4. Store durable lessons in ACE memory only after verification.

## Configuration

### Source of Truth

The workspace `.env` file is the single runtime source of truth for ACE
embedding configuration. MCP client JSON launches the server only; it must not
duplicate provider, model, Qdrant, or LLM values. `.ace/.ace.json` is onboarding
metadata only and does not override `.env` runtime settings.

### Environment Variables

```bash
# Code embedding configuration (local via LM Studio)
ACE_CODE_EMBEDDING_PROVIDER=local
ACE_LOCAL_EMBEDDING_URL=http://localhost:1234
ACE_LOCAL_CODE_MODEL=text-embedding-jina-embeddings-v2-base-code
ACE_LOCAL_CODE_DIM=768
ACE_CODE_EMBEDDING_MODEL=local

# AST Chunking (enabled by default)
ACE_ENABLE_AST_CHUNKING=true       # Enable AST-based semantic chunking
ACE_AST_MAX_LINES=80               # Max lines per chunk (default: 80)
ACE_AST_OVERLAP_LINES=10           # Overlap between chunks (default: 10)

# Embedding parallelism (match LM Studio's Max Concurrent Predictions)
ACE_EMBEDDING_PARALLEL=4           # Default: 4 workers

# Memory embedding configuration (unchanged)
ACE_EMBEDDING_MODEL=qwen3-embedding-8b
ACE_EMBEDDING_DIM=4096

# Retrieval-time LLM features disabled by default for fast deterministic MCP tools
ACE_LLM_EXPANSION=false
ACE_LLM_FILTERING=false
ACE_TYPO_LLM_CORRECTION=false
ACE_TYPO_GLM_VALIDATION=false
ACE_TYPO_AUTO_LEARN=false
```

### Programmatic Configuration

```python
from ace.config import get_embedding_provider_config

# Get code embedding config (auto-detects provider from env)
config = get_embedding_provider_config()
print(config["model"])      # text-embedding-jina-embeddings-v2-base-code
print(config["dimension"])  # 768
print(config["is_local"])   # True for local/Jina
```

## Usage

### Indexing Code

```python
from ace.code_indexer import CodeIndexer

# Index current workspace with AST-based chunking
indexer = CodeIndexer(workspace_path=".")
indexer.index_workspace()  # Uses AST chunking + parallel embedding
```

### Searching Code

```python
from ace.code_retrieval import CodeRetrieval

retriever = CodeRetrieval()

# Search for code
results = retriever.search(
    query="class UnifiedMemoryIndex Qdrant namespace",
    limit=5,
    exclude_tests=True,  # Skip test files
)

# Format like Auggie MCP
formatted = retriever.format_auggie_style(results)
print(formatted)
```

## AST Chunking

ACE uses **AST-based semantic chunking** (enabled by default) to ensure code fits within Jina's 8192 token context limit:

| Setting | Default | Description |
|---------|---------|-------------|
| `ACE_ENABLE_AST_CHUNKING` | `true` | Enable AST-based chunking |
| `ACE_AST_MAX_LINES` | `80` | Max lines per chunk |
| `ACE_AST_OVERLAP_LINES` | `10` | Overlap for context continuity |
| `JINA_SAFE_CHAR_LIMIT` | `6000` | Truncate chunks exceeding this (~4600 tokens) |

**Why AST Chunking?**
- **Problem**: Without chunking, entire files sent as single chunks → token overflow
- **Solution**: Parse AST, extract functions/classes/methods as semantic units
- **Result**: 485 chunks from 79 files (vs. 79 chunks without AST chunking)

## Parallelism Configuration

ACE embedding parallelism must match LM Studio's **Max Concurrent Predictions** setting:

```bash
# In LM Studio: Settings → Model → Max Concurrent Predictions = 4
# In ACE:
ACE_EMBEDDING_PARALLEL=4  # Must match LM Studio setting
```

**Mismatch Impact:**
- Too high: Requests queue, potential timeouts
- Too low: Under-utilizes GPU/CPU

## Search Strategy

ACE code retrieval uses **dense-only search** (not hybrid):

- **Dense vectors**: Jina embeddings capture semantic code similarity
- **No BM25**: Sparse/BM25 actually hurts code retrieval

## Collections

ACE maintains two Qdrant collections:

| Collection | Purpose | Embedding Model | Dimension |
|------------|---------|-----------------|-----------|
| ace_memories_hybrid | Lessons, preferences | Qwen3-Embedding-8B | 4096 |
| {workspace}_code_context | Code snippets | Jina-v2-base-code | 768 |

## Performance

- **Indexing**: ~30 seconds for 79 files with AST chunking + parallel embedding
- **Search latency**: <50ms for dense-only vector search
- **Memory**: ~100MB per 10,000 code chunks in Qdrant

## Troubleshooting

### LM Studio Not Running

1. Start LM Studio and load `text-embedding-jina-embeddings-v2-base-code`
2. Verify server is running at `http://localhost:1234/v1`
3. Test: `curl http://localhost:1234/v1/models`

### Token Overflow Warnings

If you see "Number of tokens exceeds model context length":
1. Verify AST chunking is enabled: `ACE_ENABLE_AST_CHUNKING=true`
2. Reduce max lines: `ACE_AST_MAX_LINES=60`
3. Re-index workspace after changing settings

### No Results for Code Queries

1. Check collection exists: `ace_{workspace}_code_context`
2. Verify LM Studio is running with Jina model loaded
3. Re-index workspace: `python -c "from ace.code_indexer import CodeIndexer; CodeIndexer('.').index_workspace()"`

## Migration from Voyage-code-3

If migrating from cloud Voyage embeddings:

1. **Dimension Change**: 1024d → 768d (requires re-indexing)
2. **Delete old collection**: Drop `{workspace}_code_context` in Qdrant
3. **Re-index**: Run `CodeIndexer(workspace).index_workspace()`
4. **Remove VOYAGE_API_KEY**: No longer needed

