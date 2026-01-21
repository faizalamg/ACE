# ACE Code Embedding Switch

## What This Is

ACE allows switching between code embedding models (Voyage, Jina, Nomic) with a single configuration change. Each model uses its own Qdrant collection based on embedding dimensions, and switching automatically handles collection indexing.

## Core Value

**Single config change, automatic everything else** - The user changes one setting and all embeddings, collections, and retrieval adapt without manual steps.

## Requirements

### Validated

- Provider selection via `ACE_CODE_EMBEDDING_PROVIDER` env var (voyage|local|nomic)
- Multi-provider support in `EmbeddingProviderConfig` with `is_code_*()` methods
- Per-model collection naming via `QdrantConfig.get_code_collection_name()`
- LM Studio `/v1/embeddings` endpoint for local models

### Active

- [ ] Add `code_embedding_model` field to `.ace/.ace.json` config file
- [ ] Config file overrides env var for model selection (config > env var priority)
- [ ] Re-index trigger on model switch detection
- [ ] Sync wait-for-index behavior when switching to empty collection
- [ ] Non-breaking: old collection used until new collection is ready
- [ ] Error-only behavior if LM Studio unavailable (no fallback to Voyage)

### Out of Scope

- [Text/memory embedding switching] — Only code embeddings in v1
- [Automatic fallback to Voyage] — If local model unavailable, fail fast
- [Cloud-only or local-only mode] — Always support both
- [Model-specific config tuning] — Use model defaults

## Context

ACE is a Python-based framework for building self-improving AI agents. The codebase already has partial infrastructure for multi-provider code embeddings:

- **config.py**: `EmbeddingProviderConfig` supports "local", "nomic", "voyage"
- **config.py**: `LocalEmbeddingConfig` for LM Studio (URL: `http://localhost:1234`)
- **config.py**: `NomicCodeEmbeddingConfig` (3584d, q8_0 quantized)
- **code_indexer.py**: `_get_embedder()` dispatches to correct provider
- **code_retrieval.py**: Same `_get_embedder()` pattern for queries
- **config.py**: `QdrantConfig.get_code_collection_name()` returns model-specific collection

Current gaps:
- Config file not supported (only env vars)
- No re-index trigger on switch
- No sync/wait-for-index on empty collection

## Constraints

- **Tech stack**: Python 3.11+, existing codebase patterns
- **Existing patterns**: Use `is_code_local()`, `is_code_nomic()`, `is_code_voyage()` methods
- **LM Studio endpoint**: `/v1/embeddings` (OpenAI/Ollama compatible)
- **Qdrant**: Each model has separate collection due to dimensions:
  - Voyage: 1024d → `ace_code_context_voyage`
  - Jina: 768d → `ace_code_context_jina`
  - Nomic: 3584d → `ace_code_context_nomic`
- **No breaking**: Old collection queries while new collection populates

## Key Decisions

| Decision | Rationale | Outcome |
|-----------|-----------|----------|
| Config in .ace/.ace.json | Already used for ACE workspace settings | — Pending |
| Config overrides env var | User preference beats default | — Pending |
| Sync wait-for-index | Non-breaking, user gets results during indexing | — Pending |
| Error-only LM Studio failure | Fail fast if local model unavailable | — Pending |
| Single model field | `code_embedding_model`: "voyage" | "jina" | "nomic" | — Pending |

---
*Last updated: 2026-01-21 after initialization*
