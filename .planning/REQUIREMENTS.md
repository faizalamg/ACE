# Requirements: ACE Code Embedding Switch

**Defined:** 2026-01-21
**Core Value:** Single config change, automatic everything else

## v1 Requirements

### Configuration

- [ ] **CONFIG-01**: User can set code embedding model via .ace/.ace.json config file
- [ ] **CONFIG-02**: Config file value overrides environment variable (config priority > env var)
- [ ] **CONFIG-03**: config.json includes code_embedding_model field ("voyage" | "jina" | "nomic")

### Collection Management

- [ ] **COLL-01**: Each embedding model uses dedicated Qdrant collection
  - Voyage: `ace_code_context_voyage` (1024d)
  - Jina: `ace_code_context_jina` (768d)
  - Nomic: `ace_code_context_nomic` (3584d)
- [ ] **COLL-02**: `get_code_collection_name()` in QdrantConfig returns correct collection based on provider

### Switching Behavior

- [ ] **SWITCH-01**: Non-breaking switch between models
- [ ] **SWITCH-02**: When switching to model with empty collection, query existing collection while new collection populates
- [ ] **SWITCH-03**: Re-index trigger occurs when model changes
- [ ] **SWITCH-04**: Lazy re-indexing - background populate while serving from old collection

### LM Studio Integration

- [ ] **LM-01**: LM Studio endpoint pattern detection (/v1/embeddings)
- [ ] **LM-02**: Fallback behavior if LM Studio is unreachable (error only, no fallback)
- [ ] **LM-03**: Both local models (Jina, Nomic) configurable in LM Studio

### Code Indexer

- [ ] **IDX-01**: CodeIndexer uses config file for model selection
- [ ] **IDX-02**: CodeIndexer._get_embedder() respects config file provider
- [ ] **IDX-03**: CodeIndexer respects per-model collection names

### Code Retrieval

- [ ] **RET-01**: CodeRetrieval uses config file for model selection
- [ ] **RET-02**: CodeRetrieval._get_embedder() respects config file provider
- [ ] **RET-03**: CodeRetrieval queries correct collection based on selected model

## v2 Requirements

### Performance (Deferred)

- [ ] **PERF-01**: Batch embedding optimizations for local models
- [ ] **PERF-02**: Caching of embeddings for repeated queries
- [ ] **PERF-03**: Parallel re-indexing for faster switches

### Additional Models (Deferred)

- [ ] **PERF-04**: Support for additional code embedding models
- [ ] **PERF-05**: Model validation and health checks

## Out of Scope

| Feature | Reason |
|----------|--------|
| Text/memory embedding switching | Out of scope for v1 - only code embeddings |
| Automatic fallback to Voyage | User wants error-only behavior if LM Studio unavailable |
| Cloud-only or local-only mode | Always support both Voyage and local models |
| Model-specific tuning | Use model defaults, avoid per-model config complexity |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CONFIG-01 | Phase 1 | Pending |
| CONFIG-02 | Phase 1 | Pending |
| CONFIG-03 | Phase 1 | Pending |
| COLL-01 | Phase 1 | Pending |
| COLL-02 | Phase 1 | Pending |
| SWITCH-01 | Phase 2 | Pending |
| SWITCH-02 | Phase 2 | Pending |
| SWITCH-03 | Phase 2 | Pending |
| SWITCH-04 | Phase 2 | Pending |
| LM-01 | Phase 1 | Pending |
| LM-02 | Phase 1 | Pending |
| LM-03 | Phase 1 | Pending |
| IDX-01 | Phase 2 | Pending |
| IDX-02 | Phase 2 | Pending |
| IDX-03 | Phase 2 | Pending |
| RET-01 | Phase 2 | Pending |
| RET-02 | Phase 2 | Pending |
| RET-03 | Phase 2 | Pending |

**Coverage:**
- v1 requirements: 16 total
- Mapped to phases: 16
- Unmapped: 0 ✓

---
*Requirements defined: 2026-01-21*
*Last updated: 2026-01-21 after initial definition*
