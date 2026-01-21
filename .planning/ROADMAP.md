# Roadmap: ACE Code Embedding Switch

**Created:** 2026-01-21
**Milestone:** v1 - Non-breaking model switch with config file support

## Overview

Implement single-switch code embedding model selection with per-model Qdrant collections and non-breaking switching behavior.

| Phase | Goal | Requirements | Plans | Status |
|--------|------|-------------|-------|--------|
| 1 | Configuration Layer | CONFIG-01, CONFIG-02, CONFIG-03 | 4 | Pending |
| 2 | Switching Behavior | SWITCH-01, SWITCH-02, SWITCH-03, SWITCH-04 | 3 | Pending |
| 3 | LM Studio Integration | LM-01, LM-02, LM-03 | 3 | Pending |
| 4 | Code Indexer Integration | IDX-01, IDX-02, IDX-03 | 3 | Pending |
| 5 | Code Retrieval Integration | RET-01, RET-02, RET-03 | 3 | Pending |
| 6 | Testing and Validation | All above | 4 | Pending |

**Total:** 6 phases | **20 plans** | **Coverage:** 100% ✓

---

## Phase 1: Configuration Layer

**Goal:** Add .ace/.ace.json config file support with code_embedding_model field that overrides env vars.

**Requirements:** CONFIG-01, CONFIG-02, CONFIG-03

**Success Criteria:**
1. .ace/.ace.json loads and merges with env var config
2. code_embedding_model field supports "voyage" | "jina" | "nomic"
3. Config value has priority over env var

### 1.1 Extend AceConfig dataclass
Add code_embedding_model field to ACEConfig with default from env var.

### 1.2 Implement .ace.json loading
Load .ace/.ace.json and merge with env-based config.

### 1.3 Update provider resolution
EmbeddingProviderConfig uses config file value when present.

### 1.4 Add config tests
Test config loading and priority order.

---

## Phase 2: Switching Behavior

**Goal:** Non-breaking model switch with old collection query while new collection populates.

**Requirements:** SWITCH-01, SWITCH-02, SWITCH-03, SWITCH-04

**Success Criteria:**
1. Switch detects collection change and triggers re-index
2. Queries use old collection until new collection ready
3. Background re-indexing doesn't block queries
4. Switch fails fast if LM Studio unavailable (no fallback)

### 2.1 Add switch detection
Detect when code_embedding_model changes and collection needs change.

### 2.2 Implement sync wait-for-index
Check if target collection has vectors, wait if empty.

### 2.3 Background re-index trigger
Spawn background re-index when switch detected.

### 2.4 Error-only behavior on LM Studio failure
Fail fast when LM Studio unreachable, no Voyage fallback.

---

## Phase 3: LM Studio Integration

**Goal:** Detect LM Studio endpoint pattern and handle connection errors appropriately.

**Requirements:** LM-01, LM-02, LM-03

**Success Criteria:**
1. /v1/embeddings endpoint pattern detected
2. Both local models (Jina, Nomic) use LM Studio
3. Connection error fails fast with clear message

### 3.1 Add endpoint pattern detection
Detect LM Studio /v1/embeddings vs other formats.

### 3.2 Support multiple local models
Both Jina and Nomic models configurable in LM Studio.

### 3.3 Error-only failure mode
Fail fast when LM Studio unreachable, no Voyage fallback.

---

## Phase 4: Code Indexer Integration

**Goal:** CodeIndexer respects config file for model selection and uses correct collection.

**Requirements:** IDX-01, IDX-02, IDX-03

**Success Criteria:**
1. CodeIndexer reads config file for provider
2. Correct collection name used (jina/nomic/voyage)
3. Collection dimension matches model output

### 4.1 Add config file reading
CodeIndexer loads .ace/.ace.json code_embedding_model.

### 4.2 Respect collection naming
Use get_code_collection_name() based on provider.

### 4.3 Verify dimension matching
Ensure collection dimension matches model output dimension.

---

## Phase 5: Code Retrieval Integration

**Goal:** CodeRetrieval respects config file for model selection and queries correct collection.

**Requirements:** RET-01, RET-02, RET-03

**Success Criteria:**
1. CodeRetrieval reads config file for provider
2. Queries use correct collection based on model
3. _get_embedder() returns correct embedder for model

### 5.1 Add config file reading
CodeRetrieval loads .ace/.ace.json code_embedding_model.

### 5.2 Use correct collection name
Query collection matches selected model.

### 5.3 Verify embedder selection
_get_embedder() returns correct embedder (Jina/Nomic/Voyage).

---

## Phase 6: Testing and Validation

**Goal:** Verify all functionality works correctly with non-breaking switches.

**Success Criteria:**
1. All unit tests pass
2. Switch from Voyage to Jina works non-breaking
3. Switch from Jina to Nomic works non-breaking
4. Config file overrides env var correctly

### 6.1 Write unit tests
Test config loading, provider resolution, collection naming.

### 6.2 Integration tests for switch
Test full switch workflow with collection fallback.

### 6.3 Test config priority
Verify config > env var ordering.

### 6.4 Manual UAT verification
User validates real switch behavior.

---

*Roadmap created: 2026-01-21*
