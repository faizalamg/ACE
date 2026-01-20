# Technical Concerns

**Analysis Date:** 2026-01-19

## Technical Debt

### Versioned Modules
Multiple prompt versions coexist:
- `prompts.py` (original)
- `prompts_v2.py` (iteration)
- `prompts_v2_1.py` (current)

**Impact:** Confusion about which to use, maintenance overhead
**Recommendation:** Consolidate to single prompts module with version parameter

### Large Module Files
Several modules exceed recommended size:
- `retrieval.py` (~1000 lines)
- `playbook.py` (~1000 lines)
- `unified_memory.py` (~800 lines)

**Impact:** Harder to navigate and maintain
**Recommendation:** Consider splitting into submodules

## Known Issues

### Empty Tool Calls in Subagents
Background agents (gsd-codebase-mapper) sometimes make empty tool calls without required parameters.

**Symptoms:** Tool parameter validation errors
**Root Cause:** Context inheritance issue in subagent prompts
**Workaround:** Manual document creation when agents fail

### Qdrant Connection Handling
No automatic reconnection if Qdrant becomes unavailable mid-session.

**Impact:** Operations fail silently or with cryptic errors
**Recommendation:** Add connection health checks and retry logic

## Security Considerations

### API Key Handling
- Keys loaded via environment variables (good)
- No key rotation mechanism
- Keys passed through LiteLLM abstraction

**Recommendation:** Consider adding key rotation support

### Input Validation
- Query inputs sanitized before Qdrant operations
- No explicit SQL injection concerns (uses vector DB)
- Prompt injection risk mitigated by structured outputs

### Multi-tenancy
- `multitenancy.py` provides tenant isolation
- Tenant data stored in `tenant_data/` directory
- Collection naming includes tenant ID

## Performance Considerations

### Embedding Model Loading
- sentence-transformers models load slowly (~5-10s)
- Lazy loading helps but first query is slow

**Mitigation:** Preload models at server startup

### Reranking Latency
- Cross-encoder reranking adds ~100-500ms per query
- Configurable via `ACE_ENABLE_RERANKING` env var

**Trade-off:** Precision vs latency

### Qdrant Query Performance
- Hybrid search (dense + BM25) is more expensive
- Index size grows with memory count

**Recommendation:** Monitor collection size, consider pruning old memories

## Fragile Areas

### JSON Parsing in Roles
Generator, Reflector, Curator all parse LLM JSON responses:
```python
def _safe_json_loads(text: str) -> Dict[str, Any]:
    # Strip markdown code blocks
    # Parse JSON
    # Retry on failure
```

**Risk:** LLM output format changes break parsing
**Mitigation:** Retry logic with modified prompts

### Configuration Hierarchy
Multiple config sources with priority:
1. Environment variables
2. Config files
3. Defaults

**Risk:** Unexpected behavior if configs conflict
**Recommendation:** Add config validation and logging

### Feature Detection
```python
from .features import has_opik, has_litellm

if has_litellm():
    from .llm_providers import LiteLLMClient
```

**Risk:** Import failures cause silent feature degradation
**Recommendation:** Log when features are unavailable

## Improvement Opportunities

### Missing Features
- [ ] Streaming support for LLM responses
- [ ] Batch operations for bulk memory storage
- [ ] Memory export/import functionality
- [ ] Admin UI for memory management

### Code Quality
- [ ] Increase type coverage for internal modules
- [ ] Add property-based testing for retrieval
- [ ] Benchmark suite for performance regression

---

*Concerns analysis: 2026-01-19*
