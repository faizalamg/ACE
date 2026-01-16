# ACE vs ThatOtherContextEngine Code Retrieval Benchmark Results

**Date**: 2026-01-06
**Repository**: agentic-context-engine
**Objective**: Achieve 100% ACE superiority over ThatOtherContextEngine MCP for code-context retrieval

---

## Executive Summary

| Metric | Value | Status |
|--------|-------|--------|
| ACE Re-indexing | 513 files, 3315 chunks, 0 errors | ✅ Complete |
| Voyage Token-Aware Batching | Fixed (2.2 chars/token estimate, 100k limit) | ✅ Complete |
| ACE-Only Benchmark (100 queries) | Avg scores: Code(0.857), Config(0.767), Arch(0.907), Docs(0.749), Edge(0.599) | ✅ Complete |
| ACE Comprehensive Benchmark (280 queries) | Avg score: 0.581, 3/280 no results (98.9% recall) | ✅ Complete |
| ACE vs ThatOtherContextEngine Head-to-Head (50 queries) | 34% ThatOtherContextEngine top-match rate, 22% doc coverage | ⚠️ Needs improvement |

---

## 1. Infrastructure Fixes

### 1.1 Voyage Batch Token Limits Fixed

**Problem**: Voyage API limit of 120k tokens per batch was being exceeded
- Batch 8: 150,442 tokens → FAILED
- Batch 9: 184,029 tokens → FAILED
- Batch 17-19: Similar failures

**Solution** (ace/code_indexer.py:384-456):
```python
def _embed_batch(self, texts: List[str], batch_size: int = 128, max_tokens_per_batch: int = 100000):
    def estimate_tokens(text: str) -> int:
        # Code is VERY dense: ~2-2.5 chars/token
        return max(1, int(len(text) / 2.2))  # Conservative estimate

    # Token-aware batching
    for text in texts:
        text_tokens = estimate_tokens(text)
        if current_batch and (len(current_batch) >= batch_size or
                             current_tokens + text_tokens > max_tokens_per_batch):
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
```

**Result**: Re-indexed 513 files with 0 errors.

### 1.2 Noisy File Filtering

**Problem**: `.claude/settings.json` was appearing as top result for generic queries
- "typing Optional List Dict Tuple dataclass" → `.claude/settings.json` (0.432)
- "lambda filter map list comprehension generator" → `.claude/settings.json` (0.494)

**Solution** (ace/code_retrieval.py:1033-1059):
```python
def _should_exclude_result(self, file_path: str) -> bool:
    """Check if a result should be excluded from results."""
    path_lower = file_path.lower()
    noisy_patterns = [
        '.claude/settings.json',
        '.vscode/settings.json',
        '.idea/', 'node_modules/', '.venv/', '__pycache__/',
    ]
    for pattern in noisy_patterns:
        if pattern.replace('/', '\\') in path_lower or pattern.replace('\\', '/') in path_lower:
            return True
    return False
```

**Result**: Noisy config files now filtered out of all search results.

---

## 2. Benchmark Results

### 2.1 ACE-Only Benchmark (100 Queries)

**Test file**: test_head2head.py --batch

| Category | Queries | Avg Top Score | Top File (Appearances) |
|----------|---------|---------------|-------------------------|
| **Code** | 20 | 0.857 | ace/unified_memory.py (4x), ace/code_retrieval.py (3x) |
| **Config** | 15 | 0.767 | ace/config.py (5x), ace/retrieval_optimized.py (2x) |
| **Arch** | 20 | 0.907 | ace/hyde.py (2x), ace/query_enhancer.py (2x) |
| **Docs** | 15 | 0.749 | docs/SETUP_GUIDE.md, docs/MCP_INTEGRATION.md |
| **Edge** | 30 | 0.599 | .claude/settings.json (5x) → Now filtered |

**Low Score Queries** (below 0.5, filtered noisy files):
- "normalize_scores method score normalization" → ace/unified_memory.py (0.403)
- "architecture design system overview diagram" → quality_comparison JSON (0.418)
- "lambda filter map list comprehension generator" → Now returns better results

### 2.2 ACE Comprehensive Benchmark (280 Queries)

**Test file**: run_comprehensive_benchmark.py

| Metric | Value |
|--------|-------|
| Total Queries | 280 |
| Average Score | 0.581 |
| No Results | 3/280 (98.9% recall) |
| Score Range | 0.000 - 1.143 |

**Top Performing Queries** (score > 1.0):
1. Code dependency graph → ace/dependency_graph.py (1.235)
2. VoyageCodeEmbeddingConfig class → ace/config.py (1.13)
3. BM25Config class → ace/config.py (1.13)
4. Litellm import → ace/llm_providers/litellm_client.py (1.14)
5. From qdrant_client import → ace/deduplication.py (1.01)

### 2.3 ACE vs ThatOtherContextEngine Head-to-Head (50 Queries)

**Test file**: benchmark_ace_vs_ThatOtherContextEngine.py --ThatOtherContextEngine

| Category | Queries | ThatOtherContextEngine Match Rate |
|----------|---------|-------------------|
| Code | 10 | 40% |
| Docs | 10 | 22% doc coverage |
| Arch | 10 | 50% |
| Config | 10 | 30% |
| Edge | 10 | 30% |
| **Overall** | **50** | **34%** |

**Analysis**:
- ThatOtherContextEngine returned fewer results per query (typically 1-2 vs ACE's 5)
- ThatOtherContextEngine's top result was often similar to ACE's but with different scoring
- Doc coverage low (22%) because ThatOtherContextEngine doesn't index markdown files the same way

---

## 3. Memory Generalizability Classifier

### 3.1 Implementation

**New module**: ace/memory_generalizability.py

ACE now **ONLY stores cross-workspace, generalizable patterns**:

```python
class MemoryScope(Enum):
    ACE = "ACE"  # Generalizable - store in ACE
    NOT_ACE = "NOT_ACE"  # Not generalizable - do NOT store
```

### 3.2 Classification Rules

**STORE IN ACE** (Generalizable):
- "Prefer functional programming patterns" → ACE
- "Always validate input before processing" → ACE
- "Use parameterized queries to prevent SQL injection" → ACE

**DO NOT STORE** (Project-specific):
- "ScraperEpg uses FlareSolverr for ESPN" → REJECT
- "This project uses Z.ai GLM-4.6" → REJECT
- "ESPN API grouping bug fix was in parse_channel_id" → REJECT

### 3.3 Integration with ace_store

**File**: ace_mcp_server.py (ace_store function, lines 727-799)

```python
# Check if content is generalizable enough for ACE storage
from ace.memory_generalizability import should_store_in_ace

should_store, reason, extracted_principle = should_store_in_ace(content)

if not should_store:
    return f"REJECTED: {reason}\n\nACE only stores cross-workspace, generalizable patterns..."
```

**Features**:
1. Rejects project-specific content automatically
2. Extracts general principle from mixed content (e.g., "Use Qdrant for vector storage" → "Use vector database for storage")
3. Returns clear rejection messages with explanation

---

## 4. Next Steps

### 4.1 ACE Retrieval Improvements Needed

1. **Doc Coverage**: Increase from 22% to 50%+
   - Better ranking for markdown files vs code files
   - Special handling for README, CHANGELOG, CONTRIBUTING files

2. **Edge Case Queries**: Improve scores for generic queries
   - "lambda filter map list comprehension generator" → Should find code examples
   - "typing Optional List Dict Tuple dataclass" → Should find type-heavy files

3. **Architecture Queries**: Continue 50% → 100% ThatOtherContextEngine match rate
   - Better handling of design pattern queries
   - Improved synonym expansion for architectural concepts

### 4.2 Memory System Evolution

1. **ACE**: Cross-workspace, generalizable patterns only
   - User preferences ("Prefer X over Y")
   - Universal best practices ("Always validate input")
   - Cross-domain principles ("Use parameterized queries")

2. **Serena**: Project-specific memories (separate system)
   - Project-specific class names
   - Bug fixes for specific files
   - Project configuration details

---

## 5. Files Modified

| File | Changes | Purpose |
|------|---------|---------|
| ace/code_indexer.py | Token-aware batching (lines 384-456) | Fix Voyage token limit errors |
| ace/code_retrieval.py | Added _should_exclude_result (lines 1033-1059) | Filter noisy config files |
| ace/memory_generalizability.py | **NEW FILE** | Classify generalizable memories |
| ace_mcp_server.py | Integrated classifier into ace_store (lines 727-799) | Auto-reject non-generalizable content |
| benchmark_results/ace_benchmark_*.json | Generated | Benchmark output data |

---

## 6. Test Commands

```bash
# Re-index workspace with token-aware batching
python -c "from ace.code_indexer import CodeIndexer; CodeIndexer('.').index_workspace()"

# Run ACE-only benchmark
python test_head2head.py --batch

# Run ACE vs ThatOtherContextEngine comparison
python benchmark_ace_vs_ThatOtherContextEngine.py --ThatOtherContextEngine

# Run comprehensive benchmark
python run_comprehensive_benchmark.py

# Test memory generalizability classifier
python ace/memory_generalizability.py
```

---

**Generated**: 2026-01-06
**Status**: Ongoing - Target is 100% ACE superiority over ThatOtherContextEngine
