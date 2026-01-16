# m1rl0k/Context-Engine Integration Analysis

## Executive Summary

**Feasibility: HIGH** - The m1rl0k/Context-Engine shares significant architectural overlap with ACE, making integration feasible. Both use Qdrant, hybrid search (dense+BM25), and cross-encoder reranking.

## Key Features for ACE Enhancement

### 1. Code Context Injection (HIGH VALUE)
- **ctx.py CLI**: Context-aware prompt enhancer that retrieves code and rewrites prompts
- **Adaptive sizing**: Short/vague queries get more context automatically
- **Implementation**: `scripts/ctx.py` lines 626-647 (`_adaptive_context_sizing`)

### 2. AST-Based Semantic Chunking (HIGH VALUE)
- **ASTAnalyzer**: Language-aware symbol extraction with call graphs
- **Semantic chunking**: Preserves code boundaries (functions/classes)
- **Gap filling**: Module-level code between symbols
- **Implementation**: `scripts/ast_analyzer.py` - `chunk_semantic()` method

### 3. ReFRAG Micro-Chunking (MEDIUM VALUE)
- **16-token windows** with 8-token stride
- **Span budgeting**: Global token budget management
- **Gate-first filtering**: Mini-vector pre-filtering

### 4. Pattern Detection (MEDIUM VALUE)
- **Structural search**: Find similar code patterns via AST analysis
- **64-dim pattern vector**: WL graph kernel, CFG fingerprint, SimHash
- **Cross-language**: Python pattern finds Go/Rust/Java equivalents

### 5. Memory-Code Blending (HIGH VALUE)
- **context_search**: Blends code + memory in single response
- **Weighted fusion**: Configurable memory vs code weight

## Architecture Comparison

| Feature | ACE | Context-Engine |
|---------|-----|----------------|
| Vector DB | Qdrant | Qdrant |
| Embeddings | BGE-base | BGE-base/Qwen3 |
| Hybrid Search | BM25 + Dense | BM25 + Dense |
| Reranker | ms-marco-MiniLM | Cross-encoder |
| Query Expansion | Adaptive/LLM | Semantic expansion |
| Code Chunking | Line-based | AST semantic |
| Memory System | Yes | Yes (memory_store) |
| MCP Transport | SSE | SSE + HTTP RMCP |

## Integration Recommendations

### Phase 1: AST Semantic Chunking (1-2 weeks)
```python
# Port from scripts/ast_analyzer.py
from ace.code_chunker import ASTChunker

chunker = ASTChunker()
chunks = chunker.chunk_semantic(content, language="python", max_lines=120)
```

**Benefits**: Better code context boundaries, improved retrieval precision

### Phase 2: Context Injection for Claude (1 week)
```python
# Adapt from scripts/ctx.py
def enhance_prompt_with_code_context(query: str, limit: int = 5) -> str:
    context = ace.retrieve(query, limit=limit)
    return f"Context:\n{context}\n\nQuery: {query}"
```

**Benefits**: Richer prompts for code-related tasks

### Phase 3: Pattern Detection (2-3 weeks)
```python
# Port from scripts/pattern_detection/search.py
def find_similar_patterns(code_snippet: str) -> List[Match]:
    signature = extract_pattern_signature(code_snippet)
    return search_by_signature(signature)
```

**Benefits**: Find structurally similar code across codebase

## API Endpoints to Leverage

| Endpoint | Port | Purpose |
|----------|------|---------|
| Memory Server | 8000/8002 | Knowledge base storage |
| Indexer Server | 8001/8003 | Code search + indexing |

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Dependency bloat | Cherry-pick only needed modules |
| Performance overhead | AST parsing is expensive - cache results |
| Language support | Tree-sitter coverage varies |

## Conclusion

**Recommended**: Integrate AST semantic chunking and context injection as Phase 1. Pattern detection can follow as Phase 2 if retrieval precision needs further improvement.

**MIT Licensed** - Safe for integration.
