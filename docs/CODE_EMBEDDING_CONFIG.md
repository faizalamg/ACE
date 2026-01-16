# ACE Code Embedding Configuration

## Overview

ACE uses a **dual-embedding architecture** for optimal retrieval quality:

1. **Memory Embeddings**: `Qwen3-Embedding-8B` (4096d) for lessons, preferences, corrections
2. **Code Embeddings**: `Voyage-code-3` (1024d) for code context retrieval

This separation allows each domain to use embeddings specifically trained for its content type.

## Why Voyage-code-3 for Code?

After benchmarking multiple embedding models for code retrieval:

| Model | Dimension | API | Code Quality |
|-------|-----------|-----|--------------|
| Qwen3-Embedding-8B | 4096 | Local | Good general |
| nomic-embed-code | 768 | Local | Better code |
| **Voyage-code-3** | **1024** | **Cloud API** | **Best code retrieval** |

**Key findings:**
- Voyage-code-3 achieves **100% match** with ThatOtherContextEngine MCP on file ranking
- Specifically trained on code corpora (vs. general-purpose embeddings)
- 1024d dimension provides optimal balance of quality and performance
- **Batch embedding support**: 100x speedup with batch API calls

## Configuration

### Environment Variables

```bash
# REQUIRED: Voyage API key (get from https://www.voyageai.com/)
VOYAGE_API_KEY=your_api_key_here

# Memory embedding configuration (unchanged)
ACE_EMBEDDING_MODEL=text-embedding-qwen3-embedding-8b
ACE_EMBEDDING_DIM=4096
```

### Programmatic Configuration

```python
from ace.config import VoyageCodeEmbeddingConfig

config = VoyageCodeEmbeddingConfig()
print(config.model)      # voyage-code-3
print(config.dimension)  # 1024
print(config.is_configured())  # True if VOYAGE_API_KEY is set
```

## Usage

### Indexing Code

```python
from ace.code_indexer import CodeIndexer

# Index current workspace with batch embedding
indexer = CodeIndexer(workspace_path=".")
indexer.index_workspace()  # Uses batch embedding for 100x speedup
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

# Format like ThatOtherContextEngine MCP
formatted = retriever.format_ThatOtherContextEngine_style(results)
print(formatted)
```

## Search Strategy

ACE code retrieval uses **dense-only search** (not hybrid):

- **Dense vectors**: Voyage-code-3 embeddings capture semantic code similarity
- **No BM25**: Sparse/BM25 actually hurts code retrieval

## Collections

ACE maintains two Qdrant collections:

| Collection | Purpose | Embedding Model | Dimension |
|------------|---------|-----------------|-----------|
| ace_memories_hybrid | Lessons, preferences | Qwen3-Embedding-8B | 4096 |
| {workspace}_code_context | Code snippets | Voyage-code-3 | 1024 |

## Performance

- **Indexing**: ~65 seconds for 440 files with batch embedding (100x faster than sequential)
- **Search latency**: <50ms for dense-only vector search
- **Memory**: ~100MB per 10,000 code chunks in Qdrant

## Troubleshooting

### VOYAGE_API_KEY not set

1. Get API key from https://www.voyageai.com/
2. Set environment variable: export VOYAGE_API_KEY=your_key
3. Verify: python -c "from ace.config import VoyageCodeEmbeddingConfig; print(VoyageCodeEmbeddingConfig().is_configured())"

### No results for code queries

1. Check collection exists
2. Verify VOYAGE_API_KEY is set
3. Re-index workspace if needed

## Comparison with ThatOtherContextEngine MCP

| Feature | ThatOtherContextEngine MCP | ACE Code Context |
|---------|------------|------------------|
| File ranking | Accurate | Accurate (100% match) |
| Embeddings | Proprietary | Voyage-code-3 (1024d) |
| Search | Unknown | Dense-only (optimal for code) |
