# Architecture

**Analysis Date:** 2026-01-19

## Pattern

**Type:** Modular Framework with Plugin Architecture

**Core Pattern:**
- Three-Role ACE Pipeline: Generator → Reflector → Curator
- Playbook-based strategy accumulation and learning
- Semantic scaffolding for intelligent retrieval

## Layers

### Core Layer (`ace/`)
- `playbook.py` - Strategy storage (Bullet, EnrichedBullet, Playbook)
- `delta.py` - Delta operations for playbook mutations
- `roles.py` - Generator, Reflector, Curator components
- `llm.py` - LLM client abstraction

### Retrieval Layer
- `retrieval.py` - SmartBulletIndex, IntentClassifier
- `qdrant_retrieval.py` - Vector database retrieval
- `unified_memory.py` - Cross-namespace memory retrieval
- `reranker.py` - Cross-encoder reranking

### Code Intelligence Layer
- `code_chunker.py` - AST-aware code chunking
- `code_indexer.py` - Code indexing for semantic search
- `code_retrieval.py` - Code-specific retrieval
- `code_analysis.py` - Tree-sitter code analysis
- `dependency_graph.py` - Import/call graph analysis

### Integration Layer (`ace/integrations/`)
- `litellm.py` - ACELiteLLM quick-start agent
- `browser_use.py` - ACEAgent for browser automation
- `langchain.py` - ACELangChain for complex workflows
- `base.py` - wrap_playbook_context utility

### MCP Server Layer
- `ace_mcp_server.py` - FastMCP server exposing ACE tools
- Tools: ace_retrieve, ace_store, ace_search, ace_stats, ace_tag

## Data Flow

```
User Query
    ↓
[MCP Server] → ace_retrieve
    ↓
[UnifiedMemoryIndex] → Qdrant hybrid search (dense + BM25)
    ↓
[SmartBulletIndex] → Multi-stage retrieval:
    1. Intent classification
    2. Namespace filtering
    3. Semantic scoring
    4. ELF adjustments (decay, golden rules, quality)
    5. Optional cross-encoder reranking
    ↓
[Results] → Formatted context for LLM injection
```

## Key Abstractions

### Playbook
Central strategy store with:
- Bullet CRUD operations
- Delta-based mutation (ADD, UPDATE, TAG, REMOVE)
- TOON serialization for token-efficient prompts
- Golden rule auto-promotion

### UnifiedBullet
Memory entry with:
- Namespace (USER_PREFS, TASK_STRATEGIES, PROJECT_SPECIFIC)
- Source tracking (user input, task execution, migration)
- Effectiveness scoring (helpful/harmful counts)
- Semantic scaffolding metadata

### SmartBulletIndex
Retrieval orchestrator with:
- Intent-based routing
- Dynamic weighting (similarity vs outcomes)
- Trigger pattern matching
- Session-aware effectiveness

## Entry Points

### MCP Server
- `ace_mcp_server.py:main()` - FastMCP server entry
- Tools exposed to Claude Code, VS Code, Cursor

### CLI/Scripts
- `scripts/ace_cli.py` - Command-line interface
- `scripts/run_ace_mcp.py` - MCP server launcher

### Programmatic
- `ace/__init__.py` - Public API exports
- `ace.integrations` - Framework integrations

## Configuration

**Config Files:**
- `ace/config.py` - Centralized configuration dataclasses
- `.env` - Environment variable overrides
- `.ace/.ace.json` - Workspace-specific config

**Key Config Classes:**
- `LLMConfig` - Model, API base, API key
- `QdrantConfig` - Host, port, collection settings
- `ELFConfig` - Golden rules, decay, thresholds
- `RetrievalConfig` - Reranking, embeddings, limits

---

*Architecture analysis: 2026-01-19*
