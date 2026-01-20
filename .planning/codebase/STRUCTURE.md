# Directory Structure

**Analysis Date:** 2026-01-19

## Root Layout

```
agentic-context-engine/
├── ace/                    # Core framework package
│   ├── integrations/       # External framework integrations
│   ├── llm_providers/      # LLM client implementations
│   ├── observability/      # Tracing, metrics, health
│   └── embedding_finetuning/  # Embedding model training
├── tests/                  # Test suite (pytest)
│   └── integrations/       # Integration-specific tests
├── scripts/                # CLI and utility scripts
├── docs/                   # Documentation
├── tenant_data/            # Multi-tenant data storage
├── .planning/              # GSD planning artifacts
└── .ace/                   # Workspace configuration
```

## Core Package (`ace/`)

### Main Modules
| File | Purpose | Lines |
|------|---------|-------|
| `playbook.py` | Strategy storage, Bullet/EnrichedBullet | ~1000 |
| `roles.py` | Generator, Reflector, Curator | ~800 |
| `retrieval.py` | SmartBulletIndex, IntentClassifier | ~1000 |
| `unified_memory.py` | Cross-namespace memory | ~800 |
| `qdrant_retrieval.py` | Vector DB operations | ~600 |

### Code Intelligence
| File | Purpose |
|------|---------|
| `code_chunker.py` | AST-aware code chunking |
| `code_indexer.py` | Code indexing for retrieval |
| `code_retrieval.py` | Code-specific search |
| `code_analysis.py` | Tree-sitter parsing |
| `dependency_graph.py` | Import/call graphs |
| `pattern_detector.py` | Code pattern detection |

### Retrieval & Search
| File | Purpose |
|------|---------|
| `retrieval.py` | Core retrieval orchestration |
| `retrieval_optimized.py` | Performance-optimized retrieval |
| `retrieval_bandit.py` | LinUCB adaptive selection |
| `retrieval_presets.py` | Preset configurations |
| `retrieval_caching.py` | Result caching |
| `reranker.py` | Cross-encoder reranking |
| `hyde.py` | Hypothetical document embeddings |

### Configuration & Utils
| File | Purpose |
|------|---------|
| `config.py` | Centralized configuration |
| `features.py` | Feature detection (has_litellm, etc.) |
| `caching.py` | General caching utilities |
| `resilience.py` | Retry logic, circuit breakers |

## Key Locations

### Entry Points
- `ace/__init__.py` - Public API exports
- `ace_mcp_server.py` - MCP server (standalone)
- `scripts/ace_cli.py` - CLI interface
- `scripts/run_ace_mcp.py` - MCP launcher

### Configuration
- `ace/config.py` - Config dataclasses
- `.env` - Environment overrides
- `.ace/.ace.json` - Workspace config
- `pyproject.toml` - Package metadata

### Tests
- `tests/conftest.py` - Shared fixtures
- `tests/test_*.py` - Unit/integration tests
- `tests/integrations/` - Framework-specific tests

## Naming Conventions

### Files
- Snake_case for all Python files
- `test_` prefix for test files
- `_v2`, `_v2_1` suffix for versioned modules

### Classes
- PascalCase for all classes
- `*Output` suffix for dataclasses (GeneratorOutput)
- `*Config` suffix for configuration classes

### Functions
- snake_case for all functions
- `_private` prefix for internal methods
- `get_*` for accessors
- `apply_*` for mutators

### Constants
- UPPER_SNAKE_CASE
- Module-level constants at top of file

---

*Structure analysis: 2026-01-19*
