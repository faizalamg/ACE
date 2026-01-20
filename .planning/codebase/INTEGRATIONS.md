# External Integrations

**Analysis Date:** 2026-01-19

## APIs & External Services

**LLM Providers:**
- OpenAI (GPT models) - Primary and query enhancement
  - SDK/Client: LiteLLM (`ace/llm_providers/litellm_client.py`)
  - Auth: `OPENAI_API_KEY`

- Anthropic (Claude models) - LLM option
  - SDK/Client: LiteLLM
  - Auth: `ANTHROPIC_API_KEY`

- Z.ai GLM - Default LLM for query enhancement/learning
  - SDK/Client: LiteLLM (OpenAI-compatible endpoint)
  - Auth: `ZAI_API_KEY`
  - API Base: `https://api.z.ai/api/coding/paas/v4`

- Google (Gemini models) - LLM option
  - SDK/Client: LiteLLM
  - Auth: `GOOGLE_API_KEY`

- Cohere (Command models) - LLM and reranking option
  - SDK/Client: LiteLLM
  - Auth: `COHERE_API_KEY`

- Azure OpenAI - Enterprise LLM option
  - SDK/Client: LiteLLM
  - Auth: `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION`

- AWS Bedrock - AWS-hosted models
  - SDK/Client: LiteLLM
  - Auth: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION_NAME`

- 100+ providers via LiteLLM - Unified LLM access

**Code Embedding:**
- Voyage AI - Code-specific embeddings for search
  - SDK/Client: `voyageai` (`voyageai>=0.3.0`)
  - Auth: `VOYAGE_API_KEY`
  - Model: voyage-code-3 (1024d, 32K context)

## Data Storage

**Vector Database:**
- Qdrant - Primary vector storage for memories and code
  - Connection: `ACE_QDRANT_URL` (default: http://localhost:6333)
  - Client: `qdrant-client>=1.16.1`
  - Collections:
    - `ace_memories_hybrid` (4096d) - Memory/lessons/preferences
    - `ace_code_context` (1024d) - Code chunks
    - `ace_bullets` (4096d) - Bullet points
  - Workspace-specific collections: `{workspace_name}_code_context`
  - Auth: `ACE_QDRANT_API_KEY` (for Qdrant Cloud)
  - gRPC: Optional via `ACE_QDRANT_GRPC=true` (port 6334)

**Hybrid Search:**
- Dense vectors (Cosine similarity)
- BM25 sparse vectors for keyword matching

**Caching:**
- HTTP-based retrieval caching via `ace/caching.py`
- Tenant-specific data storage in `tenant_data/`

## Authentication & Identity

**Auth Provider:**
- Custom implementation via `pyjwt>=2.10.1`

**JWT Configuration:**
- Token-based authentication for Qdrant Cloud and API access

## Monitoring & Observability

**Error Tracking:**
- Opik 1.8.0+ - Enterprise LLM tracing and evaluation
  - Implementation: `ace/observability/opik_integration.py`
  - Features: Bullet evolution tracking, playbook updates, role performance metrics
  - Auto-configuration for local use
  - LiteLLM callback integration for automatic token/cost tracking

**Metrics:**
- prometheus-client 0.23.1+ - Prometheus metrics collection

**Logging:**
- Python logging module (structured JSON logs)
- Level configuration via environment

## CI/CD & Deployment

**Hosting:**
- Docker support for Qdrant
- Qdrant Cloud available (free tier: 1GB)

**CI Pipeline:**
- pre-commit hooks for code quality
- Git hooks via `.pre-commit-config.yaml`
- GitHub Actions workflow support

## Environment Configuration

**Required env vars for core functionality:**
- `ACE_QDRANT_URL` - Qdrant server URL (default: localhost:6333)
- `VOYAGE_API_KEY` - Voyage AI for code embeddings
- `ZAI_API_KEY` - Z.ai GLM for query enhancement

**Optional env vars:**
- `ACE_QDRANT_API_KEY` - Qdrant Cloud authentication
- `OPENAI_API_KEY` - OpenAI models
- `ANTHROPIC_API_KEY` - Claude models
- `GOOGLE_API_KEY` - Gemini models
- `COHERE_API_KEY` - Cohere models

**Feature flags:**
- `ACE_FEATURE_MEMORIES=true` - Enable memory storage/retrieval
- `ACE_FEATURE_CODE_CONTEXT=true` - Enable code workspace indexing
- `ACE_FEATURE_MCP_SERVER=true` - Enable MCP server endpoints
- `ACE_FEATURE_LLM=true` - Enable LLM-based features
- `ACE_FEATURE_RERANKING=true` - Enable cross-encoder reranking
- `ACE_FEATURE_HYBRID=true` - Enable hybrid search

**Secrets location:**
- `.env` file (loaded via python-dotenv)
- Environment variables

**Configuration files:**
- `.env.example` - Template with all available options
- `ace/config.py` - Centralized configuration dataclasses

## Webhooks & Callbacks

**Incoming:**
- MCP (Model Context Protocol) - For IDE integration
  - Server: `ace_mcp_server.py` using FastMCP
  - Tools exposed:
    - `ace_retrieve` - Retrieve context from memory/code
    - `ace_store` - Store new memories/lessons
    - `ace_search` - Search memories with filters
    - `ace_stats` - Get memory statistics
    - `ace_tag` - Tag memories as helpful/harmful
    - `ace_onboard` - Onboard workspace for code indexing
    - `ace_workspace_info` - Get workspace configuration
    - `ace_enhance_prompt` - Enhance user prompts with context

**Outgoing:**
- LiteLLM callbacks - LLM provider abstraction
- Opik callbacks - Span tracking for token aggregation

**Browser Automation:**
- browser-use 0.9.1+ - Web browsing agent framework
  - Integration: `ace/integrations/browser_use.py`
  - Usage: ACEAgent wraps browser-use Agent with learning
  - Auto-learns from browser execution traces

**LangChain Integration:**
- LangChain Runnable wrapper - `ace/integrations/langchain.py`
  - Usage: ACELangChain wraps any LangChain Runnable
  - Automatic learning from execution

---

*Integration audit: 2026-01-19*
