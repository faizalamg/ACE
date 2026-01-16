# ACE MCP Server Integration

**Use ACE with any MCP-compatible agent: VS Code Copilot, Claude Desktop, Cursor, or any custom MCP client.**

The MCP (Model Context Protocol) server provides AI agents with persistent memory that improves over time.

---

## Quick Start (3 minutes)

```bash
# 1. Install ACE with MCP support
pip install ace-framework mcp

# 2. Set your API key
export ZAI_API_KEY="your-key-here"   # Recommended
# Or: export OPENAI_API_KEY="your-key-here"

# 3. Start Qdrant vector database
docker run -d -p 6333:6333 qdrant/qdrant

# 4. Configure your IDE (see below)
```

---

## IDE Configuration

### VS Code / GitHub Copilot

Add to `.vscode/mcp.json` (workspace) or VS Code User Settings:

```json
{
  "mcpServers": {
    "ace": {
      "command": "python",
      "args": ["/path/to/agentic-context-engine/ace_mcp_server.py"],
      "type": "stdio"
    }
  }
}
```

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "ace": {
      "command": "python",
      "args": ["/path/to/agentic-context-engine/ace_mcp_server.py"]
    }
  }
}
```

### Cursor

Add to MCP settings:

```json
{
  "ace": {
    "command": "python",
    "args": ["/path/to/ace_mcp_server.py"],
    "transport": "stdio"
  }
}
```

---

## Dependencies

### Required

| Component | Purpose | Setup |
|-----------|---------|-------|
| **ACE Framework** | Memory & learning | `pip install ace-framework mcp` |
| **Qdrant** | Vector storage | `docker run -d -p 6333:6333 qdrant/qdrant` |
| **LLM Provider** | AI inference | Z.ai (default) or OpenAI |

### API Keys

| Provider | Environment Variable | Model | Notes |
|----------|---------------------|-------|-------|
| **Z.ai (Default)** | `ZAI_API_KEY` | GLM-4.7 | Best quality |
| OpenAI | `OPENAI_API_KEY` | gpt-4o-mini | Alternative |
| Anthropic | `ANTHROPIC_API_KEY` | claude-3-haiku | Alternative |

### Optional Dependencies

| Component | Purpose | Setup |
|-----------|---------|-------|
| **Voyage AI** | Code embeddings (94% R@1) | `pip install voyageai` + `VOYAGE_API_KEY` |
| **Opik** | Observability & costs | `pip install ace-framework[observability]` |
| **LM Studio** | Local embeddings | Download from lmstudio.ai |

**Substitution:** Any component can be replaced with alternatives:
- Qdrant → Any vector DB (Pinecone, Weaviate, Milvus)
- Z.ai → Any OpenAI-compatible API
- Voyage → OpenAI embeddings or local models

---

## Available MCP Tools

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `ace_retrieve` | Get relevant memories + code | **First call on every prompt** |
| `ace_store` | Save lessons/preferences | After learning something |
| `ace_search` | Filtered memory search | Find specific categories |
| `ace_stats` | View memory statistics | Debug/monitoring |
| `ace_tag` | Mark memory helpful/harmful | Improve retrieval |
| `ace_onboard` | Initialize workspace | First use in new project |
| `ace_workspace_info` | Check workspace status | Verify setup |

### Tool Parameters

```python
# ace_retrieve - Get relevant context
ace_retrieve(query="authentication error handling", limit=5)

# ace_store - Save a lesson
ace_store(
    content="Always validate JWT tokens before database calls",
    category="CORRECTION",
    severity=8,
    namespace="task_strategies"
)

# ace_search - Filtered search
ace_search(query="error handling", category="DEBUGGING", min_severity=5)
```

---

## Agent Instructions Template

Add to your agent's instruction file (`.cursorrules`, `copilot-instructions.md`, system prompt):

```markdown
# ACE Memory Protocol

## Mandatory: Retrieve Before Every Task
BEFORE starting ANY task, call:
ace_retrieve(query="<task keywords>")

## Store When Learning
- User corrects you → ace_store(content="lesson", category="CORRECTION", severity=8)
- User preference → ace_store(content="preference", category="PREFERENCE", severity=7)
- Bug fix pattern → ace_store(content="pattern", category="DEBUGGING", severity=6)

## Memory Categories

| Category | Severity | Use For |
|----------|----------|---------|
| DIRECTIVE | 9-10 | Critical rules |
| CORRECTION | 7-8 | Mistakes to avoid |
| PREFERENCE | 6-8 | User preferences |
| WORKFLOW | 5-6 | Process patterns |
| DEBUGGING | 5-6 | Bug fix patterns |

## Trigger Words
Auto-retrieve: "recurring", "again", "same issue", "remember"
Auto-store: "I prefer", "always", "never", "remember this"
```

---

## Workspace Onboarding

On first use, ACE automatically:
1. Creates `.ace/.ace.json` configuration
2. Indexes all code files for semantic search
3. Creates workspace-specific Qdrant collection

**Manual onboarding:**
```bash
python -c "from ace.code_indexer import CodeIndexer; CodeIndexer().index_workspace('/path/to/workspace')"
```

Or call the `ace_onboard` MCP tool.

---

## Configuration

### Environment Variables

```bash
# Required
export ZAI_API_KEY="your-key"              # Or OPENAI_API_KEY
export QDRANT_URL="http://localhost:6333"  # Default

# Optional
export VOYAGE_API_KEY="your-key"           # Code embeddings
export ACE_DEBUG="true"                    # Debug logging
export OPIK_API_KEY="your-key"             # Observability
```

### Workspace Configuration

ACE creates `.ace/.ace.json` in each workspace:
```json
{
  "workspace_name": "my-project",
  "workspace_path": "/path/to/my-project",
  "collection_name": "my-project_code",
  "onboarded_at": "2025-01-07T12:00:00Z"
}
```

---

## Usage Examples

### VS Code Copilot Chat

```
User: "I keep getting the same authentication error"

[Agent calls ace_retrieve(query="authentication error")]

Agent: "Based on previous sessions, this error occurs when JWT tokens 
aren't validated before database calls. Here's the fix..."
```

### Store User Preference

```
User: "I prefer TypeScript over JavaScript for all new code"

[Agent calls ace_store(content="User prefers TypeScript", category="PREFERENCE")]

Agent: "Got it! I'll use TypeScript for all new files going forward."
```

---

## Troubleshooting

### Verify MCP Server

```bash
# Test server manually
python ace_mcp_server.py

# Should output:
# ACE MCP Server started
```

### Check Qdrant Connection

```bash
curl http://localhost:6333/collections
```

### Common Issues

| Issue | Solution |
|-------|----------|
| "Connection refused" | Start Qdrant: `docker run -d -p 6333:6333 qdrant/qdrant` |
| "No memories found" | Call `ace_onboard` to initialize workspace |
| "Invalid API key" | Verify `ZAI_API_KEY` or `OPENAI_API_KEY` is set |
| MCP tools not appearing | Restart IDE after adding MCP config |

---

## Security Best Practices

**NEVER commit secrets to version control!**

1. **Store API keys in environment variables** - Use `.env` files or system env
2. **Add `.env` to `.gitignore`** - Prevent accidental commits
3. **Use workspace-level configs** - Keep API keys out of project repos
4. **Rotate keys regularly** - Especially if exposed

Example `.env` file (gitignored):
```bash
ZAI_API_KEY=your-actual-key
VOYAGE_API_KEY=your-voyage-key
OPIK_API_KEY=your-opik-key
```

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     Your IDE                                   │
│  (VS Code / Claude Desktop / Cursor)                           │
├────────────────────────────────────────────────────────────────┤
│                           │                                    │
│                    MCP Protocol                                │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                  ACE MCP Server                          │ │
│  │  ace_retrieve | ace_store | ace_search | ace_onboard     │ │
│  └──────────────────────────────────────────────────────────┘ │
│                           │                                    │
│              ┌────────────┴────────────┐                       │
│              │                         │                       │
│              ▼                         ▼                       │
│  ┌────────────────────┐   ┌────────────────────────┐          │
│  │   Qdrant           │   │   LLM Provider         │          │
│  │ (Vector Storage)   │   │  (Z.ai/OpenAI/etc)     │          │
│  │                    │   │                        │          │
│  │ - Code chunks      │   │ - Embedding generation │          │
│  │ - Memory index     │   │ - Reranking            │          │
│  │ - Workspace data   │   │ - Inference            │          │
│  └────────────────────┘   └────────────────────────┘          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Related Documentation

- [Main README](./README.md) - ACE Framework overview
- [Claude Code CLI Guide](./README_ClaudeCodeCLI.md) - For Claude Code users
- [VS Code Integration](./VSCODE_INTEGRATION.md) - Detailed VS Code setup
- [Contributing](./CONTRIBUTING.md) - How to contribute

---

## Summary

| Step | Action |
|------|--------|
| **1. Install** | `pip install ace-framework mcp` |
| **2. Start Qdrant** | `docker run -d -p 6333:6333 qdrant/qdrant` |
| **3. Configure IDE** | Add MCP server config (see above) |
| **4. Set API Key** | `export ZAI_API_KEY="your-key"` |
| **5. Use** | Agent automatically calls ACE tools |

ACE gives your AI agent persistent memory that improves over time - works with any MCP-compatible client!
