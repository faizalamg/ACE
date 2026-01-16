# ACE MCP Integration Guide

This guide explains how to use ACE (Agentic Context Engine) as an MCP server with GitHub Copilot, Claude Desktop, Cursor, and other MCP-compatible clients.

## Overview

ACE MCP Server exposes the unified memory system as 6 MCP tools:

| Tool | Description | Primary Use |
|------|-------------|-------------|
| `ace_retrieve` | Query memories with semantic search + cross-encoder reranking | Get context before tasks |
| `ace_store` | Store new memories with automatic deduplication | Save lessons/preferences |
| `ace_search` | Filtered search by category/severity | Find specific memories |
| `ace_stats` | Collection statistics | Debug/monitor |
| `ace_tag` | Feedback tagging (helpful/harmful) | Improve retrieval quality |
| `ace_enhance_prompt` | Enhance prompts via LLM with ACE context injection | Transform vague prompts into structured specs |

## Quick Start

### 1. Prerequisites

```bash
# Install ACE with MCP support
pip install ace-framework mcp

# Ensure Qdrant is running (for vector storage)
docker run -p 6333:6333 qdrant/qdrant
```

### 2. VS Code Copilot Setup

Add to your VS Code `settings.json` or workspace `.vscode/mcp.json`:

```json
{
  "mcpServers": {
    "ace": {
      "command": "python",
      "args": ["path/to/agentic-context-engine/ace_mcp_server.py"],
      "type": "stdio"
    }
  }
}
```

Or with full path (Windows example):

```json
{
  "mcpServers": {
    "ace": {
      "command": "D:\\path\\to\\venv\\Scripts\\python.exe",
      "args": ["D:\\path\\to\\agentic-context-engine\\ace_mcp_server.py"],
      "type": "stdio"
    }
  }
}
```

### 3. Claude Desktop Setup

Add to Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

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

### 4. Cursor Setup

Add to Cursor MCP settings:

```json
{
  "ace": {
    "command": "python",
    "args": ["path/to/ace_mcp_server.py"],
    "transport": "stdio"
  }
}
```

## Tool Reference

### ace_retrieve

Query memories with semantic search. Returns formatted context with relevance scores.

**Parameters:**
- `query` (required): Natural language query
- `namespace`: Filter by `user_prefs`, `task_strategies`, `project_specific`, or `all`
- `limit`: Max results (default: 5, max: 20)

**Example:**
```json
{
  "query": "coding style preferences",
  "namespace": "user_prefs",
  "limit": 5
}
```

**Response:**
```
**User Preferences:**
[!] [PREF] [PREFERENCE] Use TypeScript for new projects [x3]
[!] [PREF] [DIRECTIVE] Always add JSDoc comments to public APIs
```

### ace_store

Store a new memory with automatic deduplication.

**Parameters:**
- `content` (required): The lesson, preference, or pattern to remember
- `namespace`: Category (`user_prefs`, `task_strategies`, `project_specific`)
- `section`: Sub-category (e.g., `communication`, `architecture`, `testing`)
- `severity`: Importance 1-10 (10=critical directive)
- `category`: Type (`PREFERENCE`, `CORRECTION`, `DIRECTIVE`, `WORKFLOW`, `ARCHITECTURE`, `DEBUGGING`, `SECURITY`)

**Example:**
```json
{
  "content": "User prefers concise explanations without emoji",
  "namespace": "user_prefs",
  "category": "PREFERENCE",
  "severity": 8
}
```

**Response:**
```
Memory stored successfully. ID: abc123-def456
```

Or if similar memory exists:
```
Memory reinforced (similar exists). Reinforcement count: 3. Similarity: 0.92
```

### ace_search

Filtered search with category and severity constraints.

**Parameters:**
- `query` (required): Search query
- `category`: Filter by category (`PREFERENCE`, `CORRECTION`, `DIRECTIVE`, etc.)
- `min_severity`: Minimum severity level (1-10)
- `limit`: Max results (default: 10)

**Example:**
```json
{
  "query": "error handling",
  "category": "DEBUGGING",
  "min_severity": 5,
  "limit": 5
}
```

**Response (JSON):**
```json
[
  {
    "id": "abc123",
    "content": "Always use try-catch for async operations",
    "category": "DEBUGGING",
    "severity": 7,
    "namespace": "task_strategies",
    "helpful": 5,
    "harmful": 0
  }
]
```

### ace_stats

Get collection statistics.

**Parameters:** None

**Response:**
```
ACE Unified Memory Statistics
=============================
Collection: ace_memories_hybrid
Total Points: 2,977
Status: GREEN
Vectors Config: {'dense': VectorParams(size=4096, ...)}

Memory types tracked:
- User preferences (directives, communication style)
- Task strategies (coding patterns, debugging approaches)
- Project-specific (architecture, codebase patterns)
- Corrections and lessons learned
```

### ace_tag

Tag a memory as helpful or harmful for retrieval quality feedback.

**Parameters:**
- `memory_id` (required): ID of the memory to tag
- `tag` (required): `helpful` or `harmful`

**Example:**
```json
{
  "memory_id": "abc123-def456",
  "tag": "helpful"
}
```

**Response:**
```
Memory abc123-def456 tagged as helpful
```

### ace_enhance_prompt

Enhance a vague or incomplete prompt into a structured, comprehensive specification using LLM with ACE context injection. This tool transforms user prompts by adding context from ACE memories, git history, open files, and chat history.

**Parameters:**
- `prompt` (required): The user prompt to enhance
- `include_memories` (default: true): Include relevant ACE memories as context
- `include_git_commits` (default: false): Include recent git commit history
- `include_git_status` (default: false): Include current git status (staged/unstaged changes)
- `open_files` (optional): List of file paths currently open in editor
- `chat_history` (optional): Recent conversation history for context
- `custom_context` (optional): Any additional context to inject
- `workspace_path` (optional): Workspace path for git operations
- `provider` (default: "zai"): LLM provider (`zai`, `openai`, `anthropic`, `lmstudio`)
- `model` (optional): Model override (e.g., `gpt-4o`, `claude-sonnet-4-20250514`)
- `max_tokens` (default: 8192): Maximum response tokens
- `temperature` (default: 0.7): Response creativity (0.0-1.0)

**Example - Basic Enhancement:**
```json
{
  "prompt": "add user auth"
}
```

**Example - Full Context:**
```json
{
  "prompt": "fix the login bug",
  "include_memories": true,
  "include_git_commits": true,
  "include_git_status": true,
  "open_files": ["src/auth/login.ts", "src/types/user.ts"],
  "chat_history": "User: The login fails after password reset...",
  "workspace_path": "/path/to/project",
  "provider": "openai",
  "model": "gpt-4o"
}
```

**Response Format:**
The enhanced prompt follows a structured format:
```
## OBJECTIVE
[Clear goal extracted from original prompt]

## CONTEXT
[Business/technical context including any project-specific information]

## REQUIREMENTS
### Functional Requirements
- [Detailed list based on original prompt + context]

### Technical Requirements
- [Implementation constraints and considerations]

### Constraints
- [Any limitations or boundaries to respect]

## ACCEPTANCE CRITERIA
- [Measurable criteria for completion]
```

**Context Sources:**
| Source | Description | Parameter |
|--------|-------------|-----------|
| ACE Memories | Relevant preferences, patterns, lessons | `include_memories=true` |
| Git Commits | Recent commit messages/history | `include_git_commits=true` |
| Git Status | Current staged/unstaged changes | `include_git_status=true` |
| Open Files | Files currently in editor | `open_files=["path/to/file"]` |
| Chat History | Recent conversation context | `chat_history="..."` |
| Custom Context | Any additional information | `custom_context="..."` |

**LLM Providers:**
| Provider | Default Model | API Endpoint |
|----------|---------------|--------------|
| `zai` | `glm-4.7` | `https://api.z.ai/v1` |
| `openai` | `gpt-4o` | `https://api.openai.com/v1` |
| `anthropic` | `claude-sonnet-4-20250514` | `https://api.anthropic.com/v1` |
| `lmstudio` | Required in `model` param | `http://localhost:1234/v1` |

**Environment Variables:**
- `ZAI_API_KEY` - API key for Z.AI provider
- `OPENAI_API_KEY` - API key for OpenAI provider
- `ANTHROPIC_API_KEY` - API key for Anthropic provider

## Usage Patterns

### Before Starting a Task

```
Use ace_retrieve with query describing what you're about to do.
This surfaces relevant past lessons and user preferences.
```

### After Learning Something

```
When user corrects you or expresses a preference, use ace_store
to save it for future sessions.
```

### Trigger Words

These phrases should trigger automatic `ace_retrieve`:
- "recurring", "again", "same issue"
- "last time", "before", "remember"
- "we discussed"

These phrases should trigger `ace_store`:
- "I prefer", "always", "never"
- "Remember this", "note that"

## Architecture

```
┌─────────────────────────┐
│   MCP Client            │
│   (Copilot/Claude/etc)  │
└───────────┬─────────────┘
            │ MCP Protocol (stdio)
┌───────────▼─────────────┐
│   ace_mcp_server.py     │
│   6 tools exposed       │
└───────────┬─────────────┘
            │
┌───────────▼─────────────┐      ┌─────────────────────┐
│   UnifiedMemoryIndex    │      │   LiteLLMClient     │
│   - Vector search       │      │   - Prompt enhance  │
│   - Cross-encoder       │      │   - Multi-provider  │
│   - Deduplication       │      │   (zai/openai/etc)  │
└───────────┬─────────────┘      └─────────────────────┘
            │
┌───────────▼─────────────┐
│   Qdrant + Embeddings   │
│   (4096-dim vectors)    │
└─────────────────────────┘
```

## Troubleshooting

### Server Not Starting

1. Check Python path is correct in MCP config
2. Ensure Qdrant is running: `curl http://localhost:6333/health`
3. Check logs: `python ace_mcp_server.py 2>&1`

### No Results from ace_retrieve

1. Check Qdrant has data: use `ace_stats` tool
2. Lower the threshold (default 0.35 may be too high for some queries)
3. Try broader queries

### Slow Responses

1. Cross-encoder reranking adds ~50ms per query
2. First query loads model (~2s warm-up)
3. Subsequent queries are fast (~100-200ms total)

## Testing

Run the test suite:

```bash
pytest tests/test_ace_mcp_server.py -v
```

Manual testing:

```python
from ace_mcp_server import handle_retrieve, handle_store
import asyncio

# Test retrieve
result = asyncio.run(handle_retrieve({"query": "coding preferences", "limit": 3}))
print(result[0].text)

# Test store
result = asyncio.run(handle_store({"content": "Test memory", "category": "DEBUGGING"}))
print(result[0].text)
```

## Integration with Copilot Instructions

Add to your `copilot-instructions.md`:

```markdown
# ACE Memory System [P0-M]

## ACE MCP Tools
| Tool | When to Use |
|------|-------------|
| `mcp_ace_ace_retrieve` | Before starting tasks, when encountering issues |
| `mcp_ace_ace_store` | After learning something, when user corrects you |

## Mandatory ACE Usage
- User says "recurring|again|same issue" → ace_retrieve first
- User expresses preference "I prefer|always|never" → ace_store
- Complete complex task → store lessons learned
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ACE_QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `ACE_UNIFIED_COLLECTION` | `ace_memories_hybrid` | Collection name |
| `ACE_EMBEDDING_URL` | `http://localhost:1234` | Embedding server |
| `ACE_EMBEDDING_MODEL` | `text-embedding-qwen3-embedding-8b` | Model name |
| `ACE_EMBEDDING_DIM` | `4096` | Vector dimension |

## Auto-Indexing and File Watching

### Background File Watcher Daemon

ACE includes a persistent file watcher daemon that automatically reindexes your code when files change.

**Auto-Start Behavior:**
- **SessionStart Hook**: Automatically starts watcher if workspace is onboarded
- **Onboarding Hook**: Starts watcher after initial indexing completes
- Works for both Claude Code hooks and MCP server

**Daemon Commands:**
```bash
# Check if watcher is running
python ace/file_watcher_daemon.py status /path/to/workspace

# Start watcher manually
python ace/file_watcher_daemon.py start /path/to/workspace

# Stop watcher
python ace/file_watcher_daemon.py stop /path/to/workspace

# List all watched workspaces
python ace/file_watcher_daemon.py list
```

**File Persistence:**
- `.ace/.watcher.pid` - Process ID for lifecycle management
- `.ace/.watcher.log` - Daemon activity log
- `~/.claude/.ace/.watched_workspaces` - Tracked workspaces

**How It Works:**
```
┌─────────────────────────────────────────────────────────────────────┐
│                         ACE File Watching                          │
├─────────────────────────────────────────────────────────────────────┤
│  SessionStart Hook ───► ensure_file_watcher_running()              │
│                           │                                         │
│                           ▼                                         │
│                    Check if daemon running                          │
│                           │                                         │
│                           ├─ Yes ► Nothing                           │
│                           │                                         │
│                           └─ No ► Start daemon in background        │
│                                                                     │
│  Daemon (Background Process)                                         │
│    │                                                                │
│    ├──► Monitors workspace for file changes (watchdog)             │
│    │                                                                │
│    ├──► On file created ──► index_file()                            │
│    │                                                                │
│    ├──► On file modified ──► update_file()                          │
│    │                                                                │
│    └──► On file deleted ──► remove_file()                           │
└─────────────────────────────────────────────────────────────────────┘
```

**Requirements:**
- `watchdog` package for file system monitoring (optional but recommended)
- `pip install watchdog`

### Workspace Onboarding

When a new workspace is detected, ACE automatically:

1. **Creates `.ace/.ace.json`** with workspace metadata
2. **Indexes all code files** using `CodeIndexer`
3. **Starts file watcher daemon** in background
4. **Creates workspace-specific Qdrant collection** (e.g., `my-project_code_context`)

**Auto-Onboarding Triggers:**
- First `ace_retrieve` call in MCP server
- First `UserPromptSubmit` hook invocation in Claude Code
- Manual call to `ace_onboard` MCP tool

## Security Notes

- ACE stores memories locally in Qdrant
- No data is sent to external services (except embedding server if configured)
- MCP uses stdio transport (no network exposure)
- File watcher daemon runs as local subprocess only
- Consider access controls for multi-user environments
