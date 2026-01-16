# Claude Code ACE Integration Guide

This guide explains how to set up ACE (Agentic Context Engine) hooks for Claude Code, enabling automatic memory retrieval, intelligent learning from code edits, and seamless context management.

## Overview

ACE hooks provide:

| Feature | Description | Hook |
|---------|-------------|------|
| **Memory Retrieval** | Query unified memory before tasks | `ace_inject_context.py` |
| **Code + Memory Blended Search** | Semantic code search + memory retrieval | ACE MCP server |
| **Learning from Edits** | Auto-extract patterns from code changes | `ace_learn_from_edit.py` |
| **Session Context** | Load learned patterns at session start | `ace_session_start.py` |
| **Debug Loop Prevention** | Detect repetitive failure patterns | `ace_detect_debug_loop.py` |
| **Display Results** | Show retrieved memories inline | `ace_display_retrieve_results.py` |
| **P0 Protocol** | Enforce ACE-first workflow | `p0_protocol.py` |

## Prerequisites

### 1. Install ACE Framework

```bash
# Install from PyPI
pip install ace-framework

# Or install from source (for development)
git clone https://github.com/kayba-ai/agentic-context-engine
cd agentic-context-engine
pip install -e .
```

### 2. Start Qdrant Vector Database

```bash
# Docker (recommended)
docker run -d -p 6333:6333 qdrant/qdrant

# Or download binary
# https://github.com/qdrant/qdrant/releases
```

### 3. Configure Embedding Server

ACE needs an embedding server for semantic search. Two options:

**Option A: LM Studio (Local)**
```bash
# Download LM Studio: https://lmstudio.ai/
# Start server with embedding model (e.g., text-embedding-qwen3-embedding-8b)
# Default: http://localhost:1234
```

**Option B: Voyage AI (Cloud)**
```bash
# Get API key: https://voyageai.com/
export VOYAGE_API_KEY="your-api-key"
```

### 4. Configure Claude Desktop Settings

Add to `~/.claude/settings.json`:

```json
{
  "env": {
    "QDRANT_URL": "http://localhost:6333",
    "ACE_EMBEDDING_URL": "http://192.168.10.64:1234",
    "ACE_EMBEDDING_MODEL": "text-embedding-qwen3-embedding-8b",
    "VOYAGE_API_KEY": "your-voyage-api-key-if-using"
  }
}
```

## Hook Installation

### Quick Install (Copy All Hooks)

```bash
# Create hooks directory if it doesn't exist
mkdir -p ~/.claude/hooks

# Copy all ACE hooks from ACE repository
cp agentic-context-engine/hooks/*.py ~/.claude/hooks/

# Or copy specific hooks (see individual setup below)
```

### Individual Hook Setup

#### 1. Memory Retrieval Hook (`ace_inject_context.py`)

**Purpose**: Injects relevant memories from Qdrant before each prompt.

**Install**:
```bash
cp ace_inject_context.py ~/.claude/hooks/
```

**Configuration** (edit the file):
```python
# Line 95-98: Configure Qdrant and embedding URLs
unified_index = UnifiedMemoryIndex(
    qdrant_url="http://localhost:6333",
    embedding_url="http://192.168.10.64:1234"  # Your LM Studio URL
)
```

#### 2. Learning from Edits Hook (`ace_learn_from_edit.py`)

**Purpose**: Extracts universal coding lessons from Edit/Write operations.

**Install**:
```bash
cp ace_learn_from_edit.py ~/.claude/hooks/
```

**Configuration** (edit lines 284-301):
```python
# Line 286: Embedding URL for memory indexing
embedding_url = "http://192.168.10.64:1234"  # Your LM Studio URL

# Line 298-301: UnifiedMemoryIndex config
index = UnifiedMemoryIndex(
    qdrant_url="http://localhost:6333",
    embedding_url=embedding_url
)
```

**Dependencies**:
- Z.AI API key in `~/.claude/settings.json` for LLM analysis
- See lines 262-279 for credential loading

#### 3. Session Start Hook (`ace_session_start.py`)

**Purpose**: Loads learned patterns and executes P0 startup protocol.

**Install**:
```bash
cp ace_session_start.py ~/.claude/hooks/
```

**Configuration** (edit lines 279-282):
```python
# Line 279-282: UnifiedMemoryIndex config
index = UnifiedMemoryIndex(
    qdrant_url="http://localhost:6333",
    embedding_url="http://192.168.10.64:1234"
)
```

#### 4. Display Retrieve Results Hook (`ace_display_retrieve_results.py`)

**Purpose**: Shows retrieved memories and code files inline in chat.

**Install**:
```bash
cp ace_display_retrieve_results.py ~/.claude/hooks/
```

**No configuration needed** - parses ACE retrieve output automatically.

#### 5. P0 Protocol Hook (`p0_protocol.py`)

**Purpose**: Enforces ACE-first workflow (retrieve before coding).

**Install**:
```bash
cp p0_protocol.py ~/.claude/hooks/
```

**Disable** (if needed):
```bash
# Create flag file
touch ~/.claude/.ace-disabled

# Or set environment variable
export ACE_DISABLED=1
```

#### 6. Optional Hooks

| Hook | Purpose | Required |
|------|---------|----------|
| `ace_detect_debug_loop.py` | Detect repetitive failure patterns | No |
| `ace_learn_from_feedback.py` | Learn from explicit user feedback | No |
| `ace_session_end.py` | Session cleanup and stats | No |
| `ace_code_retrieval.py` | Legacy code search (use MCP instead) | No |

## Hook Configuration File

Each hook may need configuration for your environment. Edit these values:

```python
# Common configuration values (edit in each hook file)
QDRANT_URL = "http://localhost:6333"           # Qdrant server
EMBEDDING_URL = "http://192.168.10.64:1234"    # LM Studio or Voyage
EMBEDDING_MODEL = "text-embedding-qwen3-embedding-8b"
COLLECTION_NAME = "ace_unified"                # Qdrant collection
ZAI_API_KEY = "your-zai-api-key"              # For LLM analysis
```

## User-Level CLAUDE.md Template

Add this to `~/.claude/CLAUDE.md` (user-scoped instructions):

```markdown
# ACE Memory System [P0-M]

## ACE MCP Tools

| Tool | When to Use |
|------|-------------|
| `mcp__ace__ace_retrieve` | Before starting tasks, when encountering issues |
| `mcp__ace__ace_store` | After learning something, when user corrects you |
| `mcp__ace__ace_search` | Find specific memories by category/severity |
| `mcp__ace__ace_stats` | Check memory collection statistics |

## Mandatory ACE Usage

### Before ANY Task
ALWAYS call `mcp__ace__ace_retrieve` with a query describing what you're about to do.

### Trigger Words
These phrases require `ace_retrieve` first:
- "recurring", "again", "same issue"
- "last time", "before", "remember"
- "we discussed"

These phrases require `ace_store` after:
- "I prefer", "always", "never"
- "Remember this", "note that"

### After Learning Something
Store the lesson using `mcp__ace__ace_store`:
```
content: "Clear description of the lesson learned"
namespace: "user_prefs" or "task_strategies"
category: "PREFERENCE", "CORRECTION", "DIRECTIVE", "WORKFLOW", "ARCHITECTURE", "DEBUGGING", or "SECURITY"
severity: 1-10 (10=critical directive)
```

## Memory Storage Policy

| Memory Type | Store In | Examples |
|-------------|----------|----------|
| User preferences | ACE (user_prefs) | "Prefer functional style", "No emojis" |
| General patterns | ACE (task_strategies) | "SQL injection prevention", "async/await patterns" |
| Cross-project lessons | ACE (task_strategies) | "Always validate input", "Use parameterized queries" |
| Project-specific | ACE (project_specific) | "Uses FastAPI + Qdrant", "ACE playbook format" |

**Project-specific memories** are automatically scoped to your current workspace - they won't appear in other projects.

## What ACE Retrieve Returns

ACE retrieve returns BLENDED results:
1. **Code Context**: File paths with line numbers (Auggie-style)
2. **Memory Context**: User preferences, task strategies, project lessons

Example output:
```
[ACE Memories Retrieved:
  Log service failures with context (request ID, stack trace)
  Always validate input before database queries
]
[ACE Code Context:
  ace/unified_memory.py [789-850]
  ace/code_retrieval.py [100-150]
]
```

## P0 Protocol

The P0 protocol hook enforces:
1. `ace_retrieve` MUST be called before using Grep/Glob for semantic queries
2. Tool blocking until ACE retrieve is called
3. Violation detection and warnings
```

## MCP Server Configuration

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "ace": {
      "command": "python",
      "args": ["D:\\path\\to\\agentic-context-engine\\ace_mcp_server.py"],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "VOYAGE_API_KEY": "your-api-key-if-using"
      }
    }
  }
}
```

## Verification

### Test ACE MCP Connection

```bash
# Check if tools are available
echo 'Test ACE connection'
```

Then in Claude Code, call:
```
mcp__ace__ace_stats
```

Expected output:
```
ACE Unified Memory Statistics
Collection: ace_unified
Total Points: [number]
Status: GREEN
```

### Test Hook Loading

In Claude Code, check startup output for:
```
Running SessionStart hooks… (1/x done)
  ⎿  SessionStart:ace_session_start.py hook success
```

### Test Memory Retrieval

```
mcp__ace__ace_retrieve(query="test query", limit=3)
```

Expected output (Auggie-compatible format):
```
The following code sections were retrieved:
Path: [files]
     1	[code lines]

**User Preferences:**
[*] [PREF] [preferences]

**Task Strategies:**
[*] [STRAT] [strategies]
```

## Troubleshooting

### Issue: "ACE retrieve returns no results"

**Solutions**:
1. Check Qdrant is running: `curl http://localhost:6333/health`
2. Check collection exists: `mcp__ace__ace_stats`
3. Lower threshold: `mcp__ace__ace_retrieve(query="...", threshold=0.2)`
4. Verify embedding server is accessible

### Issue: "Hook error on PostToolUse:Edit"

**Check**:
1. Review `~/.claude/ace_learning_errors.log`
2. Verify Z.AI_API_KEY is in `~/.claude/settings.json`
3. Check embedding URL is correct in hook file

### Issue: "P0 protocol blocking tools"

**Solutions**:
1. Call `mcp__ace__ace_retrieve` first with relevant query
2. Or disable P0: `touch ~/.claude/.ace-disabled`

### Issue: "ImportError: No module named 'ace'"

**Solution**:
```bash
pip install ace-framework
# or add ACE to path in hook (see sys.path.insert line)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Claude Code                              │
├─────────────────────────────────────────────────────────────────┤
│  Hooks (UserPromptSubmit / PostToolUse / SessionStart)          │
│    │                                                              │
│    ├──► ace_inject_context.py ──► Retrieve memories             │
│    ├──► ace_learn_from_edit.py ───► Extract & store lessons     │
│    ├──► ace_session_start.py ────► Load context at startup      │
│    ├──► ace_display_retrieve_results.py ──► Show results        │
│    └──► p0_protocol.py ───────────► Enforce ACE-first           │
└─────────────────────────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────────┐
│                    ACE MCP Server (stdio)                        │
│  Tools: ace_retrieve, ace_store, ace_search, ace_stats          │
└───────────────────────────────────┬─────────────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────────┐
│                    UnifiedMemoryIndex                            │
│  - Qdrant vector storage (4096-dim hybrids)                     │
│  - Cross-encoder reranking (95%+ precision)                     │
│  - Namespace filtering (user_prefs, task_strategies, project)   │
└─────────────────────────────────────────────────────────────────┘
```

## Best Practices

1. **Before complex tasks**: Always call `ace_retrieve` first
2. **After corrections**: Store lessons in `ace_store`
3. **Memory quality**: Store only universal, actionable lessons
4. **Namespace discipline**: Use correct namespace (user_prefs vs task_strategies)
5. **Regular maintenance**: Review and prune low-quality memories

## File Locations

| File | Location |
|------|----------|
| ACE Hooks | `~/.claude/hooks/` |
| User Config | `~/.claude/CLAUDE.md` |
| MCP Config | `~/.claude.json` |
| Settings | `~/.claude/settings.json` |
| Debug Logs | `~/.claude/ace_*.log` |

## Further Reading

- `CLAUDE_CODE_ACE_INTEGRATION.md` - Detailed hook architecture
- `MCP_INTEGRATION.md` - MCP server configuration
- `COMPLETE_GUIDE_TO_ACE.md` - Full ACE framework guide
