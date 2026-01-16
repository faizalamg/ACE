# CLAUDE.md User Template for ACE

Add this content to your user-level `~/.claude/CLAUDE.md` file to enable ACE integration with Claude Code.

---

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

### Trigger Words - Must Call ace_retrieve First
- "recurring", "again", "same issue", "last time", "before", "remember"
- "we discussed", "help me understand", "how does", "why is"

### Trigger Words - Must Call ace_store After
- "I prefer", "always", "never", "Remember this", "note that"
- User provides correction or expresses preference

### After Learning Something
Store the lesson using `mcp__ace__ace_store`:
```
content: "Clear description of the lesson learned"
namespace: "user_prefs" or "task_strategies" or "project_specific"
category: "PREFERENCE", "CORRECTION", "DIRECTIVE", "WORKFLOW", "ARCHITECTURE", "DEBUGGING", "SECURITY"
severity: 1-10 (10=critical directive, 7-9=prevents bugs, 5-6=improves quality, 1-4=nice to have)
```

## Memory Storage Policy

**CRITICAL**: Store the RIGHT type of memory in the RIGHT place.

| Memory Type | Store In | Examples |
|-------------|----------|----------|
| User preferences | ACE (user_prefs) | "Prefer functional style", "No emojis", "Short explanations" |
| General coding patterns | ACE (task_strategies) | "SQL injection prevention", "async/await patterns", "Input validation" |
| Cross-project lessons | ACE (task_strategies) | "Always validate input", "Use parameterized queries" |
| Project-specific architecture | ACE (project_specific) | "Uses FastAPI + Qdrant", "ACE playbook format", "Project API endpoints" |
| Project-specific bug fixes | ACE (project_specific) | "Specific file/class fix", "Project config patterns" |

**Project-specific memories** are stored in the `project_specific` namespace and are automatically scoped to the current workspace. They will NOT appear when working in other projects.

**DO store in ACE** (universal patterns):
- "Validate regex patterns with test cases before compiling"
- "Use parameterized queries to prevent SQL injection"
- "Log authentication failures with user context for audit trails"
- "Cache expensive computations at module level, not function level"

## What ACE Retrieve Returns

ACE retrieve returns BLENDED results from TWO sources:

1. **Code Context**: File paths with line numbers (workspace-specific, Auggie-compatible format)
2. **Memory Context**: User preferences, task strategies, project lessons

Example output:
```
The following code sections were retrieved:
Path: ace/unified_memory.py
   789	class UnifiedMemoryIndex:
   790	    def __init__(self, qdrant_url=None, embedding_url=None):

Path: ace/code_retrieval.py
   100	def search(self, query, limit=10):
   101	    results = self._client.search(...)

**User Preferences:**
[*] [PREF] Use Qdrant vector database for semantic memory storage
[!] [PREF] Always validate tool server connections before executing scripts

**Task Strategies:**
[+] [STRAT] Log errors with contextual metadata (user ID, request ID, stack trace)
```

## Namespaces Explained

- `user_prefs`: Your communication style, preferences, directives
- `task_strategies`: Universal coding patterns, debugging approaches, best practices
- `project_specific`: Project architecture, config patterns (workspace-scoped, auto-isolated)
- `all`: Search all namespaces

## P0 Protocol

The P0 protocol hook enforces ACE-first workflow:
1. `ace_retrieve` MUST be called before Grep/Glob for semantic queries
2. Tools are blocked until ACE retrieve is called
3. Violations trigger warnings

### P0 Intent Detection

These patterns trigger ACE-first enforcement:
- "how does", "what happens", "why is", "where is"
- "locate", "find", "analyze", "debug", "investigate"
- "tell me about", "understand", "confused"

### Direct Action (Bypass P0)

These patterns bypass enforcement:
- "run", "execute", "build", "git commit"
- Exact file paths or symbol names

### Disable P0 (if needed)

```bash
# Create flag file
touch ~/.claude/.ace-disabled

# Or set environment variable
export ACE_DISABLED=1
```

## ACE Hooks Summary

Your Claude Code has these ACE hooks installed:

| Hook | Event | Purpose |
|------|-------|---------|
| `ace_inject_context.py` | UserPromptSubmit | Inject relevant memories before each prompt |
| `ace_learn_from_edit.py` | PostToolUse | Extract lessons from Edit/Write operations |
| `ace_session_start.py` | SessionStart | Load learned patterns at session start |
| `ace_display_retrieve_results.py` | PostToolUse | Show retrieved memories inline |
| `p0_protocol.py` | UserPromptSubmit/PreToolUse | Enforce ACE-first workflow |
| `ace_detect_debug_loop.py` | PostToolUse | Detect repetitive failure patterns |

## Quick Reference

```bash
# Check ACE status
mcp__ace__ace_stats

# Retrieve memories
mcp__ace__ace_retrieve(query="your query here", limit=5)

# Store a lesson
mcp__ace__ace_store(
    content="Lesson learned",
    namespace="task_strategies",
    category="DEBUGGING",
    severity=7
)

# Search specific category
mcp__ace__ace_search(
    query="error handling",
    category="DEBUGGING",
    min_severity=5
)
```

## Common Issues

**ACE returns no results:**
- Check Qdrant is running: `curl http://localhost:6333/health`
- Call `mcp__ace__ace_stats` to check collection has data
- Lower threshold: `mcp__ace__ace_retrieve(..., threshold=0.2)`

**Hook error on Edit:**
- Check `~/.claude/ace_learning_errors.log`
- Verify ZAI_API_KEY in `~/.claude/settings.json`

**P0 blocking tools:**
- Call `mcp__ace__ace_retrieve` first with relevant query
- Or disable: `touch ~/.claude/.ace-disabled`

**Import error:**
- Run `pip install ace-framework`
- Or add ACE to sys.path in hook file
