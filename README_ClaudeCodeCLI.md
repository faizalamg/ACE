# ACE + Claude Code CLI Integration

**Automatic learning for Claude Code CLI using native hooks - no wrappers, no manual prompts, just use `claude` normally!**

## Quick Start (2 minutes)

```bash
# 1. Install ACE framework
pip install ace-framework

# 2. Set your API key
export ZAI_API_KEY="your-key-here"   # Recommended (Z.ai GLM-4.7)
# Or: export OPENAI_API_KEY="your-key-here"

# 3. Start Qdrant vector database
docker run -d -p 6333:6333 qdrant/qdrant

# 4. Use Claude Code CLI normally!
claude
```

The hooks run automatically. No configuration needed.

---

## How It Works

ACE uses Claude Code's native hooks system to inject learning capabilities:

| Hook | Trigger | Action |
|------|---------|--------|
| **SessionStart** | You run `claude` | Loads all learned strategies into context |
| **UserPromptSubmit** | You type a prompt | Injects relevant strategies for that task |
| **PostToolUse** | Claude edits a file | Learns from successful operations |
| **Stop** | Session ends | Logs statistics |

### What You'll See

**On session start:**
```
📚 ACE Learned Strategies

## Previously Helpful Strategies:
- Use TypeScript for new files
- Add error handling with try-catch
```

**When you submit prompts:**
```
> "edit main.py to add error handling"

📚 Relevant strategies:
- Always use try-except-finally pattern
- Log errors with structured logging
```

**After successful edits:**
```
Learning from Edit operation
```

---

## Dependencies

### Required

| Component | Purpose | Setup |
|-----------|---------|-------|
| **ACE Framework** | Learning engine | `pip install ace-framework` |
| **Qdrant** | Vector storage | `docker run -d -p 6333:6333 qdrant/qdrant` |
| **LLM Provider** | AI inference | Z.ai (default) or OpenAI |

### API Keys

| Provider | Environment Variable | Model | Notes |
|----------|---------------------|-------|-------|
| **Z.ai (Default)** | `ZAI_API_KEY` | GLM-4.7 | Best quality, 2 concurrency limit |
| OpenAI | `OPENAI_API_KEY` | gpt-4o-mini | Alternative |
| Anthropic | `ANTHROPIC_API_KEY` | claude-3-haiku | Pass model explicitly |

### Optional

| Component | Purpose | Setup |
|-----------|---------|-------|
| **Opik** | Observability & cost tracking | `pip install ace-framework[observability]` |
| **LM Studio** | Local embeddings | Download from lmstudio.ai |

---

## Configuration

### Files Created

```
.claude/
├── settings.json              # Hook configuration (auto-configured)
├── ace_playbook.json          # Learned strategies (auto-created)
├── ace_sessions.log           # Session history
└── hooks/
    ├── ace_session_start.py   # SessionStart hook
    ├── ace_inject_context.py  # UserPromptSubmit hook
    ├── ace_learn_from_edit.py # PostToolUse hook
    └── ace_session_end.py     # Stop hook
```

### Environment Variables

```bash
# Required
export ZAI_API_KEY="your-key"              # Or OPENAI_API_KEY
export QDRANT_URL="http://localhost:6333"  # Default

# Optional
export ACE_DEBUG="true"                    # Enable debug logging
export OPIK_API_KEY="your-key"             # For observability
```

---

## Usage

### Basic Usage

```bash
# Just start Claude normally
claude

# Your conversations automatically improve over time
```

### Monitor Learning

```bash
# View ACE stats
python .vscode/ace_helper.py stats

# View learned strategies
cat .claude/ace_playbook.json

# View session log
cat .claude/ace_sessions.log
```

### Test Your Setup

```bash
python test_claude_code_hooks.py
```

Expected output:
```
ACE Claude Code Hooks Test
=====================================
 ACE Framework
 API Key
 Settings Config
 SessionStart Hook
 UserPromptSubmit Hook
 PostToolUse Hook
 Stop Hook

Score: 7/7 tests passed
All tests passed! Your Claude Code hooks are ready.
```

---

## Troubleshooting

### Check hooks are registered

```bash
claude
> /hooks
```

Should show:
- SessionStart: ace_session_start.py
- UserPromptSubmit: ace_inject_context.py
- PostToolUse: ace_learn_from_edit.py
- Stop: ace_session_end.py

### Test hooks manually

```bash
# Test session start
python .claude/hooks/ace_session_start.py

# Test prompt injection
echo '{"prompt": "test"}' | python .claude/hooks/ace_inject_context.py
```

### Enable debug mode

```bash
claude --debug
```

### Common Issues

| Issue | Solution |
|-------|----------|
| "No module named ace" | Run `pip install ace-framework` |
| "Connection refused" | Start Qdrant: `docker run -d -p 6333:6333 qdrant/qdrant` |
| "Invalid API key" | Check `echo $ZAI_API_KEY` or `echo $OPENAI_API_KEY` |
| Hooks not firing | Check `.claude/settings.json` has correct paths |

---

## Security Best Practices

**NEVER commit secrets to version control!**

1. **Store API keys in environment variables** or `.env` files
2. **Add `.env` to `.gitignore`**
3. **Use secrets management** in CI/CD (GitHub Secrets, etc.)
4. **Rotate keys regularly** if exposed

Example `.env` file (add to `.gitignore`):
```bash
ZAI_API_KEY=your-actual-key
QDRANT_URL=http://localhost:6333
OPIK_API_KEY=your-opik-key
```

---

## Data Flow

```
SessionStart ──> Load playbook into context
       │
UserPromptSubmit ──> Inject relevant strategies
       │
PostToolUse ──> Learn from Write/Edit ──> Generator → Reflector → Curator
       │                                              │
       │                                   Update Playbook
       │
Stop ──> Log statistics
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Claude Code CLI                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐     ┌────────────────────────────────┐    │
│  │ .claude/     │     │        ACE Framework           │    │
│  │   hooks/     │────>│  Generator → Reflector → Curator│    │
│  │              │     └────────────────────────────────┘    │
│  │ - session    │                    │                       │
│  │ - inject     │                    ▼                       │
│  │ - learn      │     ┌────────────────────────────────┐    │
│  │ - end        │     │   Qdrant (Vector Database)     │    │
│  └──────────────┘     │   UnifiedMemoryIndex           │    │
│                       └────────────────────────────────┘    │
│                                      │                       │
│                                      ▼                       │
│                       ┌────────────────────────────────┐    │
│                       │     LLM Provider               │    │
│                       │  (Z.ai / OpenAI / Anthropic)   │    │
│                       └────────────────────────────────┘    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Related Documentation

- [Main README](./README.md) - ACE Framework overview
- [MCP Agent Guide](./README_MCPAgent.md) - For VS Code, Cursor, Claude Desktop
- [VSCODE Integration](./VSCODE_INTEGRATION.md) - VS Code specific setup
- [Contributing](./CONTRIBUTING.md) - How to contribute

---

## Summary

| What | How |
|------|-----|
| **Install** | `pip install ace-framework` |
| **Configure** | Set `ZAI_API_KEY` or `OPENAI_API_KEY` |
| **Use** | Just run `claude` normally |
| **Monitor** | `python .vscode/ace_helper.py stats` |

ACE makes Claude Code CLI learn from every session automatically - no changes to your workflow needed!
