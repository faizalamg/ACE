# Quick Start: ACE with Claude Code CLI

Get automatic learning from Claude Code CLI in **2 minutes**.

## ⚡ Installation

```powershell
# 1. Install ACE framework
pip install ace-framework

# 2. Set your API key
$env:OPENAI_API_KEY = "your-key-here"
```

That's it! The hooks are already configured in `.claude/settings.json`.

## 🎯 How to Use

### Just Start Claude Code CLI Normally!

```powershell
# That's literally it
claude
```

The ACE hooks run automatically:
- ✅ **SessionStart**: Loads learned strategies
- ✅ **UserPromptSubmit**: Injects relevant context for each prompt
- ✅ **PostToolUse**: Learns from successful Write/Edit operations
- ✅ **Stop**: Logs session statistics

### What You'll See

**On session start:**
```
📚 ACE Learned Strategies

## Previously Helpful Strategies:
- Use TypeScript for new files
- Add error handling with try-catch
...
```

**When you submit prompts:**
```
> "edit main.py to add error handling"

📚 Relevant learned strategies for this task:
- Always use try-except-finally pattern
- Log errors with structured logging
```

**After successful edits:**
```
✅ ACE learned from Edit operation
```

## 📊 Monitor Learning

```powershell
# View stats
python .vscode/ace_helper.py stats

# View playbook
cat .claude/ace_playbook.json

# View session log
cat .claude/ace_sessions.log
```

## 🔧 Troubleshooting

### Check hooks are registered:

```powershell
claude
> /hooks
```

Should show:
- SessionStart: ace_session_start.py
- UserPromptSubmit: ace_inject_context.py
- PostToolUse: ace_learn_from_edit.py
- Stop: ace_session_end.py

### Test hooks manually:

```powershell
# Test session start
python .claude/hooks/ace_session_start.py

# Test prompt injection
echo '{"prompt": "test"}' | python .claude/hooks/ace_inject_context.py
```

### Enable debug mode:

```powershell
claude --debug
```

## 📚 Full Documentation

See [CLAUDE_CODE_INTEGRATION.md](./CLAUDE_CODE_INTEGRATION.md) for:
- How hooks work
- Configuration options
- Advanced usage
- Troubleshooting
- Best practices

## 🎉 That's All!

No wrappers, no scripts to remember - just use `claude` normally and ACE learns automatically! 🚀
