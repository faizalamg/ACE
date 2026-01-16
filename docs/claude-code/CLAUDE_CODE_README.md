# 🎯 ACE + Claude Code CLI Integration

**Automatic learning for Claude Code CLI using native hooks!**

This integration uses Claude Code's built-in hooks system to automatically inject learned strategies and learn from successful operations - **no wrapper scripts, no manual prompts, just use `claude` normally!**

## ⚡ Quick Start

```powershell
# 1. Install ACE
pip install ace-framework

# 2. Set API key
$env:OPENAI_API_KEY = "your-key-here"

# 3. Use Claude Code CLI normally!
claude
```

**That's it!** The hooks run automatically. See [QUICKSTART_CLAUDE_CODE.md](./QUICKSTART_CLAUDE_CODE.md) for details.

## 🎯 What It Does

| Hook | Trigger | Action |
|------|---------|--------|
| **SessionStart** | You run `claude` | Loads all learned strategies into context |
| **UserPromptSubmit** | You type a prompt | Injects relevant strategies for that specific task |
| **PostToolUse** | Claude edits a file | Learns from the successful operation |
| **Stop** | Session ends | Logs statistics and session summary |

## 📁 Files Created

```
.claude/
├── settings.json              # ✅ Already configured
├── ace_playbook.json          # Auto-created on first learning
├── ace_sessions.log           # Session history
└── hooks/
    ├── ace_session_start.py   # ✅ Ready to use
    ├── ace_inject_context.py  # ✅ Ready to use
    ├── ace_learn_from_edit.py # ✅ Ready to use
    └── ace_session_end.py     # ✅ Ready to use
```

## 🧪 Test Your Setup

```powershell
python test_claude_code_hooks.py
```

Expected output:
```
🧪 ACE Claude Code Hooks Test
=====================================
✅ ACE Framework
✅ API Key
✅ Settings Config
✅ SessionStart Hook
✅ UserPromptSubmit Hook
✅ PostToolUse Hook
✅ Stop Hook

📈 Score: 7/7 tests passed
🎉 All tests passed! Your Claude Code hooks are ready.
```

## 📚 Documentation

- **[QUICKSTART_CLAUDE_CODE.md](./QUICKSTART_CLAUDE_CODE.md)** - 2-minute setup guide
- **[CLAUDE_CODE_INTEGRATION.md](./CLAUDE_CODE_INTEGRATION.md)** - Complete integration guide
- **[docs/COMPLETE_GUIDE_TO_ACE.md](./docs/COMPLETE_GUIDE_TO_ACE.md)** - ACE Framework deep dive

## 💡 Usage Example

```powershell
# Start Claude Code CLI
claude

# You see:
📚 ACE Learned Strategies

## Previously Helpful Strategies:
- Use TypeScript for new files
- Add error handling with try-catch
- Write unit tests for new functions

# You type:
> "edit main.py to add better error handling"

# Claude sees:
📚 Relevant learned strategies for this task:
- Always use try-except-finally pattern
- Log errors with structured logging
- Return error codes for CLI tools

# After Claude edits the file successfully:
✅ ACE learned from Edit operation

# The learning persists for next time!
```

## 🔧 Monitor & Manage

```powershell
# View learned strategies
python .vscode/ace_helper.py stats

# View playbook
cat .claude/ace_playbook.json

# View session history
cat .claude/ace_sessions.log

# Clear playbook (start fresh)
rm .claude/ace_playbook.json
```

## 🎛️ Configuration

Edit `.claude/settings.json` to customize:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      // Comment this out to disable per-prompt injection
      // (still loads strategies at session start)
    ],
    "PostToolUse": [
      {
        "matcher": "Write|Edit|Bash",  // Learn from more tools
        "hooks": [...]
      }
    ]
  }
}
```

## 🤔 Why This Approach?

| Alternative | Issue | Our Solution |
|-------------|-------|--------------|
| Wrapper scripts | Have to remember to use wrapper | ✅ Just use `claude` normally |
| Manual prompts | Have to copy/paste context | ✅ Automatic injection |
| External learning | Separate learning step | ✅ Learns automatically |
| Complex setup | Many files to configure | ✅ Already configured |

**Using Claude Code's native hooks = Zero friction!**

## 🚨 Important Notes

- **Project-specific learning**: Each `.claude/` directory has its own playbook
- **Non-blocking**: If hooks fail, Claude Code continues normally
- **Privacy**: All learning happens locally (only LLM API calls for reflection)
- **Commit playbook**: Share learnings with your team by committing `.claude/ace_playbook.json`

## 🎓 How It Works

```
┌─────────────────────────────────────────────────────────┐
│ You run: claude                                         │
│ ↓                                                       │
│ SessionStart hook loads learned strategies              │
│ ↓                                                       │
│ You see: 📚 ACE Learned Strategies                      │
│ ↓                                                       │
│ You type: "edit main.py"                               │
│ ↓                                                       │
│ UserPromptSubmit hook injects relevant context          │
│ ↓                                                       │
│ Claude sees your prompt + relevant strategies           │
│ ↓                                                       │
│ Claude edits the file successfully                      │
│ ↓                                                       │
│ PostToolUse hook triggers                               │
│ ↓                                                       │
│ ACE Reflector analyzes what worked                      │
│ ↓                                                       │
│ ACE Curator updates playbook                            │
│ ↓                                                       │
│ Strategy saved for next time! 🎉                        │
└─────────────────────────────────────────────────────────┘
```

## 🎉 Success Criteria

You'll know it's working when:

1. ✅ You see learned strategies at session start
2. ✅ You see relevant context injected for each prompt
3. ✅ You see "✅ ACE learned from Edit operation" after edits
4. ✅ `.claude/ace_playbook.json` grows over time
5. ✅ Strategies become more relevant to your codebase

## 🤝 Contributing

Found a bug? Have an idea? Open an issue or PR!

## 📜 License

Same as ACE Framework (see LICENSE file)

---

**Ready to try it?** Run `python test_claude_code_hooks.py` to verify your setup! 🚀
