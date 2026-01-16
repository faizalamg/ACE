# MCP Integration Guide for ACE

**Universal integration for VS Code, Claude Desktop, Cursor, and other MCP-compatible clients.**

This guide explains how to integrate ACE (Agentic Context Engine) as an MCP server for blended retrieval of code context and memory context.

> **Note:** This document covers the MCP server integration. For VS Code-specific features (hooks, tasks, helper scripts), see [VS Code Specific Features](#vs-code-specific-features) below.

---

## Table of Contents

1. [Universal MCP Setup](#universal-mcp-setup)
2. [Client-Specific Configuration](#client-specific-configuration)
3. [Workspace Detection](#workspace-detection)
4. [VS Code Specific Features](#vs-code-specific-features)

---

## Universal MCP Setup

### Prerequisites

```bash
# Install ACE with MCP support
pip install ace-framework mcp

# Ensure Qdrant is running (for vector storage)
docker run -d -p 6333:6333 qdrant/qdrant

# Set API keys (for embeddings and LLM)
export VOYAGE_API_KEY="your-voyage-key"  # For code embeddings
export ZAI_API_KEY="your-zai-key"        # Or OPENAI_API_KEY
```

### Client-Specific Configuration

The ACE MCP server uses the **MCP `list_roots()` protocol** for automatic workspace detection, which is supported by all MCP-compliant clients. No client-specific environment variables are required in most cases.

---

## Client-Specific Configuration

### VS Code (Copilot/Insiders)

**Location:** `%APPDATA%\Code - Insiders\User\mcp.json` (Windows) or `~/.config/Code - Insiders/User/mcp.json` (Linux/Mac)

```json
{
  "servers": {
    "ace": {
      "command": "D:\\path\\to\\agentic-context-engine\\.venv\\Scripts\\python.exe",
      "args": ["D:\\path\\to\\agentic-context-engine\\ace_mcp_server.py"],
      "type": "stdio",
      "env": {
        "VOYAGE_API_KEY": "${env:VOYAGE_API_KEY}",
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

**Note:** VS Code supports `${workspaceFolder}` but it's **not required** since ACE uses `list_roots()` for automatic detection.

### Claude Desktop

**Location:** `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows)

```json
{
  "mcpServers": {
    "ace": {
      "command": "python",
      "args": ["/path/to/agentic-context-engine/ace_mcp_server.py"],
      "env": {
        "VOYAGE_API_KEY": "${VOYAGE_API_KEY}",
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

**Note:** Claude Desktop automatically provides workspace information via the MCP protocol.

### Cursor

**Location:** Cursor Settings → MCP Servers

```json
{
  "ace": {
    "command": "python",
    "args": ["path/to/agentic-context-engine/ace_mcp_server.py"],
    "env": {
      "VOYAGE_API_KEY": "${VOYAGE_API_KEY}",
      "QDRANT_URL": "http://localhost:6333"
    }
  }
}
```

---

## Workspace Detection

ACE uses a robust fallback system for workspace detection:

| Priority | Method | Client Support |
|----------|--------|----------------|
| 1 | `MCP list_roots()` protocol | ✅ Any MCP-compliant client |
| 2 | `ACE_WORKSPACE_PATH` env var | ✅ Manual override |
| 3 | `MCP_WORKSPACE_FOLDER` env var | ✅ Legacy fallback |
| 4 | Project marker detection (`.git`, `package.json`, etc.) | ✅ Universal |
| 5 | Current working directory | ✅ Universal |

### When to Set Environment Variables

You only need to set `ACE_WORKSPACE_PATH` if:
- Your MCP client doesn't support `list_roots()`
- You want to explicitly override auto-detection
- You're running the server outside of an MCP client context

Example:
```json
"env": {
  "ACE_WORKSPACE_PATH": "/absolute/path/to/your/workspace"
}
```

---

## VS Code Specific Features

### Step 1: Install ACE in Your Project

```bash
# In your project directory
pip install ace-framework

# Or for development with all features
pip install ace-framework[all]
```

### Step 2: Create ACE Helper Script

Create `.vscode/ace_helper.py` in your workspace:

```python
#!/usr/bin/env python3
"""
ACE Helper for VS Code Integration
Automatically learns from Claude Code CLI interactions
"""

import os
import json
from pathlib import Path
from datetime import datetime
from ace import Playbook, LiteLLMClient, Reflector, Curator
from ace.roles import GeneratorOutput

class VSCodeACEHelper:
    """Wraps Claude Code CLI with automatic ACE learning."""
    
    def __init__(self, playbook_path: str = ".vscode/ace_playbook.json"):
        self.playbook_path = Path(playbook_path)
        self.playbook_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load or create playbook
        if self.playbook_path.exists():
            self.playbook = Playbook.load_from_file(str(self.playbook_path))
            print(f"📚 Loaded playbook with {len(self.playbook.bullets())} strategies")
        else:
            self.playbook = Playbook()
            print("📚 Created new playbook")
        
        # Setup ACE learning components
        llm = LiteLLMClient(
            model=os.getenv("ACE_MODEL", "gpt-4o-mini"),
            max_tokens=2048
        )
        self.reflector = Reflector(llm)
        self.curator = Curator(llm)
    
    def get_context_prompt(self) -> str:
        """Get playbook context to inject into prompts."""
        if not self.playbook.bullets():
            return ""
        
        from ace.integrations.base import wrap_playbook_context
        return f"\n\n{wrap_playbook_context(self.playbook)}"
    
    def learn_from_interaction(
        self,
        task: str,
        response: str,
        success: bool = True,
        feedback: str = ""
    ):
        """Learn from a Claude Code CLI interaction."""
        try:
            # Create adapter for ACE
            generator_output = GeneratorOutput(
                reasoning=f"Task: {task}",
                final_answer=response,
                bullet_ids=[],
                raw={"success": success}
            )
            
            # Build feedback
            if not feedback:
                feedback = f"Task {'succeeded' if success else 'failed'}"
            
            # Reflect
            reflection = self.reflector.reflect(
                question=task,
                generator_output=generator_output,
                playbook=self.playbook,
                feedback=feedback
            )
            
            # Curate
            curator_output = self.curator.curate(
                reflection=reflection,
                playbook=self.playbook,
                question_context=f"VS Code development task: {task}",
                progress=f"Learning from interaction"
            )
            
            # Update playbook
            self.playbook.apply_delta(curator_output.delta)
            
            # Save
            self.save()
            
            print(f"✅ Learned from interaction. Playbook now has {len(self.playbook.bullets())} strategies")
            
        except Exception as e:
            print(f"⚠️ Learning failed (non-critical): {e}")
    
    def save(self):
        """Save playbook to disk."""
        self.playbook.save_to_file(str(self.playbook_path))
    
    def get_stats(self) -> dict:
        """Get playbook statistics."""
        return {
            "total_strategies": len(self.playbook.bullets()),
            "playbook_stats": self.playbook.stats(),
            "location": str(self.playbook_path)
        }

# CLI interface for manual use
if __name__ == "__main__":
    import sys
    
    helper = VSCodeACEHelper()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "context":
            # Print context for injecting into prompts
            print(helper.get_context_prompt())
        
        elif command == "learn":
            # Learn from interaction
            if len(sys.argv) >= 4:
                task = sys.argv[2]
                response = sys.argv[3]
                success = sys.argv[4].lower() == "true" if len(sys.argv) > 4 else True
                helper.learn_from_interaction(task, response, success)
            else:
                print("Usage: ace_helper.py learn <task> <response> [success]")
        
        elif command == "stats":
            # Show statistics
            stats = helper.get_stats()
            print(json.dumps(stats, indent=2))
        
        else:
            print(f"Unknown command: {command}")
            print("Commands: context, learn, stats")
    else:
        # Show current stats
        stats = helper.get_stats()
        print(f"📊 ACE Statistics:")
        print(f"  • Total strategies: {stats['total_strategies']}")
        print(f"  • Location: {stats['location']}")
        print(f"\nRun with 'context', 'learn', or 'stats' command")
```

### Step 3: Configure VS Code Settings

Create/update `.vscode/settings.json`:

```json
{
  "ace.enabled": true,
  "ace.playbookPath": ".vscode/ace_playbook.json",
  "ace.autoLearn": true,
  "ace.model": "gpt-4o-mini",
  
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  
  "files.associations": {
    "ace_playbook.json": "json"
  },
  
  "files.watcherExclude": {
    "**/.git/objects/**": true,
    "**/.git/subtree-cache/**": true,
    "**/node_modules/**": true,
    "**/.vscode/ace_playbook.json": false
  }
}
```

### Step 4: Create Tasks for ACE Operations

Create/update `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "ACE: Show Context",
      "type": "shell",
      "command": "python .vscode/ace_helper.py context",
      "problemMatcher": [],
      "presentation": {
        "reveal": "always",
        "panel": "shared"
      }
    },
    {
      "label": "ACE: Show Stats",
      "type": "shell",
      "command": "python .vscode/ace_helper.py stats",
      "problemMatcher": [],
      "presentation": {
        "reveal": "always",
        "panel": "shared"
      }
    },
    {
      "label": "ACE: Clear Playbook",
      "type": "shell",
      "command": "rm -f .vscode/ace_playbook.json && echo 'Playbook cleared'",
      "windows": {
        "command": "if exist .vscode\\ace_playbook.json del .vscode\\ace_playbook.json && echo Playbook cleared"
      },
      "problemMatcher": [],
      "presentation": {
        "reveal": "always",
        "panel": "shared"
      }
    }
  ]
}
```

---

## Automatic Integration Patterns

### Pattern 1: Git Commit Hook Learning

Learn from successful commits (code that passed your review):

Create `.git/hooks/post-commit`:

```bash
#!/bin/bash
# Learn from each commit

COMMIT_MSG=$(git log -1 --pretty=%B)
CHANGED_FILES=$(git diff --name-only HEAD~1 HEAD)

# Extract task from commit message
TASK="$COMMIT_MSG"

# Learn from this successful change
python .vscode/ace_helper.py learn \
  "$TASK" \
  "Changed files: $CHANGED_FILES" \
  "true"

echo "✅ ACE learned from this commit"
```

Make it executable:
```bash
chmod +x .git/hooks/post-commit
```

### Pattern 2: Test Success Learning

Learn when tests pass:

Create `.vscode/test_with_learning.py`:

```python
#!/usr/bin/env python3
"""Run tests and learn from results."""

import subprocess
import sys
from pathlib import Path

# Import ACE helper
sys.path.insert(0, str(Path(__file__).parent))
from ace_helper import VSCodeACEHelper

def run_tests_with_learning():
    """Run pytest and learn from results."""
    helper = VSCodeACEHelper()
    
    # Run tests
    result = subprocess.run(
        ["pytest", "-v"],
        capture_output=True,
        text=True
    )
    
    success = result.returncode == 0
    
    # Extract test summary
    output_lines = result.stdout.split('\n')
    summary = '\n'.join(output_lines[-5:])
    
    # Learn from test run
    helper.learn_from_interaction(
        task="Run test suite",
        response=summary,
        success=success,
        feedback=f"Tests {'passed' if success else 'failed'}. {summary}"
    )
    
    print(result.stdout)
    print(result.stderr)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_tests_with_learning())
```

Add to `.vscode/tasks.json`:
```json
{
  "label": "Test with ACE Learning",
  "type": "shell",
  "command": "python .vscode/test_with_learning.py",
  "group": {
    "kind": "test",
    "isDefault": true
  }
}
```

### Pattern 3: Build Success Learning

Create `.vscode/build_with_learning.py`:

```python
#!/usr/bin/env python3
"""Build project and learn from results."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ace_helper import VSCodeACEHelper

def build_with_learning():
    """Run build and learn from results."""
    helper = VSCodeACEHelper()
    
    # Run build
    result = subprocess.run(
        ["python", "-m", "build"],
        capture_output=True,
        text=True
    )
    
    success = result.returncode == 0
    
    # Learn
    helper.learn_from_interaction(
        task="Build project",
        response=result.stdout + result.stderr,
        success=success,
        feedback=f"Build {'succeeded' if success else 'failed'}"
    )
    
    print(result.stdout)
    print(result.stderr)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(build_with_learning())
```

---

## Claude Code CLI Integration

### Method 1: Wrapper Script (Recommended)

Create `claude_with_ace.py` in your workspace:

```python
#!/usr/bin/env python3
"""
Claude Code CLI wrapper with ACE learning.
Usage: python claude_with_ace.py "your prompt here"
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# Add .vscode to path for ace_helper
sys.path.insert(0, str(Path(__file__).parent / ".vscode"))
from ace_helper import VSCodeACEHelper

def claude_with_ace(prompt: str, auto_approve: bool = False):
    """Run Claude Code CLI with ACE learning."""
    
    helper = VSCodeACEHelper()
    
    # Get ACE context
    ace_context = helper.get_context_prompt()
    
    # Enhance prompt with ACE context
    if ace_context:
        enhanced_prompt = f"{prompt}{ace_context}"
        print(f"📚 Injected {len(helper.playbook.bullets())} learned strategies")
    else:
        enhanced_prompt = prompt
        print("📚 No learned strategies yet")
    
    # Run Claude Code CLI
    cmd = ["claude", "code"]
    if auto_approve:
        cmd.append("--auto-approve")
    cmd.append(enhanced_prompt)
    
    print(f"\n🤖 Running: {' '.join(cmd[:2])} \"{prompt}...\"\n")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    
    # Print output
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    # Learn from interaction
    success = result.returncode == 0
    helper.learn_from_interaction(
        task=prompt,
        response=result.stdout[:1000],  # First 1000 chars
        success=success,
        feedback=f"Claude Code CLI {'succeeded' if success else 'failed'}"
    )
    
    return result.returncode

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python claude_with_ace.py 'your prompt here'")
        print("       python claude_with_ace.py 'your prompt' --auto-approve")
        sys.exit(1)
    
    prompt = sys.argv[1]
    auto_approve = "--auto-approve" in sys.argv
    
    sys.exit(claude_with_ace(prompt, auto_approve))
```

**Usage:**
```bash
# Use instead of `claude code`
python claude_with_ace.py "Add error handling to main.py"
python claude_with_ace.py "Write tests for user service" --auto-approve
```

### Method 2: Shell Alias (Quick Access)

Add to your shell config (`~/.bashrc`, `~/.zshrc`, or PowerShell profile):

**Bash/Zsh:**
```bash
# ACE-enhanced Claude Code CLI
alias ace-claude='python $(pwd)/claude_with_ace.py'
alias acc='python $(pwd)/claude_with_ace.py'  # Short version

# Examples:
# acc "Fix the bug in auth.py"
# acc "Add logging to all functions" --auto-approve
```

**PowerShell:**
```powershell
# Add to $PROFILE
function ace-claude {
    python "$PWD\claude_with_ace.py" $args
}

Set-Alias acc ace-claude

# Usage:
# acc "Fix the bug in auth.py"
# acc "Add logging to all functions" --auto-approve
```

### Method 3: VS Code Keyboard Shortcuts

Add to `.vscode/keybindings.json`:

```json
[
  {
    "key": "ctrl+shift+a",
    "command": "workbench.action.terminal.sendSequence",
    "args": {
      "text": "python claude_with_ace.py '${selectedText}'\n"
    },
    "when": "editorHasSelection"
  },
  {
    "key": "ctrl+alt+a",
    "command": "workbench.action.tasks.runTask",
    "args": "ACE: Show Stats"
  },
  {
    "key": "ctrl+alt+c",
    "command": "workbench.action.tasks.runTask",
    "args": "ACE: Show Context"
  }
]
```

**Usage:**
1. Select text in editor (e.g., "Add error handling here")
2. Press `Ctrl+Shift+A`
3. Claude Code CLI runs with selected text + ACE context
4. ACE learns from the result

---

## Automated Learning Workflows

### Workflow 1: Continuous Learning from Development

Create `.vscode/watch_and_learn.py`:

```python
#!/usr/bin/env python3
"""
Watch for file changes and learn from successful patterns.
Run in background during development.
"""

import time
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

sys.path.insert(0, str(Path(__file__).parent))
from ace_helper import VSCodeACEHelper

class DevelopmentWatcher(FileSystemEventHandler):
    """Watch for successful development patterns."""
    
    def __init__(self):
        self.helper = VSCodeACEHelper()
        self.last_modified = {}
    
    def on_modified(self, event):
        """Learn from file modifications."""
        if event.is_directory:
            return
        
        path = Path(event.src_path)
        
        # Only track Python files
        if path.suffix != '.py':
            return
        
        # Debounce (avoid learning from same file too frequently)
        now = time.time()
        if path in self.last_modified:
            if now - self.last_modified[path] < 60:  # 60 second cooldown
                return
        
        self.last_modified[path] = now
        
        # Simple heuristic: If file was modified and no errors, it's a success
        try:
            with open(path, 'r') as f:
                content = f.read()
            
            # Learn from this successful modification
            self.helper.learn_from_interaction(
                task=f"Modified {path.name}",
                response=f"File modified successfully. Size: {len(content)} chars",
                success=True,
                feedback="File modification completed without errors"
            )
            
            print(f"✅ Learned from: {path.name}")
            
        except Exception as e:
            print(f"⚠️ Could not learn from {path.name}: {e}")

def watch_directory(directory: str = "."):
    """Start watching directory for changes."""
    event_handler = DevelopmentWatcher()
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=True)
    observer.start()
    
    print(f"👀 Watching {directory} for development patterns...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n✋ Stopped watching")
    
    observer.join()

if __name__ == "__main__":
    watch_directory()
```

**Start watching:**
```bash
pip install watchdog
python .vscode/watch_and_learn.py
```

### Workflow 2: PR Review Learning

Create `.github/workflows/ace_learn_from_pr.yml`:

```yaml
name: ACE Learn from PR

on:
  pull_request:
    types: [closed]

jobs:
  learn:
    if: github.event.pull_request.merged == true
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install ACE
        run: pip install ace-framework
      
      - name: Learn from merged PR
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python .vscode/ace_helper.py learn \
            "${{ github.event.pull_request.title }}" \
            "PR merged successfully. Changed files: ${{ github.event.pull_request.changed_files }}" \
            "true"
      
      - name: Commit updated playbook
        run: |
          git config user.name "ACE Bot"
          git config user.email "ace@example.com"
          git add .vscode/ace_playbook.json
          git commit -m "ACE: Learned from PR #${{ github.event.pull_request.number }}" || true
          git push
```

### Workflow 3: Daily Learning Summary

Create `.vscode/daily_summary.py`:

```python
#!/usr/bin/env python3
"""Generate daily learning summary."""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from ace_helper import VSCodeACEHelper

def generate_summary():
    """Generate summary of learned strategies."""
    helper = VSCodeACEHelper()
    
    bullets = helper.playbook.bullets()
    
    print(f"\n📊 ACE Daily Summary - {datetime.now().strftime('%Y-%m-%d')}")
    print(f"=" * 60)
    print(f"\n📚 Total Strategies: {len(bullets)}")
    
    # Group by section
    sections = {}
    for bullet in bullets:
        sections.setdefault(bullet.section, []).append(bullet)
    
    print(f"\n📁 Sections: {len(sections)}")
    for section, section_bullets in sorted(sections.items()):
        print(f"\n  {section.upper()}")
        for bullet in sorted(section_bullets, 
                            key=lambda b: b.helpful, 
                            reverse=True)[:5]:
            score = bullet.helpful - bullet.harmful
            print(f"    • [{bullet.id}] {bullet.content[:60]}...")
            print(f"      Score: +{bullet.helpful}/-{bullet.harmful} (net: {score})")
    
    # Top strategies
    print(f"\n🏆 Top 5 Strategies:")
    top = sorted(bullets, 
                key=lambda b: b.helpful - b.harmful, 
                reverse=True)[:5]
    
    for i, bullet in enumerate(top, 1):
        score = bullet.helpful - bullet.harmful
        print(f"  {i}. [{bullet.id}] {bullet.content[:70]}...")
        print(f"     Score: +{bullet.helpful}/-{bullet.harmful} (net: {score})")
    
    print(f"\n" + "=" * 60)

if __name__ == "__main__":
    generate_summary()
```

Add to `.vscode/tasks.json`:
```json
{
  "label": "ACE: Daily Summary",
  "type": "shell",
  "command": "python .vscode/daily_summary.py",
  "problemMatcher": []
}
```

---

## Advanced Configurations

### Configuration 1: Project-Specific Playbooks

Create `.vscode/ace_config.json`:

```json
{
  "playbooks": {
    "backend": ".vscode/playbooks/backend_ace.json",
    "frontend": ".vscode/playbooks/frontend_ace.json",
    "tests": ".vscode/playbooks/tests_ace.json"
  },
  "auto_select": {
    "*.py": "backend",
    "*.ts": "frontend",
    "test_*.py": "tests"
  },
  "learning": {
    "enabled": true,
    "min_confidence": 0.7,
    "auto_save": true
  }
}
```

### Configuration 2: Team Shared Playbook

**Option A: Git-tracked playbook**
```bash
# Track playbook in git for team sharing
git add .vscode/ace_playbook.json
git commit -m "Update team ACE playbook"
git push
```

**Option B: Remote playbook sync**
```python
# .vscode/sync_playbook.py
import requests
import json
from pathlib import Path

TEAM_PLAYBOOK_URL = "https://your-server.com/team/playbook.json"

def sync_playbook():
    """Sync with team playbook."""
    local = Path(".vscode/ace_playbook.json")
    
    # Download team playbook
    response = requests.get(TEAM_PLAYBOOK_URL)
    team_playbook = response.json()
    
    # Merge with local
    if local.exists():
        with open(local) as f:
            local_playbook = json.load(f)
        
        # Merge logic here
        merged = merge_playbooks(local_playbook, team_playbook)
    else:
        merged = team_playbook
    
    # Save merged
    with open(local, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print("✅ Synced with team playbook")

if __name__ == "__main__":
    sync_playbook()
```

### Configuration 3: Environment-Specific Models

Update `.vscode/ace_helper.py` to use different models:

```python
# In VSCodeACEHelper.__init__
model = os.getenv("ACE_MODEL")
if not model:
    # Auto-select based on environment
    if os.getenv("CI"):
        model = "gpt-4o-mini"  # Fast for CI
    elif os.getenv("PRODUCTION"):
        model = "gpt-4"  # Best quality for production
    else:
        model = "gpt-4o-mini"  # Default for development

llm = LiteLLMClient(model=model, max_tokens=2048)
```

---

## VS Code Extensions & Shortcuts

### Recommended Extensions

Install these VS Code extensions for better ACE integration:

1. **Python** (ms-python.python) - Python language support
2. **Task Runner** (sana-ajani.taskrunnercode) - Quick task execution
3. **Run on Save** (emeraldwalk.runonsave) - Auto-run ACE learning

### Run on Save Configuration

Add to `.vscode/settings.json`:

```json
{
  "emeraldwalk.runonsave": {
    "commands": [
      {
        "match": "\\.py$",
        "cmd": "python .vscode/ace_helper.py learn 'File saved: ${file}' 'Success' true"
      }
    ]
  }
}
```

### Snippets for Quick ACE Usage

Create `.vscode/ace.code-snippets`:

```json
{
  "ACE Learn from Interaction": {
    "prefix": "ace-learn",
    "body": [
      "from ace_helper import VSCodeACEHelper",
      "helper = VSCodeACEHelper()",
      "helper.learn_from_interaction(",
      "    task='${1:task_description}',",
      "    response='${2:response}',",
      "    success=${3:True},",
      "    feedback='${4:feedback}'",
      ")"
    ],
    "description": "Learn from an interaction"
  },
  "ACE Get Context": {
    "prefix": "ace-context",
    "body": [
      "from ace_helper import VSCodeACEHelper",
      "helper = VSCodeACEHelper()",
      "context = helper.get_context_prompt()",
      "# Use context in your prompt"
    ],
    "description": "Get ACE context for prompts"
  }
}
```

---

## Quick Reference

### Common Commands

```bash
# Show learned strategies
python .vscode/ace_helper.py stats

# Get context for manual prompt
python .vscode/ace_helper.py context

# Learn from manual interaction
python .vscode/ace_helper.py learn "task" "response" "true"

# Run Claude with ACE
python claude_with_ace.py "your prompt"

# Daily summary
python .vscode/daily_summary.py

# Watch and learn (background)
python .vscode/watch_and_learn.py
```

### VS Code Tasks (Ctrl+Shift+P → "Tasks: Run Task")

- **ACE: Show Context** - Display learned strategies
- **ACE: Show Stats** - Show playbook statistics
- **ACE: Clear Playbook** - Reset learning
- **ACE: Daily Summary** - Generate daily report
- **Test with ACE Learning** - Run tests + learn

### Keyboard Shortcuts

- `Ctrl+Shift+A` - Run selected text with Claude + ACE
- `Ctrl+Alt+A` - Show ACE stats
- `Ctrl+Alt+C` - Show ACE context

---

## Troubleshooting

### Issue: "ACE helper not found"

**Solution:**
```bash
# Ensure ace_helper.py exists in .vscode/
ls .vscode/ace_helper.py

# Check Python path
python -c "import sys; print(sys.path)"
```

### Issue: "No API key found"

**Solution:**
```bash
# Set API key in environment
export OPENAI_API_KEY="your-key-here"

# Or add to .env file in project root
echo "OPENAI_API_KEY=your-key-here" >> .env
```

### Issue: "Playbook not learning"

**Solution:**
```python
# Debug learning
python -c "
from ace_helper import VSCodeACEHelper
helper = VSCodeACEHelper()
print(f'Bullets before: {len(helper.playbook.bullets())}')
helper.learn_from_interaction('test', 'response', True)
print(f'Bullets after: {len(helper.playbook.bullets())}')
"
```

### Issue: "Claude Code CLI not found"

**Solution:**
```bash
# Install Claude Code CLI
npm install -g @anthropic-ai/claude-code

# Verify installation
claude --version
```

---

## MCP Server Integration (Copilot Agent Mode)

### Overview

ACE can be integrated as an MCP (Model Context Protocol) server, enabling blended retrieval of both **code context** and **memory context** directly in VS Code Copilot's Agent Mode.

### User-Scope Configuration (Recommended)

Configure ACE MCP server in your user-scope `mcp.json` to make it available across all workspaces:

**Location**: `%APPDATA%\Code - Insiders\User\mcp.json` (Windows) or `~/.config/Code - Insiders/User/mcp.json` (Linux/Mac)

```json
{
  "servers": {
    "ace": {
      "command": "D:\\path\\to\\agentic-context-engine\\.venv\\Scripts\\python.exe",
      "args": ["D:\\path\\to\\agentic-context-engine\\ace_mcp_server.py"],
      "type": "stdio",
      "disabled": false,
      "env": {
        "VOYAGE_API_KEY": "${env:VOYAGE_API_KEY}",
        "ACE_WORKSPACE_PATH": "${workspaceFolder}",
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

### Multi-Workspace Isolation (CRITICAL for Multiple VS Code Instances)

When you have **multiple VS Code windows** open on different projects, each window needs its own isolated ACE server process. The `${workspaceFolder}` variable ensures this:

```
┌─────────────────────────────────────────────────────────────────┐
│ VS Code Window 1: D:\Projects\my-app                            │
│   └── ACE Server Process (PID 1234)                            │
│       └── ACE_WORKSPACE_PATH = D:\Projects\my-app              │
│       └── Collection: my-app_code_context                       │
├─────────────────────────────────────────────────────────────────┤
│ VS Code Window 2: D:\Projects\other-project                     │
│   └── ACE Server Process (PID 5678)  ← SEPARATE PROCESS        │
│       └── ACE_WORKSPACE_PATH = D:\Projects\other-project        │
│       └── Collection: other-project_code_context                │
└─────────────────────────────────────────────────────────────────┘
```

**Key Points:**
- Each VS Code window spawns its **own** ACE server process
- `${workspaceFolder}` is interpolated **before** the server starts
- Each process gets its own `ACE_WORKSPACE_PATH` environment variable
- Code is indexed into **separate Qdrant collections** (workspace-isolated)

### Workspace Onboarding

ACE uses `.ace/.ace.json` to track which workspaces have been onboarded:

1. **First tool call** in a new workspace triggers onboarding
2. ACE creates `.ace/.ace.json` with workspace configuration
3. Code is indexed into a workspace-specific Qdrant collection
4. Subsequent calls use the existing index (no re-indexing)

**To manually re-index a workspace:**
```bash
# Delete the ACE config to trigger re-onboarding
rm -rf .ace/
# Next ace_retrieve call will re-index
```

### Dynamic Workspace Configuration

The configuration uses VS Code's predefined variables for automatic workspace detection:

| Variable | Value | Purpose |
|----------|-------|---------|
| `ACE_WORKSPACE_PATH` | `${workspaceFolder}` | Auto-detects current workspace for indexing |
| `${env:VOYAGE_API_KEY}` | Your Voyage API key | References system environment variable |

**Example**: When you open `D:\Projects\my-app`:
- `ACE_WORKSPACE_PATH` → `D:\Projects\my-app`
- Collection name → `my-app_code_context` (derived from folder name)

### How It Works

1. **Startup**: Server logs workspace path, PID, and onboarding status
2. **First Query**: ACE checks if workspace has `.ace/.ace.json`
3. **Auto-Onboard**: If not onboarded, creates config and indexes workspace
4. **Blended Retrieval**: Returns both:
   - **Code Context**: Semantic code search results (ThatOtherContextEngine-style format)
   - **Memory Context**: Relevant preferences, lessons, and directives

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `ace_retrieve` | Retrieve blended code + memory context |
| `ace_store` | Store new memories/lessons |
| `ace_search` | Search memories with filters |
| `ace_stats` | Get memory collection statistics |
| `ace_tag` | Tag memories as helpful/harmful |
| `ace_onboard` | Manually onboard a workspace |
| `ace_workspace_info` | Show current workspace configuration |

### Workspace vs User-Scope Config

| Config Location | When to Use |
|-----------------|-------------|
| `.vscode/mcp.json` (workspace) | Project-specific configuration, shared with team via git |
| User-scope `mcp.json` | Personal configuration across all workspaces |

**Note**: User-scope config with `${workspaceFolder}` variables provides the best balance - single configuration that adapts to each workspace automatically.

### Troubleshooting MCP Server

1. **Reload VS Code** after changing `mcp.json` (env vars require restart)
2. **Check MCP output**: Command Palette → "MCP: List Servers" → "Show Output"
3. **Verify Qdrant**: Ensure Qdrant is running at `localhost:6333`
4. **Check API keys**: `VOYAGE_API_KEY` must be set for code embeddings

---

## Next Steps

1. ✅ **Set up basic integration** (Steps 1-4 above)
2. 🔄 **Choose automation pattern** (git hooks, test learning, etc.)
3. 🚀 **Start using Claude with ACE** (`python claude_with_ace.py`)
4. 📊 **Monitor learning** (`python .vscode/daily_summary.py`)
5. 🎯 **Customize for your workflow** (add your own patterns)

---

## Resources

- **ACE Documentation**: [Full Guide](COMPLETE_GUIDE_TO_ACE.md)
- **Integration Guide**: [INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md)
- **Examples**: [examples/](examples/)

**Happy Learning!**

