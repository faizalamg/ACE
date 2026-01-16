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
    
    print(f"\n🤖 Running: {' '.join(cmd[:2])} \"{prompt[:50]}...\"\n")
    
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
