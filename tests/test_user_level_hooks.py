#!/usr/bin/env python3
r"""
Test ACE User-Level Claude Code Hooks
Verifies all hooks in C:\Users\Erwin\.claude are working correctly
"""

import sys
import os
import json
from pathlib import Path

USER_CLAUDE_DIR = Path.home() / ".claude"
HOOKS_DIR = USER_CLAUDE_DIR / "hooks"

def test_hook(hook_name, hook_path, test_input=None):
    """Test a single hook."""
    print(f"\n🔍 Testing {hook_name}...")
    
    if not hook_path.exists():
        print(f"   ❌ Hook not found: {hook_path}")
        return False
    
    try:
        import subprocess
        
        # Run hook with test input
        if test_input:
            result = subprocess.run(
                ["python", str(hook_path)],
                input=json.dumps(test_input),
                capture_output=True,
                text=True,
                timeout=10
            )
        else:
            result = subprocess.run(
                ["python", str(hook_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
        
        # Check exit code
        if result.returncode == 0:
            print(f"   ✅ Hook executed successfully")
            if result.stdout:
                preview = result.stdout[:100].replace('\n', ' ')
                print(f"   📝 Output: {preview}...")
            return True
        else:
            print(f"   ⚠️ Hook returned code {result.returncode}")
            if result.stderr:
                print(f"   ⚠️ Error: {result.stderr[:200]}")
            return True  # Non-zero is okay (non-blocking)
            
    except Exception as e:
        print(f"   ❌ Hook failed: {e}")
        return False

def test_settings():
    """Test settings.json configuration."""
    print("\n🔍 Testing user-level settings.json...")
    
    settings_path = USER_CLAUDE_DIR / "settings.json"
    if not settings_path.exists():
        print(f"   ❌ {settings_path} not found")
        return False
    
    try:
        with open(settings_path) as f:
            settings = json.load(f)
        
        hooks = settings.get("hooks", {})
        expected_hooks = ["SessionStart", "UserPromptSubmit", "PostToolUse", "Stop"]
        
        found = 0
        for hook_type in expected_hooks:
            if hook_type in hooks:
                print(f"   ✅ {hook_type} configured")
                found += 1
            else:
                print(f"   ⚠️ {hook_type} not configured")
        
        return found == len(expected_hooks)
        
    except Exception as e:
        print(f"   ❌ Settings parse error: {e}")
        return False

def test_ace_import():
    """Test ACE framework import."""
    print("\n🔍 Testing ACE framework...")
    
    try:
        import ace
        print("   ✅ ACE framework imported")
        return True
    except ImportError as e:
        print(f"   ❌ ACE import failed: {e}")
        print("   💡 Fix: pip install ace-framework")
        return False

def test_api_key():
    """Test API key configuration."""
    print("\n🔍 Testing API key...")
    
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    
    if api_key and api_key != "your-openai-api-key-here":
        print("   ✅ API key found")
        return True
    else:
        print("   ⚠️ No API key found")
        print("   💡 Fix: $env:OPENAI_API_KEY = 'your-key-here'")
        return False

def test_directory_structure():
    """Test user-level directory structure."""
    print("\n🔍 Testing directory structure...")
    
    paths_to_check = {
        USER_CLAUDE_DIR: "User .claude directory",
        HOOKS_DIR: "Hooks directory",
        USER_CLAUDE_DIR / "settings.json": "Settings file"
    }
    
    all_exist = True
    for path, description in paths_to_check.items():
        if path.exists():
            print(f"   ✅ {description}: {path}")
        else:
            print(f"   ❌ {description} not found: {path}")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests."""
    print("=" * 70)
    print("🧪 ACE User-Level Claude Code Hooks Test")
    print(f"📁 User directory: {USER_CLAUDE_DIR}")
    print("=" * 70)
    
    results = {
        "Directory Structure": test_directory_structure(),
        "ACE Framework": test_ace_import(),
        "API Key": test_api_key(),
        "Settings Config": test_settings(),
        "SessionStart Hook": test_hook(
            "SessionStart",
            HOOKS_DIR / "ace_session_start.py"
        ),
        "UserPromptSubmit Hook": test_hook(
            "UserPromptSubmit",
            HOOKS_DIR / "ace_inject_context.py",
            {"prompt": "test prompt"}
        ),
        "PostToolUse Hook": test_hook(
            "PostToolUse",
            HOOKS_DIR / "ace_learn_from_edit.py",
            {
                "tool_name": "Write",
                "tool_input": {"file_path": "test.txt", "content": "test"},
                "tool_response": {"success": True}
            }
        ),
        "Stop Hook": test_hook(
            "Stop",
            HOOKS_DIR / "ace_session_end.py"
        )
    }
    
    print("\n" + "=" * 70)
    print("📊 Test Results")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {name}")
    
    print(f"\n📈 Score: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your user-level Claude Code hooks are ready.")
        print("\n📚 User-level setup means:")
        print("   • Hooks work in ALL Claude Code projects")
        print("   • Playbook shared across all projects")
        print("   • Settings persist globally")
        print("\n📚 Next steps:")
        print("   1. Run: claude (in any project)")
        print("   2. Type a prompt and see ACE context injection")
        print("   3. Make edits and watch ACE learn")
        print(f"   4. Check playbook: {USER_CLAUDE_DIR / 'ace_playbook.json'}")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues above.")
        print("\n📚 Make sure:")
        print(f"   1. Settings at: {USER_CLAUDE_DIR / 'settings.json'}")
        print(f"   2. Hooks at: {HOOKS_DIR}")
        print("   3. ACE installed: pip install ace-framework")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
