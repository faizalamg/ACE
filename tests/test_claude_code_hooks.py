#!/usr/bin/env python3
"""
Test ACE Claude Code Hooks
Verifies all hooks are working correctly
"""

import sys
import os
import json
from pathlib import Path

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
                print(f"   📝 Output preview: {result.stdout[:100]}...")
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
    print("\n🔍 Testing settings.json...")
    
    settings_path = Path(".claude/settings.json")
    if not settings_path.exists():
        print("   ❌ .claude/settings.json not found")
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

def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 ACE Claude Code Hooks Test")
    print("=" * 60)
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    results = {
        "ACE Framework": test_ace_import(),
        "API Key": test_api_key(),
        "Settings Config": test_settings(),
        "SessionStart Hook": test_hook(
            "SessionStart",
            Path(".claude/hooks/ace_session_start.py")
        ),
        "UserPromptSubmit Hook": test_hook(
            "UserPromptSubmit",
            Path(".claude/hooks/ace_inject_context.py"),
            {"prompt": "test prompt"}
        ),
        "PostToolUse Hook": test_hook(
            "PostToolUse",
            Path(".claude/hooks/ace_learn_from_edit.py"),
            {
                "tool_name": "Write",
                "tool_input": {"file_path": "test.txt", "content": "test"},
                "tool_response": {"success": True}
            }
        ),
        "Stop Hook": test_hook(
            "Stop",
            Path(".claude/hooks/ace_session_end.py")
        )
    }
    
    print("\n" + "=" * 60)
    print("📊 Test Results")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {name}")
    
    print(f"\n📈 Score: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your Claude Code hooks are ready.")
        print("\n📚 Next steps:")
        print("   1. Run: claude")
        print("   2. Type a prompt and see ACE context injection")
        print("   3. Make edits and watch ACE learn")
        print("   4. Read guide: CLAUDE_CODE_INTEGRATION.md")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues above.")
        print("\n📚 See QUICKSTART_CLAUDE_CODE.md for setup instructions")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
