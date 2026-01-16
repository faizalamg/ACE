#!/usr/bin/env python3
"""
Test ACE VS Code Integration Setup
Verifies all components are working correctly
"""

import sys
import os
from pathlib import Path

def test_ace_import():
    """Test ACE framework import."""
    print("🔍 Testing ACE framework import...")
    try:
        import ace
        print("   ✅ ACE framework imported successfully")
        return True
    except ImportError as e:
        print(f"   ❌ Failed to import ACE: {e}")
        print("   💡 Fix: pip install ace-framework")
        return False

def test_api_key():
    """Test API key configuration."""
    print("\n🔍 Testing API key configuration...")
    
    # Check for API key in environment or .env
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    
    if api_key and api_key != "your-openai-api-key-here":
        print("   ✅ API key found")
        return True
    else:
        print("   ⚠️ No API key found")
        print("   💡 Fix: export OPENAI_API_KEY='your-key-here'")
        print("   💡 Or: Add to .env file")
        return False

def test_ace_helper():
    """Test ACE helper script."""
    print("\n🔍 Testing ACE helper script...")
    
    helper_path = Path(".vscode/ace_helper.py")
    if not helper_path.exists():
        print("   ❌ ACE helper not found")
        print("   💡 File should be at: .vscode/ace_helper.py")
        return False
    
    try:
        # Test import
        sys.path.insert(0, str(Path(".vscode")))
        from ace_helper import VSCodeACEHelper
        
        # Test instantiation
        helper = VSCodeACEHelper()
        stats = helper.get_stats()
        
        print(f"   ✅ ACE helper working")
        print(f"   📊 Current strategies: {stats['total_strategies']}")
        return True
    except Exception as e:
        print(f"   ❌ ACE helper error: {e}")
        return False

def test_claude_wrapper():
    """Test Claude wrapper script."""
    print("\n🔍 Testing Claude wrapper script...")
    
    wrapper_path = Path("claude_with_ace.py")
    if not wrapper_path.exists():
        print("   ❌ Claude wrapper not found")
        print("   💡 File should be at: claude_with_ace.py")
        return False
    
    print("   ✅ Claude wrapper exists")
    print("   💡 Test: python claude_with_ace.py 'echo test'")
    return True

def test_vscode_config():
    """Test VS Code configuration files."""
    print("\n🔍 Testing VS Code configuration...")
    
    files = {
        ".vscode/settings.json": "VS Code settings",
        ".vscode/tasks.json": "VS Code tasks",
        ".vscode/daily_summary.py": "Daily summary script"
    }
    
    all_exist = True
    for file_path, description in files.items():
        if Path(file_path).exists():
            print(f"   ✅ {description}")
        else:
            print(f"   ⚠️ Missing: {description}")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 ACE VS Code Integration Test")
    print("=" * 60)
    
    results = {
        "ACE Framework": test_ace_import(),
        "API Key": test_api_key(),
        "ACE Helper": test_ace_helper(),
        "Claude Wrapper": test_claude_wrapper(),
        "VS Code Config": test_vscode_config()
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
        print("\n🎉 All tests passed! You're ready to use ACE with VS Code.")
        print("\n📚 Next steps:")
        print("   1. Try: python claude_with_ace.py 'your prompt here'")
        print("   2. View stats: python .vscode/ace_helper.py stats")
        print("   3. Read guide: QUICKSTART_VSCODE.md")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues above.")
        print("\n📚 See QUICKSTART_VSCODE.md for setup instructions")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
