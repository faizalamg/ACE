#!/usr/bin/env python3
"""
Setup verification script for ACE Framework development environment.
"""

import sys
import subprocess
import os
from pathlib import Path


def run_command(cmd, description, check=True):
    """Run a command and return success status."""
    print(f"\n🔍 {description}")
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check,
                              capture_output=True, text=True, cwd=Path(__file__).parent)
        if result.stdout.strip():
            print(f"✅ Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr.strip():
            print(f"   Stderr: {e.stderr.strip()}")
        return False


def main():
    print("🚀 ACE Framework Development Environment Verification")
    print("=" * 60)

    # Check Python version
    python_version = sys.version_info
    print(f"\n📋 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version < (3, 11):
        print("❌ Python 3.11+ required")
        return False
    else:
        print("✅ Python version OK")

    # Check if UV is available
    has_uv = run_command("uv --version", "Checking UV installation", False)

    # Check if in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print(f"\n📋 Virtual environment: {'✅ Active' if in_venv else '⚠️  Not active'}")

    # Test core imports
    print(f"\n🔍 Testing core imports")
    try:
        import ace
        print(f"✅ ACE Framework imported: v{getattr(ace, '__version__', 'unknown')}")

        from ace import LiteLLMClient, Generator, Reflector, Curator, Playbook
        print("✅ Core components imported")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    # Check environment variables
    print(f"\n🔍 Checking environment variables")
    env_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    for var in env_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: Set ({'*' * 8}...{value[-4:]})")
        else:
            print(f"⚠️  {var}: Not set (optional for basic testing)")

    # Test basic functionality
    print(f"\n🔍 Testing basic functionality")
    try:
        from ace.playbook import Playbook
        from ace.llm_providers.litellm_client import LiteLLMClient

        # Test playbook creation
        playbook = Playbook()
        print(f"✅ Playbook created: {len(playbook.bullets())} bullets")

        # Test LLM client (without API call)
        llm = LiteLLMClient(model="gpt-3.5-turbo")
        print(f"✅ LLM client created: {llm.model}")

    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

    # Check test suite
    test_success = run_command("uv run python -m pytest tests/ --collect-only -q",
                             "Checking test suite discovery", False)

    # Check code formatting
    format_success = run_command("uv run black --check ace/ --diff",
                               "Checking code formatting", False)

    # Check type hints
    type_success = run_command("uv run mypy ace/ --no-error-summary",
                             "Type checking", False)

    # Summary
    print(f"\n{'='*60}")
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)

    checks = [
        ("Python Version", python_version >= (3, 11)),
        ("Core Imports", True),
        ("Basic Functionality", True),
        ("Test Suite", test_success),
        ("Code Formatting", format_success),
        ("Type Checking", type_success),
    ]

    passed = sum(1 for _, success in checks if success)
    total = len(checks)

    for name, success in checks:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} {name}")

    print(f"\n{'='*60}")
    print(f"Overall: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 Setup verification PASSED! You're ready to develop ACE Framework.")
    else:
        print("⚠️  Some checks failed. Review the output above for details.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)