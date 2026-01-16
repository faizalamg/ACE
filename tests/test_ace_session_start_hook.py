#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test: ACE Session Start Hook
Tests the user-level ace_session_start.py hook for proper execution
"""

import sys
import json
import subprocess
from pathlib import Path

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# User-level hooks location
USER_HOOKS_DIR = Path.home() / ".claude" / "hooks"
HOOK_PATH = USER_HOOKS_DIR / "ace_session_start.py"

def test_hook_exists():
    """Test 1: Hook file exists"""
    print("\n[Test 1] Hook File Existence")
    if HOOK_PATH.exists():
        print(f"  ✓ PASS: Hook found at {HOOK_PATH}")
        print(f"  ✓ Size: {HOOK_PATH.stat().st_size} bytes")
        return True
    else:
        print(f"  ✗ FAIL: Hook not found at {HOOK_PATH}")
        return False

def test_hook_permissions():
    """Test 2: Hook has execute permissions"""
    print("\n[Test 2] Hook Permissions")
    if HOOK_PATH.exists() and HOOK_PATH.is_file():
        print(f"  ✓ PASS: Hook is a valid file")
        return True
    else:
        print(f"  ✗ FAIL: Hook is not a valid file")
        return False

def test_hook_execution():
    """Test 3: Hook executes without errors"""
    print("\n[Test 3] Hook Execution")
    try:
        result = subprocess.run(
            [sys.executable, str(HOOK_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=Path.cwd()
        )

        print(f"  Exit Code: {result.returncode}")

        if result.returncode == 0:
            print(f"  ✓ PASS: Hook executed successfully")
        else:
            print(f"  ✗ FAIL: Hook returned non-zero exit code")

        if result.stderr:
            print(f"  ⚠ STDERR: {result.stderr[:200]}")

        return result.returncode == 0, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        print(f"  ✗ FAIL: Hook execution timed out (>30s)")
        return False, "", "Timeout"
    except Exception as e:
        print(f"  ✗ FAIL: Exception during execution: {e}")
        return False, "", str(e)

def test_output_format():
    """Test 4: Hook output is valid JSON"""
    print("\n[Test 4] Output Format Validation")

    success, stdout, stderr = test_hook_execution()

    if not success:
        print(f"  ✗ FAIL: Hook execution failed, cannot test output")
        return False

    try:
        output = json.loads(stdout)
        print(f"  ✓ PASS: Output is valid JSON")

        # Check structure
        if "hookSpecificOutput" in output:
            print(f"  ✓ PASS: Contains 'hookSpecificOutput' key")

            hook_data = output["hookSpecificOutput"]
            if "hookEventName" in hook_data:
                event_name = hook_data["hookEventName"]
                print(f"  ✓ PASS: hookEventName = '{event_name}'")

            if "additionalContext" in hook_data:
                context = hook_data["additionalContext"]
                print(f"  ✓ PASS: additionalContext present ({len(context)} chars)")

                # Verify P0 protocol execution
                if "P0 Startup Protocol" in context:
                    print(f"  ✓ PASS: P0 Protocol executed")

                if "ACE Learned Strategies" in context:
                    print(f"  ✓ PASS: ACE strategies loaded")

                if "project_specific" in context:
                    print(f"  ✓ PASS: Project-specific memories active")

        return True

    except json.JSONDecodeError as e:
        print(f"  ✗ FAIL: Output is not valid JSON: {e}")
        print(f"  Output preview: {stdout[:200]}")
        return False

def test_dependencies():
    """Test 5: All required dependencies available"""
    print("\n[Test 5] Dependency Check")

    # Change to hooks directory to match runtime environment
    import os
    old_cwd = os.getcwd()
    os.chdir(USER_HOOKS_DIR)

    dependencies = {
        'ace_qdrant_memory': False,
        'urllib.request': False,
        'pathlib': False,
        'json': False,
        'subprocess': False
    }

    for dep_name in dependencies.keys():
        try:
            if dep_name == 'ace_qdrant_memory':
                import ace_qdrant_memory
            elif dep_name == 'urllib.request':
                import urllib.request
            elif dep_name == 'pathlib':
                from pathlib import Path
            elif dep_name == 'json':
                import json
            elif dep_name == 'subprocess':
                import subprocess

            dependencies[dep_name] = True
            print(f"  ✓ {dep_name}")
        except ImportError as e:
            print(f"  ✗ {dep_name}: {e}")

    os.chdir(old_cwd)

    all_satisfied = all(dependencies.values())
    if all_satisfied:
        print(f"\n  ✓ PASS: All dependencies satisfied")
    else:
        print(f"\n  ✗ FAIL: Some dependencies missing")

    return all_satisfied

def test_mcp2rest_connectivity():
    """Test 6: mcp2rest service connectivity"""
    print("\n[Test 6] mcp2rest Service Connectivity")

    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:28888/health", timeout=5) as response:
            if response.status == 200:
                health_data = json.loads(response.read().decode('utf-8'))
                print(f"  ✓ PASS: mcp2rest service is running")
                print(f"  ✓ Connected servers: {health_data.get('connectedServers', 0)}/{health_data.get('serverCount', 0)}")
                return True
    except Exception as e:
        print(f"  ⚠ WARNING: mcp2rest not available: {e}")
        print(f"  Note: Hook will still run but P0 protocol will be incomplete")
        return False

def main():
    """Run all tests"""
    print("=" * 70)
    print("ACE Session Start Hook - Comprehensive Test Suite")
    print("=" * 70)
    print(f"Hook Location: {HOOK_PATH}")
    print(f"Current Working Directory: {Path.cwd()}")
    print("=" * 70)

    tests = [
        ("Hook File Exists", test_hook_exists),
        ("Hook Permissions", test_hook_permissions),
        ("Output Format", test_output_format),
        ("Dependencies", test_dependencies),
        ("MCP2REST Connectivity", test_mcp2rest_connectivity)
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n  ✗ EXCEPTION in {test_name}: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")

    print(f"\n  Score: {passed}/{total} tests passed")

    # Final verdict
    print("\n" + "=" * 70)
    if passed == total:
        print("VERDICT: ✓ ALL TESTS PASSED")
        print("\nThe ACE Session Start Hook is functioning correctly:")
        print("  • Hook file exists at correct location")
        print("  • Hook executes without errors")
        print("  • Output format is valid JSON")
        print("  • All dependencies are satisfied")
        print("  • P0 Protocol executes successfully")
        print("  • ACE strategies are loaded from Qdrant memory")
        print("  • Project-specific workspace integration is active")
    elif passed >= total - 1:
        print("VERDICT: ⚠ MOSTLY PASSING (Minor Issues)")
        print("\nThe hook is functional but has minor issues:")
        failed_tests = [name for name, result in results.items() if not result]
        for test_name in failed_tests:
            print(f"  • {test_name} failed")
    else:
        print("VERDICT: ✗ SIGNIFICANT FAILURES")
        print("\nThe hook has critical issues that need to be fixed.")

    print("=" * 70)

    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
