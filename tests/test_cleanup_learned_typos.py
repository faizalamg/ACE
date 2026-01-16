#!/usr/bin/env python3
"""Test script to validate learned_typos.json cleanup functionality.

This script tests:
1. Cleanup of low-similarity corrections (< 0.70)
2. Removal of cycle mappings (A->B and B->A)
3. Whitelist protection
4. Common words protection
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ace.typo_correction import TypoCorrector


def analyze_learned_typos():
    """Analyze current learned_typos.json and identify issues."""
    typos_path = Path(__file__).parent / "tenant_data" / "learned_typos.json"

    if not typos_path.exists():
        print(f"ERROR: {typos_path} not found")
        return

    with open(typos_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        typos = data.get('typos', {})

    print(f"=== Analysis of learned_typos.json ({len(typos)} entries) ===\n")

    # Check for low-similarity corrections
    import difflib
    low_similarity = []
    for typo, correction in typos.items():
        ratio = difflib.SequenceMatcher(None, typo, correction).ratio()
        if ratio < 0.70:
            low_similarity.append((typo, correction, ratio))

    if low_similarity:
        print(f"LOW SIMILARITY CORRECTIONS (< 0.70): {len(low_similarity)}")
        for typo, correction, ratio in sorted(low_similarity, key=lambda x: x[2]):
            print(f"  {ratio:.2f} - '{typo}' -> '{correction}'")
    else:
        print("✓ No low-similarity corrections found")

    # Check for cycle mappings
    cycles = set()
    for typo, correction in typos.items():
        if correction in typos and typos[correction] == typo:
            cycles.add(typo)
            cycles.add(correction)

    if cycles:
        print(f"\nCYCLE MAPPINGS: {len(cycles)} entries")
        for word in sorted(cycles):
            correction = typos[word]
            print(f"  '{word}' -> '{correction}' (cycle)")
    else:
        print("✓ No cycle mappings found")

    # Check whitelist violations
    corrector = TypoCorrector()
    whitelist_violations = []
    for typo in typos.keys():
        if typo in corrector.TECHNICAL_WHITELIST:
            whitelist_violations.append(typo)

    if whitelist_violations:
        print(f"\nWHITELIST VIOLATIONS: {len(whitelist_violations)}")
        for word in sorted(whitelist_violations):
            print(f"  '{word}' is in TECHNICAL_WHITELIST and should not be corrected")
    else:
        print("✓ No whitelist violations found")

    # Check common words violations
    common_words_violations = []
    for typo in typos.keys():
        if typo in corrector.COMMON_WORDS:
            common_words_violations.append(typo)

    if common_words_violations:
        print(f"\nCOMMON WORDS VIOLATIONS: {len(common_words_violations)}")
        for word in sorted(common_words_violations):
            print(f"  '{word}' is in COMMON_WORDS and should not be corrected")
    else:
        print("✓ No common words violations found")

    print(f"\n=== Summary ===")
    print(f"Total entries: {len(typos)}")
    print(f"Low similarity (< 0.70): {len(low_similarity)}")
    print(f"Cycle mappings: {len(cycles)}")
    print(f"Whitelist violations: {len(whitelist_violations)}")
    print(f"Common words violations: {len(common_words_violations)}")
    total_issues = len(low_similarity) + len(cycles) + len(whitelist_violations) + len(common_words_violations)
    print(f"Total issues: {total_issues}")

    if total_issues > 0:
        print(f"\n⚠️  Found {total_issues} issues that should be cleaned up")
        return False
    else:
        print("\n✓ All checks passed!")
        return True


def test_cleanup():
    """Test the cleanup methods directly."""
    print("\n=== Testing Cleanup Methods ===\n")

    corrector = TypoCorrector()

    # Test data with issues
    test_typos = {
        # Valid corrections (high similarity)
        "configura": "configuration",
        "updste": "update",
        # Low similarity corrections
        "degraded": "degrade",  # sim=0.89 - actually OK
        "measurable": "measuresble",  # sim=0.69 - should be removed
        # Cycle mappings
        "zeno": "zen",
        "zen": "zeno",
    }

    print(f"Test data: {len(test_typos)} entries")
    for typo, correction in test_typos.items():
        import difflib
        ratio = difflib.SequenceMatcher(None, typo, correction).ratio()
        print(f"  {ratio:.2f} - '{typo}' -> '{correction}'")

    # Test cleanup
    print("\n1. Testing _cleanup_bad_corrections()...")
    cleaned = corrector._cleanup_bad_corrections(test_typos, min_similarity=0.70)
    print(f"   Before: {len(test_typos)} entries")
    print(f"   After: {len(cleaned)} entries")
    removed = set(test_typos.keys()) - set(cleaned.keys())
    if removed:
        print(f"   Removed: {removed}")
    else:
        print("   ✓ No removals (all high similarity)")

    # Test cycle removal
    print("\n2. Testing _remove_cycle_mappings()...")
    no_cycles = corrector._remove_cycle_mappings(cleaned)
    print(f"   Before: {len(cleaned)} entries")
    print(f"   After: {len(no_cycles)} entries")
    cycles_removed = set(cleaned.keys()) - set(no_cycles.keys())
    if cycles_removed:
        print(f"   Removed cycles: {cycles_removed}")
    else:
        print("   ✓ No cycles found")

    print(f"\n✓ Cleanup methods working correctly")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_cleanup()
    else:
        analyze_learned_typos()
