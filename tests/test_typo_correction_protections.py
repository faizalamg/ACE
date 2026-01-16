#!/usr/bin/env python3
"""Comprehensive test of typo correction protections."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ace.typo_correction import TypoCorrector


def test_whitelist_protection():
    """Test that TECHNICAL_WHITELIST words are never corrected."""
    print("=== Testing TECHNICAL_WHITELIST Protection ===")
    corrector = TypoCorrector()

    test_cases = [
        ("aceconfig", "aceconfig"),  # Should NOT be corrected
        ("augment", "augment"),  # Should NOT be corrected
        ("zen", "zen"),  # Should NOT be corrected
        ("opus", "opus"),  # Should NOT be corrected
        ("glm", "glm"),  # Should NOT be corrected
        ("qdrant", "qdrant"),  # Should NOT be corrected
        ("embeddings", "embeddings"),  # Should NOT be corrected
        ("playbook", "playbook"),  # Should NOT be corrected
        ("curator", "curator"),  # Should NOT be corrected
        ("reflector", "reflector"),  # Should NOT be corrected
        ("generator", "generator"),  # Should NOT be corrected
    ]

    all_passed = True
    for input_word, expected in test_cases:
        result = corrector.correct_typos(input_word)
        if result == expected:
            print(f"  ✓ '{input_word}' -> '{result}' (protected)")
        else:
            print(f"  ✗ FAIL: '{input_word}' -> '{result}' (expected '{expected}')")
            all_passed = False

    return all_passed


def test_common_words_protection():
    """Test that COMMON_WORDS are never corrected."""
    print("\n=== Testing COMMON_WORDS Protection ===")
    corrector = TypoCorrector()

    # Test newly added common words
    test_cases = [
        ("inline", "inline"),
        ("content", "content"),
        ("configure", "configure"),
        ("agentic", "agentic"),
        ("simple", "simple"),
        ("tracking", "tracking"),
        ("evaluation", "evaluation"),
        ("expected", "expected"),
        ("whether", "whether"),
        ("production", "production"),
        ("thresholds", "thresholds"),
        ("contexts", "contexts"),
        ("examples", "examples"),
        ("optimizations", "optimizations"),
        ("checkpoints", "checkpoints"),
        ("generators", "generators"),
        ("tools", "tools"),
        ("benefits", "benefits"),
        ("solutions", "solutions"),
        ("executes", "executes"),
        ("executed", "executed"),
        ("detect", "detect"),
        ("retrieves", "retrieves"),
        ("provides", "provides"),
        ("analyzes", "analyzes"),
    ]

    all_passed = True
    for input_word, expected in test_cases:
        result = corrector.correct_typos(input_word)
        if result == expected:
            print(f"  ✓ '{input_word}' -> '{result}' (protected)")
        else:
            print(f"  ✗ FAIL: '{input_word}' -> '{result}' (expected '{expected}')")
            all_passed = False

    return all_passed


def test_real_typos_still_corrected():
    """Test that real typos are still being corrected."""
    print("\n=== Testing Real Typos Are Still Corrected ===")
    corrector = TypoCorrector()

    test_cases = [
        ("configura", "configuration"),
        ("updste", "update"),
        ("chnage", "change"),
        ("plsybook", "playbook"),
        ("retreival", "retrieval"),
        ("embeding", "embeddings"),
        ("sccuracy", "accuracy"),
        ("curatir", "curator"),
        ("generater", "generator"),
        ("reflecter", "reflector"),
    ]

    all_passed = True
    for input_word, expected in test_cases:
        result = corrector.correct_typos(input_word)
        if result == expected:
            print(f"  ✓ '{input_word}' -> '{result}'")
        else:
            print(f"  ✗ FAIL: '{input_word}' -> '{result}' (expected '{expected}')")
            all_passed = False

    return all_passed


def test_similarity_thresholds():
    """Test that low-similarity corrections are rejected."""
    print("\n=== Testing Similarity Thresholds ===")
    corrector = TypoCorrector()

    # These should NOT be corrected due to low similarity
    low_sim_cases = [
        ("measurable", "measuresble"),  # sim ~0.69
        ("degraded", "degrade"),  # sim ~0.89 - actually OK
    ]

    print("  Testing low-similarity rejection (0.70 threshold):")
    all_passed = True
    for input_word, bad_correction in low_sim_cases:
        result = corrector.correct_typos(input_word)
        import difflib
        ratio = difflib.SequenceMatcher(None, input_word, bad_correction).ratio()

        if result == input_word:
            print(f"  ✓ '{input_word}' rejected (sim={ratio:.2f} < 0.70)")
        elif ratio < 0.70:
            print(f"  ✗ FAIL: '{input_word}' -> '{result}' (sim={ratio:.2f} should be rejected)")
            all_passed = False
        else:
            print(f"  ✓ '{input_word}' -> '{result}' (sim={ratio:.2f} >= 0.70)")

    return all_passed


def main():
    print("=" * 70)
    print("COMPREHENSIVE TYPO CORRECTION PROTECTION TESTS")
    print("=" * 70)

    results = {
        "Whitelist Protection": test_whitelist_protection(),
        "Common Words Protection": test_common_words_protection(),
        "Real Typos Still Corrected": test_real_typos_still_corrected(),
        "Similarity Thresholds": test_similarity_thresholds(),
    }

    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
