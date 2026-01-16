#!/usr/bin/env python
"""Test query enhancement impact on retrieval quality."""

from ace.unified_memory import UnifiedMemoryIndex
from ace.query_enhancer import enhance_query, get_enhanced_query
from ace.config import get_config


def run_enhancement_comparison():
    """Compare retrieval quality with and without query enhancement."""
    config = get_config()
    index = UnifiedMemoryIndex(
        collection_name=config.qdrant.unified_collection,
        qdrant_url=config.qdrant.url,
    )
    
    # Test cases: (original_query, expected_keywords)
    TEST_CASES = [
        # Previously failing vague queries
        ("fix it", ["fix", "error", "debug", "bug"]),
        ("slow", ["slow", "performance", "optim", "latency"]),
        ("pool", ["pool", "connection"]),
        ("lock", ["lock", "deadlock", "mutex"]),
        ("token", ["token", "JWT", "auth"]),
        
        # Previously weak domain queries
        ("memory leak", ["memory", "leak"]),
        ("deadlock prevention", ["deadlock", "lock", "thread"]),
        
        # Multi-domain
        ("secure fast API", ["secure", "API", "fast", "perform"]),
    ]
    
    print("=" * 80)
    print("QUERY ENHANCEMENT IMPACT ON RETRIEVAL")
    print("=" * 80)
    print()
    
    original_relevant = 0
    enhanced_relevant = 0
    total_results = 0
    
    for original_query, keywords in TEST_CASES:
        # Get enhanced query
        enhanced = enhance_query(original_query, verbose=True)
        enhanced_query = enhanced["enhanced"]
        
        print(f"Original Query: \"{original_query}\"")
        print(f"Enhanced Query: \"{enhanced_query}\"")
        if enhanced.get("added_terms"):
            print(f"Added Terms: {enhanced['added_terms']}")
        print("-" * 60)
        
        # Retrieve with original query
        original_results = index.retrieve(query=original_query, limit=3)
        
        # Retrieve with enhanced query
        enhanced_results = index.retrieve(query=enhanced_query, limit=3)
        
        # Count relevant results for original
        print("  ORIGINAL RESULTS:")
        orig_count = 0
        for r in original_results:
            content = r.content[:100]
            is_relevant = any(kw.lower() in content.lower() for kw in keywords)
            status = "OK" if is_relevant else "MISS"
            if is_relevant:
                orig_count += 1
            print(f"    [{status}] {content}...")
        
        # Count relevant results for enhanced
        print("  ENHANCED RESULTS:")
        enh_count = 0
        for r in enhanced_results:
            content = r.content[:100]
            is_relevant = any(kw.lower() in content.lower() for kw in keywords)
            status = "OK" if is_relevant else "MISS"
            if is_relevant:
                enh_count += 1
            print(f"    [{status}] {content}...")
        
        # Update totals
        original_relevant += orig_count
        enhanced_relevant += enh_count
        total_results += max(len(original_results), len(enhanced_results))
        
        # Show improvement
        improvement = "IMPROVED" if enh_count > orig_count else "SAME" if enh_count == orig_count else "WORSE"
        print(f"\n  Result: Original={orig_count}, Enhanced={enh_count} [{improvement}]")
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    orig_precision = 100 * original_relevant / total_results if total_results > 0 else 0
    enh_precision = 100 * enhanced_relevant / total_results if total_results > 0 else 0
    
    print(f"Original Precision: {original_relevant}/{total_results} ({orig_precision:.1f}%)")
    print(f"Enhanced Precision: {enhanced_relevant}/{total_results} ({enh_precision:.1f}%)")
    print(f"Improvement: {enh_precision - orig_precision:+.1f}%")
    
    if enh_precision > orig_precision:
        print("\n[PASS] Query enhancement IMPROVES retrieval quality!")
    elif enh_precision == orig_precision:
        print("\n[NEUTRAL] Query enhancement has no effect")
    else:
        print("\n[FAIL] Query enhancement REDUCES retrieval quality")


if __name__ == "__main__":
    run_enhancement_comparison()
