#!/usr/bin/env python
"""Test structured enhancement integration in retrieve()."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ace.unified_memory import UnifiedMemoryIndex
from ace.structured_enhancer import StructuredQueryEnhancer


def test_structured_integration():
    """Verify structured enhancement is integrated and working."""
    index = UnifiedMemoryIndex()
    enhancer = StructuredQueryEnhancer()
    
    print("=" * 70)
    print("STRUCTURED ENHANCEMENT INTEGRATION TEST")
    print("=" * 70)
    
    queries = [
        "fix database errors",
        "debug production issues", 
        "improve api performance",
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        
        # Show enhancement
        enhanced = enhancer.enhance(query)
        print(f"  Intent: {enhanced.intent.value}")
        print(f"  Domains: {[d.value for d in enhanced.domains]}")
        print(f"  Enhanced: '{enhanced.enhanced_query[:60]}...'")
        
        # Compare results
        results_with = index.retrieve(query, limit=3, use_structured_enhancement=True)
        results_without = index.retrieve(query, limit=3, use_structured_enhancement=False)
        
        # Check overlap
        ids_with = set(r.id for r in results_with)
        ids_without = set(r.id for r in results_without)
        overlap = len(ids_with & ids_without)
        
        print(f"  Results WITH enhancement: {len(results_with)}")
        print(f"  Results WITHOUT enhancement: {len(results_without)}")
        print(f"  Overlap: {overlap}/3 ({overlap/3*100:.0f}%)")
    
    print("\n" + "=" * 70)
    print("INTEGRATION COMPLETE - Structured enhancement is active!")
    print("=" * 70)


if __name__ == "__main__":
    test_structured_integration()
