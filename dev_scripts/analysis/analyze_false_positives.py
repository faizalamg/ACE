"""Analyze ACE false positives and memory performance."""

from ace.code_retrieval import CodeRetrieval
from ace.unified_memory import UnifiedMemoryIndex

def analyze_error_handling_query():
    """Analyze why fibonacci.py ranks high for error handling query."""
    print("=" * 60)
    print("ANALYZING: 'error handling patterns in Python with try except'")
    print("=" * 60)
    
    r = CodeRetrieval()
    results = r.search('error handling patterns in Python with try except', limit=20)
    
    print("\n=== TOP 10 RESULTS ===")
    for i, res in enumerate(results[:10], 1):
        print(f"{i}. {res['file_path']}")
        print(f"   Score: {res['score']:.4f} (original: {res['original_score']:.4f})")
        print(f"   Boost applied: {res['score'] - res['original_score']:.4f}")
    
    print("\n=== FIBONACCI.PY ANALYSIS ===")
    fib = [res for res in results if 'fibonacci' in res['file_path'].lower()]
    if fib:
        res = fib[0]
        print(f"Score: {res['score']:.4f} (original: {res['original_score']:.4f})")
        print(f"Content preview:\n{res['content'][:1000]}")
    else:
        print("Not in top 20 results")

def check_memory_performance():
    """Verify memory retrieval still works."""
    print("\n" + "=" * 60)
    print("CHECKING MEMORY PERFORMANCE")
    print("=" * 60)
    
    try:
        index = UnifiedMemoryIndex()
        
        # Test query
        memories = index.retrieve("user preferences coding style", limit=5)
        
        print(f"\nMemories found: {len(memories)}")
        for i, mem in enumerate(memories[:5], 1):
            print(f"\n{i}. [{mem.namespace.value if hasattr(mem.namespace, 'value') else mem.namespace}]")
            print(f"   Content: {mem.content[:100]}...")
            print(f"   Severity: {mem.severity}")
        
        # Test blended retrieval
        print("\n=== BLENDED RETRIEVAL TEST ===")
        r = CodeRetrieval()
        blended = r.retrieve_blended("user preferences", code_limit=3, memory_limit=3)
        print(f"Code results: {len(blended['code_results'])}")
        print(f"Memory results: {len(blended['memory_results'])}")
        
        if blended['memory_results']:
            print("\nMemory results in blended:")
            for i, mem in enumerate(blended['memory_results'], 1):
                print(f"  {i}. {mem.get('content', '')[:80]}...")
        
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_doc_ranking():
    """Check why docs rank lower than code for pattern queries."""
    print("\n" + "=" * 60)
    print("ANALYZING DOC RANKING FOR PATTERN QUERY")
    print("=" * 60)
    
    r = CodeRetrieval()
    
    # This should find docs
    results = r.search('error handling patterns', limit=20)
    
    print("\nDocs in results:")
    docs = [res for res in results if res['file_path'].endswith('.md')]
    code = [res for res in results if not res['file_path'].endswith('.md')]
    
    print(f"Code files: {len(code)}, Doc files: {len(docs)}")
    
    if docs:
        print("\nDoc rankings:")
        for doc in docs[:5]:
            print(f"  {doc['file_path']}: {doc['score']:.4f} (orig: {doc['original_score']:.4f})")
    
    if code:
        print("\nTop code rankings:")
        for c in code[:5]:
            print(f"  {c['file_path']}: {c['score']:.4f} (orig: {c['original_score']:.4f})")


if __name__ == "__main__":
    analyze_error_handling_query()
    analyze_doc_ranking()
    check_memory_performance()
