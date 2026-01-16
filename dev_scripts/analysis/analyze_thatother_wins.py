#!/usr/bin/env python3
"""
Analyze ThatOtherContextEngine wins to understand root causes.
"""
import sys
sys.path.insert(0, '.')
from ace.code_retrieval import CodeRetrieval

def main():
    cr = CodeRetrieval()
    queries = [
        ('UnifiedMemoryIndex class search method', 'ace/unified_memory.py'),
        ('ASTChunker class parse method', 'ace/code_chunker.py'),
        ('Playbook class initialization', 'ace/playbook.py'),  # Fixed: class is Playbook not PlaybookManager
        ('HyDEGenerator class generate method', 'ace/hyde.py'),
        ('exception hierarchy', 'ace/resilience.py'),
    ]
    
    print("=" * 80)
    print("ANALYZING ThatOtherContextEngine WINS - FULL ACE RESULT QUALITY")
    print("=" * 80)
    
    for query, expected in queries:
        print(f"\n### QUERY: {query}")
        print(f"    Expected: {expected}")
        results = cr.search(query, limit=10)
        
        found_rank = None
        for i, r in enumerate(results[:10], 1):
            fp = r.get('file_path', r.get('file', 'N/A'))
            score = r.get('score', 0)
            is_expected = expected in fp or fp.endswith(expected.split('/')[-1])
            marker = " <-- EXPECTED" if is_expected else ""
            if is_expected and found_rank is None:
                found_rank = i
            print(f"    {i}. [{score:.3f}] {fp}{marker}")
        
        if found_rank:
            print(f"\n    ANALYSIS: Expected file found at rank {found_rank}")
            if found_rank == 1:
                print("    STATUS: ACE CORRECT (rank 1)")
            elif found_rank <= 3:
                print(f"    STATUS: ACE ACCEPTABLE (rank {found_rank})")
            else:
                print(f"    STATUS: ACE NEEDS IMPROVEMENT (rank {found_rank})")
        else:
            print("\n    ANALYSIS: Expected file NOT in top 10!")
            print("    STATUS: ACE NEEDS SIGNIFICANT IMPROVEMENT")

if __name__ == "__main__":
    main()
