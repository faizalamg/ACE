#!/usr/bin/env python3
"""Analyze ACE vs ThatOtherContextEngine gaps to identify retrieval improvements needed."""

from ace.code_retrieval import CodeRetrieval

# Test queries where ThatOtherContextEngine might be winning
ANALYSIS_QUERIES = [
    # (label, query, expected_top_file_from_ThatOtherContextEngine)
    ("embedding", "embedding vector generation voyage-code-3 model", "ace/code_retrieval.py"),
    ("deduplication", "deduplicate overlapping results similarity", "ace/deduplication.py"),
    ("reranker", "cross encoder reranker BGE ranking", "rag_training/optimizations/v1_cross_encoder_rerank.py"),
    ("dense search", "dense vector search cosine similarity", "ace/code_retrieval.py"),
    ("sparse BM25", "sparse vector BM25 term frequency", "rag_training/optimizations/v8_bm25_hybrid.py"),
    ("file watcher", "file watcher daemon monitor changes", "ace/code_indexer.py"),
]


def normalize_path(path: str) -> str:
    """Normalize path for comparison."""
    return path.replace("\\", "/").lower()


def analyze_gaps():
    """Analyze where ACE differs from ThatOtherContextEngine's expected top result."""
    r = CodeRetrieval()
    
    print("=" * 80)
    print("ACE vs ThatOtherContextEngine GAP ANALYSIS")
    print("=" * 80)
    
    gaps = []
    
    for label, query, expected_top in ANALYSIS_QUERIES:
        results = r.search(query, limit=5)
        
        ace_top = results[0]["file_path"] if results else "NO RESULTS"
        ace_top_score = results[0]["score"] if results else 0
        
        # Check if ACE found the expected file
        expected_norm = normalize_path(expected_top)
        ace_has_expected = any(
            expected_norm in normalize_path(res["file_path"]) 
            for res in results
        )
        expected_rank = -1
        expected_score = 0
        for i, res in enumerate(results, 1):
            if expected_norm in normalize_path(res["file_path"]):
                expected_rank = i
                expected_score = res["score"]
                break
        
        # Determine winner
        ace_top_norm = normalize_path(ace_top)
        if expected_norm in ace_top_norm:
            winner = "ACE"
            reason = "Top result matches ThatOtherContextEngine's"
        elif ace_has_expected:
            winner = "TIE"
            reason = f"ThatOtherContextEngine's top at rank {expected_rank} (score {expected_score:.3f})"
        else:
            winner = "ThatOtherContextEngine"
            reason = "ACE doesn't have ThatOtherContextEngine's top in top 5"
            gaps.append((label, query, expected_top, ace_top, results))
        
        print(f"\n### {label.upper()}")
        print(f"Query: {query}")
        print(f"Expected (ThatOtherContextEngine): {expected_top}")
        print(f"ACE Top: {ace_top} [{ace_top_score:.3f}]")
        print(f"Winner: {winner} - {reason}")
        
        # Show all results
        print("ACE Results:")
        for i, res in enumerate(results, 1):
            marker = "*" if expected_norm in normalize_path(res["file_path"]) else " "
            print(f"  {marker}{i}. [{res['score']:.3f}] {res['file_path']}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total queries: {len(ANALYSIS_QUERIES)}")
    print(f"Gaps (ThatOtherContextEngine wins): {len(gaps)}")
    
    if gaps:
        print("\n### GAPS TO FIX:")
        for label, query, expected, actual, results in gaps:
            print(f"  - {label}: Expected '{expected}', got '{actual}'")


if __name__ == "__main__":
    analyze_gaps()
