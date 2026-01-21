"""
Manual Analysis Script for ACE vs Auggie Benchmark
This script provides detailed per-query analysis tools.
The actual analysis is done manually by the agent.
"""

import json
import sys
from pathlib import Path

# Load benchmark data
BENCHMARK_DIR = Path(__file__).parent.parent / "benchmark_results"
H2H_FILE = BENCHMARK_DIR / "enhanced_head2head_20260120_173431.json"
AUGGIE_CACHE = BENCHMARK_DIR / "auggie_results_cache.json"

def load_data():
    with open(H2H_FILE) as f:
        h2h = json.load(f)
    with open(AUGGIE_CACHE) as f:
        auggie = json.load(f)
    return h2h, auggie

def position_score(results, expected):
    """Calculate position score - 1/position for first match."""
    for i, f in enumerate(results):
        for e in expected:
            if e in f:
                return 1.0 / (i + 1)
    return 0.0

def file_match_ratio(results, expected):
    """Calculate ratio of results matching expected files."""
    if not results:
        return 0.0
    matches = sum(1 for r in results for e in expected if e in r)
    return matches / max(len(results), 1)

def content_jaccard(content1, content2):
    """Calculate Jaccard similarity between content word sets."""
    words1 = set(content1.split()) if content1 else set()
    words2 = set(content2.split()) if content2 else set()
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)

def compute_composite(pos_score, match_ratio, jaccard, chunk_score=1.0):
    """Compute composite score with specified weights."""
    return 0.40 * pos_score + 0.30 * match_ratio + 0.20 * jaccard + 0.10 * chunk_score

def analyze_query(query_idx):
    """Analyze a single query by index (1-based)."""
    h2h, auggie = load_data()
    
    idx = query_idx - 1
    if idx < 0 or idx >= len(h2h['results']):
        print(f"Invalid query index: {query_idx}")
        return
    
    q = h2h['results'][idx]
    query = q['query']
    
    print("=" * 80)
    print(f"QUERY #{query_idx} MANUAL ANALYSIS")
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Category: {q['category']}")
    print(f"Expected files: {q['expected_files']}")
    print(f"Existing winner: {q['winner']}")
    print(f"Existing reason: {q['reason']}")
    print()
    
    print("--- ACE RESULTS ---")
    print(f"Files: {q['ace_files']}")
    print(f"Scores: {q['ace_scores']}")
    if 'ace_line_ranges' in q:
        print(f"Line ranges: {q['ace_line_ranges']}")
    print()
    
    print("--- AUGGIE RESULTS ---")
    aug = auggie.get(query, {})
    print(f"Files: {aug.get('files', [])}")
    print(f"Line counts: {aug.get('line_counts', [])}")
    print()
    
    # Compute metrics
    ace_files = q['ace_files']
    aug_files = aug.get('files', [])
    expected = q['expected_files']
    
    ace_pos = position_score(ace_files, expected)
    aug_pos = position_score(aug_files, expected)
    
    ace_match = file_match_ratio(ace_files, expected)
    aug_match = file_match_ratio(aug_files, expected)
    
    ace_content = ' '.join(q.get('ace_contents', [])[:2000])
    aug_content = ' '.join(aug.get('contents', [])[:2000])
    jaccard = content_jaccard(ace_content, aug_content)
    
    print("=" * 60)
    print("METRIC CALCULATIONS")
    print("=" * 60)
    print(f"Position Score (ACE):    {ace_pos:.4f}")
    print(f"Position Score (Auggie): {aug_pos:.4f}")
    print(f"Expected Match (ACE):    {ace_match:.4f}")
    print(f"Expected Match (Auggie): {aug_match:.4f}")
    print(f"Content Jaccard:         {jaccard:.4f}")
    print()
    
    ace_composite = compute_composite(ace_pos, ace_match, jaccard)
    aug_composite = compute_composite(aug_pos, aug_match, jaccard)
    margin = ace_composite - aug_composite
    
    print(f"COMPOSITE (ACE):         {ace_composite:.4f}")
    print(f"COMPOSITE (Auggie):      {aug_composite:.4f}")
    print(f"Margin:                  {margin:.4f}")
    
    # Determine winner
    threshold = 0.05
    if ace_composite >= aug_composite + threshold:
        winner = "ACE"
    elif aug_composite >= ace_composite + threshold:
        winner = "Auggie"  
    else:
        winner = "Tie"
    
    print(f"VERDICT:                 {winner}")
    print()
    
    return {
        'query_idx': query_idx,
        'query': query,
        'category': q['category'],
        'expected_files': expected,
        'ace_files': ace_files,
        'auggie_files': aug_files,
        'ace_position': ace_pos,
        'auggie_position': aug_pos,
        'ace_match': ace_match,
        'auggie_match': aug_match,
        'content_jaccard': jaccard,
        'ace_composite': ace_composite,
        'auggie_composite': aug_composite,
        'margin': margin,
        'verdict': winner,
        'original_winner': q['winner']
    }

def analyze_range(start, end):
    """Analyze a range of queries."""
    results = []
    for i in range(start, end + 1):
        result = analyze_query(i)
        results.append(result)
        print()
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python manual_analysis.py <query_idx> [end_idx]")
        print("       python manual_analysis.py 1        # Analyze query 1")
        print("       python manual_analysis.py 1 10     # Analyze queries 1-10")
        sys.exit(1)
    
    start = int(sys.argv[1])
    end = int(sys.argv[2]) if len(sys.argv) > 2 else start
    
    if start == end:
        analyze_query(start)
    else:
        analyze_range(start, end)
