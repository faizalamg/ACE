#!/usr/bin/env python3
"""
Memory Inventory Analysis Script
Analyzes the full memory inventory and selects diverse test samples
"""

import json
import statistics
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any

# Configuration
INVENTORY_FILE = Path(__file__).parent / "memory_inventory.json"
OUTPUT_DIR = Path(__file__).parent / "test_suite"
ANALYSIS_FILE = Path(__file__).parent / "MEMORY_ANALYSIS.md"
SELECTED_FILE = OUTPUT_DIR / "selected_memories.json"

# Selection parameters
TARGET_SAMPLES = 75  # Aim for 75 diverse samples
MIN_SAMPLES_PER_CATEGORY = 5
MIN_SAMPLES_PER_FEEDBACK_TYPE = 3


def load_inventory() -> List[Dict[str, Any]]:
    """Load memory inventory from JSON"""
    with open(INVENTORY_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_inventory(memories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Comprehensive statistical analysis"""

    # Extract payload data
    payloads = [m['payload'] for m in memories]

    # Basic counts
    total = len(memories)
    categories = [p.get('category', 'unknown') for p in payloads]
    feedback_types = [p.get('feedback_type', 'unknown') for p in payloads]
    severities = [p.get('severity', 0) for p in payloads]
    lessons = [p.get('lesson', '') for p in payloads]
    reinforcements = [p.get('reinforcement_count', 1) for p in payloads]

    # Length analysis
    lesson_lengths = [len(lesson) for lesson in lessons]
    word_counts = [len(lesson.split()) for lesson in lessons]

    # Distributions
    cat_dist = Counter(categories)
    ft_dist = Counter(feedback_types)
    sev_dist = Counter(severities)
    reinf_dist = Counter(reinforcements)

    # Context patterns
    contexts = [p.get('context', '') for p in payloads]
    file_extensions = Counter()
    for ctx in contexts:
        if ctx:
            ext = Path(ctx).suffix
            if ext:
                file_extensions[ext] += 1

    return {
        'total_memories': total,
        'category_distribution': dict(cat_dist),
        'feedback_type_distribution': dict(ft_dist),
        'severity_distribution': dict(sev_dist),
        'reinforcement_distribution': dict(reinf_dist),
        'lesson_stats': {
            'avg_characters': statistics.mean(lesson_lengths),
            'median_characters': statistics.median(lesson_lengths),
            'min_characters': min(lesson_lengths),
            'max_characters': max(lesson_lengths),
            'avg_words': statistics.mean(word_counts),
            'median_words': statistics.median(word_counts),
        },
        'file_extensions': dict(file_extensions.most_common(20)),
        'unique_categories': len(cat_dist),
        'unique_feedback_types': len(ft_dist),
    }


def generate_sample_queries(memory: Dict[str, Any]) -> List[str]:
    """Generate natural language queries that should retrieve this memory"""

    payload = memory['payload']
    lesson = payload['lesson']
    category = payload.get('category', 'unknown')
    context = payload.get('context', '')

    queries = []

    # Extract key terms from lesson
    words = lesson.lower().split()
    key_terms = [w for w in words if len(w) > 4 and w not in {
        'should', 'always', 'never', 'before', 'after', 'using', 'ensure'
    }]

    # Query 1: Direct key terms (2-3 words)
    if len(key_terms) >= 2:
        queries.append(' '.join(key_terms[:2]))

    # Query 2: Category + key concept
    if key_terms:
        queries.append(f"{category.lower()} {key_terms[0]}")

    # Query 3: Question form
    if 'validate' in lesson.lower() or 'test' in lesson.lower():
        queries.append("how to validate")
    elif 'error' in lesson.lower() or 'exception' in lesson.lower():
        queries.append("error handling")
    elif 'performance' in lesson.lower() or 'optimize' in lesson.lower():
        queries.append("performance optimization")
    elif any(word in lesson.lower() for word in ['interface', 'abstract', 'class']):
        queries.append("design patterns")

    # Query 4: File extension context if available
    if context:
        ext = Path(context).suffix.lstrip('.')
        if ext and key_terms:
            queries.append(f"{ext} {key_terms[0]}")

    # Query 5: Paraphrase key concept
    if 'chain' in lesson.lower():
        queries.append("type safety chaining")
    if 'edge case' in lesson.lower():
        queries.append("boundary conditions")
    if 'naming' in lesson.lower() or 'function name' in lesson.lower():
        queries.append("function naming conventions")

    # Remove duplicates, keep first 3
    unique_queries = []
    for q in queries:
        if q and q not in unique_queries:
            unique_queries.append(q)

    return unique_queries[:3] if unique_queries else ["related to " + category.lower()]


def select_diverse_samples(memories: List[Dict[str, Any]], stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Select diverse representative samples for testing"""

    # Organize by category and feedback_type
    by_category = defaultdict(list)
    by_feedback_type = defaultdict(list)
    by_severity = defaultdict(list)

    for mem in memories:
        payload = mem['payload']
        cat = payload.get('category', 'unknown')
        ft = payload.get('feedback_type', 'unknown')
        sev = payload.get('severity', 0)

        by_category[cat].append(mem)
        by_feedback_type[ft].append(mem)
        by_severity[sev].append(mem)

    selected = []

    # Strategy 1: Ensure minimum samples per category
    for cat, mems in by_category.items():
        sample_count = max(MIN_SAMPLES_PER_CATEGORY, int(len(mems) / len(memories) * TARGET_SAMPLES))
        selected.extend(random.sample(mems, min(sample_count, len(mems))))

    # Strategy 2: Add samples per feedback_type (if not already selected)
    for ft, mems in by_feedback_type.items():
        needed = MIN_SAMPLES_PER_FEEDBACK_TYPE
        candidates = [m for m in mems if m not in selected]
        if candidates:
            selected.extend(random.sample(candidates, min(needed, len(candidates))))

    # Strategy 3: Add high severity samples
    high_severity = [m for m in by_severity.get(9, []) + by_severity.get(8, []) if m not in selected]
    if high_severity:
        selected.extend(random.sample(high_severity, min(5, len(high_severity))))

    # Strategy 4: Add diverse lesson lengths
    all_remaining = [m for m in memories if m not in selected]
    lengths = [(len(m['payload']['lesson']), m) for m in all_remaining]
    lengths.sort(key=lambda x: x[0])

    # Add shortest
    selected.extend([m for _, m in lengths[:3]])
    # Add longest
    selected.extend([m for _, m in lengths[-3:]])

    # Strategy 5: Fill to target with random samples
    remaining_needed = TARGET_SAMPLES - len(selected)
    if remaining_needed > 0:
        candidates = [m for m in memories if m not in selected]
        if candidates:
            selected.extend(random.sample(candidates, min(remaining_needed, len(candidates))))

    # Remove duplicates
    unique_selected = []
    seen_ids = set()
    for mem in selected:
        if mem['id'] not in seen_ids:
            unique_selected.append(mem)
            seen_ids.add(mem['id'])

    return unique_selected[:TARGET_SAMPLES]


def create_test_suite(selected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create test suite with sample queries"""

    test_suite = []
    for mem in selected:
        payload = mem['payload']
        test_entry = {
            'memory_id': mem['id'],
            'content': payload['lesson'],
            'category': payload.get('category', 'unknown'),
            'feedback_type': payload.get('feedback_type', 'unknown'),
            'severity': payload.get('severity', 0),
            'context': payload.get('context', ''),
            'sample_queries': generate_sample_queries(mem)
        }
        test_suite.append(test_entry)

    return test_suite


def write_analysis_report(stats: Dict[str, Any], test_suite: List[Dict[str, Any]]):
    """Write comprehensive markdown analysis"""

    md = []
    md.append("# Memory Inventory Analysis Report\n")
    md.append(f"**Generated:** {Path(__file__).parent.name}\n")
    md.append(f"**Total Memories Analyzed:** {stats['total_memories']:,}\n")
    md.append(f"**Test Suite Size:** {len(test_suite)}\n")

    md.append("\n## 1. Overview Statistics\n")
    md.append(f"- **Unique Categories:** {stats['unique_categories']}")
    md.append(f"- **Unique Feedback Types:** {stats['unique_feedback_types']}")
    md.append(f"- **Severity Range:** {min(stats['severity_distribution'].keys())} - {max(stats['severity_distribution'].keys())}")

    md.append("\n## 2. Category Distribution\n")
    md.append("| Category | Count | Percentage |")
    md.append("|----------|-------|------------|")
    total = stats['total_memories']
    for cat, count in sorted(stats['category_distribution'].items(), key=lambda x: -x[1]):
        pct = (count / total) * 100
        md.append(f"| {cat} | {count:,} | {pct:.1f}% |")

    md.append("\n## 3. Feedback Type Distribution\n")
    md.append("| Feedback Type | Count | Percentage |")
    md.append("|---------------|-------|------------|")
    for ft, count in sorted(stats['feedback_type_distribution'].items(), key=lambda x: -x[1]):
        pct = (count / total) * 100
        md.append(f"| {ft} | {count:,} | {pct:.1f}% |")

    md.append("\n## 4. Severity Distribution\n")
    md.append("| Severity | Count | Percentage |")
    md.append("|----------|-------|------------|")
    for sev, count in sorted(stats['severity_distribution'].items()):
        pct = (count / total) * 100
        md.append(f"| {sev} | {count:,} | {pct:.1f}% |")

    md.append("\n## 5. Lesson Length Analysis\n")
    ls = stats['lesson_stats']
    md.append(f"- **Average Characters:** {ls['avg_characters']:.1f}")
    md.append(f"- **Median Characters:** {ls['median_characters']:.1f}")
    md.append(f"- **Range:** {ls['min_characters']} - {ls['max_characters']}")
    md.append(f"- **Average Words:** {ls['avg_words']:.1f}")
    md.append(f"- **Median Words:** {ls['median_words']:.1f}")

    md.append("\n## 6. Context File Types (Top 20)\n")
    md.append("| Extension | Count |")
    md.append("|-----------|-------|")
    for ext, count in sorted(stats['file_extensions'].items(), key=lambda x: -x[1])[:20]:
        md.append(f"| {ext} | {count:,} |")

    md.append("\n## 7. Test Suite Composition\n")
    test_cats = Counter(t['category'] for t in test_suite)
    md.append("| Category | Selected |")
    md.append("|----------|----------|")
    for cat, count in sorted(test_cats.items(), key=lambda x: -x[1]):
        md.append(f"| {cat} | {count} |")

    md.append("\n## 8. Sample Test Queries\n")
    md.append("Examples of generated test queries:\n")
    for i, entry in enumerate(test_suite[:10], 1):
        md.append(f"\n**{i}. {entry['category']}** (ID: {entry['memory_id']})")
        md.append(f"- **Lesson:** {entry['content'][:100]}...")
        md.append(f"- **Queries:**")
        for q in entry['sample_queries']:
            md.append(f"  - `{q}`")

    md.append("\n## 9. Recommendations\n")
    md.append("- **High-severity focus:** Ensure memories with severity 8-9 are retrievable")
    md.append("- **Category coverage:** Test suite covers all major categories")
    md.append("- **Query diversity:** Each memory has 2-3 natural language variants")
    md.append("- **Balanced distribution:** Test suite mirrors overall distribution")

    md.append("\n---\n")
    md.append("**Next Steps:**")
    md.append("1. Run baseline retrieval tests with `selected_memories.json`")
    md.append("2. Measure precision/recall for each query type")
    md.append("3. Identify weak retrieval patterns")
    md.append("4. Optimize semantic scoring weights")

    with open(ANALYSIS_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print(f"[OK] Analysis report written to: {ANALYSIS_FILE}")


def main():
    """Main execution"""
    print("Loading memory inventory...")
    memories = load_inventory()

    print(f"Analyzing {len(memories):,} memories...")
    stats = analyze_inventory(memories)

    print("Selecting diverse test samples...")
    random.seed(42)  # Reproducible selection
    selected = select_diverse_samples(memories, stats)

    print(f"Generating sample queries for {len(selected)} memories...")
    test_suite = create_test_suite(selected)

    print("Writing analysis report...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    write_analysis_report(stats, test_suite)

    print("Writing test suite...")
    with open(SELECTED_FILE, 'w', encoding='utf-8') as f:
        json.dump(test_suite, f, indent=2, ensure_ascii=False)

    print(f"[OK] Test suite written to: {SELECTED_FILE}")
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total Memories: {stats['total_memories']:,}")
    print(f"Test Suite Size: {len(test_suite)}")
    print(f"Categories: {stats['unique_categories']}")
    print(f"Feedback Types: {stats['unique_feedback_types']}")
    print(f"Average Lesson Length: {stats['lesson_stats']['avg_words']:.1f} words")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
