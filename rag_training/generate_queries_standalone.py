"""
Enhanced Rule-Based Query Generator for RAG Testing

Generates 12-15 diverse queries per memory entry using:
- NLP-based keyword extraction
- Template-based query generation
- Synonym expansion
- Question format variations
- Edge case generation

No LLM required - uses pure Python NLP techniques.
"""

import json
import re
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter
import hashlib


# ============================================================================
# CONFIGURATION
# ============================================================================

# Common stop words to filter out
STOP_WORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'just', 'don', 'now', 'and', 'but', 'or', 'if', 'because',
    'until', 'while', 'this', 'that', 'these', 'those', 'i', 'you',
    'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'whom',
    'your', 'yours', 'their', 'its', 'my', 'our', 'us', 'me', 'him',
    'her', 'about', 'also', 'any', 'get', 'put', 'use', 'make',
}

# Technical synonyms for expansion
TECH_SYNONYMS = {
    'error': ['exception', 'bug', 'issue', 'problem', 'failure'],
    'fix': ['resolve', 'repair', 'correct', 'patch', 'solve'],
    'implement': ['build', 'create', 'develop', 'code', 'write'],
    'validate': ['verify', 'check', 'confirm', 'ensure', 'test'],
    'config': ['configuration', 'settings', 'options', 'preferences'],
    'api': ['endpoint', 'interface', 'service', 'route'],
    'data': ['information', 'input', 'payload', 'content'],
    'function': ['method', 'procedure', 'routine', 'operation'],
    'class': ['type', 'model', 'entity', 'object'],
    'test': ['spec', 'verification', 'check', 'assertion'],
    'security': ['auth', 'protection', 'safety', 'access control'],
    'performance': ['speed', 'efficiency', 'optimization', 'latency'],
    'debug': ['troubleshoot', 'diagnose', 'investigate', 'trace'],
    'memory': ['storage', 'cache', 'buffer', 'allocation'],
    'async': ['asynchronous', 'concurrent', 'parallel', 'non-blocking'],
    'sync': ['synchronous', 'blocking', 'sequential'],
    'callback': ['handler', 'listener', 'hook', 'event'],
    'loop': ['iteration', 'cycle', 'repetition'],
    'pattern': ['practice', 'approach', 'technique', 'method'],
    'architecture': ['design', 'structure', 'layout', 'organization'],
    'centralize': ['consolidate', 'unify', 'aggregate', 'combine'],
    'isolate': ['separate', 'decouple', 'segregate', 'modularize'],
    'extract': ['pull out', 'separate', 'externalize', 'move'],
    'prevent': ['avoid', 'block', 'stop', 'disallow'],
    'ensure': ['guarantee', 'verify', 'confirm', 'make sure'],
}

# Category-specific query templates
CATEGORY_TEMPLATES = {
    'ARCHITECTURE': [
        "how to structure {concept}",
        "best practices for {concept} architecture",
        "{concept} design pattern",
        "when to {action} in software design",
        "organizing {concept} for maintainability",
    ],
    'DATA_VALIDATION': [
        "how to validate {concept}",
        "{concept} validation rules",
        "checking {concept} before processing",
        "input validation for {concept}",
        "sanitizing {concept}",
    ],
    'ERROR_HANDLING': [
        "handling {concept} errors",
        "what to do when {concept} fails",
        "{concept} exception handling",
        "error recovery for {concept}",
        "debugging {concept} issues",
    ],
    'TESTING': [
        "how to test {concept}",
        "writing tests for {concept}",
        "{concept} test strategy",
        "unit testing {concept}",
        "test coverage for {concept}",
    ],
    'SECURITY': [
        "securing {concept}",
        "{concept} security best practices",
        "protecting against {concept}",
        "authentication for {concept}",
        "preventing {concept} vulnerabilities",
    ],
    'PERFORMANCE': [
        "optimizing {concept}",
        "{concept} performance improvement",
        "speeding up {concept}",
        "reducing {concept} latency",
        "caching {concept}",
    ],
    'API_DESIGN': [
        "designing {concept} API",
        "{concept} endpoint structure",
        "REST API for {concept}",
        "API versioning {concept}",
        "{concept} request/response format",
    ],
    'WORKFLOW_PATTERN': [
        "workflow for {concept}",
        "{concept} process automation",
        "implementing {concept} pipeline",
        "orchestrating {concept}",
        "{concept} state management",
    ],
    'TOOL_USAGE': [
        "using {concept} effectively",
        "{concept} tool configuration",
        "integrating {concept}",
        "{concept} best practices",
        "automating with {concept}",
    ],
    'DEFAULT': [
        "how to {action}",
        "best way to {action}",
        "{concept} implementation",
        "when to {action}",
        "{concept} approach",
    ]
}

# Question format templates
QUESTION_FORMATS = {
    'what': [
        "what is {concept}",
        "what does {concept} mean",
        "what is the purpose of {concept}",
    ],
    'how': [
        "how to {action}",
        "how do I {action}",
        "how can I {action}",
        "how should I {action}",
    ],
    'why': [
        "why {action}",
        "why should I {action}",
        "why is {concept} important",
    ],
    'when': [
        "when to {action}",
        "when should I {action}",
    ],
    'where': [
        "where to {action}",
        "where should {concept} be placed",
    ],
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class GeneratedQuery:
    """A generated test query with metadata."""
    query: str
    category: str
    difficulty: str
    expected_rank: int = 1
    min_similarity: float = 0.5
    generation_method: str = "rule_based"


@dataclass
class EnhancedMemoryTestCase:
    """Memory with expanded test queries."""
    memory_id: int
    content: str
    category: str
    feedback_type: str
    severity: int
    context: str
    original_queries: List[str]
    generated_queries: List[GeneratedQuery] = field(default_factory=list)
    generation_timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "category": self.category,
            "feedback_type": self.feedback_type,
            "severity": self.severity,
            "context": self.context,
            "original_queries": self.original_queries,
            "generated_queries": [
                {
                    "query": q.query,
                    "category": q.category,
                    "difficulty": q.difficulty,
                    "expected_rank": q.expected_rank,
                    "min_similarity": q.min_similarity,
                    "generation_method": q.generation_method
                }
                for q in self.generated_queries
            ],
            "generation_timestamp": self.generation_timestamp,
            "total_queries": len(self.generated_queries)
        }


# ============================================================================
# NLP UTILITIES
# ============================================================================

def tokenize(text: str) -> List[str]:
    """Simple tokenization: split on non-alphanumeric, lowercase."""
    return re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*\b', text.lower())


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """Extract top keywords from text, filtering stop words."""
    tokens = tokenize(text)
    # Filter stop words and short tokens
    keywords = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    # Get most common
    counter = Counter(keywords)
    return [word for word, _ in counter.most_common(top_n)]


def extract_keyphrases(text: str) -> List[str]:
    """Extract 2-3 word phrases from text."""
    tokens = tokenize(text)
    phrases = []

    for i in range(len(tokens) - 1):
        if tokens[i] not in STOP_WORDS and tokens[i+1] not in STOP_WORDS:
            phrases.append(f"{tokens[i]} {tokens[i+1]}")

    for i in range(len(tokens) - 2):
        if (tokens[i] not in STOP_WORDS and
            tokens[i+2] not in STOP_WORDS):
            phrases.append(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")

    return phrases[:5]


def get_synonyms(word: str) -> List[str]:
    """Get technical synonyms for a word."""
    word_lower = word.lower()
    if word_lower in TECH_SYNONYMS:
        return TECH_SYNONYMS[word_lower]
    return []


def extract_action_words(text: str) -> List[str]:
    """Extract action verbs from text."""
    action_indicators = [
        'define', 'create', 'implement', 'use', 'validate', 'check',
        'ensure', 'prevent', 'avoid', 'isolate', 'extract', 'centralize',
        'replace', 'generate', 'favor', 'group', 'enforce', 'standardize',
        'sanitize', 'block', 'handle', 'test', 'verify', 'configure',
        'optimize', 'refactor', 'debug', 'trace', 'log', 'monitor'
    ]
    tokens = tokenize(text)
    return [t for t in tokens if t in action_indicators]


def extract_concepts(text: str) -> List[str]:
    """Extract noun-like concepts from text."""
    # Simple heuristic: words that appear as subjects/objects
    keywords = extract_keywords(text, top_n=8)
    action_words = set(extract_action_words(text))
    return [k for k in keywords if k not in action_words]


# ============================================================================
# QUERY GENERATION
# ============================================================================

class RuleBasedQueryGenerator:
    """
    Generate diverse queries using rule-based NLP techniques.

    Produces 12-15 queries per memory covering:
    - Direct keyword extraction
    - Semantic variations (synonym expansion)
    - Question formats (what, how, why, when, where)
    - Category-specific templates
    - Edge cases (short, long)
    - Implicit/scenario-based
    """

    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
        self.used_queries: Set[str] = set()

    def generate_queries(self, memory: Dict) -> List[GeneratedQuery]:
        """Generate 12-15 diverse queries for a memory."""
        content = memory.get('content', '')
        category = memory.get('category', 'DEFAULT')
        context = memory.get('context', '')

        queries = []
        self.used_queries.clear()

        # 1. DIRECT: Key phrases from content (2-3 queries)
        queries.extend(self._generate_direct_queries(content))

        # 2. KEYWORD: Just keywords, no structure (2 queries)
        queries.extend(self._generate_keyword_queries(content))

        # 3. QUESTION formats: what, how, why (4 queries)
        queries.extend(self._generate_question_queries(content))

        # 4. SEMANTIC: Synonym variations (2 queries)
        queries.extend(self._generate_semantic_queries(content))

        # 5. CATEGORY-SPECIFIC: Template-based (2 queries)
        queries.extend(self._generate_category_queries(content, category))

        # 6. EDGE CASES: Short and long (2 queries)
        queries.extend(self._generate_edge_queries(content))

        # 7. IMPLICIT/SCENARIO: Problem-oriented (2 queries)
        queries.extend(self._generate_implicit_queries(content, context))

        # 8. TECHNICAL vs CASUAL (2 queries)
        queries.extend(self._generate_register_queries(content))

        # Deduplicate and limit
        seen = set()
        unique_queries = []
        for q in queries:
            normalized = q.query.lower().strip()
            if normalized not in seen and len(normalized) > 3:
                seen.add(normalized)
                unique_queries.append(q)

        # Ensure 12-15 queries
        return unique_queries[:15]

    def _generate_direct_queries(self, content: str) -> List[GeneratedQuery]:
        """Generate queries using direct phrases from content."""
        queries = []
        keyphrases = extract_keyphrases(content)

        # Use keyphrases as direct queries
        for phrase in keyphrases[:3]:
            queries.append(GeneratedQuery(
                query=phrase,
                category="direct",
                difficulty="easy",
                expected_rank=1,
                min_similarity=0.6
            ))

        return queries

    def _generate_keyword_queries(self, content: str) -> List[GeneratedQuery]:
        """Generate keyword-only queries."""
        queries = []
        keywords = extract_keywords(content, top_n=6)

        # 2-3 keywords combined
        if len(keywords) >= 2:
            queries.append(GeneratedQuery(
                query=" ".join(keywords[:2]),
                category="keyword",
                difficulty="easy",
                expected_rank=1,
                min_similarity=0.5
            ))

        if len(keywords) >= 3:
            queries.append(GeneratedQuery(
                query=" ".join(keywords[1:4]),
                category="keyword",
                difficulty="easy",
                expected_rank=2,
                min_similarity=0.5
            ))

        return queries

    def _generate_question_queries(self, content: str) -> List[GeneratedQuery]:
        """Generate question-format queries."""
        queries = []
        actions = extract_action_words(content)
        concepts = extract_concepts(content)

        # HOW questions
        if actions:
            action = actions[0]
            if concepts:
                queries.append(GeneratedQuery(
                    query=f"how to {action} {concepts[0]}",
                    category="question_how",
                    difficulty="medium",
                    expected_rank=3,
                    min_similarity=0.4
                ))
            queries.append(GeneratedQuery(
                query=f"how do I {action}",
                category="question_how",
                difficulty="medium",
                expected_rank=5,
                min_similarity=0.3
            ))

        # WHAT questions
        if concepts:
            queries.append(GeneratedQuery(
                query=f"what is {concepts[0]}",
                category="question_what",
                difficulty="hard",
                expected_rank=5,
                min_similarity=0.3
            ))

        # WHY questions
        if actions and concepts:
            queries.append(GeneratedQuery(
                query=f"why should I {actions[0]} {concepts[0] if concepts else ''}".strip(),
                category="question_why",
                difficulty="hard",
                expected_rank=5,
                min_similarity=0.3
            ))

        return queries[:4]

    def _generate_semantic_queries(self, content: str) -> List[GeneratedQuery]:
        """Generate queries with synonym substitution."""
        queries = []
        keywords = extract_keywords(content, top_n=5)

        for keyword in keywords:
            synonyms = get_synonyms(keyword)
            if synonyms:
                # Replace keyword with a synonym
                new_query = content.lower()
                new_query = new_query.replace(keyword, synonyms[0])
                # Truncate to reasonable length
                words = new_query.split()[:8]
                if len(words) >= 3:
                    queries.append(GeneratedQuery(
                        query=" ".join(words),
                        category="semantic",
                        difficulty="medium",
                        expected_rank=3,
                        min_similarity=0.4
                    ))
                    break

        # Try another synonym variation
        for keyword in keywords:
            synonyms = get_synonyms(keyword)
            if len(synonyms) > 1:
                queries.append(GeneratedQuery(
                    query=f"{synonyms[1]} best practices",
                    category="semantic",
                    difficulty="hard",
                    expected_rank=5,
                    min_similarity=0.3
                ))
                break

        return queries[:2]

    def _generate_category_queries(self, content: str, category: str) -> List[GeneratedQuery]:
        """Generate queries using category-specific templates."""
        queries = []
        templates = CATEGORY_TEMPLATES.get(category, CATEGORY_TEMPLATES['DEFAULT'])
        concepts = extract_concepts(content)
        actions = extract_action_words(content)

        concept = concepts[0] if concepts else "code"
        action = actions[0] if actions else "implement"

        for template in templates[:2]:
            query = template.format(concept=concept, action=action)
            queries.append(GeneratedQuery(
                query=query,
                category="template",
                difficulty="medium",
                expected_rank=3,
                min_similarity=0.4
            ))

        return queries

    def _generate_edge_queries(self, content: str) -> List[GeneratedQuery]:
        """Generate edge case queries (very short, very long)."""
        queries = []
        keywords = extract_keywords(content, top_n=5)

        # SHORT: 2-3 words only
        if keywords:
            queries.append(GeneratedQuery(
                query=" ".join(keywords[:2]),
                category="edge_short",
                difficulty="hard",
                expected_rank=5,
                min_similarity=0.3
            ))

        # LONG: Detailed context-rich query
        if len(keywords) >= 3:
            long_query = f"I'm working on a project and need to know how to {' '.join(keywords[:3])} in my codebase for better maintainability"
            queries.append(GeneratedQuery(
                query=long_query,
                category="edge_long",
                difficulty="hard",
                expected_rank=5,
                min_similarity=0.3
            ))

        return queries

    def _generate_implicit_queries(self, content: str, context: str) -> List[GeneratedQuery]:
        """Generate implicit/scenario-based queries."""
        queries = []
        keywords = extract_keywords(content, top_n=3)

        # Problem-oriented (implicit)
        problems = {
            'validate': "my data is incorrect and causing bugs",
            'error': "my application keeps crashing",
            'security': "I'm worried about security vulnerabilities",
            'performance': "my app is running slowly",
            'test': "my code is breaking in production",
            'config': "settings are not working correctly",
            'api': "my API calls are failing",
            'memory': "my application is using too much memory",
            'debug': "I can't figure out what's wrong",
        }

        for keyword in keywords:
            if keyword in problems:
                queries.append(GeneratedQuery(
                    query=problems[keyword],
                    category="implicit",
                    difficulty="hard",
                    expected_rank=10,
                    min_similarity=0.2
                ))
                break

        # Scenario-based
        if keywords:
            queries.append(GeneratedQuery(
                query=f"when implementing a feature, how should I handle {keywords[0]}",
                category="scenario",
                difficulty="hard",
                expected_rank=5,
                min_similarity=0.3
            ))

        return queries[:2]

    def _generate_register_queries(self, content: str) -> List[GeneratedQuery]:
        """Generate technical vs casual register queries."""
        queries = []
        keywords = extract_keywords(content, top_n=3)
        keyword_str = " ".join(keywords[:2]) if keywords else "this"

        # TECHNICAL: Formal terminology
        queries.append(GeneratedQuery(
            query=f"implementation pattern for {keyword_str}",
            category="technical",
            difficulty="medium",
            expected_rank=5,
            min_similarity=0.3
        ))

        # CASUAL: Informal phrasing
        queries.append(GeneratedQuery(
            query=f"hey how do I deal with {keyword_str}",
            category="casual",
            difficulty="hard",
            expected_rank=10,
            min_similarity=0.2
        ))

        return queries


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_memories(
    input_file: Path,
    output_file: Path,
    seed: Optional[int] = 42
) -> Dict:
    """
    Process all memories and generate expanded queries.

    Args:
        input_file: Path to selected_memories.json
        output_file: Path to save expanded test suite
        seed: Random seed for reproducibility

    Returns:
        Statistics about the generation process
    """
    print(f"\n{'='*80}")
    print("ENHANCED RULE-BASED QUERY GENERATOR")
    print(f"{'='*80}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n")

    # Load memories
    with open(input_file) as f:
        memories = json.load(f)

    print(f"Loaded {len(memories)} memories")

    # Initialize generator
    generator = RuleBasedQueryGenerator(seed=seed)

    # Process each memory
    enhanced_memories = []
    stats = {
        "total_memories": len(memories),
        "processed": 0,
        "total_queries_generated": 0,
        "queries_per_category": {},
        "queries_per_difficulty": {"easy": 0, "medium": 0, "hard": 0},
        "start_time": datetime.now().isoformat()
    }

    for i, memory in enumerate(memories):
        print(f"[{i+1}/{len(memories)}] Processing memory {memory['memory_id']}")
        print(f"  Category: {memory.get('category', 'unknown')}")
        print(f"  Content: {memory['content'][:60]}...")

        # Generate queries
        queries = generator.generate_queries(memory)

        enhanced = EnhancedMemoryTestCase(
            memory_id=memory["memory_id"],
            content=memory["content"],
            category=memory.get("category", "unknown"),
            feedback_type=memory.get("feedback_type", "unknown"),
            severity=memory.get("severity", 5),
            context=memory.get("context", "unknown"),
            original_queries=memory.get("sample_queries", []),
            generated_queries=queries,
            generation_timestamp=datetime.now().isoformat()
        )

        enhanced_memories.append(enhanced)
        stats["processed"] += 1
        stats["total_queries_generated"] += len(queries)

        # Track category/difficulty distribution
        for q in queries:
            cat = q.category
            stats["queries_per_category"][cat] = stats["queries_per_category"].get(cat, 0) + 1
            stats["queries_per_difficulty"][q.difficulty] += 1

        print(f"  Generated {len(queries)} queries")

    # Final save
    stats["end_time"] = datetime.now().isoformat()
    stats["avg_queries_per_memory"] = stats["total_queries_generated"] / stats["processed"]

    output_data = {
        "metadata": {
            "generation_stats": stats,
            "format_version": "2.0",
            "generator": "rule_based_nlp",
            "description": "Enhanced test suite with 12-15 diverse queries per memory"
        },
        "test_cases": [m.to_dict() for m in enhanced_memories]
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Processed: {stats['processed']}/{stats['total_memories']}")
    print(f"Total queries: {stats['total_queries_generated']}")
    print(f"Avg queries/memory: {stats['avg_queries_per_memory']:.1f}")
    print(f"\nQueries by category:")
    for cat, count in sorted(stats["queries_per_category"].items()):
        print(f"  {cat}: {count}")
    print(f"\nQueries by difficulty:")
    for diff, count in stats["queries_per_difficulty"].items():
        print(f"  {diff}: {count}")
    print(f"\nSaved to: {output_file}")

    return stats


def main():
    """Run the query generator."""
    input_file = Path(__file__).parent / "test_suite" / "selected_memories.json"
    output_file = Path(__file__).parent / "test_suite" / "enhanced_test_suite.json"

    stats = process_memories(
        input_file=input_file,
        output_file=output_file,
        seed=42
    )

    return stats


if __name__ == "__main__":
    main()
