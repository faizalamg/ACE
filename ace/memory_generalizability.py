"""
Memory Generalizability Classifier for ACE Framework.

This module determines the appropriate namespace for a memory:
- Generalizable memories → user_prefs / task_strategies (cross-workspace)
- Project-specific memories → project_specific (workspace-scoped)

Both types are now stored - project-specific memories are tagged with workspace_id
for strict isolation during retrieval.

Examples:
- user_prefs/task_strategies: "Prefer functional programming patterns", "Always validate input"
- project_specific: "Use FlareSolverr for ESPN", "This project uses Z.ai GLM-4.6"
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional


class MemoryScope(Enum):
    """Determines the recommended namespace for memory storage."""
    GENERALIZABLE = "GENERALIZABLE"  # Cross-workspace: user_prefs or task_strategies
    PROJECT_SPECIFIC = "PROJECT_SPECIFIC"  # Workspace-scoped: project_specific namespace


@dataclass
class GeneralizabilityResult:
    """Result of generalizability analysis."""
    scope: MemoryScope
    confidence: float  # 0.0 to 1.0
    reasons: List[str]
    extracted_principle: Optional[str]  # The generalizable version (if applicable)
    recommended_namespace: str  # 'user_prefs', 'task_strategies', or 'project_specific'


# Project-specific indicators (suggests project_specific namespace)
PROJECT_SPECIFIC_PATTERNS = [
    # Specific file paths
    r'[a-zA-Z]:\\[\\\w.-]+',  # Windows paths like D:\path\to\file
    r'/(?:home|usr|var|opt|etc)/[\w.-]+',  # Unix paths
    r'\b(?:ace|src|lib|tests|scripts)/[\w.-]+\.(?:py|ts|js|json|yaml|md)\b',  # Relative paths with extensions

    # Specific class/function names from this codebase
    r'\b(?:CodeRetrieval|UnifiedMemoryIndex|ASTChunker|QdrantRetrieval|Playbook)\b',
    r'\b(?:VoyageCodeEmbeddingConfig|QdrantConfig|LLMConfig|BM25Config)\b',

    # Specific collection names
    r'\bace_code_context|ace_unified|ace_memories_hybrid|ace_bullets\b',

    # Specific bug fixes (mentions specific files/functions)
    r'(?:bug|fix|patch|issue)\s+(?:in|at|for)\s+(?:file|class|function|method)\s+\w+',

    # Specific numbers/ports/hosts
    r'\b(?:localhost:6333|192\.168\.\d+\.\d+|port\s+\d{3,5})\b',

    # Git commit references (only hex hashes that look like commits)
    r'\b[a-f0-9]{40}\b',  # Full git hashes only
]

# Generalizable patterns (suggests ACE)
GENERALIZABLE_PATTERNS = [
    # Universal principles
    r'\b(?:always|never|must|should|prefer|avoid)\b',

    # Best practices
    r'\b(?:best practice|design pattern|coding standard|guideline)\b',

    # Cross-domain concepts
    r'\b(?:error handling|validation|logging|testing|debugging)\b',
    r'\b(?:security|authentication|authorization|encryption)\b',
    r'\b(?:performance|optimization|scalability|efficiency)\b',

    # Universal programming concepts
    r'\b(?:DRY|KISS|YAGNI|SOLID)\b',
    r'\b(?:separation of concerns|single responsibility|open closed)\b',
    r'\b(?:dependency injection|inversion of control|factory pattern)\b',

    # Problem-solving approaches
    r'\b(?:strategy|approach|technique|method|pattern)\s+(?:for|to|of)\b',
    r'\b(?:when|how to|steps to|way to)\s+\w+\s*\.',

    # Universal mistakes to avoid
    r'\b(?:don\'t|avoid|watch out|beware|prevent)\b',
]

# Stop words - common words that don't indicate specificity
STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'must', 'shall', 'to', 'for', 'of',
    'with', 'by', 'from', 'in', 'on', 'at', 'as', 'if', 'when', 'while',
    'this', 'that', 'these', 'those', 'it', 'its', 'and', 'or', 'but',
    'not', 'no', 'yes', 'use', 'using', 'used', 'make', 'made', 'get',
    'got', 'create', 'created', 'add', 'added', 'set', 'setting', 'file',
    'code', 'function', 'class', 'method', 'variable', 'value', 'result',
    'error', 'issue', 'problem', 'solution', 'fix', 'patch', 'change',
}


def classify_memory_generalizability(content: str) -> GeneralizabilityResult:
    """
    Analyze a memory and determine if it is generalizable or project-specific.

    Args:
        content: The memory content to analyze

    Returns:
        GeneralizabilityResult with scope, confidence, and reasoning
    """
    content_lower = content.lower()
    original = content

    reasons = []
    project_specific_score = 0
    generalizable_score = 0

    # Check for project-specific indicators
    for pattern in PROJECT_SPECIFIC_PATTERNS:
        matches = len(re.findall(pattern, content, re.IGNORECASE))
        if matches > 0:
            project_specific_score += matches * 2
            reasons.append(f"Project-specific pattern: {pattern}")

    # Check for generalizable indicators
    for pattern in GENERALIZABLE_PATTERNS:
        matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
        if matches > 0:
            generalizable_score += matches
            reasons.append(f"Generalizable pattern: {pattern}")

    # Check for specific code syntax (suggests project_specific namespace)
    if re.search(r'\bdef\s+\w+\s*\(', content):
        project_specific_score += 2
        reasons.append("Contains function definition")

    if re.search(r'\bclass\s+\w+\s*:', content):
        project_specific_score += 2
        reasons.append("Contains class definition")

    # Check for "This project uses" pattern (clearly project-specific)
    if re.search(r'\bthis project uses?\b', content_lower):
        project_specific_score += 3
        reasons.append("Contains 'this project uses' pattern")

    # Check for universal principle language (suggests ACE)
    principle_starters = [
        'always', 'never', 'prefer', 'avoid', 'use', 'apply', 'implement',
        'remember', 'note', 'ensure', 'validate', 'check', 'verify'
    ]
    first_word = content_lower.strip().split()[0] if content_lower.strip() else ''
    if first_word in principle_starters:
        generalizable_score += 1
        reasons.append(f"Starts with principle keyword: '{first_word}'")

    # Check length - short principles are often more generalizable
    word_count = len([w for w in content.split() if w.lower() not in STOP_WORDS])
    if word_count <= 5 and generalizable_score > 0:
        generalizable_score += 1
        reasons.append(f"Short and principle-based ({word_count} words)")

    # Determine scope and recommended namespace
    total_indicators = project_specific_score + generalizable_score

    if project_specific_score >= 1 and generalizable_score == 0:
        # Project-specific indicators found, no generalizable patterns - store in project_specific namespace
        scope = MemoryScope.PROJECT_SPECIFIC
        confidence = min(0.9, 0.5 + (project_specific_score * 0.1))
        principle = None
        recommended_namespace = "project_specific"
    elif generalizable_score >= 2 and project_specific_score == 0:
        # Clearly generalizable - store in user_prefs/task_strategies
        scope = MemoryScope.GENERALIZABLE
        confidence = min(0.9, 0.5 + (generalizable_score * 0.1))
        principle = content
        recommended_namespace = "user_prefs"  # default, caller can override
    elif project_specific_score > 0 and generalizable_score > 0:
        # Mixed - extract the generalizable principle if possible
        principle = extract_general_principle(content)
        if principle != content and generalizable_score >= 2:
            # Has a clear generalizable principle - store extracted version as generalizable
            scope = MemoryScope.GENERALIZABLE
            confidence = min(0.8, 0.5 + (generalizable_score * 0.1))
            reasons.append("Extracted general principle from specific content")
            recommended_namespace = "user_prefs"
        else:
            # Too specific - store in project_specific namespace
            scope = MemoryScope.PROJECT_SPECIFIC
            confidence = min(0.8, 0.5 + (project_specific_score * 0.1))
            reasons.append("Project-specific content")
            recommended_namespace = "project_specific"
    elif generalizable_score == 0 and project_specific_score == 0:
        # No clear indicators - default to project_specific (safer)
        scope = MemoryScope.PROJECT_SPECIFIC
        confidence = 0.3
        principle = None
        reasons.append("No clear generalizable pattern - defaulting to project_specific")
        recommended_namespace = "project_specific"
    else:
        # Generalizable enough (has generalizable indicators but no project-specific ones)
        scope = MemoryScope.GENERALIZABLE
        confidence = min(0.7, 0.4 + (generalizable_score * 0.1))
        principle = content
        recommended_namespace = "user_prefs"

    return GeneralizabilityResult(
        scope=scope,
        confidence=confidence,
        reasons=reasons,
        extracted_principle=principle,
        recommended_namespace=recommended_namespace,
    )


def extract_general_principle(content: str) -> str:
    """
    Extract the generalizable principle from content that has mixed specificity.

    Examples:
        "Use Qdrant for vector storage" -> "Use vector database for storage"
        "Apply retry policy for Voyage API calls" -> "Apply retry policy for external API calls"
        "CodeRetrieval._apply_filename_boost boosts scores" -> "Boost retrieval scores based on filename matching"
    """
    # Replace specific names with general terms
    replacements = [
        (r'\bQdrant\b', 'vector database'),
        (r'\bVoyageAI?\b', 'external API'),
        (r'\bvoyage-code-3\b', 'code embedding model'),
        (r'\bCodeRetrieval\b', 'retrieval class'),
        (r'\bUnifiedMemoryIndex\b', 'memory index'),
        (r'\bASTChunker\b', 'code chunker'),
        (r'\b[\w:]+Config\b', 'configuration'),
        (r'\bZ\.ai\b', 'external service'),
        (r'\bGLM-?4\.?6?\b', 'LLM'),
        (r'\bgpt-?4\b', 'LLM'),
        (r'\bclaude-?3\b', 'LLM'),
        (r'\bFastMCP\b', 'MCP framework'),
        (r'\b[\w:]+:\d{4,5}\b', 'port'),  # port numbers
    ]

    principle = content
    for pattern, replacement in replacements:
        principle = re.sub(pattern, replacement, principle, flags=re.IGNORECASE)

    return principle


def should_store_in_ace(content: str, min_confidence: float = 0.5) -> Tuple[bool, str, Optional[str], str]:
    """
    Analyze content and determine storage recommendation.

    ALL content can now be stored - this function determines the appropriate namespace:
    - Generalizable content → user_prefs or task_strategies (cross-workspace)
    - Project-specific content → project_specific (workspace-scoped)

    Args:
        content: Memory content to analyze
        min_confidence: Minimum confidence threshold (currently unused, kept for API compat)

    Returns:
        Tuple of (should_store, reason, extracted_principle, recommended_namespace)
        - should_store: Always True now (all content accepted)
        - reason: Explanation of classification
        - extracted_principle: Generalized version if applicable, else None
        - recommended_namespace: 'user_prefs', 'task_strategies', or 'project_specific'
    """
    result = classify_memory_generalizability(content)

    if result.scope == MemoryScope.PROJECT_SPECIFIC:
        # Project-specific - store in project_specific namespace with workspace isolation
        return True, f"Project-specific (confidence: {result.confidence:.2f}) - will be workspace-scoped", result.extracted_principle, "project_specific"
    else:
        # Generalizable - store in cross-workspace namespace
        principle = result.extracted_principle or content
        if principle != content:
            return True, f"Generalizable (extracted principle, confidence: {result.confidence:.2f})", principle, result.recommended_namespace
        return True, f"Generalizable (confidence: {result.confidence:.2f})", principle, result.recommended_namespace


# CLI for testing
if __name__ == "__main__":
    test_cases = [
        "Prefer functional programming patterns",
        "Use Qdrant for vector storage",
        "CodeRetrieval._apply_filename_boost boosts scores",
        "Apply retry policy for Voyage API calls",
        "ScraperEpg uses FlareSolverr for ESPN",
        "Always validate input before processing",
        "This project uses Z.ai GLM-4.6",
        "Use parameterized queries to prevent SQL injection",
        "ESPN API grouping bug fix was in parse_channel_id",
    ]

    print("Memory Generalizability Classification Test")
    print("=" * 60)

    for test in test_cases:
        result = classify_memory_generalizability(test)
        print(f"\nContent: {test}")
        print(f"Scope: {result.scope.value}")
        print(f"Recommended Namespace: {result.recommended_namespace}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Reasons: {result.reasons[:3]}")
        if result.extracted_principle and result.extracted_principle != test:
            print(f"Extracted: {result.extracted_principle}")
