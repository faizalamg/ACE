"""Query Enhancer for ACE Retrieval.

Transforms vague, ambiguous user queries into expanded, domain-specific queries
that improve retrieval precision.

Based on the EnginizeAPI enhanced prompt methodology.
"""

from typing import List, Dict, Optional, Tuple
import re


# Domain keyword expansions for technical queries
DOMAIN_EXPANSIONS: Dict[str, List[str]] = {
    # Vague action words -> technical context
    "fix": ["debug", "error", "bug", "issue", "resolve", "repair", "patch", "troubleshoot"],
    "slow": ["performance", "latency", "optimization", "bottleneck", "speed", "throughput"],
    "help": ["assistance", "guidance", "support", "documentation", "tutorial"],
    "broken": ["error", "exception", "failure", "crash", "bug", "defect"],
    "bad": ["issue", "problem", "error", "incorrect", "wrong"],
    
    # Ambiguous technical terms -> expanded context
    "pool": ["connection pool", "thread pool", "object pool", "resource pool", "database connection"],
    "lock": ["deadlock", "mutex", "synchronization", "concurrent", "thread lock", "database lock"],
    "token": ["JWT token", "authentication token", "CSRF token", "session token", "API token", "access token"],
    "cache": ["caching", "cache invalidation", "cache strategy", "distributed cache", "memory cache"],
    "memory": ["memory management", "memory leak", "heap", "garbage collection", "RAM usage"],
    
    # Domain shortcuts -> full terms
    "auth": ["authentication", "authorization", "OAuth", "JWT", "session management"],
    "db": ["database", "SQL", "query", "schema", "ORM", "migration"],
    "api": ["REST API", "endpoint", "HTTP", "request", "response", "rate limiting"],
    "test": ["unit test", "integration test", "TDD", "test coverage", "mocking"],
    "ci": ["CI/CD", "continuous integration", "pipeline", "deployment", "automation"],
    "sec": ["security", "vulnerability", "encryption", "authentication", "OWASP"],
}

# Intent patterns for query classification
INTENT_PATTERNS: Dict[str, List[str]] = {
    "debug": ["fix", "error", "bug", "broken", "crash", "fail", "issue", "problem", "wrong", "not working"],
    "create": ["implement", "create", "add", "build", "make", "new", "generate", "develop"],
    "modify": ["update", "change", "modify", "edit", "alter", "adjust"],
    "refactor": ["refactor", "restructure", "clean", "organize", "simplify", "improve"],
    "analyze": ["understand", "analyze", "compare", "investigate", "evaluate", "assess", "explain", "how does"],
}

# Technical domain patterns
TECHNICAL_DOMAINS: Dict[str, List[str]] = {
    "security": ["security", "auth", "encryption", "vulnerability", "XSS", "CSRF", "SQL injection", "OWASP"],
    "performance": ["performance", "speed", "latency", "optimization", "cache", "memory", "CPU"],
    "architecture": ["architecture", "pattern", "microservice", "design", "SOLID", "DRY", "KISS"],
    "debugging": ["debug", "error", "exception", "stack trace", "log", "troubleshoot"],
    "testing": ["test", "TDD", "mock", "coverage", "assertion", "fixture"],
    "database": ["database", "SQL", "query", "index", "transaction", "schema"],
}


def classify_intent(query: str) -> str:
    """Classify the intent of a query.
    
    Args:
        query: The user's query string.
        
    Returns:
        Intent classification: debug, create, modify, refactor, or analyze.
    """
    query_lower = query.lower()
    
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if pattern in query_lower:
                return intent
    
    return "analyze"  # Default to analytical


def identify_domains(query: str) -> List[str]:
    """Identify technical domains relevant to the query.
    
    Args:
        query: The user's query string.
        
    Returns:
        List of identified domains.
    """
    query_lower = query.lower()
    domains = []
    
    for domain, keywords in TECHNICAL_DOMAINS.items():
        for keyword in keywords:
            if keyword.lower() in query_lower:
                domains.append(domain)
                break
    
    return domains or ["general"]


def expand_vague_terms(query: str) -> Tuple[str, List[str]]:
    """Expand vague terms into domain-specific keywords.
    
    Args:
        query: The user's query string.
        
    Returns:
        Tuple of (expanded query, list of added terms).
    """
    query_lower = query.lower()
    added_terms = []
    expanded_parts = [query]
    
    for vague_term, expansions in DOMAIN_EXPANSIONS.items():
        # Check if vague term is in query (as whole word)
        if re.search(rf'\b{re.escape(vague_term)}\b', query_lower):
            # Add relevant expansions
            relevant_expansions = expansions[:3]  # Limit to top 3
            added_terms.extend(relevant_expansions)
            expanded_parts.extend(relevant_expansions)
    
    if added_terms:
        expanded_query = " ".join(expanded_parts)
        return expanded_query, added_terms
    
    return query, []


def enhance_query(query: str, verbose: bool = False) -> Dict:
    """Enhance a vague query for better retrieval.
    
    Args:
        query: The user's original query.
        verbose: If True, include detailed enhancement info.
        
    Returns:
        Dictionary with enhanced query and metadata.
    """
    # Step 1: Classify intent
    intent = classify_intent(query)
    
    # Step 2: Identify domains
    domains = identify_domains(query)
    
    # Step 3: Expand vague terms
    expanded_query, added_terms = expand_vague_terms(query)
    
    # Step 4: Build enhanced query with domain context
    if domains and domains != ["general"]:
        domain_context = " ".join(domains)
        enhanced_query = f"{expanded_query} {domain_context}"
    else:
        enhanced_query = expanded_query
    
    result = {
        "original": query,
        "enhanced": enhanced_query.strip(),
        "intent": intent,
        "domains": domains,
    }
    
    if verbose:
        result["added_terms"] = added_terms
        result["expansion_applied"] = bool(added_terms)
    
    return result


def get_enhanced_query(query: str) -> str:
    """Simple interface: enhance a query and return the enhanced version.
    
    Args:
        query: The user's original query.
        
    Returns:
        Enhanced query string.
    """
    result = enhance_query(query)
    return result["enhanced"]


# Test the enhancer
if __name__ == "__main__":
    test_queries = [
        "fix it",
        "slow",
        "help",
        "pool",
        "lock",
        "token",
        "memory leak",
        "deadlock prevention",
        "secure fast API",
    ]
    
    print("QUERY ENHANCEMENT EXAMPLES")
    print("=" * 70)
    
    for query in test_queries:
        result = enhance_query(query, verbose=True)
        print(f"\nOriginal: '{result['original']}'")
        print(f"Enhanced: '{result['enhanced']}'")
        print(f"Intent: {result['intent']}, Domains: {result['domains']}")
        if result.get('added_terms'):
            print(f"Added terms: {result['added_terms']}")
