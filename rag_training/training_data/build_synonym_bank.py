"""
Step 5: Build Domain-Specific Synonym Bank for Query Expansion

This script extracts key terms from the memory corpus and builds a synonym bank
for SHORT query expansion (NOT HyDE-style long hypotheticals).

Usage:
    python build_synonym_bank.py

Output: synonym_bank.json
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime
import re

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def extract_key_terms(text: str) -> list[str]:
    """Extract meaningful terms from text."""
    # Remove special characters, keep alphanumeric
    text = re.sub(r'[^\w\s\-]', ' ', text.lower())
    words = text.split()

    # Filter stopwords and short words
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them',
        'we', 'us', 'our', 'you', 'your', 'i', 'me', 'my', 'he', 'him',
        'she', 'her', 'if', 'then', 'else', 'when', 'where', 'which',
        'what', 'who', 'how', 'why', 'all', 'any', 'both', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'not', 'only',
        'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now',
        'use', 'using', 'used', 'always', 'never', 'prefer', 'like',
        'want', 'need', 'make', 'get', 'set', 'put', 'add', 'new'
    }

    return [w for w in words if len(w) >= 3 and w not in stopwords]


def build_synonym_groups() -> dict:
    """Build domain-specific synonym groups."""
    # Programming/tech synonyms
    return {
        # Languages
        "typescript": ["ts", "javascript", "js", "ecmascript"],
        "python": ["py", "python3"],
        "javascript": ["js", "typescript", "ts", "node", "nodejs"],

        # Async patterns
        "async": ["asynchronous", "await", "promise", "concurrent"],
        "await": ["async", "promise", "then"],
        "promise": ["async", "await", "future", "deferred"],

        # Error handling
        "error": ["exception", "failure", "bug", "issue", "problem"],
        "exception": ["error", "throw", "catch", "try"],
        "debug": ["troubleshoot", "diagnose", "fix", "investigate"],
        "fix": ["repair", "resolve", "patch", "correct"],

        # API patterns
        "api": ["endpoint", "rest", "http", "service"],
        "endpoint": ["api", "route", "url", "path"],
        "request": ["fetch", "call", "invoke", "query"],
        "response": ["result", "reply", "output", "return"],

        # Data patterns
        "cache": ["memoize", "store", "buffer", "memory"],
        "validate": ["check", "verify", "sanitize", "confirm"],
        "sanitize": ["clean", "escape", "validate", "filter"],

        # Config patterns
        "config": ["configuration", "settings", "options", "params"],
        "settings": ["config", "preferences", "options"],
        "environment": ["env", "config", "variable"],

        # Testing
        "test": ["spec", "unit", "integration", "check"],
        "mock": ["stub", "fake", "spy", "simulate"],

        # Performance
        "optimize": ["improve", "enhance", "speed", "performance"],
        "performance": ["speed", "latency", "throughput", "efficiency"],
        "fast": ["quick", "rapid", "efficient", "performant"],
        "slow": ["latency", "delay", "lag", "bottleneck"],

        # Security
        "security": ["auth", "authentication", "authorization", "secure"],
        "auth": ["authentication", "login", "credentials", "token"],
        "token": ["jwt", "bearer", "credential", "key"],

        # Database
        "database": ["db", "datastore", "storage", "persistence"],
        "query": ["select", "fetch", "retrieve", "search"],

        # Architecture
        "component": ["module", "service", "class", "unit"],
        "module": ["component", "package", "library", "plugin"],
        "service": ["api", "endpoint", "backend", "server"],

        # Workflow
        "deploy": ["release", "publish", "ship", "launch"],
        "build": ["compile", "bundle", "package", "construct"],
        "refactor": ["restructure", "reorganize", "clean", "improve"],

        # UI
        "ui": ["interface", "frontend", "client", "view"],
        "frontend": ["ui", "client", "web", "browser"],
        "backend": ["server", "api", "service", "database"],
    }


def main():
    print("=" * 60)
    print("Step 5: Build Domain-Specific Synonym Bank")
    print("=" * 60)

    # Connect to Qdrant to extract terms from memories
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        print("\n[1/4] Connected to Qdrant")
    except Exception as e:
        print(f"[WARNING] Could not connect to Qdrant: {e}")
        print("         Using predefined synonym bank only")
        client = None

    # Get predefined synonyms
    synonym_groups = build_synonym_groups()
    print(f"\n[2/4] Loaded {len(synonym_groups)} predefined synonym groups")

    # Extract terms from memories if Qdrant available
    term_counts = Counter()
    if client:
        try:
            print("\n[3/4] Extracting terms from memory corpus...")
            # Scroll through all memories
            offset = None
            total_memories = 0

            while True:
                results = client.scroll(
                    collection_name="ace_memories_hybrid",
                    limit=100,
                    offset=offset,
                    with_payload=True
                )

                points, next_offset = results
                if not points:
                    break

                for point in points:
                    payload = point.payload or {}
                    # Extract from lesson field
                    lesson = payload.get('lesson', '') or payload.get('text', '')
                    if lesson:
                        terms = extract_key_terms(lesson)
                        term_counts.update(terms)
                        total_memories += 1

                offset = next_offset
                if not offset:
                    break

            print(f"      Processed {total_memories} memories")
            print(f"      Found {len(term_counts)} unique terms")

        except Exception as e:
            print(f"[WARNING] Error extracting terms: {e}")

    # Build final synonym bank
    print("\n[4/4] Building final synonym bank...")

    # Get top frequent terms not already in synonym groups
    top_terms = [term for term, count in term_counts.most_common(100)
                 if term not in synonym_groups and count >= 3]

    # Create expansion mappings (term -> expansions)
    expansion_map = {}

    for term, synonyms in synonym_groups.items():
        expansion_map[term] = synonyms[:3]  # Max 3 synonyms per term

    # Add reverse mappings
    for term, synonyms in list(synonym_groups.items()):
        for syn in synonyms:
            if syn not in expansion_map:
                expansion_map[syn] = [term]

    # Add frequent terms with no synonyms (for future expansion)
    frequent_terms = {
        "entries": len(expansion_map),
        "top_corpus_terms": top_terms[:50],
        "coverage_stats": {
            "predefined_groups": len(synonym_groups),
            "total_mappings": len(expansion_map),
            "corpus_terms_analyzed": len(term_counts)
        }
    }

    # Save synonym bank
    output_file = Path(__file__).parent / "synonym_bank.json"

    bank_data = {
        "version": "1.0",
        "created": datetime.now().isoformat(),
        "description": "Domain-specific synonym bank for SHORT query expansion",
        "usage": "Expand query with 1-2 synonyms max, NOT HyDE-style long text",
        "expansion_map": expansion_map,
        "metadata": frequent_terms
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(bank_data, f, indent=2)

    print("\n" + "=" * 60)
    print("SYNONYM BANK COMPLETE")
    print("=" * 60)
    print(f"Output: {output_file}")
    print(f"Total mappings: {len(expansion_map)}")
    print(f"Predefined groups: {len(synonym_groups)}")
    if term_counts:
        print(f"Top corpus terms: {', '.join(top_terms[:10])}")

    return str(output_file)


if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n[SUCCESS] Synonym bank ready at: {result}")
    else:
        print("\n[FAILED] Could not build synonym bank")
        sys.exit(1)
