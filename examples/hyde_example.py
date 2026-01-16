"""HyDE (Hypothetical Document Embeddings) Usage Example.

This example demonstrates how to use HyDE to improve retrieval accuracy
for short, ambiguous queries by generating hypothetical answer documents.

Requirements:
- Qdrant running at http://localhost:6333
- LM Studio embeddings at http://localhost:1234
- ZAI_API_KEY or OPENAI_API_KEY in .env
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ace.hyde import HyDEGenerator, HyDEConfig
from ace.hyde_retrieval import HyDEEnhancedRetriever
from ace.llm_providers.litellm_client import LiteLLMClient


def main():
    """Demonstrate HyDE usage."""
    print("=" * 80)
    print("HyDE (Hypothetical Document Embeddings) Example")
    print("=" * 80)

    # Check API key
    if not os.getenv("ZAI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\nERROR: No API key found!")
        print("Set ZAI_API_KEY (for Z.ai GLM) or OPENAI_API_KEY in .env file")
        return

    # Initialize LLM client
    print("\n1. Initializing LLM client (Z.ai GLM-4.6)...")
    llm_client = LiteLLMClient(model="openai/glm-4.6")

    # Configure HyDE
    print("\n2. Configuring HyDE...")
    hyde_config = HyDEConfig(
        num_hypotheticals=3,  # Generate 3 hypothetical documents
        max_tokens=150,       # Max tokens per hypothetical
        temperature=0.7,      # Higher temp for diversity
        cache_enabled=True    # Cache for repeated queries
    )

    # Initialize HyDE generator
    hyde_generator = HyDEGenerator(
        llm_client=llm_client,
        config=hyde_config
    )

    # Example 1: Generate hypothetical documents
    print("\n3. Example 1: Generating Hypothetical Documents")
    print("-" * 80)

    query = "How to fix memory leak?"
    print(f"\nQuery: {query}")

    hypotheticals = hyde_generator.generate_hypotheticals(query)

    print(f"\nGenerated {len(hypotheticals)} hypothetical documents:")
    for i, hyp in enumerate(hypotheticals, 1):
        print(f"\n[{i}] {hyp}")

    # Example 2: HyDE-enhanced retrieval
    print("\n\n4. Example 2: HyDE-Enhanced Retrieval")
    print("-" * 80)

    # Initialize retriever
    retriever = HyDEEnhancedRetriever(
        hyde_generator=hyde_generator,
        qdrant_url="http://localhost:6333",
        embedding_url="http://localhost:1234",
        collection_name="ace_memories_hybrid"
    )

    # Test queries (short/ambiguous vs specific)
    test_queries = [
        ("fix auth error", True),  # Short/ambiguous -> HyDE enabled
        ("ImportError: cannot import name 'Playbook' from ace.playbook", False)  # Specific -> HyDE disabled
    ]

    print("\nTesting query classification and retrieval:")

    for query, expected_hyde in test_queries:
        print(f"\n\nQuery: {query}")

        # Auto-detect HyDE usage
        should_use_hyde = retriever._should_use_hyde(query)
        print(f"HyDE auto-enabled: {should_use_hyde} (expected: {expected_hyde})")

        # Retrieve (simulated - requires Qdrant populated)
        try:
            results = retriever.retrieve(query, limit=5)
            print(f"Retrieved {len(results)} results")

            if results:
                print("\nTop 3 results:")
                for i, result in enumerate(results[:3], 1):
                    print(f"  [{i}] (score={result.score:.3f}) {result.content[:60]}...")
        except Exception as e:
            print(f"Retrieval error (Qdrant may not be populated): {e}")

    # Example 3: Cache statistics
    print("\n\n5. Example 3: Cache Statistics")
    print("-" * 80)

    cache_stats = hyde_generator.get_cache_stats()
    print("\nCache statistics:")
    for key, value in cache_stats.items():
        print(f"  {key}: {value}")

    # Example 4: Direct hypothetical generation comparison
    print("\n\n6. Example 4: Query Expansion Comparison")
    print("-" * 80)

    queries = [
        "fix bug",
        "optimize performance",
        "debug issue"
    ]

    for query in queries:
        print(f"\nOriginal query: {query}")
        hypotheticals = hyde_generator.generate_hypotheticals(query, num_docs=2)
        print("Expanded to:")
        for i, hyp in enumerate(hypotheticals, 1):
            print(f"  [{i}] {hyp[:100]}...")

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
