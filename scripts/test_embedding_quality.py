#!/usr/bin/env python3
"""
Test embedding quality for semantic similarity evaluation.

Baseline: Average Precision 75.6%
Current model: text-embedding-qwen3-embedding-8b (768 dims)
LM Studio endpoint: http://localhost:1234

Tests:
1. Semantic similarity for related concepts
2. Semantic dissimilarity for unrelated concepts
3. Cosine similarity scoring
"""

import httpx
import numpy as np
from typing import List
import sys
from pathlib import Path

# Add ace module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ace.config import EmbeddingConfig

# Configuration
EMBEDDING_CONFIG = EmbeddingConfig()
EMBEDDING_URL = EMBEDDING_CONFIG.url
EMBEDDING_MODEL = EMBEDDING_CONFIG.model
EMBEDDING_DIM = EMBEDDING_CONFIG.dimension

print(f"Testing Embedding Quality")
print(f"=" * 60)
print(f"Model: {EMBEDDING_MODEL}")
print(f"Endpoint: {EMBEDDING_URL}")
print(f"Dimensions: {EMBEDDING_DIM}")
print()


def get_embedding(text: str) -> List[float]:
    """Get embedding vector from LM Studio."""
    # Add EOS token for Qwen models (from unified_memory.py pattern)
    if "qwen" in EMBEDDING_MODEL.lower() and not text.endswith("</s>"):
        text = f"{text}</s>"

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{EMBEDDING_URL}/v1/embeddings",
                json={"model": EMBEDDING_MODEL, "input": text[:8000]}
            )
            resp.raise_for_status()
            embedding = resp.json()["data"][0]["embedding"]

            # Validate dimension
            if len(embedding) != EMBEDDING_DIM:
                print(f"WARNING: Expected {EMBEDDING_DIM} dims, got {len(embedding)}")

            return embedding
    except Exception as e:
        print(f"ERROR getting embedding: {e}")
        raise


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    # Normalize
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    # Cosine similarity
    return float(np.dot(v1_norm, v2_norm))


def test_similarity_pair(text1: str, text2: str, expected_similarity: str) -> float:
    """
    Test similarity between two texts.

    Args:
        text1: First text
        text2: Second text
        expected_similarity: "SIMILAR" or "DIFFERENT"

    Returns:
        Cosine similarity score (0.0-1.0)
    """
    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"Expected: {expected_similarity}")

    # Get embeddings
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)

    # Calculate similarity
    similarity = cosine_similarity(emb1, emb2)

    # Interpret result
    threshold = 0.7  # Typical threshold for "similar"
    if expected_similarity == "SIMILAR":
        status = "PASS" if similarity >= threshold else "FAIL"
    else:  # DIFFERENT
        status = "PASS" if similarity < threshold else "FAIL"

    print(f"Similarity: {similarity:.4f} {status}")
    print()

    return similarity


def main():
    """Run embedding quality tests."""

    print("Test 1: Semantic Similarity (Related Concepts)")
    print("-" * 60)

    # Test 1: Similar concepts should have high similarity
    sim1 = test_similarity_pair(
        "system wiring",
        "architecture configuration",
        "SIMILAR"
    )

    print("Test 2: Semantic Dissimilarity (Unrelated Concepts)")
    print("-" * 60)

    # Test 2: Unrelated concepts should have low similarity
    sim2 = test_similarity_pair(
        "system wiring",
        "TypeScript preference",
        "DIFFERENT"
    )

    print("Test 3: Code Concepts (Related)")
    print("-" * 60)

    # Test 3: Related code concepts
    sim3 = test_similarity_pair(
        "function refactoring best practices",
        "methods for improving code quality",
        "SIMILAR"
    )

    print("Test 4: Task Types (Different)")
    print("-" * 60)

    # Test 4: Different task types
    sim4 = test_similarity_pair(
        "database optimization strategies",
        "UI component styling guidelines",
        "DIFFERENT"
    )

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test 1 (system wiring vs architecture config): {sim1:.4f}")
    print(f"Test 2 (system wiring vs TypeScript pref):     {sim2:.4f}")
    print(f"Test 3 (refactoring vs code quality):          {sim3:.4f}")
    print(f"Test 4 (database vs UI styling):               {sim4:.4f}")
    print()

    # Analysis
    print("ANALYSIS")
    print("-" * 60)

    similar_avg = (sim1 + sim3) / 2
    different_avg = (sim2 + sim4) / 2

    print(f"Average similarity for RELATED pairs:   {similar_avg:.4f}")
    print(f"Average similarity for UNRELATED pairs: {different_avg:.4f}")
    print(f"Separation margin: {similar_avg - different_avg:.4f}")
    print()

    # Quality assessment
    if similar_avg >= 0.7 and different_avg <= 0.5:
        quality = "EXCELLENT"
    elif similar_avg >= 0.6 and different_avg <= 0.6:
        quality = "GOOD"
    elif similar_avg >= 0.5:
        quality = "FAIR"
    else:
        quality = "POOR"

    print(f"Embedding Quality: {quality}")
    print()

    # Recommendations
    print("RECOMMENDATIONS")
    print("-" * 60)

    if quality in ["EXCELLENT", "GOOD"]:
        print("[PASS] Current model performs well for semantic similarity")
        print("[PASS] Separation between related/unrelated concepts is clear")
        print()
        print("Suggested improvements:")
        print("- Test with more domain-specific pairs")
        print("- Evaluate retrieval precision on real queries")
        print("- Consider fine-tuning if domain expertise needed")
    else:
        print("[WARN] Model shows weak semantic discrimination")
        print()
        print("Alternative models to test:")
        print("1. OpenAI text-embedding-3-small (1536 dims)")
        print("   - Pro: Strong semantic understanding, stable API")
        print("   - Con: Requires OpenAI API key, costs ~$0.02/1M tokens")
        print()
        print("2. BGE-large-en-v1.5 (1024 dims)")
        print("   - Pro: Open source, can run locally")
        print("   - Con: Requires model download (~1.3GB)")
        print()
        print("3. Nomic-embed-text-v1.5 (768 dims)")
        print("   - Pro: Same dimensions, training data transparency")
        print("   - Con: Requires model download")

    print()
    print("EMBEDDING MODEL INFO")
    print("-" * 60)
    print(f"Current: {EMBEDDING_MODEL}")
    print(f"Dimensions: {EMBEDDING_DIM}")
    print(f"Endpoint: {EMBEDDING_URL}")
    print()
    print("Note: LM Studio supports multiple embedding models.")
    print("Load alternative models via LM Studio UI to test.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
