#!/usr/bin/env python
"""
Semantic Similarity Scorer for ACE Retrieval Quality Measurement.

Uses embedding cosine similarity instead of keyword matching to measure
how relevant retrieved results are to the original query.

This provides a more accurate quality metric than keyword-based precision.
"""

import os
import sys
import numpy as np
from typing import List, Tuple, Optional
import httpx

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ace.config import EmbeddingConfig


class SemanticSimilarityScorer:
    """Score query-result relevance using embedding cosine similarity."""
    
    def __init__(self, embedding_url: Optional[str] = None, model: Optional[str] = None):
        """Initialize with embedding server configuration."""
        config = EmbeddingConfig()
        self.embedding_url = embedding_url or config.url
        self.model = model or config.model
        self._cache = {}
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding vector for text."""
        cache_key = text[:500]  # Truncate for cache key
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            response = httpx.post(
                f"{self.embedding_url}/v1/embeddings",
                json={"model": self.model, "input": text},
                timeout=30.0,
            )
            if response.status_code == 200:
                data = response.json()
                embedding = np.array(data["data"][0]["embedding"])
                self._cache[cache_key] = embedding
                return embedding
        except Exception as e:
            print(f"Embedding error: {e}")
        return None
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def score_result(self, query: str, result_content: str) -> float:
        """
        Score how relevant a result is to the query using semantic similarity.
        
        Returns:
            Similarity score from 0.0 (unrelated) to 1.0 (identical meaning)
        """
        query_emb = self._get_embedding(query)
        result_emb = self._get_embedding(result_content)
        
        if query_emb is None or result_emb is None:
            return 0.0
        
        return self.cosine_similarity(query_emb, result_emb)
    
    def score_results(
        self, query: str, results: List[str], threshold: float = 0.5
    ) -> Tuple[float, int, int]:
        """
        Score a list of results against a query.
        
        Args:
            query: Original search query
            results: List of result contents
            threshold: Minimum similarity to count as "relevant"
        
        Returns:
            Tuple of (average_similarity, relevant_count, total_count)
        """
        if not results:
            return 0.0, 0, 0
        
        scores = []
        relevant = 0
        
        for result in results:
            score = self.score_result(query, result)
            scores.append(score)
            if score >= threshold:
                relevant += 1
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return avg_score, relevant, len(results)


def test_semantic_scoring():
    """Test the semantic similarity scorer."""
    scorer = SemanticSimilarityScorer()
    
    print("=" * 80)
    print("SEMANTIC SIMILARITY SCORING TEST")
    print("=" * 80)
    
    # Test queries and expected relevant content
    test_cases = [
        (
            "how to handle API errors",
            [
                "Implement retry logic with exponential backoff for transient failures",  # Relevant
                "Use try-catch blocks to handle exceptions gracefully",  # Relevant
                "Configure logging to track all events",  # Less relevant
            ]
        ),
        (
            "improve performance",
            [
                "Use caching to reduce redundant computations",  # Relevant
                "Benchmark before and after optimizations",  # Relevant
                "Validate user input at entry points",  # Not relevant
            ]
        ),
        (
            "fix database errors",
            [
                "Use parameterized queries to prevent SQL injection",  # Relevant
                "Implement connection pooling for resilience",  # Relevant
                "Deploy to staging before production",  # Not relevant
            ]
        ),
    ]
    
    for query, results in test_cases:
        print(f"\nQuery: \"{query}\"")
        print("-" * 40)
        
        avg_sim, relevant, total = scorer.score_results(query, results, threshold=0.5)
        
        for i, result in enumerate(results):
            score = scorer.score_result(query, result)
            marker = "[OK]" if score >= 0.5 else "[MISS]"
            print(f"  {marker} ({score:.2f}) {result[:60]}...")
        
        print(f"\nAverage Similarity: {avg_sim:.3f}")
        print(f"Relevant (>=0.5): {relevant}/{total}")
    
    print("\n" + "=" * 80)
    print("SEMANTIC SCORING READY FOR USE")
    print("=" * 80)


if __name__ == "__main__":
    test_semantic_scoring()
