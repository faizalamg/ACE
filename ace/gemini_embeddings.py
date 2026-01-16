"""Gemini Embedding Client for ACE Framework.

Provides embeddings using Google's gemini-embedding-001 model
with proper task type optimization for retrieval (document vs query).

Usage:
    from ace.gemini_embeddings import GeminiEmbeddingClient

    client = GeminiEmbeddingClient(api_key="your-api-key")

    # For indexing documents
    doc_embedding = client.embed_document("This is a document about...")

    # For search queries
    query_embedding = client.embed_query("How do I fix...")
"""

import os
import logging
from typing import List, Optional, Literal
from functools import lru_cache
import httpx

logger = logging.getLogger(__name__)

# Gemini API Configuration
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768  # Recommended dimension

# Task types for different use cases
TaskType = Literal[
    "RETRIEVAL_DOCUMENT",  # For indexing documents
    "RETRIEVAL_QUERY",     # For search queries
    "SEMANTIC_SIMILARITY", # For similarity comparisons
    "CLASSIFICATION",      # For classification tasks
    "CLUSTERING",          # For clustering tasks
]


class GeminiEmbeddingClient:
    """Client for Google Gemini embedding API.

    Attributes:
        api_key: Gemini API key
        model: Model name (default: gemini-embedding-001)
        dimension: Output embedding dimension (default: 768)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GEMINI_MODEL,
        dimension: int = EMBEDDING_DIM,
        timeout: float = 30.0,
    ):
        """Initialize Gemini embedding client.

        Args:
            api_key: Gemini API key (or set GEMINI_API_KEY env var)
            model: Model name
            dimension: Output dimension (128-3072, recommended: 768)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY env var or pass api_key parameter."
            )

        self.model = model
        self.dimension = dimension
        self.timeout = timeout

        # HTTP client with connection pooling
        self.client = httpx.Client(
            timeout=timeout,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
        )

        # Endpoints
        self.embed_url = f"{GEMINI_API_BASE}/models/{model}:embedContent"
        self.batch_url = f"{GEMINI_API_BASE}/models/{model}:batchEmbedContents"

        logger.info(f"Gemini embedding client initialized: {model}, dim={dimension}")

    def _embed_single(
        self,
        text: str,
        task_type: TaskType = "SEMANTIC_SIMILARITY",
    ) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text (max 2048 tokens)
            task_type: Embedding task type

        Returns:
            Embedding vector (list of floats)
        """
        # Truncate to avoid token limit
        text = text[:8000]  # Approximate char limit for 2048 tokens

        payload = {
            "model": f"models/{self.model}",
            "content": {
                "parts": [{"text": text}]
            },
            "outputDimensionality": self.dimension,
            "taskType": task_type,
        }

        try:
            response = self.client.post(self.embed_url, json=payload)

            if response.status_code == 200:
                data = response.json()
                embedding = data.get("embedding", {}).get("values", [])
                if len(embedding) != self.dimension:
                    logger.warning(
                        f"Unexpected embedding dim: {len(embedding)} vs {self.dimension}"
                    )
                return embedding
            else:
                logger.error(
                    f"Gemini API error {response.status_code}: {response.text}"
                )
                return [0.0] * self.dimension

        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
            return [0.0] * self.dimension

    def embed_document(self, text: str) -> List[float]:
        """Generate embedding optimized for document indexing.

        Args:
            text: Document text to embed

        Returns:
            768-dim embedding vector
        """
        return self._embed_single(text, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding optimized for search queries.

        Args:
            text: Query text to embed

        Returns:
            768-dim embedding vector
        """
        return self._embed_single(text, task_type="RETRIEVAL_QUERY")

    def embed_batch(
        self,
        texts: List[str],
        task_type: TaskType = "RETRIEVAL_DOCUMENT",
        batch_size: int = 100,
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            task_type: Embedding task type
            batch_size: Max texts per batch request

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Build batch request
            requests = []
            for text in batch:
                requests.append({
                    "model": f"models/{self.model}",
                    "content": {
                        "parts": [{"text": text[:8000]}]
                    },
                    "outputDimensionality": self.dimension,
                    "taskType": task_type,
                })

            payload = {"requests": requests}

            try:
                response = self.client.post(self.batch_url, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    embeddings = data.get("embeddings", [])
                    for emb in embeddings:
                        values = emb.get("values", [0.0] * self.dimension)
                        all_embeddings.append(values)
                else:
                    logger.error(
                        f"Batch embed error {response.status_code}: {response.text}"
                    )
                    # Add zero vectors for failed batch
                    all_embeddings.extend([[0.0] * self.dimension] * len(batch))

            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                all_embeddings.extend([[0.0] * self.dimension] * len(batch))

        return all_embeddings

    def verify_consistency(self, text: str) -> dict:
        """Verify embedding consistency by generating same text twice.

        Args:
            text: Text to verify

        Returns:
            Dict with embeddings and cosine similarity
        """
        import numpy as np

        emb1 = self.embed_document(text)
        emb2 = self.embed_document(text)

        # Compute cosine similarity
        a = np.array(emb1)
        b = np.array(emb2)

        if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0:
            cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        else:
            cosine_sim = 0.0

        return {
            "embedding_1": emb1[:5],  # First 5 values
            "embedding_2": emb2[:5],
            "cosine_similarity": float(cosine_sim),
            "dimension": len(emb1),
            "consistent": cosine_sim > 0.999,  # Should be nearly identical
        }

    def close(self):
        """Close HTTP client."""
        if hasattr(self, "client"):
            self.client.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience function for quick embedding
def get_gemini_embedding(
    text: str,
    task_type: TaskType = "SEMANTIC_SIMILARITY",
    api_key: Optional[str] = None,
) -> List[float]:
    """Quick embedding without client management.

    Args:
        text: Text to embed
        task_type: Embedding task type
        api_key: Optional API key

    Returns:
        Embedding vector
    """
    with GeminiEmbeddingClient(api_key=api_key) as client:
        return client._embed_single(text, task_type)


if __name__ == "__main__":
    # Test the client
    import sys

    api_key = sys.argv[1] if len(sys.argv) > 1 else os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Usage: python gemini_embeddings.py <api_key>")
        print("Or set GEMINI_API_KEY environment variable")
        sys.exit(1)

    print("=" * 60)
    print("Gemini Embedding Client Test")
    print("=" * 60)

    client = GeminiEmbeddingClient(api_key=api_key)

    # Test single embedding
    test_text = "Define root-level config files to explicitly declare project boundaries."
    print(f"\nTest text: {test_text}")

    doc_emb = client.embed_document(test_text)
    print(f"\nDocument embedding (first 5): {doc_emb[:5]}")
    print(f"Dimension: {len(doc_emb)}")

    query_emb = client.embed_query(test_text)
    print(f"\nQuery embedding (first 5): {query_emb[:5]}")

    # Verify consistency
    print("\n" + "=" * 60)
    print("Consistency Verification")
    print("=" * 60)

    result = client.verify_consistency(test_text)
    print(f"Cosine similarity (same text): {result['cosine_similarity']:.6f}")
    print(f"Consistent: {result['consistent']}")

    client.close()
    print("\nTest complete!")
