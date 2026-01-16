"""OpenAI Embedding Client for ACE Framework.

Provides embeddings using OpenAI's text-embedding-3-large model
with configurable dimensions.

Usage:
    from ace.openai_embeddings import OpenAIEmbeddingClient

    client = OpenAIEmbeddingClient(api_key="your-api-key")

    # Get embedding
    embedding = client.get_embedding("This is a document about...")
"""

import os
import logging
from typing import List, Optional
import httpx

logger = logging.getLogger(__name__)

# OpenAI API Configuration
OPENAI_API_BASE = "https://api.openai.com/v1"
EMBEDDING_MODEL = "text-embedding-3-small"  # Cheaper, 62.3% MTEB
EMBEDDING_DIM = 1536  # Native dimension for text-embedding-3-small


class OpenAIEmbeddingClient:
    """Client for OpenAI embedding API.

    Attributes:
        api_key: OpenAI API key
        model: Model name (default: text-embedding-3-large)
        dimension: Output embedding dimension (default: 768)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = EMBEDDING_MODEL,
        dimension: int = EMBEDDING_DIM,
        timeout: float = 60.0,
    ):
        """Initialize OpenAI embedding client.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model name
            dimension: Output dimension (256, 512, 768, 1024, 1536, or 3072)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter."
            )

        self.model = model
        self.dimension = dimension
        self.timeout = timeout

        # HTTP client with connection pooling
        self.client = httpx.Client(
            timeout=timeout,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        # Endpoint
        self.embed_url = f"{OPENAI_API_BASE}/embeddings"

        logger.info(f"OpenAI embedding client initialized: {model}, dim={dimension}")

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text (max 8191 tokens)

        Returns:
            Embedding vector (list of floats)
        """
        # Truncate to avoid token limit
        text = text[:30000]  # Approximate char limit

        payload = {
            "model": self.model,
            "input": text,
            "dimensions": self.dimension,
        }

        try:
            response = self.client.post(self.embed_url, json=payload)

            if response.status_code == 200:
                data = response.json()
                embedding = data.get("data", [{}])[0].get("embedding", [])
                if len(embedding) != self.dimension:
                    logger.warning(
                        f"Unexpected embedding dim: {len(embedding)} vs {self.dimension}"
                    )
                return embedding
            else:
                logger.error(
                    f"OpenAI API error {response.status_code}: {response.text}"
                )
                return [0.0] * self.dimension

        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            return [0.0] * self.dimension

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 100,
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Max texts per batch request

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Truncate texts
            batch = [t[:30000] for t in batch]

            payload = {
                "model": self.model,
                "input": batch,
                "dimensions": self.dimension,
            }

            try:
                response = self.client.post(self.embed_url, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    embeddings = data.get("data", [])
                    # Sort by index to maintain order
                    embeddings.sort(key=lambda x: x.get("index", 0))
                    for emb in embeddings:
                        values = emb.get("embedding", [0.0] * self.dimension)
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

        emb1 = self.get_embedding(text)
        emb2 = self.get_embedding(text)

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


if __name__ == "__main__":
    # Test the client
    import sys

    api_key = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Usage: python openai_embeddings.py <api_key>")
        print("Or set OPENAI_API_KEY environment variable")
        sys.exit(1)

    print("=" * 60)
    print("OpenAI Embedding Client Test")
    print("=" * 60)

    client = OpenAIEmbeddingClient(api_key=api_key)

    # Test single embedding
    test_text = "Define root-level config files to explicitly declare project boundaries."
    print(f"\nTest text: {test_text}")

    emb = client.get_embedding(test_text)
    print(f"\nEmbedding (first 5): {emb[:5]}")
    print(f"Dimension: {len(emb)}")

    # Verify consistency
    print("\n" + "=" * 60)
    print("Consistency Verification")
    print("=" * 60)

    result = client.verify_consistency(test_text)
    print(f"Cosine similarity (same text): {result['cosine_similarity']:.6f}")
    print(f"Consistent: {result['consistent']}")

    client.close()
    print("\nTest complete!")
