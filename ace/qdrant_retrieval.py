"""Vector-based bullet retrieval using Qdrant hybrid search.

This module provides QdrantBulletIndex for O(1) semantic retrieval of playbook
bullets using Qdrant vector database with hybrid search (dense + BM25 sparse).

Phase 1: Vector Search Integration for ACE Fortune 100 Production Readiness.

Key features:
- Dense embeddings via LM Studio (nomic-embed-text-v1.5, 768-dim)
- BM25 sparse vectors for keyword matching (technical terms)
- Hybrid search with RRF fusion for best of both approaches
- Seamless integration with existing SmartBulletIndex
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .playbook import Bullet, EnrichedBullet, Playbook

# Import centralized configuration
from .config import EmbeddingConfig, QdrantConfig, BM25Config

# Load centralized configuration for defaults (used by async_retrieval.py)
_embedding_config = EmbeddingConfig()
_qdrant_config = QdrantConfig()
_bm25_config = BM25Config()

DEFAULT_QDRANT_URL = _qdrant_config.url
DEFAULT_EMBEDDING_URL = _embedding_config.url
DEFAULT_EMBEDDING_MODEL = _embedding_config.model
EMBEDDING_DIM = _embedding_config.dimension
DEFAULT_COLLECTION = _qdrant_config.bullets_collection

# BM25 parameters from centralized config
BM25_K1 = _bm25_config.k1
BM25_B = _bm25_config.b
AVG_DOC_LENGTH = _bm25_config.avg_doc_length

# Technical programming stopwords
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
    'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have',
    'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
    'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'it', 'its',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'what',
    'which', 'who', 'whom', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
    'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now'
}


@dataclass
class QdrantScoredResult:
    """Result from Qdrant hybrid search with score and payload."""

    bullet_id: str
    content: str
    section: str
    score: float
    task_types: List[str]
    trigger_patterns: List[str]
    payload: Dict[str, Any]


class QdrantBulletIndex:
    """Vector-based bullet retrieval using Qdrant hybrid search.

    Provides O(1) semantic retrieval using:
    - Dense vectors from LM Studio (snowflake-arctic-embed-m-v1.5)
    - BM25 sparse vectors for keyword matching
    - Hybrid search with RRF fusion

    Example:
        >>> from ace.qdrant_retrieval import QdrantBulletIndex
        >>> from ace import Playbook
        >>>
        >>> index = QdrantBulletIndex()
        >>> playbook = Playbook.load_from_file("playbook.json")
        >>> index.index_playbook(playbook)
        >>>
        >>> results = index.retrieve("how do I debug this error?")
        >>> for r in results:
        ...     print(f"{r.score:.2f}: {r.content[:50]}")
    """

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        embedding_url: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: Optional[str] = None,
    ) -> None:
        """Initialize QdrantBulletIndex.

        Args:
            qdrant_url: Qdrant server URL (default: from QdrantConfig)
            embedding_url: LM Studio embedding server URL (default: from EmbeddingConfig)
            collection_name: Qdrant collection name for bullets
            embedding_model: Embedding model name (default: from EmbeddingConfig)
        """
        # Load centralized configuration
        _embedding_config = EmbeddingConfig()
        _qdrant_config = QdrantConfig()
        
        self._qdrant_url = qdrant_url or _qdrant_config.url
        self._embedding_url = embedding_url or _embedding_config.url
        self._model = embedding_model or _embedding_config.model
        self._embedding_dim = _embedding_config.dimension
        self._collection = collection_name
        self._client = httpx.Client(timeout=30.0)
        self._collection_initialized = False

    def _ensure_collection(self) -> bool:
        """Ensure Qdrant collection exists with correct schema.

        Creates collection with named vectors (dense + sparse) if not exists.

        Returns:
            True if collection is ready, False on error.
        """
        if self._collection_initialized:
            return True

        try:
            # Check if collection exists
            resp = self._client.get(
                f"{self._qdrant_url}/collections/{self._collection}"
            )

            if resp.status_code == 200:
                self._collection_initialized = True
                return True

            # Collection doesn't exist - create it
            resp = self._client.put(
                f"{self._qdrant_url}/collections/{self._collection}",
                json={
                    "vectors": {
                        "dense": {
                            "size": self._embedding_dim,
                            "distance": "Cosine"
                        }
                    },
                    "sparse_vectors": {
                        "sparse": {}
                    }
                }
            )

            if resp.status_code in [200, 201]:
                self._collection_initialized = True
                return True

            return False

        except Exception:
            return False

    def _get_embedding(self, text: str) -> List[float]:
        """Get dense embedding from LM Studio with automatic EOS token handling.

        Args:
            text: Text to embed (truncated to 8000 chars)

        Returns:
            768-dimensional embedding vector.

        Raises:
            RuntimeError: If embedding request fails.
        """
        try:
            # Add EOS token for Qwen models to fix GGUF tokenizer warning
            # This ensures proper sentence boundary detection in embeddings
            if "qwen" in self._model.lower() and not text.endswith("</s>"):
                text = f"{text}</s>"
            
            resp = self._client.post(
                f"{self._embedding_url}/v1/embeddings",
                json={
                    "model": self._model,
                    "input": text[:8000]
                }
            )
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}")

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """Tokenize text for BM25, preserving technical terms.

        Handles:
        - CamelCase splitting
        - snake_case splitting
        - Technical term preservation
        - Stopword removal

        Args:
            text: Text to tokenize

        Returns:
            List of tokens for BM25.
        """
        # Split CamelCase
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        # Split snake_case
        text = text.replace('_', ' ')
        # Extract alphanumeric tokens
        tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
        # Filter stopwords and short tokens
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        return tokens

    def _compute_bm25_sparse(self, text: str) -> Dict[str, Any]:
        """Compute BM25-style sparse vector for Qdrant.

        Args:
            text: Text to vectorize

        Returns:
            Dict with 'indices' (term hashes) and 'values' (BM25 weights).
        """
        tokens = self._tokenize_for_bm25(text)
        if not tokens:
            return {"indices": [], "values": []}

        tf = Counter(tokens)
        doc_length = len(tokens)

        indices = []
        values = []

        for term, freq in tf.items():
            # Consistent hash for term -> index
            term_hash = int(hashlib.sha256(term.encode()).hexdigest()[:8], 16)
            indices.append(term_hash)

            # BM25 term weight
            tf_weight = (freq * (BM25_K1 + 1)) / (
                freq + BM25_K1 * (1 - BM25_B + BM25_B * doc_length / AVG_DOC_LENGTH)
            )
            values.append(float(tf_weight))

        return {"indices": indices, "values": values}

    def _bullet_to_embedding_text(self, bullet: "Bullet") -> str:
        """Generate optimized text for embedding from bullet.

        Combines content with metadata for better semantic matching.

        Args:
            bullet: Bullet to generate text for

        Returns:
            Optimized text string for embedding.
        """
        from .playbook import EnrichedBullet

        parts = [bullet.content]

        if isinstance(bullet, EnrichedBullet):
            # Add trigger patterns as keywords
            if bullet.trigger_patterns:
                parts.append(" ".join(bullet.trigger_patterns))
            # Add task types
            if bullet.task_types:
                parts.append(" ".join(bullet.task_types))
            # Use custom embedding_text if provided
            if bullet.embedding_text:
                parts = [bullet.embedding_text]

        return " ".join(parts)

    def index_bullet(self, bullet: "Bullet") -> None:
        """Index a single bullet to Qdrant.

        Creates both dense and sparse vectors for hybrid search.

        Args:
            bullet: Bullet to index (Bullet or EnrichedBullet)
        """
        from .playbook import EnrichedBullet

        self._ensure_collection()

        # Generate embedding text
        embedding_text = self._bullet_to_embedding_text(bullet)

        # Get dense embedding
        dense_vector = self._get_embedding(embedding_text)

        # Compute sparse BM25 vector
        sparse_vector = self._compute_bm25_sparse(embedding_text)

        # Build payload
        payload = {
            "bullet_id": bullet.id,
            "content": bullet.content,
            "section": bullet.section,
            "helpful": bullet.helpful,
            "harmful": bullet.harmful,
            "neutral": bullet.neutral,
        }

        if isinstance(bullet, EnrichedBullet):
            payload.update({
                "task_types": bullet.task_types,
                "trigger_patterns": bullet.trigger_patterns,
                "domains": bullet.domains,
                "complexity_level": bullet.complexity_level,
                "anti_patterns": bullet.anti_patterns,
                "effectiveness_score": bullet.effectiveness_score,
            })

        # Generate numeric ID from bullet_id hash
        point_id = int(hashlib.sha256(bullet.id.encode()).hexdigest()[:8], 16)

        # Build point with named vectors
        point_data: Dict[str, Any] = {
            "id": point_id,
            "vector": {
                "dense": dense_vector
            },
            "payload": payload
        }

        # Add sparse vector if available
        if sparse_vector.get("indices"):
            point_data["vector"]["sparse"] = sparse_vector

        # Upsert to Qdrant
        self._client.put(
            f"{self._qdrant_url}/collections/{self._collection}/points",
            json={"points": [point_data]}
        )

    def index_playbook(self, playbook: "Playbook", batch_size: int = 50) -> int:
        """Index all bullets from a playbook.

        Uses batch upsert for efficiency.

        Args:
            playbook: Playbook to index
            batch_size: Number of bullets per batch upsert

        Returns:
            Number of bullets indexed.
        """
        from .playbook import EnrichedBullet

        self._ensure_collection()

        bullets = list(playbook.bullets())
        points = []
        indexed_count = 0

        for bullet in bullets:
            # Generate embedding text
            embedding_text = self._bullet_to_embedding_text(bullet)

            try:
                # Get dense embedding
                dense_vector = self._get_embedding(embedding_text)

                # Compute sparse BM25 vector
                sparse_vector = self._compute_bm25_sparse(embedding_text)

                # Build payload
                payload = {
                    "bullet_id": bullet.id,
                    "content": bullet.content,
                    "section": bullet.section,
                    "helpful": bullet.helpful,
                    "harmful": bullet.harmful,
                    "neutral": bullet.neutral,
                }

                if isinstance(bullet, EnrichedBullet):
                    payload.update({
                        "task_types": bullet.task_types,
                        "trigger_patterns": bullet.trigger_patterns,
                        "domains": bullet.domains,
                        "complexity_level": bullet.complexity_level,
                        "effectiveness_score": bullet.effectiveness_score,
                    })

                # Generate numeric ID
                point_id = int(hashlib.sha256(bullet.id.encode()).hexdigest()[:8], 16)

                point_data: Dict[str, Any] = {
                    "id": point_id,
                    "vector": {"dense": dense_vector},
                    "payload": payload
                }

                if sparse_vector.get("indices"):
                    point_data["vector"]["sparse"] = sparse_vector

                points.append(point_data)
                indexed_count += 1

                # Batch upsert when batch is full
                if len(points) >= batch_size:
                    self._client.put(
                        f"{self._qdrant_url}/collections/{self._collection}/points",
                        json={"points": points}
                    )
                    points = []

            except Exception:
                continue  # Skip failed bullets

        # Upsert remaining points
        if points:
            self._client.put(
                f"{self._qdrant_url}/collections/{self._collection}/points",
                json={"points": points}
            )

        return indexed_count

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        query_type: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[QdrantScoredResult]:
        """Retrieve bullets using hybrid search.

        Combines dense semantic search with BM25 keyword matching
        using Reciprocal Rank Fusion (RRF).

        Args:
            query: Natural language query
            limit: Maximum number of results
            query_type: Optional query type for filtering
            min_score: Minimum score threshold

        Returns:
            List of QdrantScoredResult ordered by score descending.
        """
        self._ensure_collection()

        # Get query vectors
        query_dense = self._get_embedding(query)
        query_sparse = self._compute_bm25_sparse(query)

        # Build hybrid query with prefetch + RRF fusion
        hybrid_query: Dict[str, Any] = {
            "prefetch": [
                {
                    "query": query_dense,
                    "using": "dense",
                    "limit": limit * 3
                }
            ],
            "query": {"fusion": "rrf"},
            "limit": limit,
            "with_payload": True
        }

        # Add sparse prefetch if we have terms
        if query_sparse.get("indices"):
            hybrid_query["prefetch"].append({
                "query": {
                    "indices": query_sparse["indices"],
                    "values": query_sparse["values"]
                },
                "using": "sparse",
                "limit": limit * 3
            })

        try:
            resp = self._client.post(
                f"{self._qdrant_url}/collections/{self._collection}/points/query",
                json=hybrid_query
            )

            if resp.status_code != 200:
                return []

            result_data = resp.json().get("result", {})
            points = result_data.get("points", [])

            results = []
            for point in points:
                score = point.get("score", 0.0)
                if score < min_score:
                    continue

                payload = point.get("payload", {})

                # Filter by query_type if specified
                if query_type:
                    task_types = payload.get("task_types", [])
                    if task_types and query_type not in task_types:
                        continue

                results.append(QdrantScoredResult(
                    bullet_id=payload.get("bullet_id", ""),
                    content=payload.get("content", ""),
                    section=payload.get("section", ""),
                    score=score,
                    task_types=payload.get("task_types", []),
                    trigger_patterns=payload.get("trigger_patterns", []),
                    payload=payload
                ))

            return results[:limit]

        except Exception:
            return []

    def delete_bullet(self, bullet_id: str) -> bool:
        """Delete a bullet from the index.

        Args:
            bullet_id: ID of bullet to delete

        Returns:
            True if deleted, False on error.
        """
        point_id = int(hashlib.sha256(bullet_id.encode()).hexdigest()[:8], 16)

        try:
            resp = self._client.post(
                f"{self._qdrant_url}/collections/{self._collection}/points/delete",
                json={"points": [point_id]}
            )
            return resp.status_code == 200
        except Exception:
            return False

    def clear(self) -> bool:
        """Clear all bullets from the index.

        Returns:
            True if cleared, False on error.
        """
        try:
            # Delete and recreate collection
            self._client.delete(
                f"{self._qdrant_url}/collections/{self._collection}"
            )
            self._collection_initialized = False
            return self._ensure_collection()
        except Exception:
            return False

    def count(self) -> int:
        """Get number of indexed bullets.

        Returns:
            Number of bullets in index.
        """
        try:
            resp = self._client.get(
                f"{self._qdrant_url}/collections/{self._collection}"
            )
            if resp.status_code == 200:
                return resp.json().get("result", {}).get("points_count", 0)
        except Exception:
            pass
        return 0

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "QdrantBulletIndex":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()
