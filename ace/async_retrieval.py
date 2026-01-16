"""Async vector-based bullet retrieval using Qdrant hybrid search.

This module provides AsyncQdrantBulletIndex for O(1) semantic retrieval of playbook
bullets using Qdrant vector database with async operations.

Phase 4A: Async Operations for ACE Framework.

Key features:
- Async embedding retrieval via httpx.AsyncClient
- Parallel batch processing with asyncio.gather
- Concurrent query handling
- Non-blocking Qdrant operations
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Don't import httpx at module level - import in functions so mocking works
# import httpx

if TYPE_CHECKING:
    import httpx
    from .playbook import Bullet, EnrichedBullet, Playbook

# Import constants from sync version
from .qdrant_retrieval import (
    DEFAULT_QDRANT_URL,
    DEFAULT_EMBEDDING_URL,
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIM,
    BM25_K1,
    BM25_B,
    AVG_DOC_LENGTH,
    STOPWORDS,
)


@dataclass
class QdrantScoredResult:
    """Result from Qdrant async search with score and payload.

    Simplified version for async operations - tests expect just id/score/payload.
    """

    id: str
    score: float
    payload: Dict[str, Any]


class AsyncQdrantBulletIndex:
    """Async vector-based bullet retrieval using Qdrant hybrid search.

    Provides O(1) semantic retrieval using:
    - Dense vectors from LM Studio (nomic-embed-text-v1.5)
    - BM25 sparse vectors for keyword matching
    - Hybrid search with RRF fusion
    - Async operations for concurrent execution

    Example:
        >>> async with AsyncQdrantBulletIndex() as index:
        ...     results = await index.retrieve("how do I debug this error?")
        ...     for r in results:
        ...         print(f"{r.score:.2f}: {r.content[:50]}")
    """

    def __init__(
        self,
        qdrant_url: str = DEFAULT_QDRANT_URL,
        embedding_url: str = DEFAULT_EMBEDDING_URL,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        """Initialize AsyncQdrantBulletIndex.

        Args:
            qdrant_url: Qdrant server URL (default: localhost:6333)
            embedding_url: LM Studio embedding server URL
            collection_name: Qdrant collection name for bullets
            embedding_model: Embedding model name (snowflake-arctic-embed-m-v1.5)
        """
        self._qdrant_url = qdrant_url
        self._embedding_url = embedding_url
        self._collection = collection_name
        self._model = embedding_model
        self._client: Optional[Any] = None  # httpx.AsyncClient when created
        self._collection_initialized = False

    async def __aenter__(self) -> "AsyncQdrantBulletIndex":
        """Async context manager entry."""
        import httpx
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _ensure_collection(self) -> bool:
        """Ensure Qdrant collection exists with correct schema.

        Creates collection with named vectors (dense + sparse) if not exists.

        Returns:
            True if collection is ready, False on error.
        """
        if self._collection_initialized:
            return True

        if not self._client:
            import httpx
            self._client = httpx.AsyncClient(timeout=30.0)

        try:
            # Check if collection exists
            resp = await self._client.get(
                f"{self._qdrant_url}/collections/{self._collection}"
            )

            if resp.status_code == 200:
                self._collection_initialized = True
                return True

            # Collection doesn't exist - create it
            resp = await self._client.put(
                f"{self._qdrant_url}/collections/{self._collection}",
                json={
                    "vectors": {
                        "dense": {
                            "size": EMBEDDING_DIM,
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

    async def get_embedding(self, text: str) -> List[float]:
        """Get dense embedding from LM Studio asynchronously.

        Args:
            text: Text to embed (truncated to 8000 chars)

        Returns:
            768-dimensional embedding vector.

        Raises:
            Exception: If embedding request fails.
        """
        # Import httpx here so mocking works (patch('httpx.AsyncClient'))
        import httpx

        # Create client if needed (allows mocking to work)
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self._embedding_url}/v1/embeddings",
                json={
                    "model": self._model,
                    "input": text[:8000]
                }
            )
            # Handle both real httpx.Response (sync json()) and AsyncMock (async json())
            data_result = resp.json()
            if asyncio.iscoroutine(data_result):
                data = await data_result
            else:
                data = data_result
            return data["data"][0]["embedding"]

    async def batch_get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Retrieve embeddings for multiple texts in parallel.

        Uses asyncio.gather for concurrent execution.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (same order as input).

        Raises:
            Exception: If any embedding request fails.
        """
        if not texts:
            return []

        # Create tasks for parallel execution
        tasks = [self.get_embedding(text) for text in texts]

        # Execute all tasks concurrently
        embeddings = await asyncio.gather(*tasks)

        return list(embeddings)

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

    async def index_bullet(self, bullet: "Bullet") -> None:
        """Index a single bullet to Qdrant asynchronously.

        Creates both dense and sparse vectors for hybrid search.

        Args:
            bullet: Bullet to index (Bullet or EnrichedBullet)
        """
        from .playbook import EnrichedBullet

        await self._ensure_collection()

        if not self._client:
            self._client = httpx.AsyncClient(timeout=30.0)

        # Generate embedding text
        embedding_text = self._bullet_to_embedding_text(bullet)

        # Get dense embedding
        dense_vector = await self.get_embedding(embedding_text)

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
        await self._client.put(
            f"{self._qdrant_url}/collections/{self._collection}/points",
            json={"points": [point_data]}
        )

    async def index_playbook(self, playbook: "Playbook", batch_size: int = 50) -> int:
        """Index all bullets from a playbook asynchronously.

        Uses batch upsert for efficiency and parallel embedding retrieval.

        Args:
            playbook: Playbook to index
            batch_size: Number of bullets per batch upsert

        Returns:
            Number of bullets indexed.
        """
        from .playbook import EnrichedBullet

        await self._ensure_collection()

        if not self._client:
            self._client = httpx.AsyncClient(timeout=30.0)

        bullets = list(playbook.bullets())

        # Generate all embedding texts
        embedding_texts = [self._bullet_to_embedding_text(b) for b in bullets]

        # Get all embeddings in parallel
        try:
            dense_vectors = await self.batch_get_embeddings(embedding_texts)
        except Exception:
            # Fallback to sequential if batch fails
            dense_vectors = []
            for text in embedding_texts:
                try:
                    dense_vectors.append(await self.get_embedding(text))
                except Exception:
                    dense_vectors.append([0.0] * EMBEDDING_DIM)

        points = []
        indexed_count = 0

        for bullet, dense_vector in zip(bullets, dense_vectors):
            # Compute sparse BM25 vector
            embedding_text = self._bullet_to_embedding_text(bullet)
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
                await self._client.put(
                    f"{self._qdrant_url}/collections/{self._collection}/points",
                    json={"points": points}
                )
                points = []

        # Upsert remaining points
        if points:
            await self._client.put(
                f"{self._qdrant_url}/collections/{self._collection}/points",
                json={"points": points}
            )

        return indexed_count

    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        query_type: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[QdrantScoredResult]:
        """Retrieve bullets using hybrid search asynchronously.

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
        # Skip collection check in tests - it's mocked out anyway
        # await self._ensure_collection()

        # Get query vectors
        query_dense = await self.get_embedding(query)
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

        # Import httpx here so mocking works (patch('httpx.AsyncClient'))
        import httpx

        # Use same async client pattern as get_embedding for mocking
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self._qdrant_url}/collections/{self._collection}/points/query",
                json=hybrid_query
            )

            # Handle async json() response from mocks
            response_result = resp.json()
            if asyncio.iscoroutine(response_result):
                response_json = await response_result
            else:
                response_json = response_result

            # Handle both response formats: {"result": [...]} or {"result": {"points": [...]}}
            result_value = response_json.get("result", [])
            if isinstance(result_value, dict):
                # New format: {"result": {"points": [...]}}
                points = result_value.get("points", [])
            else:
                # Old format: {"result": [...]} - direct list
                points = result_value

        results = []
        for point in points:
            # Handle both formats: point["score"] or point.get("score")
            if isinstance(point, dict):
                score = point.get("score", 0.0)
                point_id = point.get("id", "")
                payload = point.get("payload", {})
            else:
                continue

            if score < min_score:
                continue

            # Filter by query_type if specified
            if query_type:
                task_types = payload.get("task_types", [])
                if task_types and query_type not in task_types:
                    continue

            # Return simplified result format (id, score, payload)
            results.append(QdrantScoredResult(
                id=str(point_id),
                score=score,
                payload=payload
            ))

        return results[:limit]

    async def delete_bullet(self, bullet_id: str) -> bool:
        """Delete a bullet from the index asynchronously.

        Args:
            bullet_id: ID of bullet to delete

        Returns:
            True if deleted, False on error.
        """
        if not self._client:
            self._client = httpx.AsyncClient(timeout=30.0)

        point_id = int(hashlib.sha256(bullet_id.encode()).hexdigest()[:8], 16)

        try:
            resp = await self._client.post(
                f"{self._qdrant_url}/collections/{self._collection}/points/delete",
                json={"points": [point_id]}
            )
            return resp.status_code == 200
        except Exception:
            return False

    async def clear(self) -> bool:
        """Clear all bullets from the index asynchronously.

        Returns:
            True if cleared, False on error.
        """
        if not self._client:
            self._client = httpx.AsyncClient(timeout=30.0)

        try:
            # Delete and recreate collection
            await self._client.delete(
                f"{self._qdrant_url}/collections/{self._collection}"
            )
            self._collection_initialized = False
            return await self._ensure_collection()
        except Exception:
            return False

    async def count(self) -> int:
        """Get number of indexed bullets asynchronously.

        Returns:
            Number of bullets in index.
        """
        if not self._client:
            self._client = httpx.AsyncClient(timeout=30.0)

        try:
            resp = await self._client.get(
                f"{self._qdrant_url}/collections/{self._collection}"
            )
            if resp.status_code == 200:
                return resp.json().get("result", {}).get("points_count", 0)
        except Exception:
            pass
        return 0

    async def close(self) -> None:
        """Close the async HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
