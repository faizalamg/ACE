"""HyDE-enhanced retrieval pipeline for ACE memory system.

Integrates HyDE (Hypothetical Document Embeddings) with existing hybrid search
infrastructure for improved retrieval accuracy on ambiguous/implicit queries.

Pipeline:
1. Query -> HyDE expansion -> Generate hypothetical documents
2. Embed hypotheticals -> Average embeddings
3. Search Qdrant with averaged embedding + BM25 sparse
4. Return results with hybrid RRF fusion

Performance target: +5-10% for implicit/scenario/template queries
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from collections import Counter
import httpx

from .config import EmbeddingConfig, QdrantConfig
from .hyde import HyDEGenerator, HyDEConfig
from .qdrant_retrieval import (
    QdrantScoredResult,
    BM25_K1,
    BM25_B,
    AVG_DOC_LENGTH,
    STOPWORDS
)

if TYPE_CHECKING:
    from .llm import LLMClient

logger = logging.getLogger(__name__)


class HyDEEnhancedRetriever:
    """HyDE-enhanced retrieval combining hypothetical document expansion with hybrid search.

    Example:
        >>> from ace.hyde_retrieval import HyDEEnhancedRetriever
        >>> from ace.hyde import HyDEConfig
        >>> from ace.llm_providers.litellm_client import LiteLLMClient
        >>>
        >>> # Initialize components
        >>> llm = LiteLLMClient(model="openai/glm-4.6")  # Z.ai GLM-4.6
        >>> config = HyDEConfig(num_hypotheticals=3)
        >>>
        >>> # Create retriever
        >>> retriever = HyDEEnhancedRetriever(
        ...     llm_client=llm,
        ...     hyde_config=config
        ... )
        >>>
        >>> # Retrieve with HyDE (auto-enabled for short/ambiguous queries)
        >>> results = retriever.retrieve("fix auth error")
        >>> for r in results:
        ...     print(f"{r.score:.2f}: {r.content[:50]}")
    """

    def __init__(
        self,
        hyde_generator: Optional[HyDEGenerator] = None,
        llm_client: Optional[LLMClient] = None,
        hyde_config: Optional[HyDEConfig] = None,
        embedding_client: Optional[Any] = None,
        qdrant_url: Optional[str] = None,
        embedding_url: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        """Initialize HyDE-enhanced retriever with optional overrides, defaults from centralized config.

        Args:
            hyde_generator: Pre-configured HyDE generator (created if None)
            llm_client: LLM client for HyDE generation
            hyde_config: HyDE configuration
            embedding_client: Embedding client (created if None)
            qdrant_url: Qdrant server URL (defaults to centralized config)
            embedding_url: Embedding server URL (defaults to centralized config)
            collection_name: Qdrant collection name (defaults to centralized config)
            embedding_model: Embedding model name (defaults to centralized config)
        """
        # Load centralized configuration
        _embedding_config = EmbeddingConfig()
        _qdrant_config = QdrantConfig()
        
        # Initialize HyDE generator
        if hyde_generator is not None:
            self.hyde_generator = hyde_generator
        else:
            self.hyde_generator = HyDEGenerator(
                llm_client=llm_client,
                config=hyde_config or HyDEConfig()
            )

        self.config = self.hyde_generator.config

        # Initialize embedding client
        if embedding_client is not None:
            self.embedding_client = embedding_client
        else:
            self.embedding_client = EmbeddingClient(
                embedding_url=embedding_url or _embedding_config.url,
                model=embedding_model or _embedding_config.model
            )

        # Qdrant configuration (use centralized config as defaults)
        self.qdrant_url = qdrant_url or _qdrant_config.url
        self.collection_name = collection_name or _qdrant_config.default_collection
        self._embedding_dim = _embedding_config.dimension

        # Query classification thresholds
        self.short_query_threshold = 4  # words
        self.long_query_threshold = 12  # words

        logger.info(
            f"HyDE-enhanced retriever initialized with collection: {self.collection_name}"
        )

    def _should_use_hyde(self, query: str) -> bool:
        """Determine if HyDE should be used based on query characteristics.

        HyDE is most beneficial for:
        - Short, ambiguous queries (< 4 words)
        - Implicit/scenario queries without specific error messages
        - Template queries (e.g., "how to...", "fix...")

        HyDE is less beneficial for:
        - Long, specific queries with exact error messages
        - Queries with specific technical terms/stack traces

        Args:
            query: User query

        Returns:
            True if HyDE should be enabled
        """
        # Word count
        word_count = len(query.split())

        # Short query -> use HyDE
        if word_count <= self.short_query_threshold:
            logger.debug(f"Short query ({word_count} words) -> HyDE enabled")
            return True

        # Long, specific query -> skip HyDE
        if word_count >= self.long_query_threshold:
            logger.debug(f"Long query ({word_count} words) -> HyDE disabled")
            return False

        # Check for specific error patterns (skip HyDE for specific errors)
        specific_patterns = [
            r'Error:',
            r'Exception:',
            r'Traceback',
            r'line \d+',
            r'File ".*"',
            r'ImportError',
            r'KeyError',
            r'ValueError',
        ]

        for pattern in specific_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                logger.debug(f"Specific error pattern detected -> HyDE disabled")
                return False

        # Medium-length query without specific error -> use HyDE
        logger.debug(f"Medium query ({word_count} words) without specific error -> HyDE enabled")
        return True

    def _average_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        """Average multiple embedding vectors.

        Args:
            embeddings: List of embedding vectors

        Returns:
            Averaged embedding vector
        """
        if not embeddings:
            raise ValueError("Cannot average empty embedding list")

        # Average each dimension
        num_embeddings = len(embeddings)
        dim = len(embeddings[0])

        averaged = [
            sum(emb[i] for emb in embeddings) / num_embeddings
            for i in range(dim)
        ]

        return averaged

    def _get_hyde_embedding(self, query: str) -> List[float]:
        """Generate HyDE-enhanced embedding for query.

        Args:
            query: User query

        Returns:
            Averaged embedding vector from hypothetical documents
        """
        # Generate hypothetical documents
        hypotheticals = self.hyde_generator.generate_hypotheticals(query)

        logger.info(f"Generated {len(hypotheticals)} hypotheticals for HyDE embedding")

        # Generate embeddings for each hypothetical (skip empty ones)
        embeddings = []
        for hyp in hypotheticals:
            # Skip empty or whitespace-only hypotheticals
            if not hyp or not hyp.strip():
                logger.warning("Skipping empty hypothetical")
                continue
            try:
                emb = self.embedding_client.get_embedding(hyp.strip())
                if emb:
                    embeddings.append(emb)
            except Exception as e:
                logger.warning(f"Failed to embed hypothetical: {e}")

        if not embeddings:
            logger.error("Failed to generate any HyDE embeddings, falling back to query")
            return self.embedding_client.get_embedding(query)

        # Average embeddings
        averaged_embedding = self._average_embeddings(embeddings)

        logger.debug(f"Averaged {len(embeddings)} HyDE embeddings")
        return averaged_embedding

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        use_hyde: Optional[bool] = None,
        **kwargs: Any
    ) -> List[QdrantScoredResult]:
        """Retrieve relevant documents with optional HyDE enhancement.

        Args:
            query: User query
            limit: Maximum number of results
            use_hyde: Force HyDE on/off (None = auto-detect)
            **kwargs: Additional search parameters

        Returns:
            List of scored results ordered by relevance
        """
        # Auto-detect HyDE usage if not specified
        if use_hyde is None:
            use_hyde = self._should_use_hyde(query)

        logger.info(f"Retrieving with HyDE={'enabled' if use_hyde else 'disabled'}")

        # Generate query embedding (with or without HyDE)
        if use_hyde:
            query_embedding = self._get_hyde_embedding(query)
        else:
            query_embedding = self.embedding_client.get_embedding(query)

        # Search Qdrant with hybrid query
        # Note: BM25 sparse vectors still use original query, not hypotheticals
        # This is intentional - HyDE helps with semantic match, BM25 with keywords
        results = self._search_qdrant(
            query_embedding=query_embedding,
            query_text=query,  # Original query for BM25
            limit=limit
        )

        return results

    def _search_qdrant(
        self,
        query_embedding: List[float],
        query_text: str,
        limit: int
    ) -> List[QdrantScoredResult]:
        """Execute hybrid search in Qdrant.

        Args:
            query_embedding: Dense embedding vector (possibly HyDE-enhanced)
            query_text: Original query text for BM25
            limit: Maximum results

        Returns:
            List of scored results
        """
        # Compute BM25 sparse vector from original query
        sparse_vector = self._compute_bm25_sparse(query_text)

        # Build hybrid query with RRF fusion
        hybrid_query: Dict[str, Any] = {
            "prefetch": [
                {
                    "query": query_embedding,
                    "using": "dense",
                    "limit": limit * 3
                }
            ],
            "query": {"fusion": "rrf"},
            "limit": limit,
            "with_payload": True
        }

        # Add sparse prefetch if available
        if sparse_vector.get("indices"):
            hybrid_query["prefetch"].append({
                "query": {
                    "indices": sparse_vector["indices"],
                    "values": sparse_vector["values"]
                },
                "using": "sparse",
                "limit": limit * 3
            })

        # Execute search
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.qdrant_url}/collections/{self.collection_name}/points/query",
                    json=hybrid_query
                )

                if response.status_code != 200:
                    logger.error(f"Qdrant search failed: {response.status_code}")
                    return []

                results_data = response.json()

        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            return []

        # Parse results - handle both old and new Qdrant response formats
        raw_result = results_data.get("result", [])
        # Qdrant v1.12+ returns {"result": {"points": [...]}} for /points/query
        if isinstance(raw_result, dict):
            points = raw_result.get("points", [])
        else:
            points = raw_result

        results = []
        for item in points:
            payload = item.get("payload", {})
            # Use Point ID as bullet_id (memory_id in test suite)
            # Fallback: payload.id > payload.bullet_id > item.id
            point_id = str(item.get("id", ""))
            bullet_id = payload.get("id", payload.get("bullet_id", point_id))
            # Content: payload.content > payload.lesson
            content = payload.get("content", payload.get("lesson", ""))
            results.append(QdrantScoredResult(
                bullet_id=str(bullet_id),
                content=content,
                section=payload.get("section", payload.get("category", "")),
                score=item.get("score", 0.0),
                task_types=payload.get("task_types", []),
                trigger_patterns=payload.get("trigger_patterns", []),
                payload=payload
            ))

        logger.info(f"Retrieved {len(results)} results from Qdrant")
        return results

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """Tokenize text for BM25, preserving technical terms.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens for BM25
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
            Dict with 'indices' (term hashes) and 'values' (BM25 weights)
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


class EmbeddingClient:
    """Simple embedding client for LM Studio compatibility."""

    def __init__(self, embedding_url: str, model: str) -> None:
        """Initialize embedding client.

        Args:
            embedding_url: Embedding server URL
            model: Embedding model name
        """
        self.embedding_url = embedding_url
        self.model = model
        self.client = httpx.Client(timeout=30.0)

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text with automatic EOS token handling.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        try:
            # Add EOS token for Qwen models to fix GGUF tokenizer warning
            # This ensures proper sentence boundary detection in embeddings
            if "qwen" in self.model.lower() and not text.endswith("</s>"):
                text = f"{text}</s>"
            
            response = self.client.post(
                f"{self.embedding_url}/v1/embeddings",
                json={
                    "model": self.model,
                    "input": text
                }
            )

            if response.status_code == 200:
                data = response.json()
                return data["data"][0]["embedding"]
            else:
                logger.error(f"Embedding generation failed: {response.status_code}")
                return [0.0] * self.embedding_dim

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return [0.0] * self.embedding_dim

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()
