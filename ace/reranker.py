"""Cross-encoder reranking for improved retrieval precision.

This module provides cross-encoder based reranking to improve the precision
of bullet retrieval. Cross-encoders process query-document pairs together,
enabling more accurate relevance scoring than bi-encoders.

Usage:
    from ace.reranker import get_reranker, CrossEncoderReranker
    
    # Using singleton (recommended)
    reranker = get_reranker()
    scores = reranker.predict("query", ["doc1", "doc2"])
    
    # Custom model
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = reranker.predict("query", ["doc1", "doc2"])

Requirements:
    pip install ace[reranking]
    # or: pip install sentence-transformers
"""

from __future__ import annotations

import os
import threading
from typing import List, Optional, TYPE_CHECKING

# CRITICAL: Prevent tokenizer parallelism deadlock when using HuggingFace tokenizers
# with asyncio/multiprocessing. This MUST be set before importing sentence-transformers.
# See: https://github.com/huggingface/tokenizers/issues/1067
# See: https://github.com/UKPLab/sentence-transformers/issues/1318
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Lazy import to avoid loading sentence-transformers unless needed
if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

# Module-level singleton with thread-safe lock
_reranker_instance: Optional["CrossEncoderReranker"] = None
_reranker_lock = threading.Lock()

# Default model - good balance of speed and accuracy
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Check if sentence-transformers is available
try:
    import sentence_transformers
    RERANKING_AVAILABLE = True
except ImportError:
    RERANKING_AVAILABLE = False


class CrossEncoderReranker:
    """Cross-encoder based reranker for improved retrieval precision.
    
    Cross-encoders process query-document pairs together using full
    transformer attention, providing more accurate relevance scores
    than bi-encoder similarity at the cost of higher latency.
    
    Best used as a second-stage reranker on top-K candidates from
    a fast first-stage retriever (like vector search).
    
    Thread Safety:
        This class uses internal locking to ensure thread-safe model loading.
        Multiple threads can safely call predict() concurrently - the model
        will only be loaded once, and all threads will wait for loading to
        complete before proceeding.
    
    Attributes:
        model_name: The HuggingFace model name/path for the cross-encoder.
        _model: Lazily-loaded CrossEncoder model instance.
    
    Example:
        >>> reranker = CrossEncoderReranker()
        >>> scores = reranker.predict(
        ...     "How to handle rate limits?",
        ...     ["Use exponential backoff", "Store API keys securely"]
        ... )
        >>> print(scores)  # [0.89, 0.23]
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        """Initialize the reranker.
        
        Args:
            model_name: HuggingFace model name/path. Default is 
                       "cross-encoder/ms-marco-MiniLM-L-6-v2" which provides
                       good speed/accuracy tradeoff.
        """
        self.model_name = model_name
        self._model: Optional["CrossEncoder"] = None
        self._model_lock = threading.Lock()
        
    def _load_model(self) -> "CrossEncoder":
        """Lazily load the cross-encoder model (thread-safe).
        
        Uses double-checked locking pattern to ensure the model is only
        loaded once even when multiple threads call this simultaneously.
        
        Returns:
            Loaded CrossEncoder model instance.
            
        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        # Fast path: model already loaded
        if self._model is not None:
            return self._model
        
        # Slow path: acquire lock and load model
        with self._model_lock:
            # Double-check after acquiring lock (another thread may have loaded it)
            if self._model is not None:
                return self._model
            
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError(
                    "Cross-encoder reranking requires sentence-transformers. "
                    "Install with: pip install ace[reranking] "
                    "or: pip install sentence-transformers"
                ) from e
            
            self._model = CrossEncoder(self.model_name)
        
        return self._model
    
    def predict(self, query: str, documents: List[str]) -> List[float]:
        """Predict relevance scores for query-document pairs.
        
        Args:
            query: The search query.
            documents: List of document texts to score against the query.
            
        Returns:
            List of relevance scores, one per document. Higher scores
            indicate higher relevance. Scores are typically in [0, 1]
            range but may vary by model.
            
        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        if not documents:
            return []
            
        model = self._load_model()
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get scores from cross-encoder - disable progress bar (batches are annoying in benchmarks)
        scores = model.predict(pairs, show_progress_bar=False)
        
        # Convert to list of floats
        return [float(s) for s in scores]
    
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: Optional[int] = None
    ) -> List[tuple[int, float]]:
        """Rerank documents and return sorted indices with scores.
        
        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_k: Optional limit on number of results to return.
            
        Returns:
            List of (original_index, score) tuples, sorted by score descending.
            
        Example:
            >>> reranker = CrossEncoderReranker()
            >>> results = reranker.rerank("rate limits", ["doc1", "doc2", "doc3"], top_k=2)
            >>> for idx, score in results:
            ...     print(f"Document {idx}: {score:.2f}")
        """
        scores = self.predict(query, documents)
        
        # Pair indices with scores and sort descending
        indexed_scores = [(i, s) for i, s in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            indexed_scores = indexed_scores[:top_k]
            
        return indexed_scores


def get_reranker(model_name: str = DEFAULT_MODEL) -> CrossEncoderReranker:
    """Get the singleton reranker instance (thread-safe).
    
    Using a singleton avoids reloading the model multiple times,
    which can be expensive (~200MB model load).
    
    Thread Safety:
        Uses double-checked locking to ensure only one instance is created
        even when called from multiple threads simultaneously.
    
    Args:
        model_name: HuggingFace model name/path. Only used on first call.
        
    Returns:
        Shared CrossEncoderReranker instance.
    """
    global _reranker_instance
    
    # Fast path: instance already exists
    if _reranker_instance is not None:
        return _reranker_instance
    
    # Slow path: acquire lock and create instance
    with _reranker_lock:
        # Double-check after acquiring lock
        if _reranker_instance is not None:
            return _reranker_instance
        
        _reranker_instance = CrossEncoderReranker(model_name)
    
    return _reranker_instance


def reset_reranker() -> None:
    """Reset the singleton reranker instance.
    
    Useful for testing or when switching models.
    """
    global _reranker_instance
    with _reranker_lock:
        _reranker_instance = None
