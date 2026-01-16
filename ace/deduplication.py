"""
Advanced Memory Deduplication System for RAG.

This module provides clustering-based deduplication for Qdrant collections:
- HDBSCAN/DBSCAN clustering for efficient duplicate detection (O(n log n) vs O(n^2))
- Multi-collection support (ace_memories_hybrid, ace_unified)
- Multiple merge strategies (keep_best, merge_content, canonical_form)
- Cluster quality metrics (silhouette score, Davies-Bouldin index)
- Dry-run mode for safe preview

Architecture:
    1. Load memories with embeddings from Qdrant (scroll API)
    2. Cluster embeddings using HDBSCAN (density-based) or DBSCAN
    3. Calculate cluster quality metrics for validation
    4. For each duplicate cluster:
       - Select best memory (highest combined score)
       - Merge counts (reinforcement, helpful, harmful)
       - Update best memory in Qdrant
       - Delete duplicates
    5. Return summary statistics

Usage:
    >>> from ace.deduplication import DeduplicationEngine, ClusteringMethod, MergeStrategy
    >>>
    >>> engine = DeduplicationEngine(
    ...     collection_name="ace_memories_hybrid"
    ... )
    >>>
    >>> # Preview what would happen
    >>> result = engine.run_deduplication(
    ...     method=ClusteringMethod.HDBSCAN,
    ...     min_cluster_size=2,
    ...     strategy=MergeStrategy.KEEP_BEST,
    ...     dry_run=True
    ... )
    >>>
    >>> # Execute deduplication
    >>> result = engine.run_deduplication(dry_run=False)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

from .config import EmbeddingConfig, QdrantConfig

# Optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointIdsList
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class ClusteringMethod(str, Enum):
    """Clustering algorithm to use for duplicate detection."""
    HDBSCAN = "hdbscan"  # Hierarchical density-based (recommended)
    DBSCAN = "dbscan"    # Density-based with fixed epsilon


class MergeStrategy(str, Enum):
    """Strategy for merging duplicate memories."""
    KEEP_BEST = "keep_best"              # Keep highest scored, merge counts
    MERGE_CONTENT = "merge_content"      # Combine unique information
    CANONICAL_FORM = "canonical_form"    # Normalize to standard form


# Clustering parameters (deduplication-specific, not config-based)
DEFAULT_MIN_CLUSTER_SIZE = 2
DEFAULT_DBSCAN_EPS = 0.08  # Distance threshold (cosine distance)
DEFAULT_DBSCAN_MIN_SAMPLES = 2


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DuplicateCluster:
    """A cluster of duplicate memories."""
    cluster_id: int
    memories: List[Dict[str, Any]]
    centroid: Optional[List[float]] = None

    def get_best_memory(self) -> Dict[str, Any]:
        """
        Select the best memory to keep based on combined score.

        Scoring criteria (weighted):
        1. Severity (30%)
        2. Reinforcement count (30%)
        3. Helpful count (20%)
        4. Content length (10%)
        5. Harmful count negative (10%)

        Returns:
            Best memory dict
        """
        def score(mem: Dict[str, Any]) -> float:
            severity = mem.get("severity", 5)
            reinforcement = mem.get("reinforcement_count", 1)
            helpful = mem.get("helpful_count", 0)
            harmful = mem.get("harmful_count", 0)
            content_len = len(mem.get("content", ""))

            return (
                severity * 0.3 +
                reinforcement * 0.3 +
                helpful * 0.2 +
                (content_len / 100) * 0.1 +
                (-harmful) * 0.1
            )

        return max(self.memories, key=score)

    def get_merged_counts(self) -> Dict[str, int]:
        """
        Calculate merged counts from all memories in cluster.

        Returns:
            Dict with total_reinforcement, total_helpful, total_harmful, max_severity
        """
        total_reinforcement = sum(m.get("reinforcement_count", 1) for m in self.memories)
        total_helpful = sum(m.get("helpful_count", 0) for m in self.memories)
        total_harmful = sum(m.get("harmful_count", 0) for m in self.memories)
        max_severity = max(m.get("severity", 5) for m in self.memories)

        return {
            "total_reinforcement": total_reinforcement,
            "total_helpful": total_helpful,
            "total_harmful": total_harmful,
            "max_severity": max_severity,
        }


@dataclass
class ClusterMetrics:
    """Cluster quality metrics for validation."""
    num_clusters: int
    num_noise_points: int
    silhouette_score: Optional[float] = None
    davies_bouldin_score: Optional[float] = None

    @classmethod
    def calculate(
        cls,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> "ClusterMetrics":
        """
        Calculate cluster quality metrics.

        Args:
            embeddings: Array of embeddings (n_samples, n_features)
            labels: Cluster labels (n_samples,)

        Returns:
            ClusterMetrics instance
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, skipping metric calculation")
            return cls(
                num_clusters=len(set(labels)) - (1 if -1 in labels else 0),
                num_noise_points=np.sum(labels == -1)
            )

        # Filter out noise points (label=-1) for metrics
        mask = labels != -1
        filtered_embeddings = embeddings[mask]
        filtered_labels = labels[mask]

        num_clusters = len(set(filtered_labels))
        num_noise = np.sum(labels == -1)

        # Calculate metrics only if we have multiple clusters
        silhouette = None
        davies_bouldin = None

        if num_clusters >= 2 and len(filtered_labels) > num_clusters:
            try:
                silhouette = silhouette_score(filtered_embeddings, filtered_labels)
                davies_bouldin = davies_bouldin_score(filtered_embeddings, filtered_labels)
            except Exception as e:
                logger.warning(f"Metric calculation failed: {e}")

        return cls(
            num_clusters=num_clusters,
            num_noise_points=num_noise,
            silhouette_score=silhouette,
            davies_bouldin_score=davies_bouldin,
        )


@dataclass
class DeduplicationResult:
    """Result of deduplication operation."""
    total_scanned: int
    duplicate_groups: int
    memories_merged: int
    memories_deleted: int
    dry_run: bool
    metrics: Optional[ClusterMetrics] = None
    cluster_details: List[Dict[str, Any]] = field(default_factory=list)


# =============================================================================
# MAIN DEDUPLICATION ENGINE
# =============================================================================

class DeduplicationEngine:
    """
    Main engine for clustering-based memory deduplication.

    Example:
        >>> engine = DeduplicationEngine(
        ...     collection_name="ace_memories_hybrid",
        ...     qdrant_url="http://localhost:6333"
        ... )
        >>> result = engine.run_deduplication(dry_run=True)
        >>> print(f"Found {result.duplicate_groups} duplicate groups")
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        embedding_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        qdrant_client: Optional[Any] = None,
    ):
        """
        Initialize DeduplicationEngine with optional overrides, defaults from centralized config.

        Args:
            collection_name: Qdrant collection name (defaults to centralized config)
            qdrant_url: Qdrant server URL (defaults to centralized config)
            embedding_url: Embedding server URL (defaults to centralized config)
            embedding_model: Embedding model name (defaults to centralized config)
            qdrant_client: Optional pre-configured Qdrant client (for testing)
        """
        # Load centralized configuration
        _embedding_config = EmbeddingConfig()
        _qdrant_config = QdrantConfig()
        
        self.collection_name = collection_name or _qdrant_config.default_collection
        self.qdrant_url = qdrant_url or _qdrant_config.url
        self.embedding_url = embedding_url or _embedding_config.url
        self.embedding_model = embedding_model or _embedding_config.model

        # Initialize Qdrant client
        if qdrant_client is not None:
            self.qdrant_client = qdrant_client
        elif QDRANT_AVAILABLE:
            self.qdrant_client = QdrantClient(url=self.qdrant_url)
        else:
            raise ImportError("qdrant-client is required for deduplication")

        # Check dependencies
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy is required for deduplication")

    def load_memories(self) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Load all memories with vectors from Qdrant.

        Returns:
            Tuple of (memories, embeddings) where:
            - memories: List of memory dicts with metadata
            - embeddings: NumPy array of shape (n_memories, embedding_dim)
        """
        logger.info(f"Loading memories from {self.collection_name}...")

        memories = []
        embeddings_list = []

        offset = None
        while True:
            points, offset = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=500,
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )

            for point in points:
                # Extract dense vector
                if isinstance(point.vector, dict):
                    vector = point.vector.get("dense", [])
                else:
                    vector = point.vector

                if not vector:
                    logger.warning(f"Point {point.id} has no dense vector, skipping")
                    continue

                # Build memory dict
                memory = {
                    "id": point.id,
                    "vector": vector,
                    **point.payload,
                }
                memories.append(memory)
                embeddings_list.append(vector)

            if offset is None:
                break

        logger.info(f"Loaded {len(memories)} memories")

        # Convert to NumPy array
        embeddings = np.array(embeddings_list)

        return memories, embeddings

    def _cluster_hdbscan(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    ) -> np.ndarray:
        """
        Cluster embeddings using HDBSCAN.

        Args:
            embeddings: Array of embeddings (n_samples, n_features)
            min_cluster_size: Minimum cluster size

        Returns:
            Array of cluster labels (n_samples,)
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError("hdbscan is required for HDBSCAN clustering")

        logger.info(f"Clustering with HDBSCAN (min_cluster_size={min_cluster_size})...")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
            min_samples=1,
        )
        labels = clusterer.fit_predict(embeddings)

        logger.info(f"HDBSCAN found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")

        return labels

    def _cluster_dbscan(
        self,
        embeddings: np.ndarray,
        eps: float = DEFAULT_DBSCAN_EPS,
        min_samples: int = DEFAULT_DBSCAN_MIN_SAMPLES,
    ) -> np.ndarray:
        """
        Cluster embeddings using DBSCAN.

        Args:
            embeddings: Array of embeddings (n_samples, n_features)
            eps: Maximum distance between samples in a cluster
            min_samples: Minimum samples in a neighborhood

        Returns:
            Array of cluster labels (n_samples,)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for DBSCAN clustering")

        logger.info(f"Clustering with DBSCAN (eps={eps}, min_samples={min_samples})...")

        clusterer = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(embeddings)

        logger.info(f"DBSCAN found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")

        return labels

    def find_duplicate_groups(
        self,
        method: ClusteringMethod = ClusteringMethod.HDBSCAN,
        min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
        dbscan_eps: float = DEFAULT_DBSCAN_EPS,
        dbscan_min_samples: int = DEFAULT_DBSCAN_MIN_SAMPLES,
    ) -> List[DuplicateCluster]:
        """
        Find duplicate groups using clustering.

        Args:
            method: Clustering method (HDBSCAN or DBSCAN)
            min_cluster_size: Minimum cluster size (HDBSCAN)
            dbscan_eps: Epsilon parameter (DBSCAN)
            dbscan_min_samples: Min samples parameter (DBSCAN)

        Returns:
            List of DuplicateCluster instances
        """
        # Load memories
        memories, embeddings = self.load_memories()

        if len(memories) == 0:
            logger.info("No memories to deduplicate")
            return []

        # Cluster embeddings
        if method == ClusteringMethod.HDBSCAN:
            labels = self._cluster_hdbscan(embeddings, min_cluster_size)
        else:
            labels = self._cluster_dbscan(embeddings, dbscan_eps, dbscan_min_samples)

        # Calculate metrics
        self.metrics = ClusterMetrics.calculate(embeddings, labels)
        logger.info(f"Cluster metrics: {self.metrics.num_clusters} clusters, "
                   f"{self.metrics.num_noise_points} noise points")

        if self.metrics.silhouette_score is not None:
            logger.info(f"  Silhouette score: {self.metrics.silhouette_score:.3f}")
        if self.metrics.davies_bouldin_score is not None:
            logger.info(f"  Davies-Bouldin index: {self.metrics.davies_bouldin_score:.3f}")

        # Group memories by cluster
        clusters: Dict[int, List[Dict[str, Any]]] = {}
        for memory, label in zip(memories, labels):
            if label == -1:  # Skip noise points
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(memory)

        # Create DuplicateCluster instances
        duplicate_clusters = []
        for cluster_id, cluster_memories in clusters.items():
            if len(cluster_memories) < 2:
                continue  # Not a duplicate (only 1 memory)

            # Calculate centroid
            cluster_embeddings = np.array([m["vector"] for m in cluster_memories])
            centroid = np.mean(cluster_embeddings, axis=0).tolist()

            duplicate_clusters.append(DuplicateCluster(
                cluster_id=cluster_id,
                memories=cluster_memories,
                centroid=centroid,
            ))

        logger.info(f"Found {len(duplicate_clusters)} duplicate groups")

        return duplicate_clusters

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from embedding server."""
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot re-embed")
            return None

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{self.embedding_url}/v1/embeddings",
                    json={"model": self.embedding_model, "input": text[:8000]}
                )
                resp.raise_for_status()
                return resp.json()["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

    def merge_cluster(
        self,
        cluster: DuplicateCluster,
        strategy: MergeStrategy = MergeStrategy.KEEP_BEST,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Merge a duplicate cluster.

        Args:
            cluster: DuplicateCluster to merge
            strategy: Merge strategy to use
            dry_run: If True, preview without modifying Qdrant

        Returns:
            Dict with merge result details
        """
        # Select best memory
        best = cluster.get_best_memory()
        duplicates = [m for m in cluster.memories if m["id"] != best["id"]]

        # Get merged counts
        merged_counts = cluster.get_merged_counts()

        # Handle different merge strategies
        if strategy == MergeStrategy.MERGE_CONTENT:
            # Combine unique sentences from all memories
            all_content = " ".join(m.get("content", "") for m in cluster.memories)
            merged_content = all_content  # TODO: Add sentence deduplication
        elif strategy == MergeStrategy.CANONICAL_FORM:
            # Use best content as canonical form
            merged_content = best.get("content", "")
        else:  # KEEP_BEST
            merged_content = best.get("content", "")

        result = {
            "cluster_id": cluster.cluster_id,
            "merged_id": best["id"],
            "deleted_ids": [d["id"] for d in duplicates],
            "merged_counts": merged_counts,
            "merged_content": merged_content,
        }

        if dry_run:
            logger.info(f"[DRY RUN] Would merge cluster {cluster.cluster_id}: "
                       f"{len(cluster.memories)} memories -> 1")
            return result

        # Update best memory with merged counts
        updated_payload = {
            **best,
            "reinforcement_count": merged_counts["total_reinforcement"],
            "helpful_count": merged_counts["total_helpful"],
            "harmful_count": merged_counts["total_harmful"],
            "severity": merged_counts["max_severity"],
            "content": merged_content,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "merged_from": [d["id"] for d in duplicates],
        }
        # Remove vector from payload (it's stored separately)
        updated_payload.pop("vector", None)

        # Get new embedding for merged content
        embedding = self._get_embedding(merged_content)
        if embedding is None:
            logger.error("Failed to get embedding, using original")
            embedding = best["vector"]

        # Update in Qdrant
        try:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[{
                    "id": best["id"],
                    "vector": {"dense": embedding},
                    "payload": updated_payload,
                }]
            )
            logger.info(f"Updated memory {best['id']}")

            # Delete duplicates
            if duplicates:
                self.qdrant_client.delete(
                    collection_name=self.collection_name,
                    points_selector=PointIdsList(points=[d["id"] for d in duplicates])
                )
                logger.info(f"Deleted {len(duplicates)} duplicates")

        except Exception as e:
            logger.error(f"Failed to merge cluster: {e}")
            raise

        return result

    def run_deduplication(
        self,
        method: ClusteringMethod = ClusteringMethod.HDBSCAN,
        min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
        dbscan_eps: float = DEFAULT_DBSCAN_EPS,
        dbscan_min_samples: int = DEFAULT_DBSCAN_MIN_SAMPLES,
        strategy: MergeStrategy = MergeStrategy.KEEP_BEST,
        dry_run: bool = False,
    ) -> DeduplicationResult:
        """
        Run full deduplication pipeline.

        Args:
            method: Clustering method
            min_cluster_size: Minimum cluster size (HDBSCAN)
            dbscan_eps: Epsilon parameter (DBSCAN)
            dbscan_min_samples: Min samples (DBSCAN)
            strategy: Merge strategy
            dry_run: Preview mode (no modifications)

        Returns:
            DeduplicationResult with summary
        """
        logger.info("=" * 60)
        logger.info(f"MEMORY DEDUPLICATION - {'DRY RUN' if dry_run else 'LIVE'}")
        logger.info("=" * 60)
        logger.info(f"Collection: {self.collection_name}")
        logger.info(f"Method: {method.value}")
        logger.info(f"Strategy: {strategy.value}")
        logger.info("")

        # Get initial count
        collection_info = self.qdrant_client.get_collection(self.collection_name)
        initial_count = collection_info.points_count
        logger.info(f"Initial memory count: {initial_count}")

        # Find duplicate groups
        clusters = self.find_duplicate_groups(
            method=method,
            min_cluster_size=min_cluster_size,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
        )

        if not clusters:
            logger.info("No duplicates found!")
            return DeduplicationResult(
                total_scanned=initial_count,
                duplicate_groups=0,
                memories_merged=0,
                memories_deleted=0,
                dry_run=dry_run,
                metrics=getattr(self, 'metrics', None),
            )

        # Merge clusters
        memories_merged = 0
        memories_deleted = 0
        cluster_details = []

        for cluster in clusters:
            try:
                result = self.merge_cluster(cluster, strategy=strategy, dry_run=dry_run)
                memories_merged += 1
                memories_deleted += len(result["deleted_ids"])
                cluster_details.append(result)
            except Exception as e:
                logger.error(f"Failed to merge cluster {cluster.cluster_id}: {e}")

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}DEDUPLICATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  Duplicate groups processed: {len(clusters)}")
        logger.info(f"  Memories merged: {memories_merged}")
        logger.info(f"  Memories deleted: {memories_deleted}")

        if not dry_run:
            final_info = self.qdrant_client.get_collection(self.collection_name)
            logger.info(f"  Final memory count: {final_info.points_count}")

        return DeduplicationResult(
            total_scanned=initial_count,
            duplicate_groups=len(clusters),
            memories_merged=memories_merged,
            memories_deleted=memories_deleted,
            dry_run=dry_run,
            metrics=getattr(self, 'metrics', None),
            cluster_details=cluster_details,
        )
