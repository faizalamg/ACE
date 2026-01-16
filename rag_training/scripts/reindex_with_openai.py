"""Re-index Qdrant Collection with OpenAI text-embedding-3-large.

This script:
1. Extracts all memory payloads from existing Qdrant collection
2. Generates fresh embeddings using OpenAI text-embedding-3-large (768 dims)
3. Recreates the collection with new vectors
4. Verifies embedding consistency

Usage:
    python reindex_with_openai.py <openai_api_key> [--replace]

Environment:
    OPENAI_API_KEY: Can also be set as environment variable
"""

import os
import sys
import json
import logging
import uuid
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import math
from collections import Counter

import httpx
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ace.openai_embeddings import OpenAIEmbeddingClient
from ace.config import get_qdrant_config

# Configuration from centralized config
_qdrant_config = get_qdrant_config()
QDRANT_URL = _qdrant_config.url
SOURCE_COLLECTION = _qdrant_config.unified_collection  # ace_memories_hybrid
TARGET_COLLECTION = _qdrant_config.memories_collection  # ace_memories_hybrid
EMBEDDING_DIM = 1536  # text-embedding-3-small native dimension (OpenAI specific)
BATCH_SIZE = 100  # OpenAI supports larger batches

# Setup logging
log_dir = Path(__file__).parent.parent / "optimization_results"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / "reindex_openai.log"),
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class MemoryPoint:
    """Represents a memory point from Qdrant."""
    id: str  # Original ID (could be UUID or int)
    payload: Dict[str, Any]
    text: str  # Combined text for embedding


def extract_text_for_embedding(payload: Dict[str, Any]) -> str:
    """Extract text to embed from payload.

    Combines lesson, category, and other relevant fields.
    """
    parts = []

    # Primary text field
    if payload.get("lesson"):
        parts.append(payload["lesson"])
    elif payload.get("text"):
        parts.append(payload["text"])
    elif payload.get("content"):
        parts.append(payload["content"])

    # Add category for context
    if payload.get("category"):
        parts.append(f"[Category: {payload['category']}]")

    # Add feedback type for semantic context
    if payload.get("feedback_type"):
        parts.append(f"[Type: {payload['feedback_type']}]")

    return " ".join(parts) if parts else ""


def get_all_memories(qdrant_url: str, collection: str) -> List[MemoryPoint]:
    """Extract all memories from Qdrant collection.

    Args:
        qdrant_url: Qdrant server URL
        collection: Collection name

    Returns:
        List of MemoryPoint objects
    """
    memories = []
    offset = None

    with httpx.Client(timeout=60.0) as client:
        while True:
            # Build scroll request
            scroll_body = {"limit": 100, "with_payload": True}
            if offset is not None:
                scroll_body["offset"] = offset

            resp = client.post(
                f"{qdrant_url}/collections/{collection}/points/scroll",
                json=scroll_body
            )

            if resp.status_code != 200:
                logger.error(f"Scroll failed: {resp.status_code} - {resp.text}")
                break

            data = resp.json()
            points = data.get("result", {}).get("points", [])

            if not points:
                break

            for point in points:
                payload = point.get("payload", {})
                text = extract_text_for_embedding(payload)

                if text:
                    # Keep original ID as string
                    original_id = point.get("id")
                    memories.append(MemoryPoint(
                        id=str(original_id),
                        payload=payload,
                        text=text
                    ))

            # Get next offset
            offset = data.get("result", {}).get("next_page_offset")
            if offset is None:
                break

            logger.info(f"Extracted {len(memories)} memories so far...")

    return memories


def compute_bm25_sparse(text: str) -> Dict[str, Any]:
    """Compute BM25 sparse vector for hybrid search.

    Args:
        text: Input text

    Returns:
        Sparse vector in Qdrant format
    """
    # Tokenize
    words = re.findall(r'\w+', text.lower())

    # Count term frequencies
    tf = Counter(words)

    # BM25 parameters
    k1 = 1.5
    b = 0.75
    avg_dl = 50  # Average document length

    # Compute BM25 weights
    indices = []
    values = []

    for word, count in tf.items():
        # Simple hash to index (Qdrant expects integer indices)
        # Use modulo to keep within safe range
        idx = abs(hash(word)) % 50000

        # BM25 term weight
        dl = len(words)
        tf_weight = (count * (k1 + 1)) / (count + k1 * (1 - b + b * dl / avg_dl))

        indices.append(idx)
        values.append(float(tf_weight))

    return {"indices": indices, "values": values}


def create_collection(qdrant_url: str, collection: str) -> bool:
    """Create new Qdrant collection for OpenAI embeddings.

    Args:
        qdrant_url: Qdrant server URL
        collection: Collection name

    Returns:
        True if successful
    """
    with httpx.Client(timeout=30.0) as client:
        # Delete if exists
        client.delete(f"{qdrant_url}/collections/{collection}")

        # Create new collection with hybrid vectors
        create_body = {
            "vectors": {
                "dense": {
                    "size": EMBEDDING_DIM,
                    "distance": "Cosine"
                }
            },
            "sparse_vectors": {
                "sparse": {}  # BM25 sparse vectors
            }
        }

        resp = client.put(
            f"{qdrant_url}/collections/{collection}",
            json=create_body
        )

        if resp.status_code == 200:
            logger.info(f"Created collection: {collection}")
            return True
        else:
            logger.error(f"Failed to create collection: {resp.text}")
            return False


def upsert_points(
    qdrant_url: str,
    collection: str,
    points: List[Dict[str, Any]]
) -> bool:
    """Upsert points to Qdrant.

    Args:
        qdrant_url: Qdrant server URL
        collection: Collection name
        points: List of point dictionaries

    Returns:
        True if successful
    """
    with httpx.Client(timeout=120.0) as client:
        resp = client.put(
            f"{qdrant_url}/collections/{collection}/points",
            json={"points": points}
        )

        if resp.status_code == 200:
            return True
        else:
            logger.error(f"Upsert failed: {resp.text}")
            return False


def is_valid_uuid(s: str) -> bool:
    """Check if string is a valid UUID."""
    try:
        uuid.UUID(s)
        return True
    except (ValueError, AttributeError):
        return False


def normalize_point_id(original_id: str) -> str:
    """Normalize point ID to valid Qdrant format (UUID).

    Args:
        original_id: Original ID string

    Returns:
        Valid UUID string
    """
    # If already valid UUID, return as-is
    if is_valid_uuid(original_id):
        return original_id

    # Otherwise, generate deterministic UUID from original ID
    # Using UUID5 with a namespace to ensure consistency
    namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # URL namespace
    return str(uuid.uuid5(namespace, original_id))


def reindex_with_openai(api_key: str) -> dict:
    """Main re-indexing function.

    Reads from SOURCE_COLLECTION (ace_unified) and writes to
    TARGET_COLLECTION (ace_memories_hybrid) with OpenAI embeddings.

    Args:
        api_key: OpenAI API key

    Returns:
        Statistics dictionary
    """
    start_time = datetime.now()
    stats = {
        "start_time": start_time.isoformat(),
        "memories_extracted": 0,
        "embeddings_generated": 0,
        "points_indexed": 0,
        "errors": [],
    }

    logger.info("=" * 60)
    logger.info("RE-INDEX WITH OPENAI text-embedding-3-small (1536 dims)")
    logger.info("=" * 60)

    # Step 1: Extract all memories
    logger.info("\n[1/4] Extracting memories from Qdrant...")
    memories = get_all_memories(QDRANT_URL, SOURCE_COLLECTION)
    stats["memories_extracted"] = len(memories)
    logger.info(f"      Extracted {len(memories)} memories")

    if not memories:
        logger.error("No memories found to re-index!")
        return stats

    # Step 2: Initialize OpenAI client
    logger.info("\n[2/4] Initializing OpenAI embedding client...")
    try:
        openai = OpenAIEmbeddingClient(api_key=api_key, dimension=EMBEDDING_DIM)
        logger.info("      OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        stats["errors"].append(str(e))
        return stats

    # Step 3: Create new collection
    target_collection = TARGET_COLLECTION
    logger.info(f"\n[3/4] Creating collection: {target_collection}...")

    if not create_collection(QDRANT_URL, target_collection):
        logger.error("Failed to create collection")
        return stats

    # Step 4: Generate embeddings and index in batches
    logger.info(f"\n[4/4] Generating OpenAI embeddings and indexing...")
    logger.info(f"      Processing {len(memories)} memories in batches of {BATCH_SIZE}")

    total_batches = math.ceil(len(memories) / BATCH_SIZE)

    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(memories))
        batch = memories[batch_start:batch_end]

        logger.info(f"      Batch {batch_idx + 1}/{total_batches}: {len(batch)} memories")

        # Extract texts
        texts = [m.text for m in batch]

        # Generate embeddings with OpenAI
        try:
            embeddings = openai.embed_batch(texts)
            stats["embeddings_generated"] += len(embeddings)
        except Exception as e:
            logger.error(f"Embedding batch failed: {e}")
            stats["errors"].append(f"Batch {batch_idx}: {e}")
            continue

        # Build points
        points = []
        for i, (memory, embedding) in enumerate(zip(batch, embeddings)):
            # Skip if embedding is all zeros (failed)
            if all(v == 0.0 for v in embedding[:10]):
                logger.warning(f"Skipping memory {memory.id} - embedding failed")
                continue

            # Normalize ID to valid UUID
            point_id = normalize_point_id(memory.id)

            # Compute sparse BM25 vector
            sparse = compute_bm25_sparse(memory.text)

            # Store original ID in payload for reference
            payload = memory.payload.copy()
            payload["_original_id"] = memory.id

            point = {
                "id": point_id,
                "vector": {
                    "dense": embedding,
                    "sparse": sparse
                },
                "payload": payload
            }
            points.append(point)

        # Upsert batch
        if points:
            if upsert_points(QDRANT_URL, target_collection, points):
                stats["points_indexed"] += len(points)
                logger.info(f"      Indexed {len(points)} points")
            else:
                logger.error(f"      Failed to index batch {batch_idx + 1}")

    # Cleanup
    openai.close()

    # Final stats
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    stats["end_time"] = end_time.isoformat()
    stats["duration_seconds"] = duration

    logger.info("\n" + "=" * 60)
    logger.info("RE-INDEXING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Collection: {target_collection}")
    logger.info(f"Memories extracted: {stats['memories_extracted']}")
    logger.info(f"Embeddings generated: {stats['embeddings_generated']}")
    logger.info(f"Points indexed: {stats['points_indexed']}")
    logger.info(f"Duration: {duration:.1f}s")
    if stats["errors"]:
        logger.warning(f"Errors: {len(stats['errors'])}")

    # Save stats
    stats_file = log_dir / "reindex_openai_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"\nStats saved to: {stats_file}")

    return stats


def verify_embeddings(api_key: str, collection: str = TARGET_COLLECTION) -> dict:
    """Verify embedding consistency after re-indexing.

    Args:
        api_key: OpenAI API key
        collection: Collection to verify

    Returns:
        Verification results
    """
    logger.info("\n" + "=" * 60)
    logger.info("EMBEDDING CONSISTENCY VERIFICATION")
    logger.info("=" * 60)

    openai = OpenAIEmbeddingClient(api_key=api_key, dimension=EMBEDDING_DIM)

    # Get a sample point from collection
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{QDRANT_URL}/collections/{collection}/points/scroll",
            json={"limit": 1, "with_payload": True, "with_vectors": True}
        )

        if resp.status_code != 200:
            logger.error(f"Failed to get sample point: {resp.text}")
            return {"error": "Failed to get sample"}

        data = resp.json()
        points = data.get("result", {}).get("points", [])

        if not points:
            logger.error("No points in collection")
            return {"error": "No points"}

        point = points[0]
        stored_vector = point.get("vector", {}).get("dense", [])
        payload = point.get("payload", {})
        text = extract_text_for_embedding(payload)

    # Generate fresh embedding
    fresh_vector = openai.get_embedding(text)
    openai.close()

    # Compare
    stored = np.array(stored_vector)
    fresh = np.array(fresh_vector)

    cosine_sim = np.dot(stored, fresh) / (np.linalg.norm(stored) * np.linalg.norm(fresh))

    result = {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "stored_vector_sample": stored_vector[:5],
        "fresh_vector_sample": fresh_vector[:5],
        "cosine_similarity": float(cosine_sim),
        "consistent": cosine_sim > 0.95,
    }

    logger.info(f"\nText: {result['text']}")
    logger.info(f"Stored vector (first 5): {result['stored_vector_sample']}")
    logger.info(f"Fresh vector (first 5): {result['fresh_vector_sample']}")
    logger.info(f"\nCosine similarity: {cosine_sim:.6f}")
    logger.info(f"Consistent: {result['consistent']}")

    if result["consistent"]:
        logger.info("\n[SUCCESS] Embeddings are consistent!")
    else:
        logger.error("\n[FAILURE] Embeddings are NOT consistent!")

    return result


if __name__ == "__main__":
    # Get API key from argument or environment
    api_key = None

    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            api_key = arg

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Usage: python reindex_with_openai.py <openai_api_key>")
        print(f"\nSource: {SOURCE_COLLECTION}")
        print(f"Target: {TARGET_COLLECTION}")
        print("\nOr set OPENAI_API_KEY environment variable")
        sys.exit(1)

    # Run re-indexing
    stats = reindex_with_openai(api_key)

    # Verify if successful
    if stats["points_indexed"] > 0:
        verify_embeddings(api_key, TARGET_COLLECTION)
