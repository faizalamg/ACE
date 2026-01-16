"""Re-index Qdrant Collection with Gemini Embeddings.

This script:
1. Extracts all memory payloads from existing Qdrant collection
2. Generates fresh embeddings using Gemini gemini-embedding-001
3. Recreates the collection with new vectors
4. Verifies embedding consistency

Usage:
    python reindex_with_gemini.py <gemini_api_key>

Environment:
    GEMINI_API_KEY: Can also be set as environment variable
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import math

import httpx
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ace.gemini_embeddings import GeminiEmbeddingClient

# Configuration
QDRANT_URL = "http://localhost:6333"
OLD_COLLECTION = "ace_memories_hybrid"
NEW_COLLECTION = "ace_memories_gemini"  # New collection with Gemini embeddings
EMBEDDING_DIM = 768
BATCH_SIZE = 50  # Gemini batch size

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent.parent / "optimization_results" / "reindex_gemini.log"),
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class MemoryPoint:
    """Represents a memory point from Qdrant."""
    id: str
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
                    memories.append(MemoryPoint(
                        id=str(point.get("id")),
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
    import re
    from collections import Counter

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
        idx = abs(hash(word)) % 100000

        # BM25 term weight
        dl = len(words)
        tf_weight = (count * (k1 + 1)) / (count + k1 * (1 - b + b * dl / avg_dl))

        indices.append(idx)
        values.append(float(tf_weight))

    return {"indices": indices, "values": values}


def create_collection(qdrant_url: str, collection: str) -> bool:
    """Create new Qdrant collection for Gemini embeddings.

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


def reindex_with_gemini(api_key: str, replace_original: bool = False) -> dict:
    """Main re-indexing function.

    Args:
        api_key: Gemini API key
        replace_original: If True, replace original collection

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
    logger.info("RE-INDEX WITH GEMINI EMBEDDINGS")
    logger.info("=" * 60)

    # Step 1: Extract all memories
    logger.info("\n[1/4] Extracting memories from Qdrant...")
    memories = get_all_memories(QDRANT_URL, OLD_COLLECTION)
    stats["memories_extracted"] = len(memories)
    logger.info(f"      Extracted {len(memories)} memories")

    if not memories:
        logger.error("No memories found to re-index!")
        return stats

    # Step 2: Initialize Gemini client
    logger.info("\n[2/4] Initializing Gemini embedding client...")
    try:
        gemini = GeminiEmbeddingClient(api_key=api_key)
        logger.info("      Gemini client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        stats["errors"].append(str(e))
        return stats

    # Step 3: Create new collection
    target_collection = OLD_COLLECTION if replace_original else NEW_COLLECTION
    logger.info(f"\n[3/4] Creating collection: {target_collection}...")

    if not create_collection(QDRANT_URL, target_collection):
        logger.error("Failed to create collection")
        return stats

    # Step 4: Generate embeddings and index in batches
    logger.info(f"\n[4/4] Generating Gemini embeddings and indexing...")
    logger.info(f"      Processing {len(memories)} memories in batches of {BATCH_SIZE}")

    total_batches = math.ceil(len(memories) / BATCH_SIZE)

    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(memories))
        batch = memories[batch_start:batch_end]

        logger.info(f"      Batch {batch_idx + 1}/{total_batches}: {len(batch)} memories")

        # Extract texts
        texts = [m.text for m in batch]

        # Generate embeddings with Gemini (document task type for indexing)
        try:
            embeddings = gemini.embed_batch(texts, task_type="RETRIEVAL_DOCUMENT")
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

            # Compute sparse BM25 vector
            sparse = compute_bm25_sparse(memory.text)

            point = {
                "id": memory.id if memory.id.isdigit() else abs(hash(memory.id)) % (2**63),
                "vector": {
                    "dense": embedding,
                    "sparse": sparse
                },
                "payload": memory.payload
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
    gemini.close()

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
    stats_file = Path(__file__).parent.parent / "optimization_results" / "reindex_gemini_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"\nStats saved to: {stats_file}")

    return stats


def verify_embeddings(api_key: str, collection: str = NEW_COLLECTION) -> dict:
    """Verify embedding consistency after re-indexing.

    Args:
        api_key: Gemini API key
        collection: Collection to verify

    Returns:
        Verification results
    """
    logger.info("\n" + "=" * 60)
    logger.info("EMBEDDING CONSISTENCY VERIFICATION")
    logger.info("=" * 60)

    gemini = GeminiEmbeddingClient(api_key=api_key)

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
    fresh_vector = gemini.embed_document(text)
    gemini.close()

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
    replace_original = False

    for arg in sys.argv[1:]:
        if arg == "--replace":
            replace_original = True
        elif not arg.startswith("--"):
            api_key = arg

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Usage: python reindex_with_gemini.py <gemini_api_key> [--replace]")
        print("\nOptions:")
        print("  --replace  Replace original collection instead of creating new one")
        print("\nOr set GEMINI_API_KEY environment variable")
        sys.exit(1)

    # Run re-indexing
    stats = reindex_with_gemini(api_key, replace_original=replace_original)

    # Verify if successful
    if stats["points_indexed"] > 0:
        target = OLD_COLLECTION if replace_original else NEW_COLLECTION
        verify_embeddings(api_key, target)
