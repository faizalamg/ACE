"""Re-index Qdrant Collection with LM Studio Embeddings.

Uses the FREE local LM Studio server for embeddings.
Available models: nomic-embed-text-v1.5, snowflake-arctic-embed-m-v1.5

Usage:
    python reindex_with_lmstudio.py [--model MODEL_NAME]

Default: text-embedding-nomic-embed-text-v1.5 (768 dims)
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

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from ace.config import get_qdrant_config, get_embedding_config

# Configuration from centralized config
_qdrant_config = get_qdrant_config()
_embedding_config = get_embedding_config()

QDRANT_URL = _qdrant_config.url
LMSTUDIO_URL = _embedding_config.url
SOURCE_COLLECTION = _qdrant_config.unified_collection  # ace_memories_hybrid
TARGET_COLLECTION = _qdrant_config.memories_collection  # ace_memories_hybrid
DEFAULT_MODEL = _embedding_config.model
EMBEDDING_DIM = _embedding_config.dimension
BATCH_SIZE = 50  # LM Studio works best with smaller batches

# Setup logging
log_dir = Path(__file__).parent.parent / "optimization_results"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / "reindex_lmstudio.log"),
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class MemoryPoint:
    """Represents a memory point from Qdrant."""
    id: str
    payload: Dict[str, Any]
    text: str


def extract_text_for_embedding(payload: Dict[str, Any]) -> str:
    """Extract text to embed from payload."""
    parts = []

    if payload.get("lesson"):
        parts.append(payload["lesson"])
    elif payload.get("text"):
        parts.append(payload["text"])
    elif payload.get("content"):
        parts.append(payload["content"])

    if payload.get("category"):
        parts.append(f"[Category: {payload['category']}]")

    if payload.get("feedback_type"):
        parts.append(f"[Type: {payload['feedback_type']}]")

    return " ".join(parts) if parts else ""


def get_all_memories(qdrant_url: str, collection: str) -> List[MemoryPoint]:
    """Extract all memories from Qdrant collection."""
    memories = []
    offset = None

    with httpx.Client(timeout=60.0) as client:
        while True:
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

            offset = data.get("result", {}).get("next_page_offset")
            if offset is None:
                break

            logger.info(f"Extracted {len(memories)} memories so far...")

    return memories


def get_lmstudio_embedding(client: httpx.Client, text: str, model: str) -> List[float]:
    """Get embedding from LM Studio."""
    try:
        resp = client.post(
            f"{LMSTUDIO_URL}/v1/embeddings",
            json={"model": model, "input": text[:8000]}
        )
        if resp.status_code == 200:
            return resp.json()["data"][0]["embedding"]
        else:
            logger.error(f"Embedding failed: {resp.status_code}")
    except Exception as e:
        logger.error(f"Embedding error: {e}")
    return [0.0] * EMBEDDING_DIM


def get_lmstudio_embeddings_batch(client: httpx.Client, texts: List[str], model: str) -> List[List[float]]:
    """Get embeddings for multiple texts from LM Studio."""
    embeddings = []
    for text in texts:
        emb = get_lmstudio_embedding(client, text, model)
        embeddings.append(emb)
    return embeddings


def compute_bm25_sparse(text: str) -> Dict[str, Any]:
    """Compute BM25 sparse vector."""
    words = re.findall(r'\w+', text.lower())
    tf = Counter(words)

    k1 = 1.5
    b = 0.75
    avg_dl = 50

    indices = []
    values = []

    for word, count in tf.items():
        idx = abs(hash(word)) % 50000
        dl = len(words)
        tf_weight = (count * (k1 + 1)) / (count + k1 * (1 - b + b * dl / avg_dl))
        indices.append(idx)
        values.append(float(tf_weight))

    return {"indices": indices, "values": values}


def create_collection(qdrant_url: str, collection: str, dim: int) -> bool:
    """Create new Qdrant collection."""
    with httpx.Client(timeout=30.0) as client:
        client.delete(f"{qdrant_url}/collections/{collection}")

        create_body = {
            "vectors": {
                "dense": {
                    "size": dim,
                    "distance": "Cosine"
                }
            },
            "sparse_vectors": {
                "sparse": {}
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


def upsert_points(qdrant_url: str, collection: str, points: List[Dict[str, Any]]) -> bool:
    """Upsert points to Qdrant."""
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


def normalize_point_id(original_id: str) -> str:
    """Normalize point ID to valid UUID."""
    try:
        uuid.UUID(original_id)
        return original_id
    except (ValueError, AttributeError):
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
        return str(uuid.uuid5(namespace, original_id))


def reindex_with_lmstudio(model: str = DEFAULT_MODEL) -> dict:
    """Main re-indexing function using LM Studio."""
    start_time = datetime.now()
    stats = {
        "start_time": start_time.isoformat(),
        "model": model,
        "memories_extracted": 0,
        "embeddings_generated": 0,
        "points_indexed": 0,
        "errors": [],
    }

    logger.info("=" * 60)
    logger.info(f"RE-INDEX WITH LM Studio: {model}")
    logger.info("=" * 60)

    # Step 1: Extract memories
    logger.info("\n[1/4] Extracting memories from Qdrant...")
    memories = get_all_memories(QDRANT_URL, SOURCE_COLLECTION)
    stats["memories_extracted"] = len(memories)
    logger.info(f"      Extracted {len(memories)} memories")

    if not memories:
        logger.error("No memories found!")
        return stats

    # Step 2: Verify LM Studio
    logger.info("\n[2/4] Verifying LM Studio connection...")
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{LMSTUDIO_URL}/v1/models")
            if resp.status_code == 200:
                models = [m["id"] for m in resp.json().get("data", [])]
                if model in models:
                    logger.info(f"      Model available: {model}")
                else:
                    logger.error(f"      Model not found: {model}")
                    logger.info(f"      Available: {models}")
                    return stats
            else:
                logger.error(f"      LM Studio not responding")
                return stats
    except Exception as e:
        logger.error(f"      LM Studio connection failed: {e}")
        return stats

    # Step 3: Create collection
    logger.info(f"\n[3/4] Creating collection: {TARGET_COLLECTION}...")
    if not create_collection(QDRANT_URL, TARGET_COLLECTION, EMBEDDING_DIM):
        return stats

    # Step 4: Generate embeddings and index
    logger.info(f"\n[4/4] Generating embeddings and indexing...")
    logger.info(f"      Processing {len(memories)} memories in batches of {BATCH_SIZE}")

    total_batches = math.ceil(len(memories) / BATCH_SIZE)

    with httpx.Client(timeout=60.0) as client:
        for batch_idx in range(total_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, len(memories))
            batch = memories[batch_start:batch_end]

            logger.info(f"      Batch {batch_idx + 1}/{total_batches}: {len(batch)} memories")

            # Generate embeddings
            embeddings = get_lmstudio_embeddings_batch(
                client,
                [m.text for m in batch],
                model
            )
            stats["embeddings_generated"] += len(embeddings)

            # Build points
            points = []
            for memory, embedding in zip(batch, embeddings):
                if all(v == 0.0 for v in embedding[:10]):
                    logger.warning(f"Skipping {memory.id} - embedding failed")
                    continue

                point_id = normalize_point_id(memory.id)
                sparse = compute_bm25_sparse(memory.text)

                payload = memory.payload.copy()
                payload["_original_id"] = memory.id

                points.append({
                    "id": point_id,
                    "vector": {
                        "dense": embedding,
                        "sparse": sparse
                    },
                    "payload": payload
                })

            # Upsert
            if points:
                if upsert_points(QDRANT_URL, TARGET_COLLECTION, points):
                    stats["points_indexed"] += len(points)
                    logger.info(f"      Indexed {len(points)} points")

    # Final stats
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    stats["end_time"] = end_time.isoformat()
    stats["duration_seconds"] = duration

    logger.info("\n" + "=" * 60)
    logger.info("RE-INDEXING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info(f"Memories extracted: {stats['memories_extracted']}")
    logger.info(f"Embeddings generated: {stats['embeddings_generated']}")
    logger.info(f"Points indexed: {stats['points_indexed']}")
    logger.info(f"Duration: {duration:.1f}s")

    # Save stats
    stats_file = log_dir / "reindex_lmstudio_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"\nStats saved to: {stats_file}")

    return stats


def verify_embeddings(model: str = DEFAULT_MODEL) -> dict:
    """Verify embedding consistency."""
    logger.info("\n" + "=" * 60)
    logger.info("EMBEDDING CONSISTENCY VERIFICATION")
    logger.info("=" * 60)

    with httpx.Client(timeout=30.0) as client:
        # Get sample from collection
        resp = client.post(
            f"{QDRANT_URL}/collections/{TARGET_COLLECTION}/points/scroll",
            json={"limit": 1, "with_payload": True, "with_vectors": True}
        )

        if resp.status_code != 200:
            return {"error": "Failed to get sample"}

        points = resp.json().get("result", {}).get("points", [])
        if not points:
            return {"error": "No points"}

        point = points[0]
        stored_vector = point.get("vector", {}).get("dense", [])
        payload = point.get("payload", {})
        text = extract_text_for_embedding(payload)

        # Generate fresh embedding
        fresh_vector = get_lmstudio_embedding(client, text, model)

    # Compare
    stored = np.array(stored_vector)
    fresh = np.array(fresh_vector)

    cosine_sim = np.dot(stored, fresh) / (np.linalg.norm(stored) * np.linalg.norm(fresh))

    result = {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "cosine_similarity": float(cosine_sim),
        "consistent": cosine_sim > 0.95,
    }

    logger.info(f"\nText: {result['text']}")
    logger.info(f"Cosine similarity: {cosine_sim:.6f}")
    logger.info(f"Consistent: {result['consistent']}")

    if result["consistent"]:
        logger.info("\n[SUCCESS] Embeddings are consistent!")
    else:
        logger.error("\n[FAILURE] Embeddings are NOT consistent!")

    return result


if __name__ == "__main__":
    model = DEFAULT_MODEL

    for arg in sys.argv[1:]:
        if arg.startswith("--model="):
            model = arg.split("=")[1]

    print(f"Using model: {model}")
    print(f"Source: {SOURCE_COLLECTION}")
    print(f"Target: {TARGET_COLLECTION}")

    stats = reindex_with_lmstudio(model)

    if stats["points_indexed"] > 0:
        verify_embeddings(model)
