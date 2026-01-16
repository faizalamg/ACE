"""Re-index Qdrant with OpenRouter qwen/qwen3-embedding-8b.

Usage:
    python reindex_with_openrouter.py <api_key>
"""

import os
import sys
import json
import logging
import uuid
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
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
OPENROUTER_URL = "https://openrouter.ai/api/v1"
SOURCE_COLLECTION = _qdrant_config.unified_collection  # ace_memories_hybrid
TARGET_COLLECTION = _qdrant_config.memories_collection  # ace_memories_hybrid
MODEL = "qwen/qwen3-embedding-8b"  # OpenRouter model name
EMBEDDING_DIM = _embedding_config.dimension
BATCH_SIZE = 50

# Setup logging
log_dir = Path(__file__).parent.parent / "optimization_results"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / "reindex_openrouter.log"),
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class MemoryPoint:
    id: str
    payload: Dict[str, Any]
    text: str


def extract_text_for_embedding(payload: Dict[str, Any]) -> str:
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
                logger.error(f"Scroll failed: {resp.status_code}")
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

            logger.info(f"Extracted {len(memories)} memories...")

    return memories


def get_openrouter_embedding(client: httpx.Client, text: str, api_key: str) -> List[float]:
    """Get embedding from OpenRouter."""
    try:
        resp = client.post(
            f"{OPENROUTER_URL}/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "input": text[:8000]
            },
            timeout=60.0
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["data"][0]["embedding"]
        else:
            logger.error(f"OpenRouter error {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.error(f"Embedding error: {e}")
    return []


def get_openrouter_embeddings_batch(client: httpx.Client, texts: List[str], api_key: str) -> List[List[float]]:
    """Get batch embeddings from OpenRouter."""
    try:
        resp = client.post(
            f"{OPENROUTER_URL}/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "input": [t[:8000] for t in texts]
            },
            timeout=120.0
        )
        if resp.status_code == 200:
            data = resp.json()
            # Sort by index to maintain order
            embeddings = sorted(data["data"], key=lambda x: x["index"])
            return [e["embedding"] for e in embeddings]
        else:
            logger.error(f"OpenRouter batch error {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.error(f"Batch embedding error: {e}")
    return [[]] * len(texts)


def compute_bm25_sparse(text: str) -> Dict[str, Any]:
    words = re.findall(r'\w+', text.lower())
    tf = Counter(words)
    k1, b, avg_dl = 1.5, 0.75, 50
    indices, values = [], []
    for word, count in tf.items():
        idx = abs(hash(word)) % 50000
        dl = len(words)
        tf_weight = (count * (k1 + 1)) / (count + k1 * (1 - b + b * dl / avg_dl))
        indices.append(idx)
        values.append(float(tf_weight))
    return {"indices": indices, "values": values}


def create_collection(qdrant_url: str, collection: str, dim: int) -> bool:
    with httpx.Client(timeout=30.0) as client:
        client.delete(f"{qdrant_url}/collections/{collection}")
        resp = client.put(
            f"{qdrant_url}/collections/{collection}",
            json={
                "vectors": {"dense": {"size": dim, "distance": "Cosine"}},
                "sparse_vectors": {"sparse": {}}
            }
        )
        if resp.status_code == 200:
            logger.info(f"Created collection: {collection} (dim={dim})")
            return True
        logger.error(f"Failed: {resp.text}")
        return False


def upsert_points(qdrant_url: str, collection: str, points: List[Dict]) -> bool:
    with httpx.Client(timeout=120.0) as client:
        resp = client.put(
            f"{qdrant_url}/collections/{collection}/points",
            json={"points": points}
        )
        return resp.status_code == 200


def normalize_point_id(original_id: str) -> str:
    try:
        uuid.UUID(original_id)
        return original_id
    except (ValueError, AttributeError):
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
        return str(uuid.uuid5(namespace, original_id))


def reindex_with_openrouter(api_key: str) -> dict:
    start_time = datetime.now()
    stats = {
        "start_time": start_time.isoformat(),
        "model": MODEL,
        "memories_extracted": 0,
        "embeddings_generated": 0,
        "points_indexed": 0,
        "embedding_dim": 0,
        "errors": [],
    }

    logger.info("=" * 60)
    logger.info(f"RE-INDEX WITH OpenRouter: {MODEL}")
    logger.info("=" * 60)

    # Step 1: Extract memories
    logger.info("\n[1/5] Extracting memories...")
    memories = get_all_memories(QDRANT_URL, SOURCE_COLLECTION)
    stats["memories_extracted"] = len(memories)
    logger.info(f"      Extracted {len(memories)} memories")

    if not memories:
        return stats

    # Step 2: Test embedding to get dimension
    logger.info("\n[2/5] Testing OpenRouter connection...")
    with httpx.Client() as client:
        test_emb = get_openrouter_embedding(client, "test", api_key)
        if not test_emb:
            logger.error("OpenRouter connection failed!")
            return stats
        dim = len(test_emb)
        stats["embedding_dim"] = dim
        logger.info(f"      Model dimension: {dim}")

    # Step 3: Create collection
    logger.info(f"\n[3/5] Creating collection: {TARGET_COLLECTION}...")
    if not create_collection(QDRANT_URL, TARGET_COLLECTION, dim):
        return stats

    # Step 4: Generate embeddings and index
    logger.info(f"\n[4/5] Generating embeddings...")
    total_batches = math.ceil(len(memories) / BATCH_SIZE)

    with httpx.Client(timeout=120.0) as client:
        for batch_idx in range(total_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, len(memories))
            batch = memories[batch_start:batch_end]

            logger.info(f"      Batch {batch_idx + 1}/{total_batches}: {len(batch)} memories")

            # Get batch embeddings
            texts = [m.text for m in batch]
            embeddings = get_openrouter_embeddings_batch(client, texts, api_key)
            stats["embeddings_generated"] += len([e for e in embeddings if e])

            # Build points
            points = []
            for memory, embedding in zip(batch, embeddings):
                if not embedding:
                    logger.warning(f"Skipping {memory.id} - no embedding")
                    continue

                point_id = normalize_point_id(memory.id)
                sparse = compute_bm25_sparse(memory.text)

                payload = memory.payload.copy()
                payload["_original_id"] = memory.id

                points.append({
                    "id": point_id,
                    "vector": {"dense": embedding, "sparse": sparse},
                    "payload": payload
                })

            if points and upsert_points(QDRANT_URL, TARGET_COLLECTION, points):
                stats["points_indexed"] += len(points)
                logger.info(f"      Indexed {len(points)} points")

    # Step 5: Verify
    logger.info("\n[5/5] Verifying...")
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{QDRANT_URL}/collections/{TARGET_COLLECTION}/points/scroll",
            json={"limit": 1, "with_payload": True, "with_vectors": True}
        )
        if resp.status_code == 200:
            points = resp.json().get("result", {}).get("points", [])
            if points:
                stored = points[0].get("vector", {}).get("dense", [])
                payload = points[0].get("payload", {})
                text = extract_text_for_embedding(payload)

                fresh = get_openrouter_embedding(client, text, api_key)
                if stored and fresh:
                    s, f = np.array(stored), np.array(fresh)
                    cosine = np.dot(s, f) / (np.linalg.norm(s) * np.linalg.norm(f))
                    logger.info(f"      Cosine similarity: {cosine:.6f}")
                    logger.info(f"      Consistent: {cosine > 0.95}")

    # Final stats
    duration = (datetime.now() - start_time).total_seconds()
    stats["duration_seconds"] = duration

    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model: {MODEL}")
    logger.info(f"Dimension: {stats['embedding_dim']}")
    logger.info(f"Memories: {stats['memories_extracted']}")
    logger.info(f"Indexed: {stats['points_indexed']}")
    logger.info(f"Duration: {duration:.1f}s")

    # Save stats
    with open(log_dir / "reindex_openrouter_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


if __name__ == "__main__":
    api_key = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        print(f"Usage: python reindex_with_openrouter.py <api_key>")
        print(f"Model: {MODEL}")
        sys.exit(1)

    reindex_with_openrouter(api_key)
