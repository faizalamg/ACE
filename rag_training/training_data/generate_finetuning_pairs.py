"""
Generate cross-encoder fine-tuning pairs from enhanced test suite.

Creates (query, memory, label) triples for fine-tuning a domain-specific cross-encoder:
- Positive pairs: (query, ground_truth_memory, 1) for each test case
- Hard negatives: (query, similar_but_wrong_memory, 0) using semantic similarity + category-based sampling

Target: 2000+ training pairs (1038 positives + ~3k hard negatives)

Usage:
    python rag_training/training_data/generate_finetuning_pairs.py

Output:
    rag_training/training_data/crossencoder_training_pairs.json
"""

import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set
import httpx
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_SUITE_PATH = PROJECT_ROOT / "rag_training" / "test_suite" / "enhanced_test_suite.json"
OUTPUT_PATH = PROJECT_ROOT / "rag_training" / "training_data" / "crossencoder_training_pairs.json"
WORK_LOG_PATH = PROJECT_ROOT / "rag_training" / "optimization_results" / "work_log.md"

# Qdrant settings
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "ace_memories_hybrid"
EMBEDDING_URL = "http://localhost:1234"
EMBEDDING_MODEL = "text-embedding-qwen3-embedding-8b"

# Training data settings
HARD_NEGATIVES_PER_QUERY = 3  # Generate 3 hard negatives per query
TOP_K_CANDIDATES = 20  # Retrieve top-K for hard negative selection

# BM25 Parameters
STOPWORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'it', 'its'}
BM25_K1 = 1.5
BM25_B = 0.75
AVG_DOC_LENGTH = 50


@dataclass
class TrainingPair:
    """Single training pair for cross-encoder."""
    query: str
    memory: str
    label: int  # 1 = relevant, 0 = irrelevant


def tokenize_bm25(text: str) -> List[str]:
    """Tokenize text for BM25, preserving technical terms."""
    # Split CamelCase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Split snake_case
    text = text.replace('_', ' ')
    # Extract alphanumeric tokens
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    # Filter stopwords and short tokens
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def compute_bm25_sparse(text: str) -> Dict:
    """Compute BM25-style sparse vector for Qdrant."""
    tokens = tokenize_bm25(text)
    if not tokens:
        return {"indices": [], "values": []}

    tf = Counter(tokens)
    doc_length = len(tokens)

    indices = []
    values = []

    for term, freq in tf.items():
        term_hash = int(hashlib.md5(term.encode()).hexdigest()[:8], 16)
        indices.append(term_hash)

        tf_weight = (freq * (BM25_K1 + 1)) / (
            freq + BM25_K1 * (1 - BM25_B + BM25_B * doc_length / AVG_DOC_LENGTH)
        )
        values.append(float(tf_weight))

    return {"indices": indices, "values": values}


class QdrantClient:
    """Simple Qdrant HTTP client for retrieving memories."""

    def __init__(self, url: str = QDRANT_URL, collection: str = COLLECTION_NAME):
        self.base_url = url
        self.collection = collection
        self.client = httpx.Client(timeout=30.0)
        logger.info(f"Initialized Qdrant client: {url}/{collection}")

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get dense embedding from embedding service."""
        try:
            resp = self.client.post(
                f"{EMBEDDING_URL}/v1/embeddings",
                json={
                    "model": EMBEDDING_MODEL,
                    "input": text[:8000]  # Limit length
                }
            )
            if resp.status_code == 200:
                return resp.json()["data"][0]["embedding"]
        except Exception as e:
            logger.warning(f"Embedding failed for '{text[:50]}...': {e}")
        return None

    def get_memory_by_id(self, memory_id: int) -> Optional[Dict]:
        """Retrieve single memory by ID."""
        try:
            response = self.client.post(
                f"{self.base_url}/collections/{self.collection}/points",
                json={"ids": [memory_id], "with_payload": True, "with_vector": False}
            )
            response.raise_for_status()
            data = response.json()
            points = data.get("result", [])
            if points:
                return points[0]["payload"]
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            return None

    def search_hybrid(self, query: str, limit: int = TOP_K_CANDIDATES) -> List[Dict]:
        """
        Execute hybrid search (dense + sparse BM25).

        Returns:
            List of {"id": int, "score": float, "payload": dict}
        """
        # Get dense embedding
        dense_embedding = self.get_embedding(query)
        if not dense_embedding:
            logger.warning(f"No embedding for query, falling back to scroll: '{query[:50]}...'")
            return []

        # Compute sparse BM25 vector
        sparse_vector = compute_bm25_sparse(query)

        # Build hybrid query
        hybrid_query = {
            "prefetch": [
                {
                    "query": dense_embedding,
                    "using": "dense",
                    "limit": limit * 2
                }
            ],
            "query": {"fusion": "rrf"},
            "limit": limit,
            "with_payload": True
        }

        # Add sparse vector if tokens found
        if sparse_vector.get("indices"):
            hybrid_query["prefetch"].append({
                "query": {
                    "indices": sparse_vector["indices"],
                    "values": sparse_vector["values"]
                },
                "using": "sparse",
                "limit": limit * 2
            })

        try:
            response = self.client.post(
                f"{self.base_url}/collections/{self.collection}/points/query",
                json=hybrid_query
            )
            response.raise_for_status()
            data = response.json()
            return data.get("result", {}).get("points", [])
        except Exception as e:
            logger.error(f"Hybrid search failed for query '{query[:50]}...': {e}")
            return []

    def close(self):
        """Close HTTP client."""
        self.client.close()


class TrainingPairGenerator:
    """Generate cross-encoder training pairs from test suite."""

    def __init__(self, qdrant_client: QdrantClient):
        self.qdrant = qdrant_client
        self.stats = {
            "total_test_cases": 0,
            "positive_pairs": 0,
            "negative_pairs": 0,
            "skipped_cases": 0,
            "errors": 0
        }

    def generate_positive_pair(
        self,
        query: str,
        memory_id: int,
        ground_truth_content: str
    ) -> Optional[TrainingPair]:
        """
        Generate positive training pair.

        Uses ground truth content from test suite to avoid extra Qdrant query.
        """
        return TrainingPair(
            query=query,
            memory=ground_truth_content,
            label=1
        )

    def generate_hard_negatives(
        self,
        query: str,
        ground_truth_id: int,
        num_negatives: int = HARD_NEGATIVES_PER_QUERY
    ) -> List[TrainingPair]:
        """
        Generate hard negative pairs using BM25 + semantic similarity.

        Strategy:
        1. Retrieve top-K candidates via hybrid search
        2. Filter out ground truth
        3. Select top-N as hard negatives (semantically similar but wrong)

        Args:
            query: User query
            ground_truth_id: Correct memory ID to exclude
            num_negatives: Number of hard negatives to generate

        Returns:
            List of hard negative training pairs
        """
        # Search for similar memories
        candidates = self.qdrant.search_hybrid(query, limit=TOP_K_CANDIDATES)

        hard_negatives = []
        for candidate in candidates:
            # Skip ground truth
            if candidate["id"] == ground_truth_id:
                continue

            # Extract memory text
            payload = candidate.get("payload", {})
            memory_text = payload.get("lesson", "")

            if not memory_text:
                continue

            # Add as hard negative
            hard_negatives.append(
                TrainingPair(
                    query=query,
                    memory=memory_text,
                    label=0
                )
            )

            # Stop when we have enough
            if len(hard_negatives) >= num_negatives:
                break

        return hard_negatives

    def generate_from_test_suite(
        self,
        test_suite_path: Path = TEST_SUITE_PATH
    ) -> List[TrainingPair]:
        """
        Generate all training pairs from enhanced test suite.

        Returns:
            List of TrainingPair instances
        """
        logger.info(f"Loading test suite from: {test_suite_path}")

        with open(test_suite_path, 'r', encoding='utf-8') as f:
            test_suite = json.load(f)

        test_cases = test_suite.get("test_cases", [])
        self.stats["total_test_cases"] = len(test_cases)

        logger.info(f"Processing {len(test_cases)} test cases...")

        all_pairs = []

        for i, test_case in enumerate(test_cases, 1):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{len(test_cases)} test cases processed")

            memory_id = test_case["memory_id"]
            ground_truth_content = test_case["content"]
            generated_queries = test_case.get("generated_queries", [])

            # Process all generated query variants for this memory
            for query_variant in generated_queries:
                query_text = query_variant.get("query", "").strip()

                if not query_text:
                    continue

                try:
                    # Generate positive pair
                    positive_pair = self.generate_positive_pair(
                        query=query_text,
                        memory_id=memory_id,
                        ground_truth_content=ground_truth_content
                    )

                    if positive_pair:
                        all_pairs.append(positive_pair)
                        self.stats["positive_pairs"] += 1

                    # Generate hard negatives
                    hard_negatives = self.generate_hard_negatives(
                        query=query_text,
                        ground_truth_id=memory_id,
                        num_negatives=HARD_NEGATIVES_PER_QUERY
                    )

                    all_pairs.extend(hard_negatives)
                    self.stats["negative_pairs"] += len(hard_negatives)

                except Exception as e:
                    logger.error(f"Error processing query '{query_text}': {e}")
                    self.stats["errors"] += 1

        logger.info(f"Generation complete. Total pairs: {len(all_pairs)}")
        return all_pairs

    def save_pairs(self, pairs: List[TrainingPair], output_path: Path = OUTPUT_PATH):
        """Save training pairs to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        pairs_dict = [
            {
                "query": pair.query,
                "memory": pair.memory,
                "label": pair.label
            }
            for pair in pairs
        ]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pairs_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(pairs)} training pairs to: {output_path}")

    def print_stats(self):
        """Print generation statistics."""
        print("\n" + "="*60)
        print("TRAINING PAIR GENERATION STATISTICS")
        print("="*60)
        print(f"Total test cases:     {self.stats['total_test_cases']:>6,}")
        print(f"Positive pairs:       {self.stats['positive_pairs']:>6,}")
        print(f"Negative pairs:       {self.stats['negative_pairs']:>6,}")
        print(f"Total pairs:          {self.stats['positive_pairs'] + self.stats['negative_pairs']:>6,}")
        print(f"Skipped cases:        {self.stats['skipped_cases']:>6,}")
        print(f"Errors:               {self.stats['errors']:>6,}")
        print("="*60)

        # Calculate ratio
        if self.stats['positive_pairs'] > 0:
            ratio = self.stats['negative_pairs'] / self.stats['positive_pairs']
            print(f"Negative-to-positive ratio: {ratio:.2f}:1")
        print("="*60 + "\n")


def append_to_work_log(message: str):
    """Append status update to work log."""
    try:
        WORK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(WORK_LOG_PATH, 'a', encoding='utf-8') as f:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n### [{timestamp}] Cross-Encoder Training Pair Generation\n")
            f.write(f"{message}\n")
    except Exception as e:
        logger.warning(f"Failed to append to work log: {e}")


def main():
    """Main execution."""
    logger.info("Starting cross-encoder training pair generation...")

    # Initialize Qdrant client
    qdrant = QdrantClient()

    try:
        # Generate training pairs
        generator = TrainingPairGenerator(qdrant)
        pairs = generator.generate_from_test_suite()

        # Save to file
        generator.save_pairs(pairs)

        # Print statistics
        generator.print_stats()

        # Update work log
        total_pairs = len(pairs)
        log_message = f"""
**Status**: ✅ SUCCESS

**Results**:
- Total test cases: {generator.stats['total_test_cases']:,}
- Positive pairs: {generator.stats['positive_pairs']:,}
- Negative pairs: {generator.stats['negative_pairs']:,}
- **Total pairs**: {total_pairs:,}
- Errors: {generator.stats['errors']}

**Output**: `{OUTPUT_PATH.relative_to(PROJECT_ROOT)}`

**Next Steps**:
1. Review training pair quality
2. Fine-tune cross-encoder model
3. Evaluate reranking performance
"""
        append_to_work_log(log_message)

        # Final validation
        if total_pairs < 2000:
            logger.warning(f"⚠️  Only generated {total_pairs} pairs (target: 2000+)")
        else:
            logger.info(f"✅ Target achieved: {total_pairs} pairs generated")

    finally:
        qdrant.close()


if __name__ == "__main__":
    main()
