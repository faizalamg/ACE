"""Generate training data from test suite for embedding fine-tuning.

This module extracts query-memory pairs from the enhanced test suite and generates
hard negatives by querying the current system for top-K wrong results.

Output format: JSON with [query, positive, negative1, negative2, ...]
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import httpx
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """Single training example with query and positive/negative pairs."""

    query: str
    positive: str  # Correct memory content
    negatives: List[str]  # Hard negative memory contents
    memory_id: int
    category: str
    difficulty: str
    original_query_category: str


class HardNegativeMiner:
    """Mine hard negatives from current retrieval system.

    Hard negatives are semantically similar but incorrect memories that the
    current system retrieves. These are more valuable for training than random
    negatives because they teach the model to distinguish subtle differences.
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "ace_memories_hybrid",
        embedding_url: str = "http://localhost:1234",
        embedding_model: str = "text-embedding-qwen3-embedding-8b",
        top_k: int = 20,
    ):
        """Initialize hard negative miner.

        Args:
            qdrant_url: Qdrant server URL
            collection_name: Qdrant collection name
            embedding_url: LM Studio embedding server URL
            embedding_model: Embedding model name
            top_k: Number of top results to fetch (for mining negatives)
        """
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.embedding_url = embedding_url
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.http_client = httpx.Client(timeout=30.0)

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding vector from LM Studio."""
        try:
            resp = self.http_client.post(
                f"{self.embedding_url}/v1/embeddings",
                json={"model": self.embedding_model, "input": text[:8000]},
            )
            if resp.status_code == 200:
                return resp.json()["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Embedding error: {e}")
        return None

    def mine_hard_negatives(
        self, query: str, correct_memory_id: int, num_negatives: int = 5
    ) -> List[str]:
        """Mine hard negatives for a query.

        Retrieves top-K results and returns incorrect memories as hard negatives.

        Args:
            query: Search query
            correct_memory_id: ID of the correct memory (to exclude)
            num_negatives: Number of hard negatives to return

        Returns:
            List of hard negative memory contents
        """
        # Get query embedding
        embedding = self.get_embedding(query)
        if not embedding:
            logger.warning(f"Failed to get embedding for query: {query[:50]}")
            return []

        # Search Qdrant (dense-only for simplicity)
        try:
            results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=embedding,
                using="dense",
                limit=self.top_k,
                with_payload=True,
            )

            # Extract incorrect memories (exclude correct one)
            hard_negatives = []
            for result in results.points:
                payload = result.payload
                # ID can be in payload or as the point ID directly
                result_id = (
                    payload.get("memory_id")
                    or payload.get("bullet_id")
                    or payload.get("original_id")
                    or result.id
                )

                # Skip correct memory
                if result_id == correct_memory_id:
                    continue

                # Content can be in "content" or "lesson" field
                content = payload.get("content") or payload.get("lesson", "")
                if content and content not in hard_negatives:
                    hard_negatives.append(content)

                # Stop when we have enough
                if len(hard_negatives) >= num_negatives:
                    break

            return hard_negatives

        except Exception as e:
            logger.error(f"Hard negative mining error: {e}")
            return []

    def close(self):
        """Cleanup resources."""
        self.http_client.close()


class TrainingDataGenerator:
    """Generate training data from enhanced test suite."""

    def __init__(
        self,
        test_suite_path: str,
        output_path: str,
        miner: HardNegativeMiner,
        max_examples: Optional[int] = None,
    ):
        """Initialize training data generator.

        Args:
            test_suite_path: Path to enhanced_test_suite.json
            output_path: Path to save training_data.json
            miner: HardNegativeMiner instance for mining hard negatives
            max_examples: Maximum examples to generate (None = all)
        """
        self.test_suite_path = Path(test_suite_path)
        self.output_path = Path(output_path)
        self.miner = miner
        self.max_examples = max_examples

    def load_test_suite(self) -> Dict:
        """Load enhanced test suite."""
        with open(self.test_suite_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def generate_training_data(
        self,
        negatives_per_example: int = 5,
        min_difficulty: Optional[str] = None,
        skip_categories: Optional[Set[str]] = None,
    ) -> List[TrainingExample]:
        """Generate training examples with hard negatives.

        Args:
            negatives_per_example: Number of hard negatives per query
            min_difficulty: Only use queries with this difficulty or harder
                          (None, "easy", "medium", "hard")
            skip_categories: Query categories to skip (e.g., {"edge_long"})

        Returns:
            List of TrainingExample instances
        """
        test_suite = self.load_test_suite()
        test_cases = test_suite["test_cases"]

        training_examples = []
        skip_categories = skip_categories or set()

        # Difficulty ordering for filtering
        difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
        min_difficulty_level = (
            difficulty_order.get(min_difficulty, -1) if min_difficulty else -1
        )

        logger.info(f"Generating training data from {len(test_cases)} memories...")
        start_time = time.time()

        for test_case in test_cases:
            memory_id = test_case["memory_id"]
            memory_content = test_case["content"]
            memory_category = test_case["category"]
            generated_queries = test_case["generated_queries"]

            # Process each generated query
            for query_obj in generated_queries:
                query = query_obj["query"]
                query_category = query_obj["category"]
                query_difficulty = query_obj["difficulty"]

                # Apply filters
                if query_category in skip_categories:
                    continue

                if (
                    min_difficulty
                    and difficulty_order.get(query_difficulty, 0)
                    < min_difficulty_level
                ):
                    continue

                # Mine hard negatives
                hard_negatives = self.miner.mine_hard_negatives(
                    query=query,
                    correct_memory_id=memory_id,
                    num_negatives=negatives_per_example,
                )

                # Only add if we got at least 2 hard negatives
                if len(hard_negatives) >= 2:
                    training_examples.append(
                        TrainingExample(
                            query=query,
                            positive=memory_content,
                            negatives=hard_negatives,
                            memory_id=memory_id,
                            category=memory_category,
                            difficulty=query_difficulty,
                            original_query_category=query_category,
                        )
                    )

                # Stop if we hit max examples
                if self.max_examples and len(training_examples) >= self.max_examples:
                    break

            if self.max_examples and len(training_examples) >= self.max_examples:
                break

        elapsed = time.time() - start_time
        logger.info(
            f"Generated {len(training_examples)} training examples in {elapsed:.2f}s"
        )

        return training_examples

    def save_training_data(self, examples: List[TrainingExample]) -> None:
        """Save training examples to JSON.

        Output format compatible with sentence-transformers:
        [
            {
                "query": "...",
                "positive": "...",
                "negatives": ["...", "..."],
                "metadata": {...}
            },
            ...
        ]
        """
        output_data = []

        for example in examples:
            output_data.append(
                {
                    "query": example.query,
                    "positive": example.positive,
                    "negatives": example.negatives,
                    "metadata": {
                        "memory_id": example.memory_id,
                        "category": example.category,
                        "difficulty": example.difficulty,
                        "query_category": example.original_query_category,
                    },
                }
            )

        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save JSON
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {
                        "total_examples": len(output_data),
                        "source": str(self.test_suite_path),
                        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    "examples": output_data,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(f"Saved {len(output_data)} examples to {self.output_path}")

    def generate_and_save(
        self,
        negatives_per_example: int = 5,
        min_difficulty: Optional[str] = None,
        skip_categories: Optional[Set[str]] = None,
    ) -> int:
        """Generate training data and save to file.

        Args:
            negatives_per_example: Number of hard negatives per query
            min_difficulty: Only use queries with this difficulty or harder
            skip_categories: Query categories to skip

        Returns:
            Number of training examples generated
        """
        examples = self.generate_training_data(
            negatives_per_example=negatives_per_example,
            min_difficulty=min_difficulty,
            skip_categories=skip_categories,
        )

        self.save_training_data(examples)
        return len(examples)


def main():
    """CLI entry point for training data generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate training data for embedding fine-tuning"
    )
    parser.add_argument(
        "--test-suite",
        default="rag_training/test_suite/enhanced_test_suite.json",
        help="Path to enhanced test suite",
    )
    parser.add_argument(
        "--output",
        default="ace/embedding_finetuning/training_data.json",
        help="Output path for training data",
    )
    parser.add_argument(
        "--negatives",
        type=int,
        default=5,
        help="Number of hard negatives per example",
    )
    parser.add_argument(
        "--min-difficulty",
        choices=["easy", "medium", "hard"],
        help="Minimum query difficulty to include",
    )
    parser.add_argument(
        "--skip-categories",
        nargs="+",
        help="Query categories to skip (e.g., edge_long casual)",
    )
    parser.add_argument(
        "--max-examples", type=int, help="Maximum number of examples to generate"
    )
    parser.add_argument(
        "--qdrant-url", default="http://localhost:6333", help="Qdrant server URL"
    )
    parser.add_argument(
        "--collection",
        default="ace_memories_hybrid",
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--embedding-url",
        default="http://localhost:1234",
        help="LM Studio embedding server URL",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create hard negative miner
    logger.info("Initializing hard negative miner...")
    miner = HardNegativeMiner(
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
        embedding_url=args.embedding_url,
    )

    try:
        # Create generator
        generator = TrainingDataGenerator(
            test_suite_path=args.test_suite,
            output_path=args.output,
            miner=miner,
            max_examples=args.max_examples,
        )

        # Generate and save
        skip_cats = set(args.skip_categories) if args.skip_categories else None
        num_examples = generator.generate_and_save(
            negatives_per_example=args.negatives,
            min_difficulty=args.min_difficulty,
            skip_categories=skip_cats,
        )

        logger.info(f"Successfully generated {num_examples} training examples!")
        logger.info(f"Output saved to: {args.output}")

    finally:
        miner.close()


if __name__ == "__main__":
    main()
