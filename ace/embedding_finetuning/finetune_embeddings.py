"""Fine-tune embedding models for domain adaptation.

Uses sentence-transformers with contrastive learning (MultipleNegativesRankingLoss)
to adapt embeddings to the ACE memory domain, improving query-memory similarity.

Base model: sentence-transformers/all-MiniLM-L6-v2 (lightweight, 384 dims)
Can be fine-tuned on CPU/GPU, outputs compatible with Qdrant.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    losses,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for embedding fine-tuning."""

    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    output_dir: str = "ace/embedding_finetuning/models/ace_finetuned"
    epochs: int = 3
    batch_size: int = 16
    warmup_steps: int = 100
    learning_rate: float = 2e-5
    train_split: float = 0.8
    eval_steps: int = 50
    save_steps: int = 100
    max_seq_length: int = 256
    device: Optional[str] = None  # None = auto-detect


class EmbeddingFineTuner:
    """Fine-tune embedding models with contrastive learning."""

    def __init__(self, config: TrainingConfig):
        """Initialize fine-tuner.

        Args:
            config: Training configuration
        """
        self.config = config
        self.device = self._get_device()
        self.model: Optional[SentenceTransformer] = None
        self.train_examples: List[InputExample] = []
        self.eval_examples: List[InputExample] = []

        logger.info(f"Using device: {self.device}")

    def _get_device(self) -> str:
        """Determine device for training."""
        if self.config.device:
            return self.config.device

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def load_training_data(self, data_path: str) -> None:
        """Load training data from JSON.

        Expected format:
        {
            "metadata": {...},
            "examples": [
                {
                    "query": "...",
                    "positive": "...",
                    "negatives": ["...", "..."],
                    "metadata": {...}
                },
                ...
            ]
        }

        Args:
            data_path: Path to training_data.json
        """
        logger.info(f"Loading training data from {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        examples_data = data["examples"]
        total_examples = len(examples_data)

        logger.info(f"Loaded {total_examples} examples")

        # Convert to InputExample format
        # For MultipleNegativesRankingLoss, we create pairs of (query, positive)
        # and the loss handles negatives automatically via in-batch negatives
        all_examples = []
        for item in examples_data:
            query = item["query"]
            positive = item["positive"]

            # Create primary positive pair
            all_examples.append(InputExample(texts=[query, positive]))

            # OPTIONAL: Also create negative pairs with explicit labels
            # This teaches the model to separate queries from hard negatives
            for negative in item["negatives"][:2]:  # Use top-2 hard negatives
                all_examples.append(
                    InputExample(texts=[query, negative], label=0.0)
                )  # Low similarity

        # Train/eval split
        split_idx = int(len(all_examples) * self.config.train_split)
        self.train_examples = all_examples[:split_idx]
        self.eval_examples = all_examples[split_idx:]

        logger.info(
            f"Split: {len(self.train_examples)} train, {len(self.eval_examples)} eval"
        )

    def initialize_model(self) -> None:
        """Load and initialize base model."""
        logger.info(f"Loading base model: {self.config.base_model}")

        self.model = SentenceTransformer(
            self.config.base_model, device=self.device
        )

        # Set max sequence length
        self.model.max_seq_length = self.config.max_seq_length

        logger.info(
            f"Model initialized with {self.model.get_sentence_embedding_dimension()} dimensions"
        )

    def train(self) -> Dict[str, float]:
        """Fine-tune the embedding model.

        Returns:
            Training metrics dictionary
        """
        if not self.model:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")

        if not self.train_examples:
            raise RuntimeError(
                "No training data loaded. Call load_training_data() first."
            )

        logger.info("Starting fine-tuning...")
        start_time = time.time()

        # Create DataLoader
        train_dataloader = DataLoader(
            self.train_examples,
            shuffle=True,
            batch_size=self.config.batch_size,
        )

        # Define loss function
        # MultipleNegativesRankingLoss:
        # - Uses in-batch negatives (efficient, no explicit negatives needed)
        # - Learns to rank positive pairs higher than negatives
        # - Ideal for query-document retrieval tasks
        train_loss = losses.MultipleNegativesRankingLoss(model=self.model)

        # Calculate training steps
        num_train_steps = len(train_dataloader) * self.config.epochs
        warmup_steps = min(self.config.warmup_steps, num_train_steps // 10)

        logger.info(f"Training for {self.config.epochs} epochs")
        logger.info(f"Total steps: {num_train_steps}, Warmup steps: {warmup_steps}")

        # Train
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.config.epochs,
            warmup_steps=warmup_steps,
            output_path=self.config.output_dir,
            show_progress_bar=True,
            optimizer_params={"lr": self.config.learning_rate},
        )

        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.2f}s")

        # Save final model
        self.save_model()

        return {
            "epochs": self.config.epochs,
            "total_steps": num_train_steps,
            "training_time_seconds": elapsed,
        }

    def save_model(self, output_path: Optional[str] = None) -> None:
        """Save fine-tuned model.

        Args:
            output_path: Optional custom output path (defaults to config.output_dir)
        """
        if not self.model:
            raise RuntimeError("No model to save")

        save_path = output_path or self.config.output_dir
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on eval set.

        Returns:
            Evaluation metrics
        """
        if not self.model or not self.eval_examples:
            logger.warning("Cannot evaluate: no model or eval data")
            return {}

        logger.info(f"Evaluating on {len(self.eval_examples)} examples...")

        # Simple evaluation: measure similarity between query-positive pairs
        correct_similarities = []

        for example in self.eval_examples[:100]:  # Sample 100 for speed
            if len(example.texts) == 2:
                query, document = example.texts
                embeddings = self.model.encode(
                    [query, document], convert_to_tensor=True
                )

                # Cosine similarity
                similarity = torch.nn.functional.cosine_similarity(
                    embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
                ).item()

                correct_similarities.append(similarity)

        if correct_similarities:
            avg_similarity = sum(correct_similarities) / len(correct_similarities)
            logger.info(f"Average query-positive similarity: {avg_similarity:.4f}")

            return {"avg_positive_similarity": avg_similarity}

        return {}


def train_embedding_model(
    training_data_path: str,
    output_dir: str,
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    device: Optional[str] = None,
) -> Tuple[str, Dict[str, float]]:
    """High-level function to fine-tune embedding model.

    Args:
        training_data_path: Path to training_data.json
        output_dir: Directory to save fine-tuned model
        base_model: Base model to fine-tune
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        device: Device to use (None = auto-detect)

    Returns:
        Tuple of (output_dir, metrics_dict)
    """
    config = TrainingConfig(
        base_model=base_model,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )

    fine_tuner = EmbeddingFineTuner(config)
    fine_tuner.load_training_data(training_data_path)
    fine_tuner.initialize_model()

    # Train
    train_metrics = fine_tuner.train()

    # Evaluate
    eval_metrics = fine_tuner.evaluate()

    # Combine metrics
    all_metrics = {**train_metrics, **eval_metrics}

    return output_dir, all_metrics


def main():
    """CLI entry point for fine-tuning."""
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune embedding model")
    parser.add_argument(
        "--training-data",
        default="ace/embedding_finetuning/training_data.json",
        help="Path to training data JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="ace/embedding_finetuning/models/ace_finetuned",
        help="Output directory for fine-tuned model",
    )
    parser.add_argument(
        "--base-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base model to fine-tune",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=100, help="Warmup steps"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        help="Device to use (auto-detect if not specified)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create config
    config = TrainingConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        device=args.device,
    )

    # Fine-tune
    fine_tuner = EmbeddingFineTuner(config)
    fine_tuner.load_training_data(args.training_data)
    fine_tuner.initialize_model()

    train_metrics = fine_tuner.train()
    eval_metrics = fine_tuner.evaluate()

    # Print results
    logger.info("\n=== Training Results ===")
    for key, value in train_metrics.items():
        logger.info(f"{key}: {value}")

    if eval_metrics:
        logger.info("\n=== Evaluation Results ===")
        for key, value in eval_metrics.items():
            logger.info(f"{key}: {value}")

    logger.info(f"\nModel saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
