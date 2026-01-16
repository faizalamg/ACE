"""Embedding fine-tuning pipeline for domain adaptation.

This module provides tools for fine-tuning embedding models on domain-specific
query-memory pairs to improve RAG retrieval performance through contrastive learning.

Components:
- data_generator.py: Generate training data from test suite with hard negatives
- finetune_embeddings.py: Fine-tune embeddings with contrastive learning
- evaluate_finetuned.py: Evaluate fine-tuned model performance
- finetuned_retrieval.py: Production retrieval using fine-tuned embeddings

Expected improvements: +15-20% retrieval accuracy from domain adaptation.

Quick Start:
    >>> # 1. Generate training data
    >>> from ace.embedding_finetuning.data_generator import (
    ...     HardNegativeMiner, TrainingDataGenerator
    ... )
    >>> miner = HardNegativeMiner()
    >>> generator = TrainingDataGenerator(
    ...     test_suite_path="rag_training/test_suite/enhanced_test_suite.json",
    ...     output_path="ace/embedding_finetuning/training_data.json",
    ...     miner=miner,
    ... )
    >>> generator.generate_and_save(negatives_per_example=5)

    >>> # 2. Fine-tune embeddings
    >>> from ace.embedding_finetuning.finetune_embeddings import train_embedding_model
    >>> output_dir, metrics = train_embedding_model(
    ...     training_data_path="ace/embedding_finetuning/training_data.json",
    ...     output_dir="ace/embedding_finetuning/models/ace_finetuned",
    ...     epochs=3,
    ... )

    >>> # 3. Use fine-tuned retrieval
    >>> from ace.embedding_finetuning.finetuned_retrieval import FineTunedRetrieval
    >>> retrieval = FineTunedRetrieval(
    ...     finetuned_model_path="ace/embedding_finetuning/models/ace_finetuned"
    ... )
    >>> results = retrieval.search("how to debug errors", limit=10)
"""

# Import key classes and functions for convenient access
from .data_generator import (
    HardNegativeMiner,
    TrainingDataGenerator,
    TrainingExample,
)
from .finetune_embeddings import (
    EmbeddingFineTuner,
    TrainingConfig,
    train_embedding_model,
)
from .evaluate_finetuned import (
    BaselineEmbeddingClient,
    EmbeddingEvaluator,
    EvaluationResult,
    AggregateMetrics,
)
from .finetuned_retrieval import (
    FineTunedRetrieval,
    create_finetuned_retrieval,
)

__all__ = [
    # Data generation
    "HardNegativeMiner",
    "TrainingDataGenerator",
    "TrainingExample",
    # Fine-tuning
    "EmbeddingFineTuner",
    "TrainingConfig",
    "train_embedding_model",
    # Evaluation
    "BaselineEmbeddingClient",
    "EmbeddingEvaluator",
    "EvaluationResult",
    "AggregateMetrics",
    # Production retrieval
    "FineTunedRetrieval",
    "create_finetuned_retrieval",
]
