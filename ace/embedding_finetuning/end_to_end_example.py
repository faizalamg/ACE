"""End-to-end example: Fine-tune embeddings and compare performance.

This script demonstrates the complete pipeline:
1. Generate training data with hard negatives
2. Fine-tune embeddings with contrastive learning
3. Evaluate performance vs baseline
4. Use fine-tuned model for retrieval

Run this script to test the complete pipeline on a subset of data.
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run end-to-end fine-tuning pipeline."""
    logger.info("=" * 80)
    logger.info("EMBEDDING FINE-TUNING PIPELINE - END-TO-END EXAMPLE")
    logger.info("=" * 80)

    # Configuration
    test_suite_path = "rag_training/test_suite/enhanced_test_suite.json"
    training_data_path = "ace/embedding_finetuning/training_data_example.json"
    model_output_path = "ace/embedding_finetuning/models/ace_finetuned_example"
    eval_output_path = (
        "rag_training/optimization_results/v5_finetuned_embeddings_example.json"
    )

    # Limit for faster testing
    max_training_examples = 200
    max_eval_queries = 100

    # =========================================================================
    # STEP 1: Generate Training Data
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Generating Training Data")
    logger.info("=" * 80)

    try:
        from ace.embedding_finetuning.data_generator import (
            HardNegativeMiner,
            TrainingDataGenerator,
        )

        # Create hard negative miner
        logger.info("Initializing hard negative miner...")
        miner = HardNegativeMiner(
            qdrant_url="http://localhost:6333",
            collection_name="ace_memories_hybrid",
            embedding_url="http://localhost:1234",
        )

        # Create generator
        generator = TrainingDataGenerator(
            test_suite_path=test_suite_path,
            output_path=training_data_path,
            miner=miner,
            max_examples=max_training_examples,
        )

        # Generate training data
        num_examples = generator.generate_and_save(
            negatives_per_example=5,
            min_difficulty="medium",  # Skip easy queries
        )

        logger.info(f"✓ Generated {num_examples} training examples")
        logger.info(f"✓ Saved to: {training_data_path}")

        miner.close()

    except Exception as e:
        logger.error(f"✗ Training data generation failed: {e}")
        logger.error(
            "Make sure Qdrant is running and the test suite exists."
        )
        return 1

    # =========================================================================
    # STEP 2: Fine-Tune Embeddings
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Fine-Tuning Embeddings")
    logger.info("=" * 80)

    try:
        from ace.embedding_finetuning.finetune_embeddings import (
            EmbeddingFineTuner,
            TrainingConfig,
        )

        # Configure training
        config = TrainingConfig(
            base_model="sentence-transformers/all-MiniLM-L6-v2",
            output_dir=model_output_path,
            epochs=2,  # Fewer epochs for faster testing
            batch_size=16,
            learning_rate=2e-5,
            warmup_steps=50,
            train_split=0.8,
        )

        # Create fine-tuner
        fine_tuner = EmbeddingFineTuner(config)
        fine_tuner.load_training_data(training_data_path)
        fine_tuner.initialize_model()

        # Train
        logger.info("Starting training (this may take a few minutes)...")
        train_metrics = fine_tuner.train()

        logger.info("\n✓ Training completed!")
        logger.info(f"  - Epochs: {train_metrics['epochs']}")
        logger.info(f"  - Total steps: {train_metrics['total_steps']}")
        logger.info(
            f"  - Time: {train_metrics['training_time_seconds']:.1f}s"
        )
        logger.info(f"  - Model saved to: {model_output_path}")

        # Evaluate on validation set
        eval_metrics = fine_tuner.evaluate()
        if eval_metrics:
            logger.info("\n✓ Validation metrics:")
            logger.info(
                f"  - Avg positive similarity: {eval_metrics['avg_positive_similarity']:.3f}"
            )

    except Exception as e:
        logger.error(f"✗ Fine-tuning failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # =========================================================================
    # STEP 3: Evaluate Performance
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Evaluating Performance")
    logger.info("=" * 80)

    try:
        from ace.embedding_finetuning.evaluate_finetuned import (
            BaselineEmbeddingClient,
            EmbeddingEvaluator,
        )
        from sentence_transformers import SentenceTransformer

        # Create evaluator
        evaluator = EmbeddingEvaluator(
            test_suite_path=test_suite_path,
            qdrant_url="http://localhost:6333",
            collection_name="ace_memories_hybrid",
        )
        evaluator.load_test_suite()

        # Load models
        logger.info("Loading baseline model (LM Studio)...")
        baseline_model = BaselineEmbeddingClient(
            embedding_url="http://localhost:1234",
            model="text-embedding-qwen3-embedding-8b",
        )

        logger.info(f"Loading fine-tuned model from {model_output_path}...")
        finetuned_model = SentenceTransformer(model_output_path)

        # Compare
        logger.info(
            f"Evaluating on {max_eval_queries} queries (this may take a minute)..."
        )
        comparison = evaluator.compare_models(
            baseline_model=baseline_model,
            finetuned_model=finetuned_model,
            output_path=eval_output_path,
            max_queries=max_eval_queries,
        )

        logger.info("\n✓ Evaluation completed!")
        logger.info(f"  - Results saved to: {eval_output_path}")

        # Print summary
        baseline = comparison["baseline"]
        finetuned = comparison["finetuned"]
        improvement = comparison["improvement"]

        logger.info("\n" + "=" * 80)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info(
            f"Baseline Recall@1:    {baseline['recall@1']:.3f}"
        )
        logger.info(
            f"Fine-tuned Recall@1:  {finetuned['recall@1']:.3f}  ({improvement['recall@1']:+.1f}%)"
        )
        logger.info("")
        logger.info(
            f"Baseline Recall@5:    {baseline['recall@5']:.3f}"
        )
        logger.info(
            f"Fine-tuned Recall@5:  {finetuned['recall@5']:.3f}  ({improvement['recall@5']:+.1f}%)"
        )
        logger.info("")
        logger.info(f"Baseline MRR:         {baseline['mrr']:.3f}")
        logger.info(
            f"Fine-tuned MRR:       {finetuned['mrr']:.3f}  ({improvement['mrr']:+.1f}%)"
        )
        logger.info("=" * 80)

        baseline_model.close()

    except Exception as e:
        logger.error(f"✗ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # =========================================================================
    # STEP 4: Test Retrieval
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Testing Retrieval")
    logger.info("=" * 80)

    try:
        from ace.embedding_finetuning.finetuned_retrieval import (
            FineTunedRetrieval,
        )

        # Create retrieval system
        retrieval = FineTunedRetrieval(
            finetuned_model_path=model_output_path,
            qdrant_url="http://localhost:6333",
            collection_name="ace_memories_hybrid",
        )

        # Test queries
        test_queries = [
            "how to debug authentication errors",
            "best practices for code organization",
            "handling edge cases in production",
        ]

        for query in test_queries:
            logger.info(f"\nQuery: '{query}'")
            results = retrieval.search(
                query=query, limit=3, threshold=0.3
            )

            for i, result in enumerate(results, 1):
                logger.info(
                    f"  {i}. [Score: {result['score']:.3f}] {result['content'][:70]}..."
                )

        retrieval.close()

        logger.info("\n✓ Retrieval test completed!")

    except Exception as e:
        logger.error(f"✗ Retrieval test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # =========================================================================
    # COMPLETION
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("1. Review evaluation results:")
    logger.info(f"   {eval_output_path}")
    logger.info("2. Use fine-tuned model in production:")
    logger.info(f"   {model_output_path}")
    logger.info("3. For full training, run without max_examples limit")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
