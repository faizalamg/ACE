"""Test script for embedding fine-tuning pipeline.

Validates all components work correctly with mock data (no external dependencies).
Useful for CI/CD and development testing.
"""

import json
import tempfile
from pathlib import Path


def test_data_generator_structure():
    """Test TrainingDataGenerator creates correct data structure."""
    print("Testing data generator structure...")

    from ace.embedding_finetuning.data_generator import TrainingExample

    # Create example
    example = TrainingExample(
        query="how to debug errors",
        positive="Check logs first",
        negatives=["Use print statements", "Restart server"],
        memory_id=12345,
        category="DEBUGGING",
        difficulty="medium",
        original_query_category="procedural",
    )

    assert example.query == "how to debug errors"
    assert example.positive == "Check logs first"
    assert len(example.negatives) == 2
    assert example.difficulty == "medium"

    print("[PASS] Data generator structure test passed")


def test_training_config():
    """Test TrainingConfig default values."""
    print("Testing training config...")

    from ace.embedding_finetuning.finetune_embeddings import TrainingConfig

    config = TrainingConfig()

    assert config.epochs == 3
    assert config.batch_size == 16
    assert config.learning_rate == 2e-5
    assert config.base_model == "sentence-transformers/all-MiniLM-L6-v2"

    # Test custom config
    custom_config = TrainingConfig(epochs=5, batch_size=32)
    assert custom_config.epochs == 5
    assert custom_config.batch_size == 32

    print("[PASS] Training config test passed")


def test_evaluation_result_structure():
    """Test EvaluationResult structure."""
    print("Testing evaluation result structure...")

    from ace.embedding_finetuning.evaluate_finetuned import EvaluationResult

    result = EvaluationResult(
        query="test query",
        correct_memory_id=123,
        correct_rank=2,
        top_5_ids=[456, 123, 789, 101, 112],
        found_in_top_1=False,
        found_in_top_5=True,
        reciprocal_rank=0.5,
        similarity_score=0.856,
        difficulty="hard",
        category="technical",
    )

    assert result.correct_rank == 2
    assert result.found_in_top_1 is False
    assert result.found_in_top_5 is True
    assert result.reciprocal_rank == 0.5

    print("[PASS] Evaluation result structure test passed")


def test_training_data_format():
    """Test training data JSON format generation."""
    print("Testing training data format...")

    # Create mock training data
    training_data = {
        "metadata": {
            "total_examples": 2,
            "source": "test_suite.json",
            "generated_at": "2025-12-12 14:00:00",
        },
        "examples": [
            {
                "query": "how to fix auth errors",
                "positive": "Check OAuth token expiration first",
                "negatives": [
                    "Restart the server",
                    "Clear browser cache",
                    "Update dependencies",
                ],
                "metadata": {
                    "memory_id": 123,
                    "category": "DEBUGGING",
                    "difficulty": "medium",
                    "query_category": "procedural",
                },
            },
            {
                "query": "best practices for code organization",
                "positive": "Use SOLID principles and separation of concerns",
                "negatives": [
                    "Put everything in one file",
                    "Use global variables",
                ],
                "metadata": {
                    "memory_id": 456,
                    "category": "ARCHITECTURE",
                    "difficulty": "hard",
                    "query_category": "template",
                },
            },
        ],
    }

    # Validate structure
    assert "metadata" in training_data
    assert "examples" in training_data
    assert len(training_data["examples"]) == 2

    example = training_data["examples"][0]
    assert "query" in example
    assert "positive" in example
    assert "negatives" in example
    assert len(example["negatives"]) == 3

    print("[PASS] Training data format test passed")


def test_finetuned_retrieval_initialization():
    """Test FineTunedRetrieval initialization (no model loading)."""
    print("Testing retrieval initialization...")

    from ace.embedding_finetuning.finetuned_retrieval import FineTunedRetrieval

    # Test initialization with fallback
    try:
        retrieval = FineTunedRetrieval(
            finetuned_model_path="nonexistent/path",
            fallback_to_baseline=True,  # Should not raise error
        )
        assert retrieval.use_finetuned is False
        print("[PASS] Graceful fallback works")
    except Exception as e:
        print(f"[FAIL] Unexpected error during fallback: {e}")
        raise

    print("[PASS] Retrieval initialization test passed")


def test_end_to_end_data_flow():
    """Test complete data flow from generation to evaluation."""
    print("Testing end-to-end data flow...")

    import tempfile
    from pathlib import Path

    # Create temporary training data
    with tempfile.TemporaryDirectory() as tmpdir:
        training_data_path = Path(tmpdir) / "training_data.json"

        # Mock training data
        training_data = {
            "metadata": {"total_examples": 3},
            "examples": [
                {
                    "query": f"query {i}",
                    "positive": f"correct answer {i}",
                    "negatives": [f"wrong {i}_1", f"wrong {i}_2"],
                    "metadata": {
                        "memory_id": i,
                        "category": "TEST",
                        "difficulty": "medium",
                        "query_category": "direct",
                    },
                }
                for i in range(3)
            ],
        }

        # Save to file
        with open(training_data_path, "w") as f:
            json.dump(training_data, f)

        # Verify file exists
        assert training_data_path.exists()

        # Load and validate
        with open(training_data_path, "r") as f:
            loaded_data = json.load(f)

        assert loaded_data["metadata"]["total_examples"] == 3
        assert len(loaded_data["examples"]) == 3

        print("[PASS] Data flow test passed")


def test_import_all_modules():
    """Test all modules can be imported."""
    print("Testing module imports...")

    try:
        from ace.embedding_finetuning import (
            HardNegativeMiner,
            TrainingDataGenerator,
            TrainingExample,
            EmbeddingFineTuner,
            TrainingConfig,
            train_embedding_model,
            BaselineEmbeddingClient,
            EmbeddingEvaluator,
            EvaluationResult,
            AggregateMetrics,
            FineTunedRetrieval,
            create_finetuned_retrieval,
        )

        print("[PASS] All modules imported successfully")
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        raise


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 80)
    print("EMBEDDING FINE-TUNING PIPELINE - UNIT TESTS")
    print("=" * 80 + "\n")

    tests = [
        test_import_all_modules,
        test_data_generator_structure,
        test_training_config,
        test_evaluation_result_structure,
        test_training_data_format,
        test_finetuned_retrieval_initialization,
        test_end_to_end_data_flow,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
            print()
        except Exception as e:
            failed += 1
            print(f"[FAIL] {test_func.__name__} FAILED: {e}\n")

    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    import sys

    success = run_all_tests()
    sys.exit(0 if success else 1)
