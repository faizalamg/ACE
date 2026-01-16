# Usage Examples - Embedding Fine-Tuning Pipeline

Practical examples for common use cases.

## Table of Contents

1. [Quick Start (Full Pipeline)](#quick-start-full-pipeline)
2. [Data Generation](#data-generation)
3. [Fine-Tuning](#fine-tuning)
4. [Evaluation](#evaluation)
5. [Production Deployment](#production-deployment)
6. [Advanced Workflows](#advanced-workflows)

---

## Quick Start (Full Pipeline)

Run the complete pipeline in one script:

```bash
# Run end-to-end example with subset of data
python ace/embedding_finetuning/end_to_end_example.py
```

**What it does:**
1. Generates training data from test suite (200 examples)
2. Fine-tunes all-MiniLM-L6-v2 model (2 epochs)
3. Evaluates baseline vs fine-tuned (100 queries)
4. Tests retrieval with sample queries

**Expected output:**
```
STEP 1: Generating Training Data
✓ Generated 200 training examples
✓ Saved to: ace/embedding_finetuning/training_data_example.json

STEP 2: Fine-Tuning Embeddings
✓ Training completed!
  - Epochs: 2
  - Total steps: 25
  - Time: 120.5s

STEP 3: Evaluating Performance
✓ Evaluation completed!

RESULTS SUMMARY
Baseline Recall@1:    0.612
Fine-tuned Recall@1:  0.734  (+19.9%)

Baseline Recall@5:    0.823
Fine-tuned Recall@5:  0.891  (+8.3%)
```

---

## Data Generation

### Example 1: Generate Full Training Dataset

```bash
python -m ace.embedding_finetuning.data_generator \
    --test-suite rag_training/test_suite/enhanced_test_suite.json \
    --output ace/embedding_finetuning/training_data.json \
    --negatives 5
```

**Output:** `training_data.json` with ~1000 examples (all 1038 queries)

### Example 2: Generate High-Quality Subset

```bash
# Only medium/hard queries, skip noisy categories
python -m ace.embedding_finetuning.data_generator \
    --test-suite rag_training/test_suite/enhanced_test_suite.json \
    --output ace/embedding_finetuning/training_data_filtered.json \
    --negatives 8 \
    --min-difficulty medium \
    --skip-categories edge_long casual implicit
```

**Why?** Medium/hard queries are more representative of real usage patterns.

### Example 3: Generate Small Test Dataset

```bash
# Limit to 100 examples for quick testing
python -m ace.embedding_finetuning.data_generator \
    --test-suite rag_training/test_suite/enhanced_test_suite.json \
    --output ace/embedding_finetuning/training_data_small.json \
    --negatives 5 \
    --max-examples 100
```

**Use case:** Rapid iteration during development.

### Example 4: Custom Hard Negative Mining

```python
from ace.embedding_finetuning.data_generator import (
    HardNegativeMiner,
    TrainingDataGenerator,
)

# Create miner with aggressive sampling
miner = HardNegativeMiner(
    qdrant_url="http://localhost:6333",
    collection_name="ace_memories_hybrid",
    top_k=50,  # Retrieve top-50 results (more hard negatives)
)

# Generate with 10 hard negatives per example
generator = TrainingDataGenerator(
    test_suite_path="rag_training/test_suite/enhanced_test_suite.json",
    output_path="ace/embedding_finetuning/training_data_aggressive.json",
    miner=miner,
)

num_examples = generator.generate_and_save(
    negatives_per_example=10,
    min_difficulty="medium",
)

print(f"Generated {num_examples} examples with 10 hard negatives each")
miner.close()
```

---

## Fine-Tuning

### Example 1: Standard Fine-Tuning

```bash
python -m ace.embedding_finetuning.finetune_embeddings \
    --training-data ace/embedding_finetuning/training_data.json \
    --output-dir ace/embedding_finetuning/models/ace_finetuned \
    --epochs 3 \
    --batch-size 16
```

**Training time:** ~5 min (GPU), ~30 min (CPU)

### Example 2: High-Quality Fine-Tuning (BGE-small)

```bash
# Use better base model (higher quality, slightly slower)
python -m ace.embedding_finetuning.finetune_embeddings \
    --base-model BAAI/bge-small-en-v1.5 \
    --training-data ace/embedding_finetuning/training_data.json \
    --output-dir ace/embedding_finetuning/models/bge_finetuned \
    --epochs 5 \
    --batch-size 16
```

**Expected improvement:** +3-5% over MiniLM (but same speed at inference)

### Example 3: Maximum Quality (E5-base)

```bash
# Use large model (highest quality, 2x slower)
python -m ace.embedding_finetuning.finetune_embeddings \
    --base-model intfloat/e5-base-v2 \
    --training-data ace/embedding_finetuning/training_data.json \
    --output-dir ace/embedding_finetuning/models/e5_finetuned \
    --epochs 5 \
    --batch-size 8 \
    --learning-rate 1e-5
```

**Use case:** Maximum accuracy, can afford higher latency.

### Example 4: Low-Memory Fine-Tuning (CPU)

```bash
# Force CPU, reduce batch size
python -m ace.embedding_finetuning.finetune_embeddings \
    --training-data ace/embedding_finetuning/training_data.json \
    --output-dir ace/embedding_finetuning/models/ace_finetuned_cpu \
    --device cpu \
    --batch-size 8 \
    --epochs 2
```

### Example 5: Programmatic Fine-Tuning with Custom Config

```python
from ace.embedding_finetuning.finetune_embeddings import (
    EmbeddingFineTuner,
    TrainingConfig,
)

# Custom configuration
config = TrainingConfig(
    base_model="sentence-transformers/all-MiniLM-L6-v2",
    output_dir="ace/embedding_finetuning/models/custom_finetuned",
    epochs=5,
    batch_size=16,
    learning_rate=2e-5,
    warmup_steps=100,
    train_split=0.8,  # 80% train, 20% validation
)

# Fine-tune
fine_tuner = EmbeddingFineTuner(config)
fine_tuner.load_training_data("ace/embedding_finetuning/training_data.json")
fine_tuner.initialize_model()

train_metrics = fine_tuner.train()
eval_metrics = fine_tuner.evaluate()

print(f"Training time: {train_metrics['training_time_seconds']:.1f}s")
print(f"Avg similarity: {eval_metrics['avg_positive_similarity']:.3f}")
```

---

## Evaluation

### Example 1: Quick Evaluation (100 queries)

```bash
python -m ace.embedding_finetuning.evaluate_finetuned \
    --test-suite rag_training/test_suite/enhanced_test_suite.json \
    --finetuned-model ace/embedding_finetuning/models/ace_finetuned \
    --output rag_training/optimization_results/v5_quick_eval.json \
    --max-queries 100
```

**Use case:** Fast validation during development.

### Example 2: Full Evaluation (All Queries)

```bash
python -m ace.embedding_finetuning.evaluate_finetuned \
    --test-suite rag_training/test_suite/enhanced_test_suite.json \
    --finetuned-model ace/embedding_finetuning/models/ace_finetuned \
    --output rag_training/optimization_results/v5_finetuned_embeddings.json
```

**Output:** Comprehensive metrics with difficulty/category breakdowns.

### Example 3: Programmatic Evaluation

```python
from ace.embedding_finetuning.evaluate_finetuned import (
    BaselineEmbeddingClient,
    EmbeddingEvaluator,
)
from sentence_transformers import SentenceTransformer

# Create evaluator
evaluator = EmbeddingEvaluator(
    test_suite_path="rag_training/test_suite/enhanced_test_suite.json",
    qdrant_url="http://localhost:6333",
)
evaluator.load_test_suite()

# Load models
baseline = BaselineEmbeddingClient(embedding_url="http://localhost:1234")
finetuned = SentenceTransformer("ace/embedding_finetuning/models/ace_finetuned")

# Evaluate
comparison = evaluator.compare_models(
    baseline_model=baseline,
    finetuned_model=finetuned,
    output_path="rag_training/optimization_results/custom_eval.json",
    max_queries=200,
)

# Access metrics
print(f"Baseline Recall@1: {comparison['baseline']['recall@1']:.3f}")
print(f"Fine-tuned Recall@1: {comparison['finetuned']['recall@1']:.3f}")
print(f"Improvement: {comparison['improvement']['recall@1']:.1f}%")

# Breakdown by difficulty
for difficulty, metrics in comparison['finetuned']['by_difficulty'].items():
    print(f"{difficulty}: Recall@1 = {metrics['recall@1']:.3f}")

baseline.close()
```

---

## Production Deployment

### Example 1: Drop-In Replacement for Baseline Retrieval

```python
from ace.embedding_finetuning.finetuned_retrieval import FineTunedRetrieval

# Replace your existing retrieval
retrieval = FineTunedRetrieval(
    finetuned_model_path="ace/embedding_finetuning/models/ace_finetuned",
    fallback_to_baseline=True,  # Graceful degradation
)

# Use exactly like before
results = retrieval.search(
    query="how to debug authentication errors",
    limit=10,
    threshold=0.3,
)

for result in results:
    print(f"[{result['score']:.3f}] {result['content']}")

retrieval.close()
```

### Example 2: Hybrid Search with Fine-Tuned Embeddings

```python
# Enable hybrid search (dense + sparse + RRF)
results = retrieval.search(
    query="best practices for error handling",
    limit=10,
    threshold=0.3,
    use_hybrid=True,  # Better quality, slightly slower
)
```

### Example 3: Batch Processing

```python
# Process multiple queries efficiently
queries = [
    "how to fix bugs",
    "code organization tips",
    "debugging strategies",
    "performance optimization",
]

batch_results = retrieval.batch_search(
    queries=queries,
    limit=5,
    threshold=0.4,
)

for query, results in zip(queries, batch_results):
    print(f"\nQuery: {query}")
    for result in results:
        print(f"  - {result['content'][:60]}...")
```

### Example 4: Integration with ACE Playbook

```python
from ace import Playbook
from ace.embedding_finetuning.finetuned_retrieval import FineTunedRetrieval

# Load playbook
playbook = Playbook.load_from_file("playbook.json")

# Create fine-tuned retrieval
retrieval = FineTunedRetrieval(
    finetuned_model_path="ace/embedding_finetuning/models/ace_finetuned"
)

# Retrieve relevant bullets for task
task_context = "Implementing authentication with OAuth"
results = retrieval.search(task_context, limit=5)

# Use retrieved bullets in Generator prompt
relevant_strategies = "\n".join([r['content'] for r in results])
print(f"Relevant strategies:\n{relevant_strategies}")
```

### Example 5: CLI Testing

```bash
# Test retrieval from command line
python -m ace.embedding_finetuning.finetuned_retrieval \
    --model ace/embedding_finetuning/models/ace_finetuned \
    --query "how to handle edge cases in production" \
    --limit 5 \
    --threshold 0.3
```

**Output:**
```
Found 5 results:

1. [Score: 0.856] Check boundary conditions first before general cases...
2. [Score: 0.782] Test edge cases with property-based testing...
3. [Score: 0.745] Document all known edge cases in test suite...
4. [Score: 0.698] Use defensive programming for edge cases...
5. [Score: 0.654] Log edge case handling for future analysis...
```

---

## Advanced Workflows

### Workflow 1: A/B Testing Fine-Tuned vs Baseline

```python
from ace.embedding_finetuning.finetuned_retrieval import FineTunedRetrieval
from ace.qdrant_retrieval import QdrantBulletIndex

# Setup both systems
baseline = QdrantBulletIndex()
finetuned = FineTunedRetrieval(
    finetuned_model_path="ace/embedding_finetuning/models/ace_finetuned"
)

# Test query
query = "debugging authentication failures"

# Get results from both
baseline_results = baseline.retrieve(query, limit=10)
finetuned_results = finetuned.search(query, limit=10)

# Compare
print("BASELINE Top-3:")
for i, r in enumerate(baseline_results[:3], 1):
    print(f"{i}. [{r.score:.3f}] {r.content[:60]}")

print("\nFINE-TUNED Top-3:")
for i, r in enumerate(finetuned_results[:3], 1):
    print(f"{i}. [{r['score']:.3f}] {r['content'][:60]}")
```

### Workflow 2: Incremental Fine-Tuning

```python
# Start with base model
base_model_path = "sentence-transformers/all-MiniLM-L6-v2"

# First round: General domain adaptation
from ace.embedding_finetuning.finetune_embeddings import train_embedding_model

round1_path, _ = train_embedding_model(
    training_data_path="ace/embedding_finetuning/training_data_general.json",
    output_dir="ace/embedding_finetuning/models/round1",
    base_model=base_model_path,
    epochs=3,
)

# Second round: Specific use case fine-tuning
round2_path, metrics = train_embedding_model(
    training_data_path="ace/embedding_finetuning/training_data_specific.json",
    output_dir="ace/embedding_finetuning/models/round2",
    base_model=round1_path,  # Use round1 as base
    epochs=2,
)

print(f"Final model saved to: {round2_path}")
```

### Workflow 3: Cross-Validation for Hyperparameter Tuning

```python
from ace.embedding_finetuning.finetune_embeddings import (
    EmbeddingFineTuner,
    TrainingConfig,
)

# Test different learning rates
learning_rates = [1e-5, 2e-5, 5e-5]
best_lr = None
best_similarity = 0

for lr in learning_rates:
    config = TrainingConfig(
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        output_dir=f"ace/embedding_finetuning/models/lr_{lr}",
        epochs=3,
        learning_rate=lr,
    )

    fine_tuner = EmbeddingFineTuner(config)
    fine_tuner.load_training_data("ace/embedding_finetuning/training_data.json")
    fine_tuner.initialize_model()

    fine_tuner.train()
    eval_metrics = fine_tuner.evaluate()

    similarity = eval_metrics['avg_positive_similarity']
    print(f"LR {lr}: Similarity = {similarity:.3f}")

    if similarity > best_similarity:
        best_similarity = similarity
        best_lr = lr

print(f"\nBest learning rate: {best_lr} (similarity: {best_similarity:.3f})")
```

### Workflow 4: Production Monitoring

```python
import time
from collections import defaultdict

# Track retrieval quality over time
quality_tracker = defaultdict(list)

def retrieve_with_monitoring(query, ground_truth_id=None):
    start = time.time()

    results = retrieval.search(query, limit=10)

    latency = (time.time() - start) * 1000  # ms

    # Track metrics
    quality_tracker['latency'].append(latency)

    if ground_truth_id:
        # Check if correct result is in top-K
        ids = [r['id'] for r in results]
        found = ground_truth_id in ids
        rank = ids.index(ground_truth_id) + 1 if found else None

        quality_tracker['found'].append(found)
        if rank:
            quality_tracker['rank'].append(rank)

    return results

# Use in production
retrieval = FineTunedRetrieval(
    finetuned_model_path="ace/embedding_finetuning/models/ace_finetuned"
)

# Example queries with known answers
test_cases = [
    ("how to debug errors", 12345),
    ("code organization", 67890),
    # ...
]

for query, ground_truth_id in test_cases:
    results = retrieve_with_monitoring(query, ground_truth_id)

# Print stats
print(f"Avg latency: {sum(quality_tracker['latency']) / len(quality_tracker['latency']):.1f}ms")
print(f"Recall@10: {sum(quality_tracker['found']) / len(quality_tracker['found']):.2%}")
print(f"Avg rank: {sum(quality_tracker['rank']) / len(quality_tracker['rank']):.1f}")
```

---

## Tips & Best Practices

### Data Quality
- **Filter noisy queries**: Skip `edge_long`, `casual`, `implicit` categories for cleaner training data
- **Focus on realistic queries**: Use `min-difficulty=medium` to exclude trivial queries
- **More hard negatives = better**: 8-10 hard negatives per example improves ranking quality

### Training
- **Start with 3 epochs**: Good balance of quality vs training time
- **Use validation split**: Set `train_split=0.8` to monitor overfitting
- **Early stopping**: Monitor validation similarity, stop if it plateaus

### Production
- **Always enable fallback**: `fallback_to_baseline=True` ensures graceful degradation
- **Monitor latency**: Fine-tuned models should be faster than baseline (smaller dims)
- **A/B test before full rollout**: Compare fine-tuned vs baseline on real traffic

### Model Selection
- **Production**: all-MiniLM-L6-v2 (fastest, 95% quality of larger models)
- **High accuracy**: BAAI/bge-small-en-v1.5 (best quality/speed trade-off)
- **Maximum quality**: intfloat/e5-base-v2 (highest accuracy, 2x slower)

---

## Troubleshooting

See [README.md#troubleshooting](README.md#troubleshooting) for common issues and solutions.
