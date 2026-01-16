"""
Step 4: Fine-tune Cross-Encoder for Domain-Specific Reranking

This script fine-tunes a cross-encoder model on preference/strategy memory pairs
to replace the generic BGE-reranker that was trained on MS-MARCO.

Usage:
    python finetune_crossencoder.py

Requirements:
    pip install sentence-transformers torch

Input: crossencoder_training_pairs.json (from Step 3)
Output: ../models/domain_crossencoder/
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def main():
    print("=" * 60)
    print("Step 4: Fine-tune Cross-Encoder for Domain Reranking")
    print("=" * 60)

    # Check for training data
    training_file = Path(__file__).parent / "crossencoder_training_pairs.json"
    if not training_file.exists():
        print(f"[ERROR] Training data not found: {training_file}")
        print("Please run generate_finetuning_pairs.py first (Step 3)")
        return None

    # Load training data
    print(f"\n[1/5] Loading training data from {training_file}...")
    with open(training_file, 'r', encoding='utf-8') as f:
        training_pairs = json.load(f)

    print(f"      Loaded {len(training_pairs)} training pairs")

    # Count positives and negatives
    positives = sum(1 for p in training_pairs if p.get('label') == 1)
    negatives = len(training_pairs) - positives
    print(f"      Positives: {positives}, Negatives: {negatives}")

    # Check for sentence-transformers
    try:
        from sentence_transformers import CrossEncoder, InputExample
        from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
        import torch
    except ImportError:
        print("\n[ERROR] sentence-transformers not installed.")
        print("Install with: pip install sentence-transformers torch")
        return None

    # Prepare training examples
    print("\n[2/5] Preparing training examples...")
    train_examples = []
    eval_examples = []

    # Split 90/10 for train/eval
    split_idx = int(len(training_pairs) * 0.9)

    for i, pair in enumerate(training_pairs):
        query = pair.get('query', '')
        memory = pair.get('memory', '')
        label = float(pair.get('label', 0))

        if not query or not memory:
            continue

        example = InputExample(texts=[query, memory], label=label)

        if i < split_idx:
            train_examples.append(example)
        else:
            eval_examples.append(example)

    print(f"      Train examples: {len(train_examples)}")
    print(f"      Eval examples: {len(eval_examples)}")

    if len(train_examples) < 100:
        print("[ERROR] Not enough training examples (need at least 100)")
        return None

    # Initialize model
    print("\n[3/5] Initializing cross-encoder model...")
    base_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    print(f"      Base model: {base_model}")

    model = CrossEncoder(base_model, num_labels=1, max_length=256)

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"      Device: {device}")

    # Training parameters
    output_dir = Path(__file__).parent.parent / "models" / "domain_crossencoder"
    output_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = 2
    batch_size = 16 if device == "cuda" else 8
    warmup_steps = int(len(train_examples) * 0.1)

    print(f"\n[4/5] Training cross-encoder...")
    print(f"      Output: {output_dir}")
    print(f"      Epochs: {num_epochs}")
    print(f"      Batch size: {batch_size}")
    print(f"      Warmup steps: {warmup_steps}")

    # Create evaluator
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(
        eval_examples,
        name="domain-eval"
    )

    # Train
    start_time = datetime.now()

    model.fit(
        train_dataloader=torch.utils.data.DataLoader(
            train_examples,
            shuffle=True,
            batch_size=batch_size
        ),
        evaluator=evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        save_best_model=True,
        show_progress_bar=True
    )

    training_time = (datetime.now() - start_time).total_seconds()
    print(f"\n      Training completed in {training_time:.1f}s")

    # Save model
    print(f"\n[5/5] Saving model to {output_dir}...")
    model.save(str(output_dir))

    # Save training metadata
    metadata = {
        "base_model": base_model,
        "training_pairs": len(training_pairs),
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "positives": positives,
        "negatives": negatives,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "training_time_seconds": training_time,
        "timestamp": datetime.now().isoformat(),
        "device": device
    }

    with open(output_dir / "training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {output_dir}")
    print(f"Training time: {training_time:.1f}s")
    print(f"Train examples: {len(train_examples)}")
    print(f"Eval examples: {len(eval_examples)}")

    return str(output_dir)


if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n[SUCCESS] Model ready at: {result}")
    else:
        print("\n[FAILED] Fine-tuning did not complete")
        sys.exit(1)
