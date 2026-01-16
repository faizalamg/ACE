"""
ACE-specific retrieval benchmark for validating RAG optimization improvements.

This benchmark evaluates retrieval quality for strategy bullets (not document chunks),
testing multi-factor scoring with metadata, trigger patterns, and effectiveness scores.

Benchmark Structure:
- Representative cases (50): Normal queries matching typical task_types/domains
- Adversarial cases (50): Designed to trick retrieval with keyword mismatches

Metrics:
- Top-1 Accuracy: Percentage of queries where top result is relevant
- MRR (Mean Reciprocal Rank): Average of 1/rank for first relevant result
- nDCG@5 (Normalized Discounted Cumulative Gain): Quality of top-5 ranking
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ace import Playbook

from ace.retrieval import SmartBulletIndex


@dataclass
class BenchmarkSample:
    """A single benchmark test case for retrieval validation.

    Attributes:
        query: The query text to retrieve bullets for
        query_type: Type of task (debugging, reasoning, security, etc.)
        relevant_bullet_ids: IDs of bullets that SHOULD be retrieved
        irrelevant_bullet_ids: IDs of bullets that should NOT be retrieved
        difficulty: Difficulty level (easy, medium, hard, adversarial)
    """
    query: str
    query_type: str
    relevant_bullet_ids: List[str]
    irrelevant_bullet_ids: List[str]
    difficulty: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkSample":
        """Create BenchmarkSample from dictionary."""
        return cls(**data)


def load_benchmark_dataset(file_path: Path) -> List[BenchmarkSample]:
    """Load benchmark dataset from JSON file.

    Args:
        file_path: Path to JSON file containing benchmark samples

    Returns:
        List of BenchmarkSample objects
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    return [BenchmarkSample.from_dict(item) for item in data]


def save_benchmark_dataset(samples: List[BenchmarkSample], file_path: Path) -> None:
    """Save benchmark dataset to JSON file.

    Args:
        samples: List of BenchmarkSample objects
        file_path: Path where JSON file should be saved
    """
    data = [sample.to_dict() for sample in samples]

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def calculate_top1_accuracy(results: List[Dict[str, Any]]) -> float:
    """Calculate Top-1 accuracy across all queries.

    Top-1 accuracy: Percentage of queries where the top-ranked result is relevant.

    Args:
        results: List of dicts with keys:
            - query: Query text
            - relevant_ids: List of relevant bullet IDs
            - retrieved_ids: List of retrieved bullet IDs (ranked)

    Returns:
        Float between 0.0 and 1.0
    """
    if not results:
        return 0.0

    hits = 0
    for result in results:
        relevant = set(result["relevant_ids"])
        retrieved = result["retrieved_ids"]

        # Check if top result is relevant
        if retrieved and retrieved[0] in relevant:
            hits += 1

    return hits / len(results)


def calculate_mrr(results: List[Dict[str, Any]]) -> float:
    """Calculate Mean Reciprocal Rank (MRR).

    MRR: Average of 1/rank for the first relevant result in each query.

    Args:
        results: List of dicts with keys:
            - query: Query text
            - relevant_ids: List of relevant bullet IDs
            - retrieved_ids: List of retrieved bullet IDs (ranked)

    Returns:
        Float between 0.0 and 1.0
    """
    if not results:
        return 0.0

    reciprocal_ranks = []

    for result in results:
        relevant = set(result["relevant_ids"])
        retrieved = result["retrieved_ids"]

        # Find rank of first relevant result
        rank = None
        for i, bullet_id in enumerate(retrieved, start=1):
            if bullet_id in relevant:
                rank = i
                break

        # Add reciprocal rank (0 if no relevant result found)
        if rank:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def calculate_ndcg_at_k(results: List[Dict[str, Any]], k: int = 5) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k (nDCG@k).

    nDCG@k: Quality metric that rewards relevant results appearing higher in ranking.
    Uses binary relevance (1 for relevant, 0 for irrelevant).

    Args:
        results: List of dicts with keys:
            - query: Query text
            - relevant_ids: List of relevant bullet IDs
            - retrieved_ids: List of retrieved bullet IDs (ranked)
        k: Cutoff for evaluation (default: 5)

    Returns:
        Float between 0.0 and 1.0
    """
    if not results:
        return 0.0

    ndcg_scores = []

    for result in results:
        relevant = set(result["relevant_ids"])
        retrieved = result["retrieved_ids"][:k]  # Only consider top-k

        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, bullet_id in enumerate(retrieved, start=1):
            if bullet_id in relevant:
                # Binary relevance: 1 if relevant, 0 otherwise
                # Discount by log2(i+1) for position
                dcg += 1.0 / math.log2(i + 1)

        # Calculate IDCG (Ideal DCG - if all relevant results were at top)
        num_relevant = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, num_relevant + 1))

        # Normalized DCG
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)

    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0


def run_benchmark(
    playbook: "Playbook",
    samples: List[BenchmarkSample],
    top_k: int = 10,
    min_effectiveness: Optional[float] = None
) -> Dict[str, Any]:
    """Run complete benchmark evaluation on a playbook.

    Args:
        playbook: ACE Playbook to evaluate
        samples: List of benchmark test samples
        top_k: Number of results to retrieve per query (default: 10)
        min_effectiveness: Optional minimum effectiveness filter (for testing filter behavior)

    Returns:
        Dict with benchmark results:
            - top1_accuracy: Top-1 accuracy metric
            - mrr: Mean Reciprocal Rank
            - ndcg_at_5: Normalized DCG at k=5
            - per_sample_results: Detailed results for each sample
    """
    index = SmartBulletIndex(playbook=playbook)

    per_sample_results = []

    for sample in samples:
        # Retrieve bullets for this query
        scored_bullets = index.retrieve(
            query=sample.query,
            task_type=sample.query_type,
            limit=top_k,
            min_effectiveness=min_effectiveness
        )

        # Extract bullet IDs in ranked order
        retrieved_ids = [sb.bullet.id for sb in scored_bullets]

        # Store result for metric calculation
        per_sample_results.append({
            "query": sample.query,
            "query_type": sample.query_type,
            "difficulty": sample.difficulty,
            "relevant_ids": sample.relevant_bullet_ids,
            "irrelevant_ids": sample.irrelevant_bullet_ids,
            "retrieved_ids": retrieved_ids,
            "scores": [sb.score for sb in scored_bullets]
        })

    # Calculate aggregate metrics
    top1_accuracy = calculate_top1_accuracy(per_sample_results)
    mrr = calculate_mrr(per_sample_results)
    ndcg_at_5 = calculate_ndcg_at_k(per_sample_results, k=5)

    return {
        "top1_accuracy": top1_accuracy,
        "mrr": mrr,
        "ndcg_at_5": ndcg_at_5,
        "num_samples": len(samples),
        "top_k": top_k,
        "per_sample_results": per_sample_results
    }


def generate_representative_cases() -> List[Dict[str, Any]]:
    """Generate 50 representative test cases for normal ACE usage.

    These cases test typical queries that match task_types, domains, and
    trigger_patterns in expected ways.

    Returns:
        List of dicts with benchmark sample structure
    """
    # Placeholder implementation - will be populated with actual test cases
    # In production, this would be manually curated or generated from real usage
    representative_cases = [
        {
            "query": "How to debug timeout errors in production?",
            "query_type": "debugging",
            "relevant_bullet_ids": ["debug_timeout", "check_logs"],
            "irrelevant_bullet_ids": ["security_xss", "optimize_query"],
            "difficulty": "easy"
        },
        {
            "query": "What steps should I follow when investigating a security breach?",
            "query_type": "security",
            "relevant_bullet_ids": ["incident_response", "forensics"],
            "irrelevant_bullet_ids": ["debug_memory_leak", "refactor_code"],
            "difficulty": "medium"
        },
        {
            "query": "How to approach solving a complex algorithmic problem?",
            "query_type": "reasoning",
            "relevant_bullet_ids": ["break_down_problem", "use_examples"],
            "irrelevant_bullet_ids": ["write_tests", "review_pr"],
            "difficulty": "medium"
        }
        # NOTE: This is a minimal placeholder. Production implementation would have 50 cases.
    ]

    return representative_cases


def generate_adversarial_cases() -> List[Dict[str, Any]]:
    """Generate 50 adversarial test cases designed to trick retrieval.

    Adversarial patterns:
    1. Keyword mismatch: Query uses synonyms, bullet uses different words
    2. Domain crossing: Query from one domain, correct bullet from another
    3. High similarity but wrong context: Similar wording but different intent
    4. Low similarity but correct strategy: Different wording, same solution

    Returns:
        List of dicts with benchmark sample structure
    """
    adversarial_cases = [
        {
            "query": "System is hanging and won't respond",  # "hanging" not "timeout"
            "query_type": "debugging",
            "relevant_bullet_ids": ["debug_timeout", "check_performance"],
            "irrelevant_bullet_ids": ["syntax_error_fix"],
            "difficulty": "adversarial"
        },
        {
            "query": "User input causes application to crash",  # Could be injection OR validation
            "query_type": "debugging",
            "relevant_bullet_ids": ["input_validation", "error_handling"],
            "irrelevant_bullet_ids": ["optimize_database"],
            "difficulty": "adversarial"
        },
        {
            "query": "Need to make code run faster",  # Generic, many possible strategies
            "query_type": "optimization",
            "relevant_bullet_ids": ["profile_first", "cache_results"],
            "irrelevant_bullet_ids": ["write_documentation"],
            "difficulty": "adversarial"
        }
        # NOTE: This is a minimal placeholder. Production implementation would have 50 cases.
    ]

    return adversarial_cases


def main():
    """Main entry point for running benchmark from command line."""
    import argparse
    from ace import Playbook

    parser = argparse.ArgumentParser(description="Run ACE retrieval benchmark")
    parser.add_argument(
        "--playbook",
        type=Path,
        help="Path to playbook JSON file (uses empty playbook if not provided)"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to benchmark dataset JSON file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save results JSON file"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve per query (default: 10)"
    )

    args = parser.parse_args()

    # Load or create playbook
    if args.playbook and args.playbook.exists():
        playbook = Playbook.load_from_file(args.playbook)
        print(f"Loaded playbook from {args.playbook}")
    else:
        playbook = Playbook()
        print("Using empty playbook")

    # Load or generate dataset
    if args.dataset and args.dataset.exists():
        samples = load_benchmark_dataset(args.dataset)
        print(f"Loaded {len(samples)} samples from {args.dataset}")
    else:
        # Generate default dataset
        print("Generating default benchmark dataset...")
        representative = generate_representative_cases()
        adversarial = generate_adversarial_cases()
        all_cases = representative + adversarial
        samples = [BenchmarkSample.from_dict(case) for case in all_cases]
        print(f"Generated {len(samples)} samples")

    # Run benchmark
    print(f"\nRunning benchmark (top_k={args.top_k})...")
    results = run_benchmark(playbook, samples, top_k=args.top_k)

    # Print results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Samples evaluated: {results['num_samples']}")
    print(f"Top-1 Accuracy:    {results['top1_accuracy']:.2%}")
    print(f"MRR:               {results['mrr']:.4f}")
    print(f"nDCG@5:            {results['ndcg_at_5']:.4f}")
    print("="*60)

    # Save results if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
