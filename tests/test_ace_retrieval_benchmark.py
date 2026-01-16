"""
Tests for ACE retrieval benchmark infrastructure.

Following TDD protocol - these tests are written FIRST before implementing the benchmark.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ace import Playbook, EnrichedBullet


class TestACERetrievalBenchmark(unittest.TestCase):
    """Test suite for ACE retrieval benchmark functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_benchmark_sample_structure(self):
        """Test that BenchmarkSample structure can be created and validated."""
        # This will fail until we implement the dataclass
        from benchmarks.ace_retrieval_benchmark import BenchmarkSample

        sample = BenchmarkSample(
            query="How to debug a timeout error?",
            query_type="debugging",
            relevant_bullet_ids=["b1", "b2"],
            irrelevant_bullet_ids=["b3", "b4"],
            difficulty="medium"
        )

        self.assertEqual(sample.query, "How to debug a timeout error?")
        self.assertEqual(sample.query_type, "debugging")
        self.assertEqual(len(sample.relevant_bullet_ids), 2)
        self.assertEqual(len(sample.irrelevant_bullet_ids), 2)
        self.assertEqual(sample.difficulty, "medium")

    def test_load_benchmark_dataset(self):
        """Test loading benchmark dataset from JSON file."""
        from benchmarks.ace_retrieval_benchmark import load_benchmark_dataset

        # Create test dataset
        test_data = [
            {
                "query": "Test query 1",
                "query_type": "debugging",
                "relevant_bullet_ids": ["b1"],
                "irrelevant_bullet_ids": ["b2"],
                "difficulty": "easy"
            },
            {
                "query": "Test query 2",
                "query_type": "reasoning",
                "relevant_bullet_ids": ["b3", "b4"],
                "irrelevant_bullet_ids": ["b5"],
                "difficulty": "hard"
            }
        ]

        test_file = self.temp_path / "test_dataset.json"
        test_file.write_text(json.dumps(test_data))

        samples = load_benchmark_dataset(test_file)

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].query, "Test query 1")
        self.assertEqual(samples[1].query_type, "reasoning")

    def test_calculate_top1_accuracy(self):
        """Test Top-1 accuracy metric calculation."""
        from benchmarks.ace_retrieval_benchmark import calculate_top1_accuracy

        # Test case: Top ranked bullet matches one relevant bullet
        results = [
            {
                "query": "q1",
                "relevant_ids": ["b1", "b2"],
                "retrieved_ids": ["b1", "b3", "b4"]  # Top-1 is b1 (match)
            },
            {
                "query": "q2",
                "relevant_ids": ["b5"],
                "retrieved_ids": ["b6", "b5", "b7"]  # Top-1 is b6 (no match)
            },
            {
                "query": "q3",
                "relevant_ids": ["b8", "b9"],
                "retrieved_ids": ["b9", "b8", "b10"]  # Top-1 is b9 (match)
            }
        ]

        # Expected: 2/3 = 0.6667
        accuracy = calculate_top1_accuracy(results)
        self.assertAlmostEqual(accuracy, 0.6667, places=4)

    def test_calculate_mrr(self):
        """Test Mean Reciprocal Rank (MRR) calculation."""
        from benchmarks.ace_retrieval_benchmark import calculate_mrr

        results = [
            {
                "query": "q1",
                "relevant_ids": ["b1"],
                "retrieved_ids": ["b1", "b2", "b3"]  # Rank 1, RR = 1.0
            },
            {
                "query": "q2",
                "relevant_ids": ["b4"],
                "retrieved_ids": ["b5", "b6", "b4"]  # Rank 3, RR = 0.333
            },
            {
                "query": "q3",
                "relevant_ids": ["b7"],
                "retrieved_ids": ["b8", "b7", "b9"]  # Rank 2, RR = 0.5
            },
            {
                "query": "q4",
                "relevant_ids": ["b10"],
                "retrieved_ids": ["b11", "b12", "b13"]  # No match, RR = 0.0
            }
        ]

        # Expected: (1.0 + 0.333 + 0.5 + 0.0) / 4 = 0.458
        mrr = calculate_mrr(results)
        self.assertAlmostEqual(mrr, 0.458, places=3)

    def test_calculate_ndcg_at_k(self):
        """Test Normalized Discounted Cumulative Gain (nDCG@k) calculation."""
        from benchmarks.ace_retrieval_benchmark import calculate_ndcg_at_k

        results = [
            {
                "query": "q1",
                "relevant_ids": ["b1", "b2", "b3"],
                "retrieved_ids": ["b1", "b3", "b4", "b2", "b5"]  # 3 relevant in top 5
            }
        ]

        # Calculate nDCG@5
        ndcg = calculate_ndcg_at_k(results, k=5)

        # nDCG should be between 0 and 1
        self.assertGreaterEqual(ndcg, 0.0)
        self.assertLessEqual(ndcg, 1.0)

    def test_run_benchmark_with_playbook(self):
        """Test running complete benchmark with a playbook."""
        from benchmarks.ace_retrieval_benchmark import run_benchmark, BenchmarkSample

        # Create test playbook
        playbook = Playbook()
        playbook.add_enriched_bullet(
            section="debugging",
            content="Check timeout settings in configuration",
            bullet_id="b1",
            task_types=["debugging"],
            domains=["backend"],
            trigger_patterns=["timeout", "slow"]
        )
        playbook.add_enriched_bullet(
            section="debugging",
            content="Review error logs for stack traces",
            bullet_id="b2",
            task_types=["debugging"],
            trigger_patterns=["error", "exception"]
        )
        playbook.add_enriched_bullet(
            section="security",
            content="Sanitize user input to prevent injection",
            bullet_id="b3",
            task_types=["security"],
            domains=["security"],
            trigger_patterns=["injection", "xss"]
        )

        # Create test samples
        samples = [
            BenchmarkSample(
                query="How to debug timeout errors?",
                query_type="debugging",
                relevant_bullet_ids=["b1"],
                irrelevant_bullet_ids=["b3"],
                difficulty="easy"
            ),
            BenchmarkSample(
                query="What causes security vulnerabilities?",
                query_type="security",
                relevant_bullet_ids=["b3"],
                irrelevant_bullet_ids=["b1", "b2"],
                difficulty="medium"
            )
        ]

        # Run benchmark
        results = run_benchmark(playbook, samples, top_k=5)

        # Validate results structure
        self.assertIn("top1_accuracy", results)
        self.assertIn("mrr", results)
        self.assertIn("ndcg_at_5", results)
        self.assertIn("per_sample_results", results)

        # Validate metric ranges
        self.assertGreaterEqual(results["top1_accuracy"], 0.0)
        self.assertLessEqual(results["top1_accuracy"], 1.0)
        self.assertGreaterEqual(results["mrr"], 0.0)
        self.assertLessEqual(results["mrr"], 1.0)

    def test_benchmark_with_effectiveness_scores(self):
        """Test that benchmark respects effectiveness scores in ranking.
        
        Note: Reranking is disabled for this test to verify pure effectiveness-based ranking.
        With reranking enabled, cross-encoder semantic scores may override effectiveness.
        """
        from benchmarks.ace_retrieval_benchmark import run_benchmark, BenchmarkSample
        from unittest.mock import patch
        from ace.config import RetrievalConfig

        playbook = Playbook()

        # Bullet with high effectiveness
        playbook.add_enriched_bullet(
            section="test",
            content="High effectiveness strategy",
            bullet_id="b_high",
            task_types=["debugging"],
            trigger_patterns=["error"]
        )
        b_high = playbook.get_bullet("b_high")
        b_high.helpful = 10
        b_high.harmful = 1

        # Bullet with low effectiveness
        playbook.add_enriched_bullet(
            section="test",
            content="Low effectiveness strategy",
            bullet_id="b_low",
            task_types=["debugging"],
            trigger_patterns=["error"]
        )
        b_low = playbook.get_bullet("b_low")
        b_low.helpful = 1
        b_low.harmful = 10

        samples = [
            BenchmarkSample(
                query="How to handle errors?",
                query_type="debugging",
                relevant_bullet_ids=["b_high"],
                irrelevant_bullet_ids=["b_low"],
                difficulty="easy"
            )
        ]

        # Disable reranking to test pure effectiveness-based ranking
        mock_config = RetrievalConfig(enable_reranking=False)
        with patch('ace.retrieval.get_retrieval_config', return_value=mock_config):
            results = run_benchmark(playbook, samples, top_k=5)

        # High effectiveness bullet should rank higher
        first_result = results["per_sample_results"][0]
        self.assertEqual(first_result["retrieved_ids"][0], "b_high")


class TestBenchmarkDataGeneration(unittest.TestCase):
    """Test benchmark dataset generation utilities."""

    def test_generate_representative_cases_structure(self):
        """Test that representative cases have correct structure."""
        from benchmarks.ace_retrieval_benchmark import generate_representative_cases

        # This will fail until implemented
        # For now, we'll test that it returns a list
        cases = generate_representative_cases()

        self.assertIsInstance(cases, list)
        self.assertGreater(len(cases), 0)

        # Check first case structure
        if len(cases) > 0:
            first_case = cases[0]
            self.assertIn("query", first_case)
            self.assertIn("query_type", first_case)
            self.assertIn("relevant_bullet_ids", first_case)
            self.assertIn("irrelevant_bullet_ids", first_case)
            self.assertIn("difficulty", first_case)

    def test_generate_adversarial_cases_structure(self):
        """Test that adversarial cases have correct structure."""
        from benchmarks.ace_retrieval_benchmark import generate_adversarial_cases

        cases = generate_adversarial_cases()

        self.assertIsInstance(cases, list)
        self.assertGreater(len(cases), 0)

        # Adversarial cases should be marked as "adversarial" difficulty
        if len(cases) > 0:
            self.assertTrue(
                any(c.get("difficulty") == "adversarial" for c in cases),
                "At least some cases should be marked as adversarial"
            )


if __name__ == "__main__":
    unittest.main()
