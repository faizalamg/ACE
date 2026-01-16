# ACE Retrieval Benchmark

**Purpose**: Validate retrieval quality improvements for Phase 1-2 of the ACE RAG optimization project.

## Overview

This benchmark evaluates ACE's strategy bullet retrieval system using:
- **Representative cases** (50): Normal queries matching typical task_types/domains
- **Adversarial cases** (50): Queries designed to trick retrieval (keyword mismatches, domain crossing, etc.)

Unlike the Reddit benchmark (which tested document chunk retrieval), this benchmark evaluates:
- Multi-factor scoring (metadata + trigger patterns + effectiveness)
- Strategy bullets (not document chunks)
- ACE-specific retrieval patterns

## Metrics

| Metric | Description | Target (Phase 1) | Target (Phase 2) |
|--------|-------------|------------------|------------------|
| **Top-1 Accuracy** | % queries where top result is relevant | 40% | 50% |
| **MRR** | Mean Reciprocal Rank (1/rank of first relevant) | 0.7 | 0.8 |
| **nDCG@5** | Ranking quality of top-5 results | 0.75 | 0.85 |

## Dataset Structure

### Representative Cases (`benchmarks/data/representative.json`)

50 normal queries testing typical ACE usage:
- Clear task_type matching (debugging, security, optimization)
- Domain-specific queries
- Trigger pattern matching

Example:
```json
{
  "query": "How to debug timeout errors in production?",
  "query_type": "debugging",
  "relevant_bullet_ids": ["debug_timeout", "check_logs"],
  "irrelevant_bullet_ids": ["security_xss", "optimize_query"],
  "difficulty": "easy"
}
```

### Adversarial Cases (`benchmarks/data/adversarial.json`)

50 challenging queries designed to expose retrieval weaknesses:

1. **Keyword mismatch**: "System is hanging" → should match "timeout" bullets
2. **Domain crossing**: Generic query could apply to multiple domains
3. **High similarity, wrong context**: Similar wording, different intent
4. **Low similarity, correct strategy**: Different wording, same solution

Example:
```json
{
  "query": "System is hanging and won't respond",
  "query_type": "debugging",
  "relevant_bullet_ids": ["debug_timeout", "check_performance"],
  "irrelevant_bullet_ids": ["syntax_error_fix"],
  "difficulty": "adversarial"
}
```

## Usage

### Run Benchmark

```bash
# With default generated dataset
uv run python benchmarks/ace_retrieval_benchmark.py

# With custom playbook and dataset
uv run python benchmarks/ace_retrieval_benchmark.py \
    --playbook path/to/playbook.json \
    --dataset benchmarks/data/representative.json \
    --output results/baseline.json \
    --top-k 10
```

### Run Tests

```bash
# Run all benchmark tests
uv run pytest tests/test_ace_retrieval_benchmark.py -v

# Run specific test
uv run pytest tests/test_ace_retrieval_benchmark.py::TestACERetrievalBenchmark::test_calculate_mrr -v
```

### Load Dataset in Code

```python
from benchmarks.ace_retrieval_benchmark import load_benchmark_dataset, run_benchmark
from ace import Playbook
from pathlib import Path

# Load dataset
samples = load_benchmark_dataset(Path("benchmarks/data/representative.json"))

# Run benchmark
playbook = Playbook.load("path/to/playbook.json")
results = run_benchmark(playbook, samples, top_k=10)

print(f"Top-1 Accuracy: {results['top1_accuracy']:.2%}")
print(f"MRR: {results['mrr']:.4f}")
print(f"nDCG@5: {results['ndcg_at_5']:.4f}")
```

## Benchmark Results Format

```json
{
  "top1_accuracy": 0.42,
  "mrr": 0.6543,
  "ndcg_at_5": 0.7231,
  "num_samples": 100,
  "top_k": 10,
  "per_sample_results": [
    {
      "query": "How to debug timeout errors?",
      "query_type": "debugging",
      "difficulty": "easy",
      "relevant_ids": ["debug_timeout", "check_logs"],
      "irrelevant_ids": ["security_xss"],
      "retrieved_ids": ["debug_timeout", "check_logs", ...],
      "scores": [0.85, 0.72, ...]
    }
  ]
}
```

## Integration with TUNINGPROJECT.md

This benchmark is part of **Phase 3A** (ACE-Specific Benchmark Creation):

| Task | Status | Notes |
|------|--------|-------|
| 3A.1 | ✅ COMPLETE | BenchmarkSample dataclass defined |
| 3A.2 | ✅ COMPLETE | 50 representative cases created |
| 3A.3 | ✅ COMPLETE | 50 adversarial cases created |
| 3A.4 | ✅ COMPLETE | Benchmark runner implemented |
| 3A.5 | ✅ COMPLETE | Metrics (Top-1, MRR, nDCG@5) implemented |

**Next Steps** (Phase 3B - Baseline Measurement):
1. Run benchmark on current ACE implementation (before Phase 1-2 changes)
2. Document baseline metrics
3. Run benchmark after Phase 1 changes (metadata, filter fix, asymmetric penalties, dynamic weights)
4. Run benchmark after Phase 2 changes (session-level tracking)
5. Generate comparison report

## Validation Workflow

```
Phase 1 Implementation
    ↓
Run Benchmark (Representative + Adversarial)
    ↓
Measure: Top-1, MRR, nDCG@5
    ↓
Compare to Baseline
    ↓
If improvement >= 15%: Phase 1 SUCCESS
    ↓
Phase 2 Implementation
    ↓
Run Benchmark Again
    ↓
If improvement >= 30% total: Phase 2 SUCCESS
```

## Adding New Test Cases

To add new benchmark samples:

1. **Representative cases**: Add to `benchmarks/data/representative.json`
   - Clear query with obvious matching task_type/domain
   - Explicit relevant/irrelevant bullet IDs
   - Difficulty: easy/medium/hard

2. **Adversarial cases**: Add to `benchmarks/data/adversarial.json`
   - Intentionally challenging queries
   - Keyword mismatches, ambiguous intent
   - Difficulty: adversarial

Example:
```json
{
  "query": "Your new query here",
  "query_type": "debugging|security|optimization|etc",
  "relevant_bullet_ids": ["id1", "id2"],
  "irrelevant_bullet_ids": ["id3", "id4"],
  "difficulty": "easy|medium|hard|adversarial"
}
```

## Files

| File | Purpose |
|------|---------|
| `benchmarks/ace_retrieval_benchmark.py` | Main benchmark implementation |
| `benchmarks/data/representative.json` | 50 normal test cases |
| `benchmarks/data/adversarial.json` | 50 challenging test cases |
| `tests/test_ace_retrieval_benchmark.py` | Test suite for benchmark |
| `benchmarks/ACE_RETRIEVAL_BENCHMARK.md` | This documentation |

## Contributing

When modifying the benchmark:

1. **Follow TDD**: Write tests FIRST, then implementation
2. **Update both datasets**: Keep representative and adversarial balanced (50 each)
3. **Document changes**: Update this README with new patterns or metrics
4. **Validate metrics**: Ensure Top-1, MRR, nDCG@5 remain comparable across runs
5. **Update TUNINGPROJECT.md**: Mark tasks as complete with date and notes

## References

- **TUNINGPROJECT.md**: Phase 3A task definitions and success criteria
- **Reddit benchmark**: https://github.com/roampal-ai/roampal/tree/master/benchmarks
- **Reddit post**: https://www.reddit.com/r/Rag/comments/1pimyb9/
- **ACE Retrieval**: `ace/retrieval.py` SmartBulletIndex implementation
