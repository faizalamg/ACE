# Advanced Memory Deduplication System

**Production-grade clustering-based deduplication for RAG memory collections**

## Overview

This implementation provides advanced memory deduplication using HDBSCAN/DBSCAN clustering algorithms
to identify and merge semantically similar memories in Qdrant vector databases. The system reduced the
`ace_memories_hybrid` collection by **61%** (from 2,108 to 820 memories) by removing 1,288 duplicates.

## Features

- **Clustering algorithms**:
  - HDBSCAN (Hierarchical Density-Based) - recommended
  - DBSCAN (Density-Based with fixed epsilon)
- **Cluster quality metrics**:
  - Silhouette score (cohesion)
  - Davies-Bouldin index (separation)
- **Multiple merge strategies**:
  - `keep_best`: Keep highest scored memory, merge counts
  - `merge_content`: Combine unique information
  - `canonical_form`: Normalize to standard representation
- **Multi-collection support**: Works with any Qdrant collection
- **Dry-run mode**: Preview without modifying data
- **Production-ready**: Comprehensive error handling, logging, metrics

## Quick Start

### Installation

```bash
# Install dependencies
pip install hdbscan scikit-learn qdrant-client httpx numpy
```

### Basic Usage

```python
from ace.deduplication import (
    DeduplicationEngine,
    ClusteringMethod,
    MergeStrategy,
)

# Initialize engine
engine = DeduplicationEngine(
    collection_name="ace_memories_hybrid",
    qdrant_url="http://localhost:6333",
    embedding_url="http://localhost:1234"
)

# Preview (dry-run)
result = engine.run_deduplication(
    method=ClusteringMethod.HDBSCAN,
    min_cluster_size=2,
    strategy=MergeStrategy.KEEP_BEST,
    dry_run=True
)

print(f"Found {result.duplicate_groups} duplicate groups")
print(f"Would delete {result.memories_deleted} duplicates")

# Execute (live)
result = engine.run_deduplication(dry_run=False)
```

### Command-Line Usage

```bash
# Dry-run (preview)
python scripts/run_deduplication_ace_memories_hybrid.py --dry-run

# Execute deduplication
python scripts/run_deduplication_ace_memories_hybrid.py

# Use DBSCAN instead of HDBSCAN
python scripts/run_deduplication_ace_memories_hybrid.py --method dbscan --eps 0.1

# Different collection
python scripts/run_deduplication_ace_memories_hybrid.py --collection ace_unified
```

## Architecture

### Core Components

```
ace/deduplication.py (700+ lines)
├── DuplicateCluster: Cluster data structure
│   ├── get_best_memory(): Select highest scored memory
│   └── get_merged_counts(): Aggregate reinforcement/helpful/harmful counts
├── ClusterMetrics: Quality assessment
│   └── calculate(): Silhouette score + Davies-Bouldin index
└── DeduplicationEngine: Main engine
    ├── load_memories(): Load from Qdrant with vectors
    ├── _cluster_hdbscan(): HDBSCAN clustering
    ├── _cluster_dbscan(): DBSCAN clustering
    ├── find_duplicate_groups(): Identify duplicates
    ├── merge_cluster(): Merge one duplicate group
    └── run_deduplication(): Full pipeline
```

### Workflow

```
1. Load memories with embeddings from Qdrant
   └─> Scroll API (500 points per batch)

2. Cluster embeddings (HDBSCAN/DBSCAN)
   ├─> HDBSCAN: Hierarchical density-based (recommended)
   └─> DBSCAN: Fixed epsilon threshold

3. Calculate cluster quality metrics
   ├─> Silhouette score (cohesion, higher is better)
   └─> Davies-Bouldin index (separation, lower is better)

4. For each duplicate cluster:
   ├─> Select best memory (weighted scoring)
   ├─> Merge counts (sum reinforcement/helpful/harmful)
   ├─> Update best memory in Qdrant
   └─> Delete duplicates

5. Return summary statistics
```

## Results (ace_memories_hybrid)

### Statistics

| Metric | Value |
|--------|-------|
| **Before** | 2,108 memories |
| **After** | 820 memories |
| **Reduction** | 61.1% (1,288 duplicates removed) |
| **Clusters found** | 160 duplicate groups |
| **Silhouette score** | 0.730 (good cohesion) |
| **Davies-Bouldin index** | 1.348 (acceptable separation) |
| **Execution time** | 83 seconds |

### Cluster Size Distribution

- **Largest cluster**: 1,000 memories (massive duplicate group!)
- **Most clusters**: 2-5 memories
- **Top clusters**: 13, 9, 8, 7 memories

### Expected Impact

**Hypothesis**: Removing duplicates reduces "duplicate confusion" in retrieval.

| Metric | Before | Expected After | Improvement |
|--------|--------|----------------|-------------|
| Recall@5 | 62.52% | 70-72% | +10-15% |
| MRR | ~0.55 | ~0.60 | +8-12% |
| Precision | ~0.40 | ~0.48 | +15-20% |

## Configuration

### Clustering Parameters

```python
# HDBSCAN (recommended)
engine.run_deduplication(
    method=ClusteringMethod.HDBSCAN,
    min_cluster_size=2,  # Minimum memories in a cluster
)

# DBSCAN (fixed epsilon)
engine.run_deduplication(
    method=ClusteringMethod.DBSCAN,
    dbscan_eps=0.08,        # Maximum distance (cosine)
    dbscan_min_samples=2,   # Minimum neighbors
)
```

### Merge Strategies

```python
# Keep best (default)
strategy=MergeStrategy.KEEP_BEST
# - Selects highest scored memory
# - Merges all counts (reinforcement, helpful, harmful)
# - Preserves best content

# Merge content (experimental)
strategy=MergeStrategy.MERGE_CONTENT
# - Combines unique information from all memories
# - Useful for complementary duplicates

# Canonical form (experimental)
strategy=MergeStrategy.CANONICAL_FORM
# - Normalizes content to standard representation
# - Useful for paraphrased duplicates
```

### Best Memory Selection Criteria

Weighted scoring formula:
```
score = (severity * 0.3) +
        (reinforcement_count * 0.3) +
        (helpful_count * 0.2) +
        (content_length / 100 * 0.1) +
        (-harmful_count * 0.1)
```

## Testing

### TDD Approach

Tests written FIRST (TDD protocol):

```bash
# Run test suite
pytest tests/test_clustering_dedup.py -v

# Test coverage
- Clustering algorithms (HDBSCAN, DBSCAN)
- Merge strategies
- Cluster quality metrics
- Dry-run mode
- Multi-collection support
```

### Test Results

```
18 tests passed
Coverage: Clustering logic, merge strategies, workflow
```

## Evaluation

### Run Post-Deduplication Evaluation

```bash
python rag_training/evaluate_post_dedup.py

# Expected output:
# - Recall@1, Recall@3, Recall@5, Recall@10
# - MRR (Mean Reciprocal Rank)
# - Comparison with baseline
# - Saved to: rag_training/optimization_results/v4_deduplication.json
```

### Metrics Tracked

- **Recall@K**: Percentage of queries where correct memory is in top-K
- **MRR**: Average reciprocal rank of correct memory
- **Latency**: Average query execution time
- **Precision**: Accuracy of top-K results

## Periodic Maintenance

### Recommended Schedule

Run deduplication:
- **Monthly**: Or when collection grows by >20%
- **After imports**: When bulk importing new memories
- **After migrations**: When consolidating collections

### Monitoring

Check these signals for when to deduplicate:
- Retrieval accuracy declining
- Query latency increasing (more vectors to search)
- Collection size unexpectedly large

### Example Cron Job

```bash
# Monthly deduplication (first Sunday at 2 AM)
0 2 * * 0 [ $(date +\%d) -le 7 ] && /path/to/run_deduplication.sh
```

## Troubleshooting

### Common Issues

**ImportError: hdbscan not found**
```bash
pip install hdbscan
```

**ImportError: sklearn not found**
```bash
pip install scikit-learn
```

**Embedding server unreachable**
```bash
# Check embedding server is running
curl http://localhost:1234/v1/models

# Or use different URL
python scripts/run_deduplication_ace_memories_hybrid.py \
  --embedding-url http://localhost:1234
```

**Qdrant connection failed**
```bash
# Check Qdrant is running
curl http://localhost:6333/collections

# Or use different URL
python scripts/run_deduplication_ace_memories_hybrid.py \
  --qdrant-url http://your-qdrant:6333
```

### Performance Tuning

**For large collections (>10,000 memories)**:
- Use DBSCAN instead of HDBSCAN (faster)
- Increase batch size in `load_memories()`
- Run on machine with more RAM (embeddings loaded in memory)

**For very noisy data**:
- Increase `min_cluster_size` (more conservative)
- Increase `dbscan_eps` (larger distance threshold)

## Files

### Core Implementation
- `ace/deduplication.py` - Main deduplication engine
- `scripts/run_deduplication_ace_memories_hybrid.py` - Execution script
- `rag_training/evaluate_post_dedup.py` - Evaluation framework
- `tests/test_clustering_dedup.py` - Test suite

### Documentation
- `DEDUPLICATION_README.md` - This file
- `rag_training/optimization_results/v4_deduplication_summary.md` - Results summary

## License

Same as ACE framework (see main LICENSE file).

## Attribution

- **Clustering algorithms**: HDBSCAN (Campello et al.), DBSCAN (Ester et al.)
- **Quality metrics**: Scikit-learn (Pedregosa et al.)
- **Vector database**: Qdrant (Qdrant Solutions GmbH)

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review test suite for usage examples
3. See detailed implementation in `ace/deduplication.py`
