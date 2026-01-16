# RAG Retrieval Performance Metrics

## Primary Metrics

### 1. Recall@K
**Definition**: Proportion of queries where the correct memory appears in top K results.

```
Recall@K = (queries with correct answer in top K) / (total queries)
```

**Targets**:
- Recall@1: >80% (correct answer is top result)
- Recall@3: >90% (correct answer in top 3)
- Recall@5: >95% (correct answer in top 5)
- Recall@10: >98% (correct answer in top 10)

### 2. Mean Reciprocal Rank (MRR)
**Definition**: Average of reciprocal ranks of the first correct answer.

```
MRR = (1/N) * sum(1/rank_i)
```

Where rank_i is the position of the first correct result for query i.

**Target**: MRR > 0.85

### 3. Normalized Discounted Cumulative Gain (NDCG)
**Definition**: Measures ranking quality considering position-based discounting.

```
DCG@K = sum(rel_i / log2(i+1)) for i in 1..K
NDCG@K = DCG@K / IDCG@K
```

**Target**: NDCG@10 > 0.90

### 4. Precision@K
**Definition**: Proportion of relevant results in top K.

```
Precision@K = (relevant results in top K) / K
```

**Target**: Precision@5 > 0.70

## Secondary Metrics

### 5. Semantic Similarity Score
**Definition**: Average cosine similarity between query and retrieved memory embeddings.

**Target**: Mean similarity > 0.65 for true positives

### 6. False Positive Rate
**Definition**: Rate of irrelevant results appearing in top K.

```
FPR@K = (queries with irrelevant top-1 result) / (total queries)
```

**Target**: FPR@1 < 5%

### 7. Query Latency
**Definition**: Time to execute search and return results.

**Targets**:
- P50 latency: <50ms
- P95 latency: <100ms
- P99 latency: <200ms

### 8. Coverage
**Definition**: Percentage of memories that are successfully retrieved by at least one test query.

**Target**: 100% coverage

## Query Category Metrics

Track metrics separately for query types:
- **Direct queries**: Exact or near-exact phrasing
- **Paraphrased queries**: Same meaning, different words
- **Technical queries**: Using domain-specific terminology
- **Casual queries**: Informal/conversational language
- **Implicit queries**: Contextual hints without explicit keywords
- **Edge cases**: Very short (<3 words) or very long (>20 words) queries

## Failure Categories

### 1. Complete Miss
Memory not in top 100 results. Indicates fundamental semantic gap.

### 2. Low Rank
Memory found but rank > 10. Indicates scoring/weighting issues.

### 3. False Positive Dominance
Irrelevant memories consistently rank higher. Indicates metadata or filter issues.

### 4. Threshold Exclusion
Memory filtered out due to score threshold. Indicates threshold misconfiguration.

### 5. Embedding Mismatch
Low cosine similarity despite semantic relevance. Indicates embedding model limitations.

## Measurement Protocol

1. **Baseline Test**: Run all test queries against current configuration
2. **Per-Change Test**: After each optimization, run full test suite
3. **Regression Check**: Verify no metric degrades by >2%
4. **A/B Test**: When comparing configurations, use statistical significance (p<0.05)

## Reporting Format

```json
{
  "timestamp": "2025-12-12T00:00:00Z",
  "configuration": "baseline_v1",
  "total_queries": 1500,
  "total_memories": 100,
  "metrics": {
    "recall_at_1": 0.75,
    "recall_at_3": 0.88,
    "recall_at_5": 0.93,
    "recall_at_10": 0.97,
    "mrr": 0.82,
    "ndcg_at_10": 0.87,
    "precision_at_5": 0.65,
    "mean_similarity": 0.68,
    "fpr_at_1": 0.08,
    "latency_p50_ms": 45,
    "latency_p95_ms": 85,
    "coverage": 0.98
  },
  "failures_by_category": {
    "complete_miss": 15,
    "low_rank": 45,
    "false_positive_dominance": 22,
    "threshold_exclusion": 8,
    "embedding_mismatch": 10
  },
  "query_category_breakdown": {
    "direct": {"recall_at_1": 0.92, "mrr": 0.94},
    "paraphrased": {"recall_at_1": 0.78, "mrr": 0.85},
    "technical": {"recall_at_1": 0.80, "mrr": 0.86},
    "casual": {"recall_at_1": 0.65, "mrr": 0.75},
    "implicit": {"recall_at_1": 0.55, "mrr": 0.68},
    "edge_cases": {"recall_at_1": 0.45, "mrr": 0.60}
  }
}
```
