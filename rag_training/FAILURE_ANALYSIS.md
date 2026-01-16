# Failure Analysis Report

**Generated:** 2025-12-12
**Baseline Recall@1:** 0.39%
**Target Recall@1:** >80%

---

## Executive Summary

The baseline evaluation revealed **catastrophic retrieval failure** with 98.5% complete misses. Root cause analysis identified a fundamental issue in how the evaluation script performed searches.

---

## Baseline Results

| Metric | Value | Target | Gap |
|--------|-------|--------|-----|
| Recall@1 | 0.39% | >80% | -79.61% |
| Recall@5 | 1.06% | >95% | -93.94% |
| Recall@10 | 1.54% | >98% | -96.46% |
| MRR | 0.0075 | >0.85 | -0.8425 |
| Complete Misses | 98.5% | <2% | +96.5% |

---

## Root Cause Analysis

### Primary Issue: Incomplete Hybrid Search

**Finding:** The baseline evaluation script only used **dense (semantic) prefetch**, completely missing the **sparse (BM25)** component that provides keyword matching.

**Evidence:**
```python
# INCORRECT (what baseline did)
search_body = {
    "prefetch": [
        {
            "query": embedding,
            "using": "dense",
            "limit": limit * 3
        }
    ],
    "query": {"fusion": "rrf"},
    ...
}

# CORRECT (what it should do)
search_body = {
    "prefetch": [
        {
            "query": dense_embedding,
            "using": "dense",
            "limit": limit * 3
        },
        {
            "query": {
                "indices": sparse_indices,
                "values": sparse_values
            },
            "using": "sparse",
            "limit": limit * 3
        }
    ],
    "query": {"fusion": "rrf"},
    ...
}
```

**Impact:** Without sparse prefetch, RRF fusion has nothing to fuse - it just returns dense results with formulaic RRF scores (0.5, 0.33, 0.25...).

### Secondary Issues Identified

1. **Formulaic RRF Scores**
   - All results show identical score patterns: 0.5, 0.33, 0.25
   - Indicates RRF is working but only on single source (dense)
   - No differentiation between relevant and irrelevant results

2. **Semantic Gap**
   - Short queries ("validate input") don't match well semantically with longer lessons
   - BM25 keyword matching essential for bridging this gap

3. **No Query Preprocessing**
   - Queries passed directly without BM25 tokenization
   - Technical terms not preserved (CamelCase, snake_case)

---

## Collection Verification

The `ace_memories_hybrid` collection is correctly configured:

```json
{
    "vectors": {
        "dense": {"size": 768, "distance": "Cosine"}
    },
    "sparse_vectors": {
        "sparse": {}
    }
}
```

Sample point shows both vectors are populated:
- Dense: 768-dimensional vector
- Sparse: BM25 indices and values present

---

## Failure Distribution by Query Type

| Query Category | Total | Recall@1 | MRR |
|----------------|-------|----------|-----|
| casual | 39 | 0.00% | 0.000 |
| direct | 223 | 0.45% | 0.006 |
| edge_long | 75 | 0.00% | 0.008 |
| implicit | 21 | 0.00% | 0.000 |
| keyword | 95 | 0.00% | 0.006 |
| question_how | 102 | 0.98% | 0.015 |
| question_what | 75 | 0.00% | 0.007 |
| question_why | 51 | 1.96% | 0.020 |
| scenario | 74 | 0.00% | 0.007 |
| semantic | 82 | 0.00% | 0.004 |
| technical | 58 | 0.00% | 0.000 |
| template | 143 | 0.70% | 0.010 |

**Observations:**
- All categories have near-zero recall
- Question formats slightly better (question_how, question_why)
- Implicit and casual completely fail

---

## Fix Strategy

### Immediate Fix: Enable Proper Hybrid Search

1. Add BM25 sparse vector computation to query pipeline
2. Include sparse prefetch in search query
3. Verify RRF fusion with both sources

### Expected Impact

Based on research (see SOTA_RESEARCH.md):
- Hybrid search (RRF): +15-30% recall improvement
- Proper BM25 tokenization: +5-10% additional
- Combined: Should reach 20-40% baseline (still far from 95% target)

### Additional Optimizations Required

After hybrid search is working:
1. **Cross-Encoder Re-ranking** (+15-25%)
2. **Query Expansion/HyDE** (+10-15%)
3. **Embedding Fine-Tuning** (+15-35%)
4. **Metadata Filtering** (+10-20%)

---

## Corrective Actions

1. **DONE** - Identified root cause (missing sparse prefetch)
2. **TODO** - Implement corrected hybrid search
3. **TODO** - Re-run baseline evaluation
4. **TODO** - Compare results
5. **TODO** - Proceed with SOTA optimizations

---

## Technical Reference

### Correct Hybrid Search Query Format

```python
def hybrid_search(query: str, limit: int = 10):
    # 1. Get dense embedding
    dense_embedding = get_embedding(query)

    # 2. Compute BM25 sparse vector
    tokens = tokenize_bm25(query)  # CamelCase split, stopword removal
    sparse = compute_bm25_weights(tokens)

    # 3. Build hybrid query
    search_body = {
        "prefetch": [
            {
                "query": dense_embedding,
                "using": "dense",
                "limit": limit * 3
            },
            {
                "query": {
                    "indices": sparse["indices"],
                    "values": sparse["values"]
                },
                "using": "sparse",
                "limit": limit * 3
            }
        ],
        "query": {"fusion": "rrf"},
        "limit": limit,
        "with_payload": True
    }

    return qdrant.query(search_body)
```

### BM25 Tokenization

```python
def tokenize_bm25(text: str) -> List[str]:
    # Split CamelCase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Split snake_case
    text = text.replace('_', ' ')
    # Extract tokens
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    # Remove stopwords
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens
```

---

## Next Steps

1. Implement corrected hybrid search in evaluation script
2. Re-run evaluation with proper hybrid search
3. Establish true baseline with hybrid working
4. Proceed with SOTA optimization iterations
