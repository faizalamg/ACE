"""
Optimization V3: BGE Reranker + Boosted Retrieval

This optimization uses:
1. BAAI/bge-reranker-base - State-of-the-art reranker (2024)
2. Increased first-stage retrieval (50 candidates)
3. Multi-query expansion with RRF fusion
4. Score-based filtering

BGE reranker is trained on diverse retrieval tasks and outperforms
ms-marco based models on many benchmarks.

Expected improvement: +10-20% over V2
"""

import json
import math
import re
import time
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Set
import httpx

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

QDRANT_URL = "http://localhost:6333"
EMBEDDING_URL = "http://localhost:1234"
COLLECTION_NAME = "ace_memories_hybrid"
EMBEDDING_MODEL = "text-embedding-qwen3-embedding-8b"

# Boosted retrieval configuration
NUM_EXPANDED_QUERIES = 4
CANDIDATES_PER_QUERY = 25  # Increased from 20
FIRST_STAGE_K = 50  # Increased from 40
FINAL_K = 10

# State-of-the-art reranker
BGE_RERANKER_MODEL = "BAAI/bge-reranker-base"

# RRF parameter
RRF_K = 60

# BM25 parameters
BM25_K1 = 1.5
BM25_B = 0.75
AVG_DOC_LENGTH = 50

STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
    'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have',
    'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
    'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'it', 'its',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'what',
    'which', 'who', 'whom', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
    'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now'
}

# Technical synonym mappings
SYNONYMS = {
    'api': ['endpoint', 'interface', 'service'],
    'config': ['configuration', 'settings', 'options'],
    'db': ['database', 'datastore', 'storage'],
    'auth': ['authentication', 'authorization', 'login', 'security'],
    'ui': ['interface', 'frontend', 'view', 'component'],
    'validate': ['verify', 'check', 'ensure', 'confirm'],
    'create': ['generate', 'build', 'construct', 'initialize'],
    'delete': ['remove', 'destroy', 'cleanup', 'clear'],
    'update': ['modify', 'change', 'edit', 'patch'],
    'fetch': ['get', 'retrieve', 'load', 'query'],
    'store': ['save', 'persist', 'cache', 'write'],
    'handle': ['process', 'manage', 'deal with', 'respond to'],
    'parse': ['process', 'interpret', 'decode', 'extract'],
    'test': ['spec', 'unit test', 'verification', 'check'],
    'mock': ['stub', 'fake', 'simulate'],
    'error': ['exception', 'failure', 'issue', 'problem', 'bug'],
    'fix': ['resolve', 'repair', 'correct', 'patch'],
    'debug': ['troubleshoot', 'diagnose', 'investigate'],
    'function': ['method', 'procedure', 'routine', 'handler'],
    'class': ['type', 'model', 'entity', 'object'],
    'module': ['package', 'library', 'component'],
    'initialize': ['setup', 'bootstrap', 'configure', 'start'],
    'cleanup': ['dispose', 'teardown', 'finalize', 'destroy'],
    'input': ['data', 'parameter', 'argument'],
    'output': ['result', 'response', 'return value'],
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class QueryResult:
    """Result of a single query evaluation."""
    query: str
    query_category: str
    difficulty: str
    expected_memory_id: int
    retrieved_ids: List[int]
    retrieved_scores: List[float]
    rank: Optional[int]
    score: Optional[float]
    success_at_1: bool
    success_at_3: bool
    success_at_5: bool
    success_at_10: bool
    reciprocal_rank: float
    latency_ms: float
    false_positives_above: int
    expanded_queries: List[str] = field(default_factory=list)
    expansion_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0


@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    timestamp: str
    configuration: Dict[str, str]
    test_suite_stats: Dict[str, Any]
    search_type: str
    optimization: str

    total_queries: int = 0
    overall_recall_at_1: float = 0.0
    overall_recall_at_3: float = 0.0
    overall_recall_at_5: float = 0.0
    overall_recall_at_10: float = 0.0
    overall_mrr: float = 0.0
    overall_ndcg_at_10: float = 0.0

    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_avg_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0

    expansion_latency_avg_ms: float = 0.0
    retrieval_latency_avg_ms: float = 0.0
    rerank_latency_avg_ms: float = 0.0

    results_by_query_category: Dict[str, Dict] = field(default_factory=dict)
    results_by_difficulty: Dict[str, Dict] = field(default_factory=dict)
    results_by_memory_category: Dict[str, Dict] = field(default_factory=dict)

    complete_misses: int = 0
    low_rank: int = 0
    false_positive_dominant: int = 0

    improvement_over_baseline: Dict[str, float] = field(default_factory=dict)
    improvement_over_v2: Dict[str, float] = field(default_factory=dict)
    memory_results: List[Dict] = field(default_factory=list)


# ============================================================================
# QUERY EXPANSION
# ============================================================================

class QueryExpander:
    """Rule-based query expansion."""

    def __init__(self):
        self.synonyms = SYNONYMS

    def expand_query(self, query: str, num_expansions: int = 3) -> List[str]:
        """Expand a query into multiple variations."""
        expansions = [query]

        synonym_version = self._expand_synonyms(query)
        if synonym_version != query:
            expansions.append(synonym_version)

        question_version = self._reformulate_question(query)
        if question_version and question_version != query:
            expansions.append(question_version)

        tech_version = self._expand_technical_terms(query)
        if tech_version != query:
            expansions.append(tech_version)

        if len(query.split()) <= 5:
            context_version = self._add_context(query)
            if context_version != query:
                expansions.append(context_version)

        seen = set()
        unique_expansions = []
        for exp in expansions:
            if exp.lower() not in seen:
                seen.add(exp.lower())
                unique_expansions.append(exp)

        return unique_expansions[:num_expansions + 1]

    def _expand_synonyms(self, query: str) -> str:
        words = query.lower().split()
        expanded_words = []
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.synonyms:
                expanded_words.append(self.synonyms[clean_word][0])
            else:
                expanded_words.append(word)
        return ' '.join(expanded_words)

    def _reformulate_question(self, query: str) -> Optional[str]:
        query_lower = query.lower().strip()
        patterns = {
            'how': ['what is the approach to', 'what is the method for'],
            'what': ['which', 'describe'],
            'why': ['what is the reason for', 'purpose of'],
            'where': ['in which location', 'which file'],
        }
        for qword, alternatives in patterns.items():
            if query_lower.startswith(qword + ' '):
                rest = query[len(qword) + 1:].strip()
                return f"{alternatives[0]} {rest}"
        if not query_lower.endswith('?') and not any(query_lower.startswith(q) for q in ['how', 'what', 'why', 'where']):
            return f"how to {query}"
        return None

    def _expand_technical_terms(self, query: str) -> str:
        expansions = {
            'api': 'API application programming interface',
            'ui': 'UI user interface',
            'db': 'database',
            'config': 'configuration',
            'auth': 'authentication',
            'env': 'environment',
        }
        words = query.split()
        expanded = []
        for word in words:
            clean = word.lower().strip('.,!?')
            if clean in expansions:
                expanded.append(expansions[clean])
            else:
                expanded.append(word)
        return ' '.join(expanded)

    def _add_context(self, query: str) -> str:
        query_lower = query.lower()
        if any(w in query_lower for w in ['validate', 'input', 'check', 'sanitize']):
            return f"{query} data validation security"
        elif any(w in query_lower for w in ['error', 'exception', 'handle']):
            return f"{query} error handling recovery"
        elif any(w in query_lower for w in ['test', 'mock', 'assert']):
            return f"{query} testing quality assurance"
        elif any(w in query_lower for w in ['api', 'endpoint', 'request']):
            return f"{query} API design integration"
        elif any(w in query_lower for w in ['config', 'setting', 'option']):
            return f"{query} configuration management"
        return f"{query} best practices"


# ============================================================================
# BM25 TOKENIZATION
# ============================================================================

def tokenize_bm25(text: str) -> List[str]:
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = text.replace('_', ' ')
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def compute_bm25_sparse(text: str) -> Dict[str, Any]:
    tokens = tokenize_bm25(text)
    if not tokens:
        return {"indices": [], "values": []}

    tf = Counter(tokens)
    doc_length = len(tokens)

    indices = []
    values = []

    for term, freq in tf.items():
        term_hash = int(hashlib.md5(term.encode()).hexdigest()[:8], 16)
        indices.append(term_hash)

        tf_weight = (freq * (BM25_K1 + 1)) / (
            freq + BM25_K1 * (1 - BM25_B + BM25_B * doc_length / AVG_DOC_LENGTH)
        )
        values.append(float(tf_weight))

    return {"indices": indices, "values": values}


# ============================================================================
# RECIPROCAL RANK FUSION
# ============================================================================

def reciprocal_rank_fusion(result_sets: List[List[Dict]], k: int = RRF_K) -> List[Dict]:
    scores: Dict[int, float] = {}
    doc_data: Dict[int, Dict] = {}

    for results in result_sets:
        for rank, doc in enumerate(results, start=1):
            doc_id = doc["id"]
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank)
            if doc_id not in doc_data:
                doc_data[doc_id] = doc

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    fused_results = []
    for doc_id, score in sorted_items:
        doc = doc_data[doc_id].copy()
        doc["rrf_score"] = score
        doc["original_score"] = doc.get("score", 0)
        doc["score"] = score
        fused_results.append(doc)

    return fused_results


# ============================================================================
# BGE RERANKER EVALUATOR
# ============================================================================

class BGERerankerEvaluator:
    """Evaluator with state-of-the-art BGE reranker."""

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        embedding_url: str = EMBEDDING_URL,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        reranker_model: str = BGE_RERANKER_MODEL
    ):
        self.qdrant_url = qdrant_url
        self.embedding_url = embedding_url
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.client = httpx.Client(timeout=60.0)
        self.expander = QueryExpander()

        if CROSS_ENCODER_AVAILABLE:
            print(f"Loading BGE reranker model: {reranker_model}")
            self.reranker = CrossEncoder(reranker_model, max_length=512)
            print("BGE reranker loaded successfully")
        else:
            self.reranker = None

    def get_embedding(self, text: str) -> Optional[List[float]]:
        try:
            resp = self.client.post(
                f"{self.embedding_url}/v1/embeddings",
                json={"model": self.embedding_model, "input": text[:8000]}
            )
            if resp.status_code == 200:
                return resp.json()["data"][0]["embedding"]
        except Exception:
            pass
        return None

    def hybrid_search_single(self, query: str, limit: int = CANDIDATES_PER_QUERY) -> List[Dict]:
        dense_embedding = self.get_embedding(query)
        if not dense_embedding:
            return []

        sparse_vector = compute_bm25_sparse(query)

        hybrid_query = {
            "prefetch": [
                {"query": dense_embedding, "using": "dense", "limit": limit * 2}
            ],
            "query": {"fusion": "rrf"},
            "limit": limit,
            "with_payload": True
        }

        if sparse_vector.get("indices"):
            hybrid_query["prefetch"].append({
                "query": {
                    "indices": sparse_vector["indices"],
                    "values": sparse_vector["values"]
                },
                "using": "sparse",
                "limit": limit * 2
            })

        try:
            resp = self.client.post(
                f"{self.qdrant_url}/collections/{self.collection_name}/points/query",
                json=hybrid_query
            )
            if resp.status_code == 200:
                return resp.json().get("result", {}).get("points", [])
        except Exception:
            pass

        return []

    def multi_query_search(self, queries: List[str], limit: int = FIRST_STAGE_K) -> Tuple[List[Dict], float]:
        start = time.perf_counter()

        all_results = []
        for query in queries:
            results = self.hybrid_search_single(query, CANDIDATES_PER_QUERY)
            if results:
                all_results.append(results)

        latency = (time.perf_counter() - start) * 1000

        if not all_results:
            return [], latency

        fused = reciprocal_rank_fusion(all_results, k=RRF_K)
        return fused[:limit], latency

    def rerank_with_bge(
        self,
        query: str,
        candidates: List[Dict],
        final_k: int = FINAL_K
    ) -> Tuple[List[Dict], float]:
        """Re-rank using BGE reranker - state-of-the-art model."""
        if not self.reranker or not candidates:
            return candidates[:final_k], 0.0

        start = time.perf_counter()

        pairs = []
        for c in candidates:
            payload = c.get("payload", {})
            doc_text = payload.get("lesson", "") or payload.get("content", "") or str(payload)
            pairs.append([query, doc_text[:1000]])

        # BGE reranker produces scores where higher is better
        scores = self.reranker.predict(pairs)

        scored_candidates = list(zip(scores, candidates))
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        reranked = []
        for score, candidate in scored_candidates[:final_k]:
            candidate["bge_score"] = float(score)
            candidate["score"] = float(score)
            candidate["reranked"] = True
            reranked.append(candidate)

        latency = (time.perf_counter() - start) * 1000
        return reranked, latency

    def search_with_expansion(
        self,
        query: str,
        limit: int = FINAL_K
    ) -> Tuple[List[Dict], List[str], float, float, float]:
        exp_start = time.perf_counter()
        expanded_queries = self.expander.expand_query(query, NUM_EXPANDED_QUERIES - 1)
        expansion_latency = (time.perf_counter() - exp_start) * 1000

        candidates, retrieval_latency = self.multi_query_search(expanded_queries, FIRST_STAGE_K)

        reranked, rerank_latency = self.rerank_with_bge(query, candidates, limit)

        return reranked, expanded_queries, expansion_latency, retrieval_latency, rerank_latency

    def evaluate_query(
        self,
        query: str,
        query_category: str,
        difficulty: str,
        expected_memory_id: int
    ) -> QueryResult:
        results, expanded_queries, exp_lat, ret_lat, rerank_lat = self.search_with_expansion(query)
        total_latency = exp_lat + ret_lat + rerank_lat

        retrieved_ids = [r["id"] for r in results]
        retrieved_scores = [r.get("score", 0) for r in results]

        rank = None
        score = None
        if expected_memory_id in retrieved_ids:
            rank = retrieved_ids.index(expected_memory_id) + 1
            score = retrieved_scores[retrieved_ids.index(expected_memory_id)]

        success_at_1 = rank == 1 if rank else False
        success_at_3 = rank is not None and rank <= 3
        success_at_5 = rank is not None and rank <= 5
        success_at_10 = rank is not None and rank <= 10
        reciprocal_rank = 1.0 / rank if rank else 0.0
        false_positives_above = rank - 1 if rank else len(retrieved_ids)

        return QueryResult(
            query=query,
            query_category=query_category,
            difficulty=difficulty,
            expected_memory_id=expected_memory_id,
            retrieved_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            rank=rank,
            score=score,
            success_at_1=success_at_1,
            success_at_3=success_at_3,
            success_at_5=success_at_5,
            success_at_10=success_at_10,
            reciprocal_rank=reciprocal_rank,
            latency_ms=total_latency,
            false_positives_above=false_positives_above,
            expanded_queries=expanded_queries,
            expansion_latency_ms=exp_lat,
            retrieval_latency_ms=ret_lat,
            rerank_latency_ms=rerank_lat
        )

    def evaluate_memory(self, test_case: Dict) -> Dict:
        memory_id = test_case["memory_id"]
        content = test_case["content"]
        category = test_case["category"]
        queries = test_case.get("generated_queries", [])

        query_results = []
        for q in queries:
            qr = self.evaluate_query(
                query=q["query"],
                query_category=q["category"],
                difficulty=q["difficulty"],
                expected_memory_id=memory_id
            )
            query_results.append(qr)

        n = len(query_results)
        if n == 0:
            return {
                "memory_id": memory_id,
                "category": category,
                "total_queries": 0,
                "recall_at_1": 0,
                "recall_at_5": 0,
                "recall_at_10": 0,
                "mrr": 0,
                "avg_latency_ms": 0,
                "query_results": []
            }

        return {
            "memory_id": memory_id,
            "category": category,
            "content": content[:100],
            "total_queries": n,
            "recall_at_1": sum(1 for r in query_results if r.success_at_1) / n,
            "recall_at_5": sum(1 for r in query_results if r.success_at_5) / n,
            "recall_at_10": sum(1 for r in query_results if r.success_at_10) / n,
            "mrr": sum(r.reciprocal_rank for r in query_results) / n,
            "avg_latency_ms": sum(r.latency_ms for r in query_results) / n,
            "avg_rerank_latency_ms": sum(r.rerank_latency_ms for r in query_results) / n,
            "query_results": query_results
        }

    def calculate_ndcg(self, query_results: List[QueryResult], k: int = 10) -> float:
        ndcg_scores = []
        for qr in query_results:
            if qr.rank is None or qr.rank > k:
                ndcg_scores.append(0.0)
                continue
            dcg = 1.0 / math.log2(qr.rank + 1)
            idcg = 1.0 / math.log2(2)
            ndcg_scores.append(dcg / idcg)
        return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    def load_previous_results(self, path: Path) -> Optional[Dict]:
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def run_evaluation(
        self,
        test_suite_path: Path,
        output_path: Path,
        baseline_path: Optional[Path] = None,
        v2_path: Optional[Path] = None
    ) -> EvaluationResult:
        print(f"\n{'='*80}")
        print("OPTIMIZATION V3: BGE RERANKER + BOOSTED RETRIEVAL")
        print(f"{'='*80}")
        print(f"Test Suite: {test_suite_path}")
        print(f"Output: {output_path}")
        print(f"Expanded Queries: {NUM_EXPANDED_QUERIES}")
        print(f"Candidates Per Query: {CANDIDATES_PER_QUERY}")
        print(f"First Stage K: {FIRST_STAGE_K}")
        print(f"Final K: {FINAL_K}")
        print(f"Reranker: {BGE_RERANKER_MODEL}")
        print(f"{'='*80}\n")

        baseline = self.load_previous_results(baseline_path) if baseline_path else None
        v2_results = self.load_previous_results(v2_path) if v2_path else None

        if baseline:
            print(f"Baseline: R@1={baseline['overall_recall_at_1']:.2%}, R@5={baseline['overall_recall_at_5']:.2%}")
        if v2_results:
            print(f"V2: R@1={v2_results['overall_recall_at_1']:.2%}, R@5={v2_results['overall_recall_at_5']:.2%}")

        with open(test_suite_path) as f:
            data = json.load(f)

        test_cases = data["test_cases"]
        metadata = data.get("metadata", {})

        print(f"\nLoaded {len(test_cases)} test cases")
        total_queries = sum(len(tc.get("generated_queries", [])) for tc in test_cases)
        print(f"Total queries: {total_queries}")

        result = EvaluationResult(
            timestamp=datetime.now().isoformat(),
            configuration={
                "qdrant_url": self.qdrant_url,
                "embedding_url": self.embedding_url,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
                "reranker_model": BGE_RERANKER_MODEL,
                "num_expanded_queries": NUM_EXPANDED_QUERIES,
                "candidates_per_query": CANDIDATES_PER_QUERY,
                "first_stage_k": FIRST_STAGE_K,
                "final_k": FINAL_K,
                "rrf_k": RRF_K,
                "bm25_k1": BM25_K1,
                "bm25_b": BM25_B
            },
            test_suite_stats=metadata.get("generation_stats", {}),
            search_type="hybrid_multiquery_bge_reranking",
            optimization="bge_reranker_v3"
        )

        all_query_results: List[QueryResult] = []
        all_latencies: List[float] = []
        all_expansion_latencies: List[float] = []
        all_retrieval_latencies: List[float] = []
        all_rerank_latencies: List[float] = []
        by_query_category = defaultdict(list)
        by_difficulty = defaultdict(list)
        by_memory_category = defaultdict(list)

        for i, tc in enumerate(test_cases):
            print(f"\n[{i+1}/{len(test_cases)}] Memory {tc['memory_id']} ({tc['category']})")
            print(f"  Content: {tc['content'][:60]}...")

            mem_result = self.evaluate_memory(tc)

            print(f"  Queries: {mem_result['total_queries']}")
            print(f"  Recall@1: {mem_result['recall_at_1']:.2%}")
            print(f"  Recall@5: {mem_result['recall_at_5']:.2%}")
            print(f"  MRR: {mem_result['mrr']:.3f}")

            for qr in mem_result["query_results"]:
                all_query_results.append(qr)
                all_latencies.append(qr.latency_ms)
                all_expansion_latencies.append(qr.expansion_latency_ms)
                all_retrieval_latencies.append(qr.retrieval_latency_ms)
                all_rerank_latencies.append(qr.rerank_latency_ms)
                by_query_category[qr.query_category].append(qr)
                by_difficulty[qr.difficulty].append(qr)
                by_memory_category[tc["category"]].append(qr)

            result.memory_results.append({
                "memory_id": mem_result["memory_id"],
                "category": mem_result["category"],
                "total_queries": mem_result["total_queries"],
                "recall_at_1": mem_result["recall_at_1"],
                "recall_at_5": mem_result["recall_at_5"],
                "recall_at_10": mem_result["recall_at_10"],
                "mrr": mem_result["mrr"],
                "avg_latency_ms": mem_result["avg_latency_ms"]
            })

        # Calculate overall metrics
        n = len(all_query_results)
        result.total_queries = n

        result.overall_recall_at_1 = sum(1 for r in all_query_results if r.success_at_1) / n
        result.overall_recall_at_3 = sum(1 for r in all_query_results if r.success_at_3) / n
        result.overall_recall_at_5 = sum(1 for r in all_query_results if r.success_at_5) / n
        result.overall_recall_at_10 = sum(1 for r in all_query_results if r.success_at_10) / n
        result.overall_mrr = sum(r.reciprocal_rank for r in all_query_results) / n
        result.overall_ndcg_at_10 = self.calculate_ndcg(all_query_results, k=10)

        # Latency stats
        all_latencies.sort()
        result.latency_min_ms = min(all_latencies)
        result.latency_max_ms = max(all_latencies)
        result.latency_avg_ms = sum(all_latencies) / len(all_latencies)
        result.latency_p50_ms = all_latencies[len(all_latencies) // 2]
        result.latency_p95_ms = all_latencies[int(len(all_latencies) * 0.95)]
        result.latency_p99_ms = all_latencies[int(len(all_latencies) * 0.99)]
        result.expansion_latency_avg_ms = sum(all_expansion_latencies) / len(all_expansion_latencies)
        result.retrieval_latency_avg_ms = sum(all_retrieval_latencies) / len(all_retrieval_latencies)
        result.rerank_latency_avg_ms = sum(all_rerank_latencies) / len(all_rerank_latencies)

        # Breakdowns
        for cat, qrs in by_query_category.items():
            n_cat = len(qrs)
            result.results_by_query_category[cat] = {
                "total": n_cat,
                "recall_at_1": sum(1 for r in qrs if r.success_at_1) / n_cat,
                "recall_at_5": sum(1 for r in qrs if r.success_at_5) / n_cat,
                "mrr": sum(r.reciprocal_rank for r in qrs) / n_cat
            }

        for diff, qrs in by_difficulty.items():
            n_diff = len(qrs)
            result.results_by_difficulty[diff] = {
                "total": n_diff,
                "recall_at_1": sum(1 for r in qrs if r.success_at_1) / n_diff,
                "recall_at_5": sum(1 for r in qrs if r.success_at_5) / n_diff,
                "mrr": sum(r.reciprocal_rank for r in qrs) / n_diff
            }

        for cat, qrs in by_memory_category.items():
            n_cat = len(qrs)
            result.results_by_memory_category[cat] = {
                "total": n_cat,
                "recall_at_1": sum(1 for r in qrs if r.success_at_1) / n_cat,
                "recall_at_5": sum(1 for r in qrs if r.success_at_5) / n_cat,
                "mrr": sum(r.reciprocal_rank for r in qrs) / n_cat
            }

        # Failure analysis
        result.complete_misses = sum(1 for r in all_query_results if r.rank is None)
        result.low_rank = sum(1 for r in all_query_results if r.rank and r.rank > 5)
        result.false_positive_dominant = sum(1 for r in all_query_results if r.rank and r.rank > 1)

        # Calculate improvements
        if baseline:
            result.improvement_over_baseline = {
                "recall_at_1_delta": result.overall_recall_at_1 - baseline["overall_recall_at_1"],
                "recall_at_5_delta": result.overall_recall_at_5 - baseline["overall_recall_at_5"],
                "mrr_delta": result.overall_mrr - baseline["overall_mrr"],
            }

        if v2_results:
            result.improvement_over_v2 = {
                "recall_at_1_delta": result.overall_recall_at_1 - v2_results["overall_recall_at_1"],
                "recall_at_5_delta": result.overall_recall_at_5 - v2_results["overall_recall_at_5"],
                "mrr_delta": result.overall_mrr - v2_results["overall_mrr"],
            }

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)

        # Print summary
        print(f"\n{'='*80}")
        print("BGE RERANKER EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"\nOVERALL METRICS:")
        print(f"  Total Queries: {result.total_queries}")
        print(f"  Recall@1: {result.overall_recall_at_1:.2%}")
        print(f"  Recall@3: {result.overall_recall_at_3:.2%}")
        print(f"  Recall@5: {result.overall_recall_at_5:.2%}")
        print(f"  Recall@10: {result.overall_recall_at_10:.2%}")
        print(f"  MRR: {result.overall_mrr:.4f}")
        print(f"  NDCG@10: {result.overall_ndcg_at_10:.4f}")

        if baseline:
            print(f"\nIMPROVEMENT OVER BASELINE:")
            imp = result.improvement_over_baseline
            print(f"  Recall@1: {baseline['overall_recall_at_1']:.2%} -> {result.overall_recall_at_1:.2%} ({imp['recall_at_1_delta']:+.2%})")
            print(f"  Recall@5: {baseline['overall_recall_at_5']:.2%} -> {result.overall_recall_at_5:.2%} ({imp['recall_at_5_delta']:+.2%})")

        if v2_results:
            print(f"\nIMPROVEMENT OVER V2:")
            imp = result.improvement_over_v2
            print(f"  Recall@1: {v2_results['overall_recall_at_1']:.2%} -> {result.overall_recall_at_1:.2%} ({imp['recall_at_1_delta']:+.2%})")
            print(f"  Recall@5: {v2_results['overall_recall_at_5']:.2%} -> {result.overall_recall_at_5:.2%} ({imp['recall_at_5_delta']:+.2%})")

        print(f"\nLATENCY BREAKDOWN:")
        print(f"  Total Avg: {result.latency_avg_ms:.1f}ms")
        print(f"  Retrieval Avg: {result.retrieval_latency_avg_ms:.1f}ms")
        print(f"  Rerank Avg: {result.rerank_latency_avg_ms:.1f}ms")
        print(f"  P50: {result.latency_p50_ms:.1f}ms")
        print(f"  P95: {result.latency_p95_ms:.1f}ms")

        print(f"\nFAILURE ANALYSIS:")
        print(f"  Complete Misses: {result.complete_misses} ({result.complete_misses/n:.1%})")
        print(f"  Low Rank (>5): {result.low_rank} ({result.low_rank/n:.1%})")

        print(f"\nBY DIFFICULTY:")
        for diff, data in sorted(result.results_by_difficulty.items()):
            print(f"  {diff}: R@1={data['recall_at_1']:.2%}, MRR={data['mrr']:.3f} (n={data['total']})")

        print(f"\nResults saved to: {output_path}")

        return result

    def close(self):
        self.client.close()


def main():
    base_path = Path(__file__).parent.parent
    test_suite = base_path / "test_suite" / "enhanced_test_suite.json"
    baseline_path = base_path / "baseline_results" / "hybrid_baseline.json"
    v2_path = base_path / "optimization_results" / "v2_query_expansion.json"
    output = base_path / "optimization_results" / "v3_bge_reranker.json"

    evaluator = BGERerankerEvaluator()
    try:
        result = evaluator.run_evaluation(test_suite, output, baseline_path, v2_path)
    finally:
        evaluator.close()

    return result


if __name__ == "__main__":
    main()
