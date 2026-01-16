"""
Fortune 100 Self-Retrieval Test - GPU-Accelerated Full Optimization

This test validates self-retrieval: can we retrieve each memory using its own content?
Uses the ACTUAL memories from the current Qdrant collection with UUID IDs.

Target: 100% self-retrieval recall (if a memory exists, querying its content should find it)
"""

import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from ace.retrieval_optimized import (
    OptimizedRetriever,
    CROSS_ENCODER_AVAILABLE,
    GPU_RERANKER_AVAILABLE,
)
from ace.config import QdrantConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_memories_from_qdrant(limit: int = 500) -> List[Dict]:
    """Load memories directly from Qdrant collection."""
    config = QdrantConfig()
    client = httpx.Client(timeout=60.0)

    memories = []
    offset = None

    while len(memories) < limit:
        payload = {
            "limit": min(100, limit - len(memories)),
            "with_payload": True,
            "with_vector": False
        }
        if offset:
            payload["offset"] = offset

        resp = client.post(
            f"{config.url}/collections/{config.memories_collection}/points/scroll",
            json=payload
        )

        if resp.status_code != 200:
            print(f"Error loading memories: {resp.status_code}")
            break

        result = resp.json().get("result", {})
        points = result.get("points", [])

        if not points:
            break

        for p in points:
            memories.append({
                "id": p["id"],
                "payload": p.get("payload", {})
            })

        offset = result.get("next_page_offset")
        if not offset:
            break

    client.close()
    return memories


@dataclass
class TestResult:
    """Result for a single self-retrieval test."""
    memory_id: str
    query: str
    query_type: str
    retrieved_ids: List[str]
    rank: Optional[int]
    success_at_1: bool
    success_at_3: bool
    success_at_5: bool
    success_at_10: bool
    latency_ms: float
    reranked: bool


class Fortune100SelfTest:
    """Fortune 100 self-retrieval test with full optimization."""

    def __init__(self):
        print("=" * 80)
        print("FORTUNE 100 SELF-RETRIEVAL TEST - FULL OPTIMIZATION")
        print("=" * 80)
        print()

        print("SYSTEM STATUS:")
        print(f"  Cross-encoder available: {CROSS_ENCODER_AVAILABLE}")
        print(f"  GPU reranker (DirectML): {GPU_RERANKER_AVAILABLE}")

        print()
        print("Initializing OptimizedRetriever with full optimization...")
        self.retriever = OptimizedRetriever(
            config={
                "enable_reranking": True,
                "num_expanded_queries": 4,
                "candidates_per_query": 30,
                "first_stage_k": 30,
                "final_k": 10,
            }
        )

        if hasattr(self.retriever, 'cross_encoder'):
            ce = self.retriever.cross_encoder
            if hasattr(ce, 'use_gpu'):
                print(f"  GPU acceleration active: {ce.use_gpu}")

        print()
        print("FEATURES ENABLED:")
        print("  - Query expansion (4 variations)")
        print("  - Multi-query retrieval with RRF fusion")
        print("  - BM25 sparse + Dense hybrid search")
        print(f"  - Cross-encoder re-ranking: {'GPU (DirectML)' if GPU_RERANKER_AVAILABLE else 'CPU'}")
        print()
        print("Test Types:")
        print("  1. exact: Full content as query")
        print("  2. first_5_words: First 5 words of content")
        print("  3. keywords: Key terms from content")
        print("  4. semantic: First sentence or meaningful chunk")
        print()
        print("Target: 95%+ Recall@5")
        print("=" * 80)

    def extract_keywords(self, text: str, n: int = 5) -> str:
        """Extract key terms from text."""
        import re
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
                     'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have',
                     'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can',
                     'this', 'that', 'these', 'those', 'it', 'its', 'i', 'you', 'he', 'she', 'we',
                     'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how', 'not', 'no'}

        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        keywords = [w for w in words if w not in stopwords]
        # Dedupe while preserving order
        seen = set()
        unique = []
        for w in keywords:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        return ' '.join(unique[:n])

    def generate_queries(self, memory: Dict) -> List[tuple]:
        """Generate test queries from a memory."""
        payload = memory.get("payload", {})
        content = payload.get("lesson", "") or payload.get("content", "")

        if not content:
            return []

        queries = []

        # 1. Exact: first 200 chars
        queries.append((content[:200].strip(), "exact"))

        # 2. First 5 words
        words = content.split()
        if len(words) >= 5:
            queries.append((' '.join(words[:5]), "first_5_words"))

        # 3. Keywords
        kw = self.extract_keywords(content, 5)
        if kw:
            queries.append((kw, "keywords"))

        # 4. First sentence
        sentences = content.replace('\n', ' ').split('.')
        if sentences and len(sentences[0]) > 10:
            queries.append((sentences[0].strip()[:100], "semantic"))

        return queries

    def test_memory(self, memory: Dict) -> List[TestResult]:
        """Test retrieval for a single memory."""
        memory_id = memory["id"]
        queries = self.generate_queries(memory)

        results = []
        for query, query_type in queries:
            start = time.perf_counter()

            search_results = self.retriever.search(query, limit=10)

            latency = (time.perf_counter() - start) * 1000

            retrieved_ids = [r.id for r in search_results]
            reranked = any(r.reranked for r in search_results) if search_results else False

            rank = None
            if memory_id in retrieved_ids:
                rank = retrieved_ids.index(memory_id) + 1

            results.append(TestResult(
                memory_id=str(memory_id),
                query=query[:100],
                query_type=query_type,
                retrieved_ids=[str(rid) for rid in retrieved_ids[:5]],
                rank=rank,
                success_at_1=rank == 1 if rank else False,
                success_at_3=rank is not None and rank <= 3,
                success_at_5=rank is not None and rank <= 5,
                success_at_10=rank is not None and rank <= 10,
                latency_ms=latency,
                reranked=reranked
            ))

        return results

    def run_test(self, sample_size: int = 500) -> Dict:
        """Run full self-retrieval test."""
        print(f"\nLoading memories from Qdrant (limit={sample_size})...")
        memories = load_memories_from_qdrant(limit=sample_size)
        print(f"Loaded {len(memories)} memories")

        if not memories:
            print("ERROR: No memories found!")
            return {}

        all_results: List[TestResult] = []
        by_query_type = defaultdict(list)

        start_time = time.time()

        for i, memory in enumerate(memories):
            progress = (i + 1) / len(memories) * 100
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (len(memories) - i - 1) if i > 0 else 0

            if (i + 1) % 50 == 0 or i == 0:
                current_r5 = sum(1 for r in all_results if r.success_at_5) / len(all_results) * 100 if all_results else 0
                print(f"[{progress:5.1f}%] Memory {i+1}/{len(memories)} | "
                      f"Current Recall@5: {current_r5:.1f}% | "
                      f"ETA: {eta:.0f}s")

            results = self.test_memory(memory)
            all_results.extend(results)

            for r in results:
                by_query_type[r.query_type].append(r)

        total_time = time.time() - start_time
        n = len(all_results)

        # Calculate metrics
        metrics = {
            "total_memories": len(memories),
            "total_queries": n,
            "total_time_sec": total_time,
            "queries_per_sec": n / total_time if total_time > 0 else 0,
            "overall": {
                "recall_at_1": sum(1 for r in all_results if r.success_at_1) / n,
                "recall_at_3": sum(1 for r in all_results if r.success_at_3) / n,
                "recall_at_5": sum(1 for r in all_results if r.success_at_5) / n,
                "recall_at_10": sum(1 for r in all_results if r.success_at_10) / n,
                "mrr": sum((1.0 / r.rank if r.rank else 0) for r in all_results) / n,
            },
            "by_query_type": {},
            "failures": []
        }

        for qtype, results in by_query_type.items():
            n_type = len(results)
            metrics["by_query_type"][qtype] = {
                "total": n_type,
                "recall_at_1": sum(1 for r in results if r.success_at_1) / n_type,
                "recall_at_5": sum(1 for r in results if r.success_at_5) / n_type,
                "mrr": sum((1.0 / r.rank if r.rank else 0) for r in results) / n_type,
            }

        # Track failures
        for r in all_results:
            if not r.success_at_5:
                metrics["failures"].append({
                    "memory_id": r.memory_id,
                    "query_type": r.query_type,
                    "query": r.query,
                    "rank": r.rank
                })

        # Print results
        print()
        print("=" * 80)
        print("FORTUNE 100 SELF-RETRIEVAL TEST COMPLETE")
        print("=" * 80)

        print(f"\nTOTAL EXECUTION TIME: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"Queries per second: {metrics['queries_per_sec']:.1f}")

        print(f"\n{'='*40}")
        print("OVERALL METRICS")
        print(f"{'='*40}")
        print(f"  Total Memories: {metrics['total_memories']}")
        print(f"  Total Queries: {metrics['total_queries']}")
        print()

        o = metrics["overall"]
        r5_status = "PASS" if o["recall_at_5"] >= 0.95 else "FAIL"
        print(f"  Recall@1:  {o['recall_at_1']:.2%}")
        print(f"  Recall@3:  {o['recall_at_3']:.2%}")
        print(f"  Recall@5:  {o['recall_at_5']:.2%}  <-- TARGET 95% [{r5_status}]")
        print(f"  Recall@10: {o['recall_at_10']:.2%}")
        print(f"  MRR:       {o['mrr']:.4f}")

        print(f"\n{'='*40}")
        print("BY QUERY TYPE")
        print(f"{'='*40}")
        for qtype, data in sorted(metrics["by_query_type"].items()):
            status = "PASS" if data['recall_at_5'] >= 0.95 else "FAIL"
            print(f"  {qtype:15s}: R@5={data['recall_at_5']:.1%} [{status}] | MRR={data['mrr']:.3f} (n={data['total']})")

        print(f"\n{'='*40}")
        print("FAILURE ANALYSIS")
        print(f"{'='*40}")
        n_failures = len(metrics["failures"])
        print(f"  Failures at @5: {n_failures} ({n_failures/n:.1%})")

        if metrics["failures"][:5]:
            print(f"\n  Sample failures:")
            for f in metrics["failures"][:5]:
                print(f"    - {f['query_type']}: rank={f['rank']} | {f['query'][:50]}...")

        print(f"\n{'='*40}")
        print("VERDICT")
        print(f"{'='*40}")
        if o["recall_at_5"] >= 0.95:
            print("  >>> FORTUNE 100 QUALITY: ACHIEVED <<<")
            print(f"  >>> Recall@5 = {o['recall_at_5']:.2%} (>= 95% target)")
        else:
            print("  >>> FORTUNE 100 QUALITY: NOT ACHIEVED <<<")
            print(f"  >>> Recall@5 = {o['recall_at_5']:.2%} (< 95% target)")
            print(f"  >>> Gap: {0.95 - o['recall_at_5']:.2%}")

        print("=" * 80)

        # Save results
        output_path = Path(__file__).parent / "optimization_results" / f"self_retrieval_gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

        return metrics


def main():
    """Run Fortune 100 self-retrieval test."""
    tester = Fortune100SelfTest()
    result = tester.run_test(sample_size=500)
    return result


if __name__ == "__main__":
    main()
