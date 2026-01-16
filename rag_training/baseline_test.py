"""
Baseline connectivity and retrieval test for RAG optimization project.

This script verifies:
1. Qdrant connectivity
2. Embedding service availability
3. Basic retrieval functionality
4. Initial performance metrics
"""

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class BaselineConfig:
    """Configuration for baseline testing."""
    qdrant_url: str = "http://localhost:6333"
    embedding_url: str = "http://localhost:1234"
    collection_name: str = "ace_memories_hybrid"
    embedding_model: str = "text-embedding-qwen3-embedding-8b"


@dataclass
class ConnectivityResult:
    """Result of connectivity test."""
    service: str
    status: str  # "ok", "error", "timeout"
    latency_ms: float
    details: Optional[str] = None


@dataclass
class RetrievalResult:
    """Result of a single retrieval test."""
    query: str
    top_results: List[Dict[str, Any]]
    latency_ms: float
    total_results: int


class BaselineTester:
    """Run baseline tests for RAG system."""

    def __init__(self, config: Optional[BaselineConfig] = None):
        self.config = config or BaselineConfig()
        self.client = httpx.Client(timeout=30.0)

    def test_qdrant_connectivity(self) -> ConnectivityResult:
        """Test Qdrant database connectivity."""
        start = time.perf_counter()
        try:
            resp = self.client.get(f"{self.config.qdrant_url}/collections")
            latency = (time.perf_counter() - start) * 1000

            if resp.status_code == 200:
                collections = resp.json().get("result", {}).get("collections", [])
                return ConnectivityResult(
                    service="qdrant",
                    status="ok",
                    latency_ms=latency,
                    details=f"Found {len(collections)} collections"
                )
            else:
                return ConnectivityResult(
                    service="qdrant",
                    status="error",
                    latency_ms=latency,
                    details=f"Status {resp.status_code}: {resp.text[:200]}"
                )
        except httpx.TimeoutException:
            return ConnectivityResult(
                service="qdrant",
                status="timeout",
                latency_ms=30000,
                details="Connection timed out"
            )
        except Exception as e:
            return ConnectivityResult(
                service="qdrant",
                status="error",
                latency_ms=(time.perf_counter() - start) * 1000,
                details=str(e)
            )

    def test_embedding_service(self) -> ConnectivityResult:
        """Test embedding service connectivity."""
        start = time.perf_counter()
        try:
            # Test with a simple embedding request
            resp = self.client.post(
                f"{self.config.embedding_url}/v1/embeddings",
                json={
                    "model": self.config.embedding_model,
                    "input": "test connectivity"
                }
            )
            latency = (time.perf_counter() - start) * 1000

            if resp.status_code == 200:
                data = resp.json()
                embedding = data.get("data", [{}])[0].get("embedding", [])
                return ConnectivityResult(
                    service="embedding",
                    status="ok",
                    latency_ms=latency,
                    details=f"Embedding dimension: {len(embedding)}"
                )
            else:
                return ConnectivityResult(
                    service="embedding",
                    status="error",
                    latency_ms=latency,
                    details=f"Status {resp.status_code}: {resp.text[:200]}"
                )
        except httpx.TimeoutException:
            return ConnectivityResult(
                service="embedding",
                status="timeout",
                latency_ms=30000,
                details="Connection timed out"
            )
        except Exception as e:
            return ConnectivityResult(
                service="embedding",
                status="error",
                latency_ms=(time.perf_counter() - start) * 1000,
                details=str(e)
            )

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory collection."""
        try:
            resp = self.client.get(
                f"{self.config.qdrant_url}/collections/{self.config.collection_name}"
            )
            if resp.status_code == 200:
                result = resp.json().get("result", {})
                return {
                    "points_count": result.get("points_count", 0),
                    "vectors_count": result.get("vectors_count", 0),
                    "indexed_vectors_count": result.get("indexed_vectors_count", 0),
                    "status": result.get("status", "unknown")
                }
        except Exception as e:
            return {"error": str(e)}
        return {}

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text."""
        try:
            resp = self.client.post(
                f"{self.config.embedding_url}/v1/embeddings",
                json={
                    "model": self.config.embedding_model,
                    "input": text
                }
            )
            if resp.status_code == 200:
                return resp.json()["data"][0]["embedding"]
        except Exception:
            pass
        return None

    def search_memories(
        self,
        query: str,
        limit: int = 10
    ) -> RetrievalResult:
        """Search memories with a query."""
        start = time.perf_counter()

        # Get query embedding
        embedding = self.get_embedding(query)
        if not embedding:
            return RetrievalResult(
                query=query,
                top_results=[],
                latency_ms=(time.perf_counter() - start) * 1000,
                total_results=0
            )

        # Execute search
        try:
            # Hybrid search using prefetch + RRF
            search_body = {
                "prefetch": [
                    {
                        "query": embedding,
                        "using": "dense",
                        "limit": limit * 3
                    }
                ],
                "query": {"fusion": "rrf"},
                "limit": limit,
                "with_payload": True
            }

            resp = self.client.post(
                f"{self.config.qdrant_url}/collections/{self.config.collection_name}/points/query",
                json=search_body
            )
            latency = (time.perf_counter() - start) * 1000

            if resp.status_code == 200:
                points = resp.json().get("result", {}).get("points", [])
                return RetrievalResult(
                    query=query,
                    top_results=[
                        {
                            "id": p.get("id"),
                            "score": p.get("score", 0),
                            "lesson": p.get("payload", {}).get("lesson", "")[:100],
                            "category": p.get("payload", {}).get("category", "")
                        }
                        for p in points
                    ],
                    latency_ms=latency,
                    total_results=len(points)
                )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return RetrievalResult(
                query=query,
                top_results=[{"error": str(e)}],
                latency_ms=latency,
                total_results=0
            )

        return RetrievalResult(
            query=query,
            top_results=[],
            latency_ms=(time.perf_counter() - start) * 1000,
            total_results=0
        )

    def run_baseline_tests(self) -> Dict[str, Any]:
        """Run all baseline tests and return results."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "configuration": asdict(self.config),
            "connectivity": {},
            "collection_stats": {},
            "sample_retrievals": []
        }

        # Connectivity tests
        print("Testing Qdrant connectivity...")
        qdrant_result = self.test_qdrant_connectivity()
        results["connectivity"]["qdrant"] = asdict(qdrant_result)
        print(f"  {qdrant_result.status}: {qdrant_result.details}")

        print("Testing embedding service...")
        embedding_result = self.test_embedding_service()
        results["connectivity"]["embedding"] = asdict(embedding_result)
        print(f"  {embedding_result.status}: {embedding_result.details}")

        # Collection stats
        if qdrant_result.status == "ok":
            print("Getting collection stats...")
            stats = self.get_collection_stats()
            results["collection_stats"] = stats
            print(f"  Points: {stats.get('points_count', 'N/A')}")

        # Sample retrievals
        if qdrant_result.status == "ok" and embedding_result.status == "ok":
            sample_queries = [
                "how to handle errors",
                "security best practices",
                "testing strategies",
                "architecture patterns",
                "configuration management",
                "validate input data",
                "debug loop prevention",
                "API design",
                "memory management",
                "code review process"
            ]

            print(f"\nRunning {len(sample_queries)} sample retrieval tests...")
            latencies = []
            for query in sample_queries:
                result = self.search_memories(query)
                latencies.append(result.latency_ms)
                results["sample_retrievals"].append({
                    "query": result.query,
                    "total_results": result.total_results,
                    "latency_ms": result.latency_ms,
                    "top_3": result.top_results[:3]
                })
                print(f"  '{query}': {result.total_results} results, {result.latency_ms:.1f}ms")

            # Summary stats
            results["latency_stats"] = {
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "avg_ms": sum(latencies) / len(latencies),
                "total_queries": len(sample_queries)
            }

        return results


def main():
    """Run baseline tests and save results."""
    tester = BaselineTester()
    results = tester.run_baseline_tests()

    # Save results
    output_path = Path(__file__).parent / "baseline_results" / "baseline_test_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n=== BASELINE TEST SUMMARY ===")
    print(f"Qdrant: {results['connectivity']['qdrant']['status']}")
    print(f"Embedding: {results['connectivity']['embedding']['status']}")
    if results.get("collection_stats"):
        print(f"Total memories: {results['collection_stats'].get('points_count', 'N/A')}")
    if results.get("latency_stats"):
        print(f"Avg latency: {results['latency_stats']['avg_ms']:.1f}ms")


if __name__ == "__main__":
    main()
