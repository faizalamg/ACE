"""Compare ACE memory retrieval with CrossEncoder off and on.

This script intentionally measures only the memory retrieval path. Full
`ace_retrieve` also performs code retrieval, which has separate embedding and
index startup costs.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ace_mcp_server import get_memory_index


DEFAULT_QUERIES = [
    "code chunking config",
    "workspace onboarding ace marker",
    "retrieval precision optimization tuning guide",
    "cross-encoder reranking precision improvement",
    "local embedding provider configuration",
    "typo correction llm validation",
    "MCP server initialize timeout lazy imports",
    "Qdrant collection dimension mismatch auto reindex",
    "AST chunking token overflow Jina embeddings",
    "OpenCode Go LLM fallback LM Studio",
    "workspace .env source of truth",
    "ACE retrieve invalid API key LiteLLM",
]


@dataclass
class QueryComparison:
    query: str
    fast_elapsed: float
    precise_elapsed: float
    fast_top: str | None
    precise_top: str | None
    fast_count: int
    precise_count: int
    top_k_overlap: float


def _result_key(result: object) -> str:
    bullet = getattr(result, "bullet", result)
    return str(getattr(bullet, "id", None) or getattr(bullet, "content", str(result))[:160])


def _retrieve(index: object, query: str, limit: int, use_cross_encoder: bool) -> tuple[float, list[object]]:
    started = time.perf_counter()
    results = index.retrieve(
        query=query,
        limit=limit,
        auto_detect_preset=True,
        use_cross_encoder=use_cross_encoder,
    )
    return time.perf_counter() - started, list(results)


def compare(query: str, index: object, limit: int) -> QueryComparison:
    fast_elapsed, fast_results = _retrieve(index, query, limit, False)
    precise_elapsed, precise_results = _retrieve(index, query, limit, True)
    fast_keys = [_result_key(result) for result in fast_results]
    precise_keys = [_result_key(result) for result in precise_results]
    overlap = len(set(fast_keys) & set(precise_keys)) / max(len(set(precise_keys)), 1)
    return QueryComparison(
        query=query,
        fast_elapsed=fast_elapsed,
        precise_elapsed=precise_elapsed,
        fast_top=fast_keys[0] if fast_keys else None,
        precise_top=precise_keys[0] if precise_keys else None,
        fast_count=len(fast_results),
        precise_count=len(precise_results),
        top_k_overlap=overlap,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ACE memory retrieval with CrossEncoder off/on.")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    index = get_memory_index()
    comparisons = [compare(query, index, args.limit) for query in DEFAULT_QUERIES]

    same_top = sum(item.fast_top == item.precise_top for item in comparisons)
    average_overlap = sum(item.top_k_overlap for item in comparisons) / len(comparisons)
    fast_total = sum(item.fast_elapsed for item in comparisons)
    precise_total = sum(item.precise_elapsed for item in comparisons)

    print(f"queries={len(comparisons)}")
    print(f"fast_total={fast_total:.3f}s")
    print(f"precision_total={precise_total:.3f}s")
    print(f"same_top={same_top}/{len(comparisons)}")
    print(f"average_top{args.limit}_overlap={average_overlap:.3f}")

    for item in comparisons:
        print(f"- query={item.query}")
        print(f"  fast={item.fast_elapsed:.3f}s count={item.fast_count} top={item.fast_top}")
        print(f"  precision={item.precise_elapsed:.3f}s count={item.precise_count} top={item.precise_top}")
        print(f"  overlap={item.top_k_overlap:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())