"""
Phase Effectiveness Benchmark - Quantitative measurement of each RAG optimization phase.

This benchmark isolates and measures the quantitative impact of each Phase 1-2 change
by running controlled experiments with pre-populated feedback fixtures.

Usage:
    python benchmarks/phase_effectiveness_benchmark.py
    python benchmarks/phase_effectiveness_benchmark.py --output results/phase_effectiveness.json
    python benchmarks/phase_effectiveness_benchmark.py --phases 1a,1c,1d --verbose

Metrics measured:
    - Top-1 Accuracy: % of queries where top result is relevant
    - MRR (Mean Reciprocal Rank): Average of 1/rank for first relevant result
    - nDCG@5: Normalized Discounted Cumulative Gain at k=5
    - Separation Score: Average score gap between relevant/irrelevant results
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ace import Playbook
from ace.playbook import EnrichedBullet, PENALTY_WEIGHTS
from ace.retrieval import SmartBulletIndex
from ace.session_tracking import SessionOutcomeTracker


# =============================================================================
# Test Data - Curated benchmark queries with known relevant bullets
# =============================================================================

@dataclass
class BenchmarkQuery:
    """A benchmark query with ground truth relevance labels."""
    query: str
    query_type: str
    relevant_bullet_patterns: List[str]  # Content patterns that indicate relevance
    irrelevant_bullet_patterns: List[str]  # Content patterns that indicate irrelevance
    difficulty: str = "medium"


# Curated benchmark queries designed to test different aspects
BENCHMARK_QUERIES = [
    # Debugging queries
    BenchmarkQuery(
        query="How to debug timeout errors in production?",
        query_type="debugging",
        relevant_bullet_patterns=["timeout", "connection", "slow response"],
        irrelevant_bullet_patterns=["syntax error", "SQL injection"],
        difficulty="easy"
    ),
    BenchmarkQuery(
        query="Application crashes with out of memory error",
        query_type="debugging",
        relevant_bullet_patterns=["memory", "leak", "OOM", "allocation"],
        irrelevant_bullet_patterns=["security", "authentication"],
        difficulty="medium"
    ),
    BenchmarkQuery(
        query="System is hanging and won't respond to requests",
        query_type="debugging",
        relevant_bullet_patterns=["timeout", "hang", "deadlock", "connection"],
        irrelevant_bullet_patterns=["compile error", "syntax"],
        difficulty="adversarial"  # Uses "hanging" not "timeout"
    ),

    # Security queries
    BenchmarkQuery(
        query="How to investigate a potential security breach?",
        query_type="security",
        relevant_bullet_patterns=["incident", "breach", "forensics", "logs"],
        irrelevant_bullet_patterns=["performance", "optimization"],
        difficulty="easy"
    ),
    BenchmarkQuery(
        query="Prevent SQL injection vulnerabilities",
        query_type="security",
        relevant_bullet_patterns=["SQL injection", "parameterized", "prepared"],
        irrelevant_bullet_patterns=["memory leak", "timeout"],
        difficulty="medium"
    ),
    BenchmarkQuery(
        query="User data might have been compromised",
        query_type="security",
        relevant_bullet_patterns=["incident", "breach", "compromise", "data"],
        irrelevant_bullet_patterns=["refactor", "cleanup"],
        difficulty="adversarial"  # Indirect language
    ),

    # Optimization queries
    BenchmarkQuery(
        query="How to improve application performance?",
        query_type="optimization",
        relevant_bullet_patterns=["profile", "bottleneck", "performance", "optimize"],
        irrelevant_bullet_patterns=["security", "authentication"],
        difficulty="easy"
    ),
    BenchmarkQuery(
        query="Database queries are running slow",
        query_type="optimization",
        relevant_bullet_patterns=["query", "index", "database", "slow"],
        irrelevant_bullet_patterns=["memory leak", "syntax error"],
        difficulty="medium"
    ),
    BenchmarkQuery(
        query="Need to make the code run faster",
        query_type="optimization",
        relevant_bullet_patterns=["profile", "bottleneck", "optimize", "performance"],
        irrelevant_bullet_patterns=["documentation", "testing"],
        difficulty="adversarial"  # Very generic
    ),

    # Testing queries
    BenchmarkQuery(
        query="How to write effective unit tests?",
        query_type="testing",
        relevant_bullet_patterns=["test", "unit", "coverage", "mock"],
        irrelevant_bullet_patterns=["deployment", "security"],
        difficulty="easy"
    ),
]


def create_comprehensive_playbook() -> Tuple[Playbook, Dict[str, List[str]]]:
    """Create a playbook with diverse bullets for comprehensive testing.

    Returns:
        Tuple of (playbook, relevance_map) where relevance_map maps
        query patterns to relevant bullet IDs.
    """
    playbook = Playbook()
    relevance_map: Dict[str, List[str]] = {}

    # --- Debugging bullets ---
    b1 = playbook.add_enriched_bullet(
        section="debugging",
        content="When debugging timeout errors, check connection pool exhaustion and retry configurations",
        task_types=["debugging", "troubleshooting"],
        domains=["backend", "networking"],
        trigger_patterns=["timeout", "connection", "slow", "hang"],
        complexity_level="medium"
    )
    relevance_map["timeout"] = relevance_map.get("timeout", []) + [b1.id]
    relevance_map["connection"] = relevance_map.get("connection", []) + [b1.id]
    relevance_map["hang"] = relevance_map.get("hang", []) + [b1.id]

    b2 = playbook.add_enriched_bullet(
        section="debugging",
        content="For memory leaks, use heap profilers to identify allocation patterns and retention",
        task_types=["debugging", "performance"],
        domains=["backend", "memory"],
        trigger_patterns=["memory", "leak", "OOM", "allocation", "heap"],
        complexity_level="hard"
    )
    relevance_map["memory"] = relevance_map.get("memory", []) + [b2.id]
    relevance_map["leak"] = relevance_map.get("leak", []) + [b2.id]
    relevance_map["OOM"] = relevance_map.get("OOM", []) + [b2.id]

    b3 = playbook.add_enriched_bullet(
        section="debugging",
        content="Check for deadlocks when system hangs by examining thread states",
        task_types=["debugging"],
        domains=["concurrency"],
        trigger_patterns=["hang", "deadlock", "frozen", "stuck"],
        complexity_level="hard"
    )
    relevance_map["hang"] = relevance_map.get("hang", []) + [b3.id]
    relevance_map["deadlock"] = relevance_map.get("deadlock", []) + [b3.id]

    # --- Security bullets ---
    b4 = playbook.add_enriched_bullet(
        section="security",
        content="For security incidents, preserve logs, isolate systems, and begin forensic analysis",
        task_types=["security", "incident_response"],
        domains=["security", "operations"],
        trigger_patterns=["breach", "incident", "compromise", "attack", "forensics"],
        complexity_level="hard"
    )
    relevance_map["breach"] = relevance_map.get("breach", []) + [b4.id]
    relevance_map["incident"] = relevance_map.get("incident", []) + [b4.id]
    relevance_map["forensics"] = relevance_map.get("forensics", []) + [b4.id]
    relevance_map["compromise"] = relevance_map.get("compromise", []) + [b4.id]

    b5 = playbook.add_enriched_bullet(
        section="security",
        content="Prevent SQL injection by using parameterized queries and prepared statements",
        task_types=["security", "implementation"],
        domains=["database", "backend"],
        trigger_patterns=["SQL injection", "parameterized", "prepared statement"],
        complexity_level="simple"
    )
    relevance_map["SQL injection"] = relevance_map.get("SQL injection", []) + [b5.id]
    relevance_map["parameterized"] = relevance_map.get("parameterized", []) + [b5.id]

    b6 = playbook.add_enriched_bullet(
        section="security",
        content="Implement proper authentication with secure token storage and rotation",
        task_types=["security", "implementation"],
        domains=["authentication"],
        trigger_patterns=["auth", "token", "authentication", "login"],
        complexity_level="medium"
    )

    # --- Optimization bullets ---
    b7 = playbook.add_enriched_bullet(
        section="optimization",
        content="Always profile before optimizing - identify actual bottlenecks first",
        task_types=["optimization", "performance"],
        domains=["general"],
        trigger_patterns=["slow", "performance", "optimize", "profile", "bottleneck"],
        complexity_level="simple"
    )
    relevance_map["profile"] = relevance_map.get("profile", []) + [b7.id]
    relevance_map["bottleneck"] = relevance_map.get("bottleneck", []) + [b7.id]
    relevance_map["optimize"] = relevance_map.get("optimize", []) + [b7.id]
    relevance_map["performance"] = relevance_map.get("performance", []) + [b7.id]

    b8 = playbook.add_enriched_bullet(
        section="optimization",
        content="Add database indexes for slow queries after analyzing query plans",
        task_types=["optimization", "database"],
        domains=["database"],
        trigger_patterns=["query", "slow", "index", "database"],
        complexity_level="medium"
    )
    relevance_map["query"] = relevance_map.get("query", []) + [b8.id]
    relevance_map["index"] = relevance_map.get("index", []) + [b8.id]
    relevance_map["database"] = relevance_map.get("database", []) + [b8.id]

    # --- Testing bullets ---
    b9 = playbook.add_enriched_bullet(
        section="testing",
        content="Write unit tests with clear arrange-act-assert structure and good coverage",
        task_types=["testing", "quality"],
        domains=["testing"],
        trigger_patterns=["test", "unit", "coverage", "TDD"],
        complexity_level="simple"
    )
    relevance_map["test"] = relevance_map.get("test", []) + [b9.id]
    relevance_map["unit"] = relevance_map.get("unit", []) + [b9.id]
    relevance_map["coverage"] = relevance_map.get("coverage", []) + [b9.id]

    # --- Irrelevant noise bullets (to test precision) ---
    b10 = playbook.add_enriched_bullet(
        section="documentation",
        content="Write clear documentation with examples and usage patterns",
        task_types=["documentation"],
        domains=["general"],
        trigger_patterns=["document", "readme", "guide"],
        complexity_level="simple"
    )

    b11 = playbook.add_enriched_bullet(
        section="refactoring",
        content="Refactor code incrementally with tests to prevent regressions",
        task_types=["refactoring"],
        domains=["general"],
        trigger_patterns=["refactor", "cleanup", "technical debt"],
        complexity_level="medium"
    )

    b12 = playbook.add_enriched_bullet(
        section="deployment",
        content="Use blue-green deployments for zero-downtime releases",
        task_types=["deployment"],
        domains=["operations"],
        trigger_patterns=["deploy", "release", "rollback"],
        complexity_level="medium"
    )

    return playbook, relevance_map


# =============================================================================
# Phase Configuration - Enable/disable features for isolated testing
# =============================================================================

@dataclass
class PhaseConfiguration:
    """Configuration for testing specific phases in isolation."""
    name: str
    description: str

    # Phase 1A: Metadata Enhancement
    enable_query_type_boost: bool = True

    # Phase 1B: Filter Fix (Trigger Override)
    enable_trigger_override: bool = True
    trigger_override_threshold: float = 0.3

    # Phase 1C: Asymmetric Penalties (applied during tagging, not retrieval)
    # This is always active since it's in the Bullet.tag() method

    # Phase 1D: Dynamic Weight Shifting
    enable_dynamic_weights: bool = True

    # Phase 2: Session Tracking
    enable_session_tracking: bool = True
    session_type: Optional[str] = None

    # Feedback signal levels for testing maturity
    helpful_signals: int = 0
    harmful_signals: int = 0
    neutral_signals: int = 0


# Predefined phase configurations for isolated testing
PHASE_CONFIGS = {
    "baseline": PhaseConfiguration(
        name="baseline",
        description="Baseline: No optimizations, cold start",
        enable_query_type_boost=False,
        enable_trigger_override=False,
        enable_dynamic_weights=False,
        enable_session_tracking=False,
        helpful_signals=0,
        harmful_signals=0
    ),
    "1a_metadata": PhaseConfiguration(
        name="1a_metadata",
        description="Phase 1A: Query type boost (+0.25)",
        enable_query_type_boost=True,
        enable_trigger_override=False,
        enable_dynamic_weights=False,
        enable_session_tracking=False,
        helpful_signals=0,
        harmful_signals=0
    ),
    "1b_filter_fix": PhaseConfiguration(
        name="1b_filter_fix",
        description="Phase 1B: Trigger override for low effectiveness",
        enable_query_type_boost=False,
        enable_trigger_override=True,
        enable_dynamic_weights=False,
        enable_session_tracking=False,
        helpful_signals=2,
        harmful_signals=8  # Low effectiveness to test filter bypass
    ),
    "1c_asymmetric": PhaseConfiguration(
        name="1c_asymmetric",
        description="Phase 1C: Asymmetric penalties (2x harmful)",
        enable_query_type_boost=False,
        enable_trigger_override=False,
        enable_dynamic_weights=True,  # Need weights to see penalty effect
        enable_session_tracking=False,
        helpful_signals=5,
        harmful_signals=5  # Equal counts, but harmful has 2x weight
    ),
    "1d_dynamic_weights": PhaseConfiguration(
        name="1d_dynamic_weights",
        description="Phase 1D: Dynamic weight shifting by maturity",
        enable_query_type_boost=False,
        enable_trigger_override=False,
        enable_dynamic_weights=True,
        enable_session_tracking=False,
        helpful_signals=10,  # Mature bullets
        harmful_signals=2
    ),
    "2_session_mismatch": PhaseConfiguration(
        name="2_session_mismatch",
        description="Phase 2: Session tracking with MISMATCHED session (bug scenario)",
        enable_query_type_boost=False,
        enable_trigger_override=False,
        enable_dynamic_weights=True,
        enable_session_tracking=True,
        session_type="debugging",  # Fixed session - causes mismatch for non-debugging queries
        helpful_signals=5,
        harmful_signals=5
    ),
    "2_session_matched": PhaseConfiguration(
        name="2_session_matched",
        description="Phase 2: Session tracking with MATCHED session (correct usage)",
        enable_query_type_boost=False,
        enable_trigger_override=False,
        enable_dynamic_weights=True,
        enable_session_tracking=True,
        session_type=None,  # Will be set per-query to match query_type
        helpful_signals=5,
        harmful_signals=5
    ),
    "all_optimizations": PhaseConfiguration(
        name="all_optimizations",
        description="All optimizations enabled",
        enable_query_type_boost=True,
        enable_trigger_override=True,
        enable_dynamic_weights=True,
        enable_session_tracking=True,
        session_type="debugging",
        helpful_signals=10,
        harmful_signals=2
    ),
    "cold_start": PhaseConfiguration(
        name="cold_start",
        description="Cold start: All features enabled, no feedback",
        enable_query_type_boost=True,
        enable_trigger_override=True,
        enable_dynamic_weights=True,
        enable_session_tracking=False,
        helpful_signals=0,
        harmful_signals=0
    ),
    "mature": PhaseConfiguration(
        name="mature",
        description="Mature: All features, high feedback",
        enable_query_type_boost=True,
        enable_trigger_override=True,
        enable_dynamic_weights=True,
        enable_session_tracking=True,
        session_type="debugging",
        helpful_signals=20,
        harmful_signals=5
    ),
}


# =============================================================================
# Metrics Calculation
# =============================================================================

@dataclass
class QueryResult:
    """Result for a single benchmark query."""
    query: str
    query_type: str
    difficulty: str
    relevant_ids: List[str]
    retrieved_ids: List[str]
    scores: List[float]
    top1_hit: bool
    first_relevant_rank: Optional[int]
    reciprocal_rank: float
    ndcg_at_5: float
    separation_score: float  # Gap between relevant and irrelevant scores


@dataclass
class PhaseMetrics:
    """Aggregate metrics for a phase configuration."""
    phase_name: str
    phase_description: str
    num_queries: int
    top1_accuracy: float
    mrr: float
    ndcg_at_5: float
    avg_separation: float
    per_difficulty: Dict[str, Dict[str, float]]
    query_results: List[QueryResult]


def calculate_ndcg(retrieved_ids: List[str], relevant_ids: set, k: int = 5) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k."""
    retrieved = retrieved_ids[:k]

    # DCG
    dcg = 0.0
    for i, bid in enumerate(retrieved, start=1):
        if bid in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)

    # Ideal DCG
    num_relevant = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, num_relevant + 1))

    return dcg / idcg if idcg > 0 else 0.0


def calculate_separation(scores: List[float], relevant_ids: set, retrieved_ids: List[str]) -> float:
    """Calculate score separation between relevant and irrelevant results."""
    relevant_scores = []
    irrelevant_scores = []

    for bid, score in zip(retrieved_ids, scores):
        if bid in relevant_ids:
            relevant_scores.append(score)
        else:
            irrelevant_scores.append(score)

    if not relevant_scores or not irrelevant_scores:
        return 0.0

    avg_relevant = sum(relevant_scores) / len(relevant_scores)
    avg_irrelevant = sum(irrelevant_scores) / len(irrelevant_scores)

    return avg_relevant - avg_irrelevant


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_phase_benchmark(
    config: PhaseConfiguration,
    playbook: Playbook,
    relevance_map: Dict[str, List[str]],
    queries: List[BenchmarkQuery],
    verbose: bool = False
) -> PhaseMetrics:
    """Run benchmark for a specific phase configuration.

    Args:
        config: Phase configuration specifying which features to enable
        playbook: Pre-populated playbook (will be modified with feedback)
        relevance_map: Mapping of patterns to relevant bullet IDs
        queries: List of benchmark queries to evaluate
        verbose: Print detailed results

    Returns:
        PhaseMetrics with aggregate and per-query results
    """
    # Apply feedback signals based on config
    for bullet in playbook.bullets():
        # Reset counters first
        bullet.helpful = 0
        bullet.harmful = 0
        bullet.neutral = 0

        # Apply configured signals
        for _ in range(config.helpful_signals):
            bullet.tag("helpful", increment=1)
        for _ in range(config.harmful_signals):
            bullet.tag("harmful", increment=1)
        for _ in range(config.neutral_signals):
            bullet.tag("neutral", increment=1)

    # Create session tracker if enabled
    session_tracker = None
    use_per_query_session = config.enable_session_tracking and config.session_type is None

    if config.enable_session_tracking:
        session_tracker = SessionOutcomeTracker()

        if config.session_type:
            # Fixed session type - populate data for that session only
            for bullet in playbook.bullets():
                if isinstance(bullet, EnrichedBullet):
                    if config.session_type in bullet.task_types:
                        # High effectiveness in matching sessions
                        for _ in range(8):
                            session_tracker.track(config.session_type, bullet.id, "worked")
                        for _ in range(2):
                            session_tracker.track(config.session_type, bullet.id, "failed")
                    else:
                        # Low effectiveness in non-matching sessions
                        for _ in range(2):
                            session_tracker.track(config.session_type, bullet.id, "worked")
                        for _ in range(8):
                            session_tracker.track(config.session_type, bullet.id, "failed")
        else:
            # Per-query session type - populate data for ALL session types
            session_types = ["debugging", "security", "optimization", "testing"]
            for bullet in playbook.bullets():
                if isinstance(bullet, EnrichedBullet):
                    for stype in session_types:
                        if stype in bullet.task_types:
                            # High effectiveness when bullet matches session type
                            for _ in range(8):
                                session_tracker.track(stype, bullet.id, "worked")
                            for _ in range(2):
                                session_tracker.track(stype, bullet.id, "failed")
                        else:
                            # Low effectiveness when bullet doesn't match session type
                            for _ in range(2):
                                session_tracker.track(stype, bullet.id, "worked")
                            for _ in range(8):
                                session_tracker.track(stype, bullet.id, "failed")

    # Create index
    index = SmartBulletIndex(playbook=playbook, session_tracker=session_tracker)

    # Run queries
    query_results: List[QueryResult] = []
    difficulty_results: Dict[str, List[QueryResult]] = {"easy": [], "medium": [], "hard": [], "adversarial": []}

    for bq in queries:
        # Determine relevant bullet IDs from patterns
        relevant_ids = set()
        for pattern in bq.relevant_bullet_patterns:
            if pattern in relevance_map:
                relevant_ids.update(relevance_map[pattern])

        # Determine session_type for this query
        if config.enable_session_tracking:
            if use_per_query_session:
                # Use query_type as session_type (correct usage)
                effective_session_type = bq.query_type
            else:
                # Use fixed session_type from config
                effective_session_type = config.session_type
        else:
            effective_session_type = None

        # Retrieve with phase-specific parameters
        results = index.retrieve(
            query=bq.query,
            task_type=bq.query_type if config.enable_query_type_boost else None,
            query_type=bq.query_type if config.enable_query_type_boost else None,
            trigger_override_threshold=config.trigger_override_threshold if config.enable_trigger_override else 1.0,
            session_type=effective_session_type,
            limit=10
        )

        retrieved_ids = [r.bullet.id for r in results]
        scores = [r.score for r in results]

        # Calculate metrics
        top1_hit = len(retrieved_ids) > 0 and retrieved_ids[0] in relevant_ids

        first_relevant_rank = None
        for i, bid in enumerate(retrieved_ids, start=1):
            if bid in relevant_ids:
                first_relevant_rank = i
                break

        reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        ndcg = calculate_ndcg(retrieved_ids, relevant_ids, k=5)
        separation = calculate_separation(scores, relevant_ids, retrieved_ids)

        qr = QueryResult(
            query=bq.query,
            query_type=bq.query_type,
            difficulty=bq.difficulty,
            relevant_ids=list(relevant_ids),
            retrieved_ids=retrieved_ids,
            scores=scores,
            top1_hit=top1_hit,
            first_relevant_rank=first_relevant_rank,
            reciprocal_rank=reciprocal_rank,
            ndcg_at_5=ndcg,
            separation_score=separation
        )
        query_results.append(qr)
        difficulty_results[bq.difficulty].append(qr)

        if verbose:
            hit_marker = "[HIT]" if top1_hit else "[MISS]"
            print(f"  {hit_marker} {bq.query[:50]}... RR={reciprocal_rank:.3f} nDCG={ndcg:.3f}")

    # Aggregate metrics
    num_queries = len(query_results)
    top1_accuracy = sum(1 for qr in query_results if qr.top1_hit) / num_queries if num_queries > 0 else 0.0
    mrr = sum(qr.reciprocal_rank for qr in query_results) / num_queries if num_queries > 0 else 0.0
    ndcg_at_5 = sum(qr.ndcg_at_5 for qr in query_results) / num_queries if num_queries > 0 else 0.0
    avg_separation = sum(qr.separation_score for qr in query_results) / num_queries if num_queries > 0 else 0.0

    # Per-difficulty metrics
    per_difficulty = {}
    for diff, results in difficulty_results.items():
        if results:
            per_difficulty[diff] = {
                "count": len(results),
                "top1_accuracy": sum(1 for qr in results if qr.top1_hit) / len(results),
                "mrr": sum(qr.reciprocal_rank for qr in results) / len(results),
                "ndcg_at_5": sum(qr.ndcg_at_5 for qr in results) / len(results)
            }

    return PhaseMetrics(
        phase_name=config.name,
        phase_description=config.description,
        num_queries=num_queries,
        top1_accuracy=top1_accuracy,
        mrr=mrr,
        ndcg_at_5=ndcg_at_5,
        avg_separation=avg_separation,
        per_difficulty=per_difficulty,
        query_results=query_results
    )


def run_all_benchmarks(
    phases: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict[str, PhaseMetrics]:
    """Run benchmarks for all specified phases.

    Args:
        phases: List of phase names to test (None = all phases)
        verbose: Print detailed results

    Returns:
        Dict mapping phase name to PhaseMetrics
    """
    # Create test playbook
    playbook, relevance_map = create_comprehensive_playbook()

    # Select phases to test
    phase_names = phases or list(PHASE_CONFIGS.keys())

    results: Dict[str, PhaseMetrics] = {}

    for phase_name in phase_names:
        if phase_name not in PHASE_CONFIGS:
            print(f"Warning: Unknown phase '{phase_name}', skipping")
            continue

        config = PHASE_CONFIGS[phase_name]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Phase: {config.name}")
            print(f"Description: {config.description}")
            print(f"{'='*60}")

        # Need fresh playbook for each phase
        playbook, relevance_map = create_comprehensive_playbook()

        metrics = run_phase_benchmark(
            config=config,
            playbook=playbook,
            relevance_map=relevance_map,
            queries=BENCHMARK_QUERIES,
            verbose=verbose
        )

        results[phase_name] = metrics

    return results


def generate_effectiveness_report(results: Dict[str, PhaseMetrics]) -> str:
    """Generate a formatted effectiveness report.

    Args:
        results: Dict mapping phase name to PhaseMetrics

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("ACE RAG OPTIMIZATION - PHASE EFFECTIVENESS REPORT")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("=" * 80)

    # Summary table
    lines.append("\n## Summary Metrics")
    lines.append("-" * 80)
    lines.append(f"{'Phase':<20} {'Top-1 Acc':>10} {'MRR':>10} {'nDCG@5':>10} {'Separation':>12}")
    lines.append("-" * 80)

    baseline_metrics = results.get("baseline")

    for phase_name, metrics in results.items():
        # Calculate delta from baseline if available
        if baseline_metrics and phase_name != "baseline":
            top1_delta = metrics.top1_accuracy - baseline_metrics.top1_accuracy
            mrr_delta = metrics.mrr - baseline_metrics.mrr
            ndcg_delta = metrics.ndcg_at_5 - baseline_metrics.ndcg_at_5
            sep_delta = metrics.avg_separation - baseline_metrics.avg_separation

            top1_str = f"{metrics.top1_accuracy:.1%} ({top1_delta:+.1%})"
            mrr_str = f"{metrics.mrr:.3f} ({mrr_delta:+.3f})"
            ndcg_str = f"{metrics.ndcg_at_5:.3f} ({ndcg_delta:+.3f})"
            sep_str = f"{metrics.avg_separation:.3f} ({sep_delta:+.3f})"
        else:
            top1_str = f"{metrics.top1_accuracy:.1%}"
            mrr_str = f"{metrics.mrr:.3f}"
            ndcg_str = f"{metrics.ndcg_at_5:.3f}"
            sep_str = f"{metrics.avg_separation:.3f}"

        lines.append(f"{phase_name:<20} {top1_str:>10} {mrr_str:>10} {ndcg_str:>10} {sep_str:>12}")

    lines.append("-" * 80)

    # Detailed breakdown
    lines.append("\n## Detailed Analysis")

    for phase_name, metrics in results.items():
        lines.append(f"\n### {phase_name}: {metrics.phase_description}")
        lines.append(f"- Top-1 Accuracy: {metrics.top1_accuracy:.1%}")
        lines.append(f"- Mean Reciprocal Rank: {metrics.mrr:.4f}")
        lines.append(f"- nDCG@5: {metrics.ndcg_at_5:.4f}")
        lines.append(f"- Avg Score Separation: {metrics.avg_separation:.4f}")

        if metrics.per_difficulty:
            lines.append("\nBy Difficulty:")
            for diff, diff_metrics in metrics.per_difficulty.items():
                lines.append(f"  {diff}: Top-1={diff_metrics['top1_accuracy']:.1%}, "
                           f"MRR={diff_metrics['mrr']:.3f}, nDCG={diff_metrics['ndcg_at_5']:.3f}")

    # Recommendations
    lines.append("\n## Effectiveness Summary")
    lines.append("-" * 80)

    if baseline_metrics:
        all_opt = results.get("all_optimizations")
        if all_opt:
            improvement = all_opt.top1_accuracy - baseline_metrics.top1_accuracy
            lines.append(f"Total improvement over baseline: {improvement:+.1%} Top-1 Accuracy")

            # Calculate per-phase contribution
            lines.append("\nPer-Phase Contribution:")
            for phase_name, metrics in results.items():
                if phase_name not in ["baseline", "all_optimizations", "cold_start", "mature"]:
                    delta = metrics.top1_accuracy - baseline_metrics.top1_accuracy
                    lines.append(f"  {phase_name}: {delta:+.1%}")

    return "\n".join(lines)


def main():
    """Main entry point for the effectiveness benchmark."""
    parser = argparse.ArgumentParser(description="Run ACE RAG phase effectiveness benchmarks")
    parser.add_argument(
        "--phases",
        type=str,
        help="Comma-separated list of phases to test (default: all)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save JSON results"
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Path to save text report"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed results"
    )
    parser.add_argument(
        "--list-phases",
        action="store_true",
        help="List available phase configurations"
    )

    args = parser.parse_args()

    if args.list_phases:
        print("Available phase configurations:")
        for name, config in PHASE_CONFIGS.items():
            print(f"  {name}: {config.description}")
        return

    # Parse phases
    phases = None
    if args.phases:
        phases = [p.strip() for p in args.phases.split(",")]

    # Run benchmarks
    print("Running phase effectiveness benchmarks...")
    results = run_all_benchmarks(phases=phases, verbose=args.verbose)

    # Generate report
    report = generate_effectiveness_report(results)
    print("\n" + report)

    # Save results if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        json_results = {
            name: {
                "phase_name": m.phase_name,
                "phase_description": m.phase_description,
                "num_queries": m.num_queries,
                "top1_accuracy": m.top1_accuracy,
                "mrr": m.mrr,
                "ndcg_at_5": m.ndcg_at_5,
                "avg_separation": m.avg_separation,
                "per_difficulty": m.per_difficulty
            }
            for name, m in results.items()
        }
        with open(args.output, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nJSON results saved to {args.output}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"Text report saved to {args.report}")


if __name__ == "__main__":
    main()
