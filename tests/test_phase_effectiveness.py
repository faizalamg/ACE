"""
Tests for measuring effectiveness of each RAG optimization phase in isolation.

This test module validates that each improvement phase can be measured independently
by using pre-populated feedback fixtures that simulate different maturity states.
"""

import json
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ace import Playbook
from ace.playbook import EnrichedBullet, PENALTY_WEIGHTS
from ace.retrieval import SmartBulletIndex
from ace.session_tracking import SessionOutcomeTracker


# =============================================================================
# Test Fixtures - Pre-populated feedback states for isolated measurement
# =============================================================================

@dataclass
class PhaseConfig:
    """Configuration for testing a specific phase in isolation."""
    name: str
    description: str
    # Feedback signals to populate
    helpful_signals: int = 0
    harmful_signals: int = 0
    # Session data
    session_data: Optional[Dict[str, Dict[str, str]]] = None
    # Feature flags
    use_query_type_boost: bool = False
    use_trigger_override: bool = False
    use_dynamic_weights: bool = True
    use_session_tracking: bool = False
    # Expected behavior changes
    min_effectiveness_filter: Optional[float] = None
    trigger_override_threshold: float = 0.3


def create_test_playbook_with_signals(
    helpful: int = 0,
    harmful: int = 0,
    neutral: int = 0
) -> Playbook:
    """Create a playbook with bullets pre-populated with feedback signals.

    Args:
        helpful: Number of helpful signals per bullet
        harmful: Number of harmful signals per bullet
        neutral: Number of neutral signals per bullet

    Returns:
        Playbook with EnrichedBullets containing specified feedback
    """
    playbook = Playbook()

    # Debugging bullets
    playbook.add_enriched_bullet(
        section="debugging",
        content="When debugging timeout errors, start by checking connection timeouts and retry configurations",
        task_types=["debugging", "troubleshooting"],
        domains=["backend", "networking"],
        trigger_patterns=["timeout", "connection", "hang", "slow"],
        anti_patterns=["syntax error", "compile error"],
        complexity_level="medium"
    )

    playbook.add_enriched_bullet(
        section="debugging",
        content="For memory leaks, use profiling tools to identify allocation patterns",
        task_types=["debugging", "performance"],
        domains=["backend", "memory"],
        trigger_patterns=["memory", "leak", "OOM", "out of memory"],
        complexity_level="hard"
    )

    # Security bullets
    playbook.add_enriched_bullet(
        section="security",
        content="For security incident response, preserve logs and isolate affected systems first",
        task_types=["security", "incident_response"],
        domains=["security", "operations"],
        trigger_patterns=["breach", "incident", "compromise", "attack"],
        complexity_level="hard"
    )

    playbook.add_enriched_bullet(
        section="security",
        content="Prevent SQL injection by using parameterized queries exclusively",
        task_types=["security", "implementation"],
        domains=["database", "backend"],
        trigger_patterns=["sql injection", "parameterized", "prepared statement"],
        complexity_level="simple"
    )

    # Optimization bullets
    playbook.add_enriched_bullet(
        section="optimization",
        content="Before optimizing, always profile to identify actual bottlenecks",
        task_types=["optimization", "performance"],
        domains=["general"],
        trigger_patterns=["slow", "performance", "optimize", "faster"],
        complexity_level="simple"
    )

    # Apply feedback signals to all bullets
    for bullet in playbook.bullets():
        for _ in range(helpful):
            bullet.tag("helpful", increment=1)  # Use explicit increment=1 for test consistency
        for _ in range(harmful):
            bullet.tag("harmful", increment=1)  # Use explicit increment=1 for test consistency
        for _ in range(neutral):
            bullet.tag("neutral", increment=1)

    return playbook


def create_session_tracker_with_data(
    session_data: Dict[str, Dict[str, str]]
) -> SessionOutcomeTracker:
    """Create a session tracker pre-populated with outcome data.

    Args:
        session_data: Dict mapping session_type:bullet_id to outcome counts
            Format: {"session_type:bullet_id": {"worked": N, "failed": M}}

    Returns:
        Pre-populated SessionOutcomeTracker
    """
    tracker = SessionOutcomeTracker(ttl_hours=24)

    for key, outcomes in session_data.items():
        parts = key.split(":", 1)
        session_type = parts[0]
        bullet_id = parts[1] if len(parts) > 1 else "unknown"

        worked = outcomes.get("worked", 0)
        failed = outcomes.get("failed", 0)

        for _ in range(worked):
            tracker.track(session_type, bullet_id, "worked")
        for _ in range(failed):
            tracker.track(session_type, bullet_id, "failed")

    return tracker


# =============================================================================
# Phase 1A: Metadata Enhancement Tests
# =============================================================================

class TestPhase1AMetadataEffectiveness(unittest.TestCase):
    """Test effectiveness of Phase 1A: Metadata Enhancement (+0.25 query_type boost)."""

    def test_query_type_boost_improves_ranking(self):
        """Query type matching should boost relevant bullets."""
        playbook = create_test_playbook_with_signals(helpful=0, harmful=0)
        index = SmartBulletIndex(playbook=playbook)

        # Query for debugging - should prioritize debugging bullets
        results_with_type = index.retrieve(
            query="How to debug timeout errors?",
            query_type="debugging",
            limit=10
        )

        # Query without type - no boost
        results_without_type = index.retrieve(
            query="How to debug timeout errors?",
            limit=10
        )

        # With query_type, debugging bullets should score higher
        top_with_type = results_with_type[0] if results_with_type else None
        top_without_type = results_without_type[0] if results_without_type else None

        self.assertIsNotNone(top_with_type)
        self.assertIn("debugging", top_with_type.bullet.task_types)

        # The boost should increase score by 0.25
        if top_with_type and top_without_type:
            # Score should be higher with query_type match
            self.assertGreaterEqual(top_with_type.score, top_without_type.score)

    def test_query_type_boost_value(self):
        """Verify +0.25 boost is applied for query_type match."""
        playbook = create_test_playbook_with_signals(helpful=0, harmful=0)
        index = SmartBulletIndex(playbook=playbook)

        # Get security bullet score without boost
        results_no_boost = index.retrieve(
            query="security",
            task_type="security",  # Filter only
            limit=10
        )

        # Get security bullet score with query_type boost
        results_with_boost = index.retrieve(
            query="security",
            task_type="security",
            query_type="security",  # This adds +0.25 boost
            limit=10
        )

        # Both should have results
        self.assertTrue(len(results_no_boost) > 0)
        self.assertTrue(len(results_with_boost) > 0)

        # With boost should be higher
        no_boost_score = results_no_boost[0].score
        with_boost_score = results_with_boost[0].score

        # The difference should include the 0.25 boost
        self.assertGreater(with_boost_score, no_boost_score)


# =============================================================================
# Phase 1B: Filter Fix (Trigger Override) Tests
# =============================================================================

class TestPhase1BFilterFixEffectiveness(unittest.TestCase):
    """Test effectiveness of Phase 1B: Trigger Override for low-effectiveness bullets."""

    def test_strong_trigger_bypasses_effectiveness_filter(self):
        """Bullets with strong trigger matches should bypass min_effectiveness filter."""
        playbook = create_test_playbook_with_signals(helpful=1, harmful=9)  # Low effectiveness
        index = SmartBulletIndex(playbook=playbook)

        # Query that strongly matches a trigger pattern
        results = index.retrieve(
            query="How to debug timeout connection issues?",  # Matches "timeout", "connection"
            min_effectiveness=0.5,  # Filter should exclude low-effectiveness bullets
            trigger_override_threshold=0.3,  # But strong triggers override
            limit=10
        )

        # Should still get results despite low effectiveness
        # because trigger score > 0.3 overrides the filter
        self.assertTrue(len(results) > 0)

    def test_weak_trigger_respects_effectiveness_filter(self):
        """Bullets with weak triggers should be filtered by min_effectiveness."""
        playbook = create_test_playbook_with_signals(helpful=1, harmful=9)
        index = SmartBulletIndex(playbook=playbook)

        # Query with no strong trigger matches
        results = index.retrieve(
            query="general question about code",  # No trigger patterns match
            min_effectiveness=0.5,
            trigger_override_threshold=0.3,
            limit=10
        )

        # Low-effectiveness bullets without strong triggers should be filtered
        for result in results:
            effectiveness = result.bullet.effectiveness_score
            # Either high effectiveness OR had trigger override
            trigger_matched = any("trigger_match" in r for r in result.match_reasons)
            if not trigger_matched:
                self.assertGreaterEqual(effectiveness, 0.5)


# =============================================================================
# Phase 1C: Asymmetric Penalties Tests
# =============================================================================

class TestPhase1CAsymmetricPenaltiesEffectiveness(unittest.TestCase):
    """Test effectiveness of Phase 1C: Asymmetric penalty weights."""

    def test_harmful_weight_is_double(self):
        """Harmful tags should have 2x weight compared to helpful."""
        self.assertEqual(PENALTY_WEIGHTS["harmful"], 2)
        self.assertEqual(PENALTY_WEIGHTS["helpful"], 1)

    def test_asymmetric_penalty_accelerates_suppression(self):
        """Harmful bullets should drop in effectiveness faster."""
        playbook = Playbook()

        playbook.add_enriched_bullet(
            section="test",
            content="Test bullet for penalty testing",
            task_types=["testing"]
        )

        bullet = list(playbook.bullets())[0]

        # Apply equal count of helpful and harmful
        for _ in range(5):
            bullet.tag("helpful")  # +1 each = 5 total
        for _ in range(5):
            bullet.tag("harmful")  # +2 each = 10 total

        # With asymmetric penalties:
        # helpful = 5, harmful = 10
        # effectiveness = 5 / (5 + 10) = 0.333
        self.assertEqual(bullet.helpful, 5)
        self.assertEqual(bullet.harmful, 10)
        self.assertAlmostEqual(bullet.effectiveness_score, 5/15, places=2)

    def test_symmetric_vs_asymmetric_comparison(self):
        """Compare effectiveness under symmetric vs asymmetric penalties."""
        # Symmetric scenario (old behavior): 5 helpful, 5 harmful
        # Would give: 5 / 10 = 0.5

        # Asymmetric scenario (new behavior): 5 helpful (1x), 5 harmful (2x)
        # Gives: 5 / 15 = 0.333

        playbook = Playbook()
        playbook.add_enriched_bullet(
            section="test",
            content="Asymmetric test",
            task_types=["testing"]
        )

        bullet = list(playbook.bullets())[0]

        # 5 helpful at 1x weight
        for _ in range(5):
            bullet.tag("helpful")

        # 5 harmful at 2x weight
        for _ in range(5):
            bullet.tag("harmful")

        # Under asymmetric: effectiveness < 0.5
        self.assertLess(bullet.effectiveness_score, 0.5)


# =============================================================================
# Phase 1D: Dynamic Weight Shifting Tests
# =============================================================================

class TestPhase1DDynamicWeightsEffectiveness(unittest.TestCase):
    """Test effectiveness of Phase 1D: Dynamic weight shifting based on maturity."""

    def test_cold_start_weights(self):
        """New bullets (0 signals) should use 0.8/0.2 similarity/outcome weights."""
        playbook = create_test_playbook_with_signals(helpful=0, harmful=0)
        index = SmartBulletIndex(playbook=playbook)

        bullet = list(playbook.bullets())[0]
        sim_weight, out_weight = index._get_dynamic_weights(bullet)

        self.assertAlmostEqual(sim_weight, 0.8, places=2)
        self.assertAlmostEqual(out_weight, 0.2, places=2)

    def test_early_stage_weights(self):
        """Early bullets (1-4 signals) should use 0.5/0.5 balanced weights."""
        playbook = create_test_playbook_with_signals(helpful=2, harmful=1)  # 3 signals
        index = SmartBulletIndex(playbook=playbook)

        bullet = list(playbook.bullets())[0]
        sim_weight, out_weight = index._get_dynamic_weights(bullet)

        self.assertAlmostEqual(sim_weight, 0.5, places=2)
        self.assertAlmostEqual(out_weight, 0.5, places=2)

    def test_mature_weights(self):
        """Mature bullets (5+ signals) should use 0.3/0.7 outcome-heavy weights."""
        playbook = create_test_playbook_with_signals(helpful=4, harmful=2)  # 6 signals
        index = SmartBulletIndex(playbook=playbook)

        bullet = list(playbook.bullets())[0]
        sim_weight, out_weight = index._get_dynamic_weights(bullet)

        self.assertAlmostEqual(sim_weight, 0.3, places=2)
        self.assertAlmostEqual(out_weight, 0.7, places=2)

    def test_weight_shift_affects_ranking(self):
        """Weight shift should change ranking for bullets with different effectiveness."""
        # Create playbook with two debugging bullets - one high, one low effectiveness
        playbook = Playbook()

        # High effectiveness bullet (will rank higher when outcomes weighted more)
        bullet1 = playbook.add_enriched_bullet(
            section="debugging",
            content="High effectiveness debugging strategy A",
            task_types=["debugging"],
            trigger_patterns=["debug", "error"]
        )

        # Low effectiveness bullet (same similarity but lower outcomes)
        bullet2 = playbook.add_enriched_bullet(
            section="debugging",
            content="Low effectiveness debugging strategy B",
            task_types=["debugging"],
            trigger_patterns=["debug", "error"]
        )

        # Apply feedback - bullet1 is better
        for _ in range(8):
            bullet1.tag("helpful", increment=1)
        for _ in range(2):
            bullet1.tag("harmful", increment=1)
        # bullet1 effectiveness: 8/10 = 0.8

        for _ in range(2):
            bullet2.tag("helpful", increment=1)
        for _ in range(8):
            bullet2.tag("harmful", increment=1)
        # bullet2 effectiveness: 2/10 = 0.2

        index = SmartBulletIndex(playbook=playbook)

        # With mature bullets, outcomes weighted 0.7 - high effectiveness should win
        results = index.retrieve(
            query="debug error",
            task_type="debugging",
            limit=5
        )

        # High effectiveness bullet should rank first
        self.assertEqual(len(results), 2)
        self.assertGreater(results[0].score, results[1].score)


# =============================================================================
# Phase 2: Session Tracking Tests
# =============================================================================

class TestPhase2SessionTrackingEffectiveness(unittest.TestCase):
    """Test effectiveness of Phase 2: Session-level context tracking."""

    def test_session_aware_retrieval_uses_session_effectiveness(self):
        """Session-aware retrieval should use session-specific effectiveness."""
        playbook = create_test_playbook_with_signals(helpful=5, harmful=5)

        # Create session tracker with divergent effectiveness
        session_data = {}
        for bullet in playbook.bullets():
            if "debugging" in bullet.task_types:
                # High effectiveness in debugging sessions
                session_data[f"debugging:{bullet.id}"] = {"worked": 8, "failed": 2}
            else:
                # Low effectiveness in debugging sessions
                session_data[f"debugging:{bullet.id}"] = {"worked": 2, "failed": 8}

        tracker = create_session_tracker_with_data(session_data)
        index = SmartBulletIndex(playbook=playbook, session_tracker=tracker)

        # Retrieve with session context
        results = index.retrieve(
            query="debug something",
            session_type="debugging",
            limit=10
        )

        # Debugging bullets should rank higher due to session effectiveness
        self.assertTrue(len(results) > 0)
        top_result = results[0]

        # Top result should be a debugging bullet (high session effectiveness)
        self.assertIn("debugging", top_result.bullet.task_types)

    def test_session_effectiveness_fallback_to_global(self):
        """Without session data, should fall back to global effectiveness."""
        playbook = create_test_playbook_with_signals(helpful=8, harmful=2)

        # Empty session tracker
        tracker = SessionOutcomeTracker()
        index = SmartBulletIndex(playbook=playbook, session_tracker=tracker)

        # Retrieve with session type but no session data
        results = index.retrieve(
            query="test query",
            session_type="unknown_session",
            limit=10
        )

        # Should still get results using global effectiveness
        self.assertTrue(len(results) > 0)

    def test_divergent_session_effectiveness(self):
        """Same bullet should have different effectiveness in different sessions."""
        playbook = Playbook()

        bullet = playbook.add_enriched_bullet(
            section="general",
            content="Multi-purpose strategy",
            task_types=["debugging", "security"]
        )

        # Set up divergent session data
        tracker = SessionOutcomeTracker()

        # High effectiveness in debugging
        for _ in range(9):
            tracker.track("debugging", bullet.id, "worked")
        tracker.track("debugging", bullet.id, "failed")

        # Low effectiveness in security
        tracker.track("security", bullet.id, "worked")
        for _ in range(9):
            tracker.track("security", bullet.id, "failed")

        # Check divergent effectiveness
        debug_eff = tracker.get_session_effectiveness("debugging", bullet.id)
        security_eff = tracker.get_session_effectiveness("security", bullet.id)

        self.assertAlmostEqual(debug_eff, 0.9, places=1)
        self.assertAlmostEqual(security_eff, 0.1, places=1)


# =============================================================================
# Comparative Benchmark: Measure All Phases Together
# =============================================================================

class TestComparativeEffectiveness(unittest.TestCase):
    """Compare effectiveness of baseline vs all optimizations."""

    def test_baseline_vs_optimized_ranking(self):
        """Optimized retrieval should produce better rankings than baseline."""
        # Create playbook with varied effectiveness
        playbook = Playbook()

        # Good bullet - high effectiveness
        good = playbook.add_enriched_bullet(
            section="debugging",
            content="Excellent debugging strategy for timeout issues",
            task_types=["debugging"],
            trigger_patterns=["timeout", "slow", "hang"]
        )
        for _ in range(10):
            good.tag("helpful", increment=1)

        # Bad bullet - low effectiveness but similar content
        bad = playbook.add_enriched_bullet(
            section="debugging",
            content="Poor debugging approach for timeout problems",
            task_types=["debugging"],
            trigger_patterns=["timeout", "slow"]
        )
        for _ in range(10):
            bad.tag("harmful", increment=1)

        # Session tracker showing divergent performance
        tracker = SessionOutcomeTracker()
        for _ in range(9):
            tracker.track("debugging", good.id, "worked")
        tracker.track("debugging", good.id, "failed")

        tracker.track("debugging", bad.id, "worked")
        for _ in range(9):
            tracker.track("debugging", bad.id, "failed")

        # Optimized retrieval
        index = SmartBulletIndex(playbook=playbook, session_tracker=tracker)

        results = index.retrieve(
            query="How to debug timeout issues?",
            query_type="debugging",
            session_type="debugging",
            limit=5
        )

        # Good bullet should rank first
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].bullet.id, good.id)
        self.assertEqual(results[1].bullet.id, bad.id)

        # Score gap should be significant
        score_gap = results[0].score - results[1].score
        self.assertGreater(score_gap, 0.1)  # At least 0.1 difference


# =============================================================================
# Effectiveness Measurement Fixtures
# =============================================================================

class TestEffectivenessMeasurement(unittest.TestCase):
    """Test the effectiveness measurement infrastructure."""

    def test_create_playbook_with_signals(self):
        """Verify fixture creates playbook with correct signal counts."""
        playbook = create_test_playbook_with_signals(helpful=5, harmful=3, neutral=2)

        for bullet in playbook.bullets():
            self.assertEqual(bullet.helpful, 5)
            self.assertEqual(bullet.harmful, 3)
            self.assertEqual(bullet.neutral, 2)

    def test_create_session_tracker_with_data(self):
        """Verify fixture creates tracker with correct outcome data."""
        session_data = {
            "debugging:bullet1": {"worked": 8, "failed": 2},
            "security:bullet2": {"worked": 3, "failed": 7}
        }

        tracker = create_session_tracker_with_data(session_data)

        debug_eff = tracker.get_session_effectiveness("debugging", "bullet1")
        security_eff = tracker.get_session_effectiveness("security", "bullet2")

        self.assertAlmostEqual(debug_eff, 0.8, places=1)
        self.assertAlmostEqual(security_eff, 0.3, places=1)


if __name__ == "__main__":
    unittest.main()
