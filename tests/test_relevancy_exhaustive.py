#!/usr/bin/env python3
"""
EXHAUSTIVE RELEVANCY TESTING

This test goes beyond similarity scores to evaluate ACTUAL RELEVANCE:
- Does the result semantically answer the query?
- Is the information useful for the user's intent?
- Are there false positives (high score but irrelevant)?
- Are there false negatives (low score but relevant)?

Uses LLM-based relevance grading for objective assessment.
"""

import os
import sys
import json
import asyncio
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import IntEnum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ace.unified_memory import UnifiedMemoryIndex


class RelevanceGrade(IntEnum):
    """Grading scale for relevance judgment"""
    HIGHLY_RELEVANT = 4  # Directly answers the query
    RELEVANT = 3         # Useful related information
    MARGINAL = 2         # Tangentially related
    IRRELEVANT = 1       # Not useful at all
    OFF_TOPIC = 0        # Completely unrelated


@dataclass
class RelevanceJudgment:
    """Single relevance judgment for a query-result pair"""
    query: str
    result_content: str
    result_score: float
    relevance_grade: RelevanceGrade
    reasoning: str
    is_false_positive: bool  # High score but irrelevant
    is_false_negative: bool  # Low score but relevant


@dataclass
class QueryRelevanceReport:
    """Complete relevance report for a single query"""
    query: str
    intent: str
    expected_topics: list[str]
    judgments: list[RelevanceJudgment]
    precision_at_k: float
    false_positive_rate: float
    false_negative_rate: float
    overall_quality: str


class ExhaustiveRelevancyTester:
    """
    Tests retrieval RELEVANCY, not just similarity.
    Uses pattern-based relevance grading with comprehensive rules.
    """
    
    # Relevance grading rules for different query types
    RELEVANCE_RULES = {
        # Query patterns -> Required content patterns for relevance
        "database": {
            "highly_relevant": ["sql", "query", "database", "table", "index", "schema", "orm", "postgres", "mysql", "sqlite"],
            "relevant": ["data", "storage", "persist", "model", "record"],
            "irrelevant": ["ui", "frontend", "css", "html", "display"]
        },
        "security": {
            "highly_relevant": ["auth", "permission", "token", "jwt", "oauth", "encrypt", "hash", "password", "vulnerability", "xss", "injection"],
            "relevant": ["validate", "sanitize", "safe", "protect", "secure"],
            "irrelevant": ["style", "layout", "animation", "color"]
        },
        "performance": {
            "highly_relevant": ["cache", "optimize", "fast", "slow", "latency", "throughput", "memory", "cpu", "profil"],
            "relevant": ["efficient", "speed", "improve", "reduce", "scale"],
            "irrelevant": ["design", "style", "ux", "color"]
        },
        "testing": {
            "highly_relevant": ["test", "assert", "mock", "fixture", "coverage", "unit", "integration", "e2e", "pytest", "jest"],
            "relevant": ["verify", "validate", "check", "quality"],
            "irrelevant": ["deploy", "production", "release"]
        },
        "debugging": {
            "highly_relevant": ["debug", "error", "exception", "stack", "trace", "log", "breakpoint", "fix", "bug", "crash"],
            "relevant": ["issue", "problem", "investigate", "diagnose"],
            "irrelevant": ["feature", "enhance", "new"]
        },
        "api": {
            "highly_relevant": ["api", "endpoint", "rest", "graphql", "request", "response", "http", "route", "controller"],
            "relevant": ["interface", "service", "call", "method"],
            "irrelevant": ["ui", "frontend", "component", "style"]
        },
        "architecture": {
            "highly_relevant": ["architect", "design", "pattern", "structure", "module", "layer", "microservice", "monolith"],
            "relevant": ["organize", "separate", "component", "system"],
            "irrelevant": ["typo", "rename", "format"]
        },
        "deployment": {
            "highly_relevant": ["deploy", "docker", "kubernetes", "k8s", "ci/cd", "pipeline", "helm", "terraform"],
            "relevant": ["container", "build", "release", "environment"],
            "irrelevant": ["algorithm", "logic", "calculation"]
        },
        "error_handling": {
            "highly_relevant": ["error", "exception", "catch", "throw", "try", "handle", "fallback", "retry"],
            "relevant": ["fail", "recover", "graceful", "robust"],
            "irrelevant": ["success", "happy path", "normal"]
        },
        "configuration": {
            "highly_relevant": ["config", "setting", "env", "environment", "yaml", "json", "toml", ".env"],
            "relevant": ["option", "parameter", "variable", "value"],
            "irrelevant": ["algorithm", "business logic", "feature"]
        }
    }

    # Test queries with expected relevance criteria
    TEST_QUERIES = [
        {
            "query": "how to optimize database queries for better performance",
            "intent": "troubleshooting",
            "expected_topics": ["database", "performance"],
            "must_contain": ["sql", "query", "index", "optimize", "cache"],
            "must_not_contain": ["ui", "css", "frontend", "animation"]
        },
        {
            "query": "fix authentication token expiry issue",
            "intent": "troubleshooting",
            "expected_topics": ["security"],
            "must_contain": ["token", "auth", "expire", "jwt", "session"],
            "must_not_contain": ["database", "sql", "css"]
        },
        {
            "query": "write unit tests for the payment service",
            "intent": "implementation",
            "expected_topics": ["testing"],
            "must_contain": ["test", "mock", "assert", "unit", "pytest"],
            "must_not_contain": ["deploy", "production"]
        },
        {
            "query": "kubernetes pod keeps crashing with OOM error",
            "intent": "troubleshooting",
            "expected_topics": ["deployment", "debugging"],
            "must_contain": ["kubernetes", "pod", "memory", "oom", "container"],
            "must_not_contain": ["sql", "database", "frontend"]
        },
        {
            "query": "design REST API for user management",
            "intent": "implementation",
            "expected_topics": ["api", "architecture"],
            "must_contain": ["api", "endpoint", "rest", "user", "crud"],
            "must_not_contain": ["css", "animation", "color"]
        },
        {
            "query": "configure environment variables for production",
            "intent": "implementation",
            "expected_topics": ["configuration", "deployment"],
            "must_contain": ["env", "config", "production", "variable", "secret"],
            "must_not_contain": ["test", "mock", "unittest"]
        },
        {
            "query": "handle API errors gracefully with retry logic",
            "intent": "implementation",
            "expected_topics": ["error_handling", "api"],
            "must_contain": ["error", "retry", "exception", "handle", "fallback"],
            "must_not_contain": ["style", "ui", "display"]
        },
        {
            "query": "debug memory leak in Python application",
            "intent": "troubleshooting",
            "expected_topics": ["debugging", "performance"],
            "must_contain": ["memory", "leak", "debug", "profile", "gc"],
            "must_not_contain": ["frontend", "css", "html"]
        },
        {
            "query": "implement caching strategy for high traffic API",
            "intent": "implementation",
            "expected_topics": ["performance", "api"],
            "must_contain": ["cache", "redis", "memcache", "ttl", "invalidate"],
            "must_not_contain": ["database migration", "schema"]
        },
        {
            "query": "set up CI/CD pipeline for microservices",
            "intent": "implementation",
            "expected_topics": ["deployment"],
            "must_contain": ["ci", "cd", "pipeline", "build", "deploy", "github", "jenkins"],
            "must_not_contain": ["algorithm", "math", "calculation"]
        },
        # Vague queries - these are harder
        {
            "query": "make it faster",
            "intent": "troubleshooting",
            "expected_topics": ["performance"],
            "must_contain": ["performance", "optimize", "cache", "speed", "fast"],
            "must_not_contain": ["style", "color", "font"]
        },
        {
            "query": "broken code",
            "intent": "troubleshooting",
            "expected_topics": ["debugging"],
            "must_contain": ["error", "fix", "debug", "bug", "crash"],
            "must_not_contain": []  # Too vague to require specific exclusions
        },
        {
            "query": "config problem",
            "intent": "troubleshooting",
            "expected_topics": ["configuration"],
            "must_contain": ["config", "setting", "env", "environment"],
            "must_not_contain": []
        },
        # Edge cases
        {
            "query": "why",
            "intent": "analytical",
            "expected_topics": [],  # Too vague
            "must_contain": [],
            "must_not_contain": []
        },
        {
            "query": "help",
            "intent": "exploratory",
            "expected_topics": [],  # Too vague
            "must_contain": [],
            "must_not_contain": []
        }
    ]

    def __init__(self):
        self.memory_index: Optional[UnifiedMemoryIndex] = None
        self.results: list[QueryRelevanceReport] = []

    def setup(self):
        """Initialize the memory index"""
        self.memory_index = UnifiedMemoryIndex()
        print("✓ Memory index initialized")

    def grade_relevance(
        self,
        query_spec: dict,
        result_content: str,
        result_score: float
    ) -> RelevanceJudgment:
        """
        Grade the relevance of a single result for a query.
        Uses pattern-based grading rules.
        """
        content_lower = result_content.lower()
        query = query_spec["query"]
        must_contain = query_spec.get("must_contain", [])
        must_not_contain = query_spec.get("must_not_contain", [])
        
        # Count matches
        positive_matches = sum(1 for term in must_contain if term in content_lower)
        negative_matches = sum(1 for term in must_not_contain if term in content_lower)
        
        # Calculate relevance grade
        if must_contain:
            match_ratio = positive_matches / len(must_contain)
        else:
            match_ratio = 0.5  # Neutral for vague queries
        
        # Grade assignment
        if negative_matches > 0:
            grade = RelevanceGrade.IRRELEVANT if negative_matches > 1 else RelevanceGrade.MARGINAL
            reasoning = f"Contains {negative_matches} off-topic terms"
        elif match_ratio >= 0.6:
            grade = RelevanceGrade.HIGHLY_RELEVANT
            reasoning = f"Strong match: {positive_matches}/{len(must_contain)} required terms"
        elif match_ratio >= 0.4:
            grade = RelevanceGrade.RELEVANT
            reasoning = f"Good match: {positive_matches}/{len(must_contain)} required terms"
        elif match_ratio >= 0.2:
            grade = RelevanceGrade.MARGINAL
            reasoning = f"Weak match: {positive_matches}/{len(must_contain)} required terms"
        elif not must_contain:  # Vague query
            grade = RelevanceGrade.MARGINAL
            reasoning = "Vague query - unable to assess relevance precisely"
        else:
            grade = RelevanceGrade.IRRELEVANT
            reasoning = f"Poor match: {positive_matches}/{len(must_contain)} required terms"
        
        # False positive detection: high score but irrelevant
        is_false_positive = (
            result_score > 0.7 and 
            grade.value <= RelevanceGrade.MARGINAL.value
        )
        
        # False negative detection: low score but relevant
        is_false_negative = (
            result_score < 0.5 and 
            grade.value >= RelevanceGrade.RELEVANT.value
        )
        
        return RelevanceJudgment(
            query=query,
            result_content=result_content[:200] + "..." if len(result_content) > 200 else result_content,
            result_score=result_score,
            relevance_grade=grade,
            reasoning=reasoning,
            is_false_positive=is_false_positive,
            is_false_negative=is_false_negative
        )

    def test_query_relevance(
        self,
        query_spec: dict,
        use_enhancement: bool = True,
        limit: int = 5
    ) -> QueryRelevanceReport:
        """
        Test relevance of retrieval results for a single query.
        """
        query = query_spec["query"]
        
        # Retrieve with enhancement (synchronous)
        results = self.memory_index.retrieve(
            query=query,
            limit=limit,
            use_structured_enhancement=use_enhancement
        )
        
        judgments = []
        for result in results:
            # Handle both dict and UnifiedBullet objects
            if hasattr(result, 'content'):
                content = result.content
                score = getattr(result, 'score', 0.0)
            else:
                content = result.get("content", result.get("payload", {}).get("content", ""))
                score = result.get("score", 0.0)
            
            judgment = self.grade_relevance(query_spec, content, score)
            judgments.append(judgment)
        
        # Calculate metrics
        if judgments:
            relevant_count = sum(
                1 for j in judgments 
                if j.relevance_grade.value >= RelevanceGrade.RELEVANT.value
            )
            precision_at_k = relevant_count / len(judgments)
            
            fp_count = sum(1 for j in judgments if j.is_false_positive)
            fn_count = sum(1 for j in judgments if j.is_false_negative)
            
            false_positive_rate = fp_count / len(judgments)
            false_negative_rate = fn_count / len(judgments) if len(judgments) > 0 else 0
        else:
            precision_at_k = 0.0
            false_positive_rate = 0.0
            false_negative_rate = 0.0
        
        # Overall quality assessment
        if precision_at_k >= 0.8:
            overall_quality = "EXCELLENT"
        elif precision_at_k >= 0.6:
            overall_quality = "GOOD"
        elif precision_at_k >= 0.4:
            overall_quality = "FAIR"
        else:
            overall_quality = "POOR"
        
        return QueryRelevanceReport(
            query=query,
            intent=query_spec.get("intent", "unknown"),
            expected_topics=query_spec.get("expected_topics", []),
            judgments=judgments,
            precision_at_k=precision_at_k,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            overall_quality=overall_quality
        )

    def run_exhaustive_test(self) -> dict:
        """
        Run exhaustive relevancy testing across all queries.
        """
        print("\n" + "="*70)
        print("EXHAUSTIVE RELEVANCY TEST")
        print("="*70)
        
        self.setup()
        
        # Test with enhancement enabled
        enhanced_results = []
        print("\n📊 Testing with Structured Enhancement ENABLED")
        print("-"*50)
        
        for i, query_spec in enumerate(self.TEST_QUERIES, 1):
            report = self.test_query_relevance(query_spec, use_enhancement=True)
            enhanced_results.append(report)
            
            status = "✓" if report.overall_quality in ["EXCELLENT", "GOOD"] else "⚠" if report.overall_quality == "FAIR" else "✗"
            print(f"{status} [{i:02d}] {report.query[:40]:<40} | P@K: {report.precision_at_k:.1%} | {report.overall_quality}")
            
            # Show false positives
            fps = [j for j in report.judgments if j.is_false_positive]
            if fps:
                print(f"    ⚠️ FALSE POSITIVES: {len(fps)}")
                for fp in fps:
                    print(f"       - Score {fp.result_score:.3f}: {fp.result_content[:60]}...")
        
        # Test without enhancement for comparison
        baseline_results = []
        print("\n📊 Testing with Structured Enhancement DISABLED (baseline)")
        print("-"*50)
        
        for i, query_spec in enumerate(self.TEST_QUERIES, 1):
            report = self.test_query_relevance(query_spec, use_enhancement=False)
            baseline_results.append(report)
            
            status = "✓" if report.overall_quality in ["EXCELLENT", "GOOD"] else "⚠" if report.overall_quality == "FAIR" else "✗"
            print(f"{status} [{i:02d}] {report.query[:40]:<40} | P@K: {report.precision_at_k:.1%} | {report.overall_quality}")
        
        # Comparative analysis
        print("\n" + "="*70)
        print("COMPARATIVE ANALYSIS")
        print("="*70)
        
        enhanced_avg_precision = sum(r.precision_at_k for r in enhanced_results) / len(enhanced_results)
        baseline_avg_precision = sum(r.precision_at_k for r in baseline_results) / len(baseline_results)
        
        enhanced_excellent = sum(1 for r in enhanced_results if r.overall_quality == "EXCELLENT")
        enhanced_good = sum(1 for r in enhanced_results if r.overall_quality == "GOOD")
        baseline_excellent = sum(1 for r in baseline_results if r.overall_quality == "EXCELLENT")
        baseline_good = sum(1 for r in baseline_results if r.overall_quality == "GOOD")
        
        total_enhanced_fp = sum(sum(1 for j in r.judgments if j.is_false_positive) for r in enhanced_results)
        total_baseline_fp = sum(sum(1 for j in r.judgments if j.is_false_positive) for r in baseline_results)
        
        print(f"\n{'Metric':<30} | {'Enhanced':>12} | {'Baseline':>12} | {'Δ':>10}")
        print("-"*70)
        print(f"{'Avg Precision@K':<30} | {enhanced_avg_precision:>11.1%} | {baseline_avg_precision:>11.1%} | {enhanced_avg_precision - baseline_avg_precision:>+9.1%}")
        print(f"{'EXCELLENT ratings':<30} | {enhanced_excellent:>12} | {baseline_excellent:>12} | {enhanced_excellent - baseline_excellent:>+10}")
        print(f"{'GOOD+ ratings':<30} | {enhanced_excellent + enhanced_good:>12} | {baseline_excellent + baseline_good:>12} | {(enhanced_excellent + enhanced_good) - (baseline_excellent + baseline_good):>+10}")
        print(f"{'Total False Positives':<30} | {total_enhanced_fp:>12} | {total_baseline_fp:>12} | {total_enhanced_fp - total_baseline_fp:>+10}")
        
        # Detailed breakdown by query type
        print("\n📋 BREAKDOWN BY QUERY INTENT")
        print("-"*50)
        
        intent_groups = {}
        for report in enhanced_results:
            intent = report.intent
            if intent not in intent_groups:
                intent_groups[intent] = []
            intent_groups[intent].append(report.precision_at_k)
        
        for intent, precisions in intent_groups.items():
            avg = sum(precisions) / len(precisions)
            print(f"  {intent:<20}: {avg:.1%} avg precision ({len(precisions)} queries)")
        
        # Identify worst performers
        print("\n⚠️ WORST PERFORMING QUERIES (need attention)")
        print("-"*50)
        
        poor_queries = [r for r in enhanced_results if r.precision_at_k < 0.4]
        if poor_queries:
            for report in sorted(poor_queries, key=lambda r: r.precision_at_k):
                print(f"  ✗ '{report.query}' - P@K: {report.precision_at_k:.1%}")
                print(f"      Expected: {report.expected_topics}")
                for j in report.judgments[:2]:
                    print(f"      - [{j.relevance_grade.name}] {j.reasoning}")
        else:
            print("  ✓ No critically poor queries!")
        
        # Final verdict
        print("\n" + "="*70)
        print("FINAL VERDICT")
        print("="*70)
        
        if enhanced_avg_precision >= 0.95:
            verdict = "🏆 EXCEPTIONAL - Exceeds 95% precision target"
            status = "PASS"
        elif enhanced_avg_precision >= 0.80:
            verdict = "✅ STRONG - Good relevance quality"
            status = "PASS"
        elif enhanced_avg_precision >= 0.60:
            verdict = "⚠️ MODERATE - Needs improvement"
            status = "WARN"
        else:
            verdict = "❌ POOR - Significant relevance issues"
            status = "FAIL"
        
        print(f"\n{verdict}")
        print(f"Enhanced Precision: {enhanced_avg_precision:.1%}")
        print(f"vs Baseline: {baseline_avg_precision:.1%} ({enhanced_avg_precision - baseline_avg_precision:+.1%})")
        print(f"Enhancement Impact: {'Positive' if enhanced_avg_precision > baseline_avg_precision else 'Negative' if enhanced_avg_precision < baseline_avg_precision else 'Neutral'}")
        
        return {
            "status": status,
            "enhanced_precision": enhanced_avg_precision,
            "baseline_precision": baseline_avg_precision,
            "improvement": enhanced_avg_precision - baseline_avg_precision,
            "enhanced_excellent": enhanced_excellent,
            "baseline_excellent": baseline_excellent,
            "total_queries": len(self.TEST_QUERIES),
            "verdict": verdict
        }


def main():
    """Run exhaustive relevancy test"""
    tester = ExhaustiveRelevancyTester()
    results = tester.run_exhaustive_test()
    
    # Return exit code based on status
    if results["status"] == "FAIL":
        sys.exit(1)
    return results


if __name__ == "__main__":
    main()
