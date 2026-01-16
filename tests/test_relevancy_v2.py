#!/usr/bin/env python3
"""
RELEVANCY TEST V2 - Tests against actual ACE memory content

The ACE system stores LESSONS, PREFERENCES, and STRATEGIES - not technical docs.
This test evaluates relevance based on what's actually in the memory database.
"""

import os
import sys
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ace.unified_memory import UnifiedMemoryIndex


class RelevanceGrade(IntEnum):
    HIGHLY_RELEVANT = 4  # Directly answers the query
    RELEVANT = 3         # Useful related information  
    MARGINAL = 2         # Tangentially related
    IRRELEVANT = 1       # Not useful
    OFF_TOPIC = 0        # Completely unrelated


@dataclass
class Result:
    content: str
    grade: RelevanceGrade
    reasoning: str


class RelevancyTesterV2:
    """
    Tests relevancy against queries that match what ACE actually stores:
    - Best practices and lessons learned
    - User preferences and directives
    - Workflow patterns and strategies
    - Error handling guidance
    - Development process tips
    """
    
    # These queries match the type of content ACE stores
    TEST_QUERIES = [
        {
            "query": "how to handle errors properly",
            "intent": "best_practice",
            "relevant_terms": ["error", "log", "trace", "context", "handle", "fail", "exception", "catch"],
            "description": "Error handling best practices"
        },
        {
            "query": "configuration management best practices",
            "intent": "best_practice",
            "relevant_terms": ["config", "environment", "variable", "secret", "api key", "setting"],
            "description": "Configuration management"
        },
        {
            "query": "what should I test in my code",
            "intent": "guidance",
            "relevant_terms": ["test", "coverage", "ci", "cd", "automation", "regression", "quality"],
            "description": "Testing guidance"
        },
        {
            "query": "how to improve performance",
            "intent": "optimization",
            "relevant_terms": ["performance", "optimize", "benchmark", "monitor", "latency", "speed", "cache"],
            "description": "Performance optimization tips"
        },
        {
            "query": "security best practices",
            "intent": "best_practice",
            "relevant_terms": ["security", "auth", "validate", "secret", "environment", "safe"],
            "description": "Security guidance"
        },
        {
            "query": "debugging tips",
            "intent": "workflow",
            "relevant_terms": ["debug", "log", "trace", "error", "stack", "investigate", "diagnose"],
            "description": "Debugging workflow"
        },
        {
            "query": "code documentation practices",
            "intent": "best_practice",
            "relevant_terms": ["doc", "sync", "design", "code", "live", "update"],
            "description": "Documentation practices"
        },
        {
            "query": "deployment automation",
            "intent": "workflow",
            "relevant_terms": ["deploy", "ci", "cd", "pipeline", "automation", "build", "release"],
            "description": "Deployment workflow"
        },
        {
            "query": "user frustration handling",
            "intent": "preference",
            "relevant_terms": ["frustration", "user", "feedback", "log", "context"],
            "description": "User experience in debugging"
        },
        {
            "query": "api key management",
            "intent": "security",
            "relevant_terms": ["api", "key", "config", "environment", "secret", "centralize"],
            "description": "API key handling"
        },
        {
            "query": "monitoring and observability",
            "intent": "best_practice",
            "relevant_terms": ["monitor", "observ", "log", "trace", "metric", "alert", "config", "blind"],
            "description": "Monitoring practices"
        },
        {
            "query": "code quality lessons",
            "intent": "guidance",
            "relevant_terms": ["quality", "clean", "maintain", "refactor", "review", "lint", "static", "test"],
            "description": "Code quality guidance"
        },
        # Vague queries
        {
            "query": "something broke",
            "intent": "troubleshooting",
            "relevant_terms": ["error", "fail", "debug", "log", "fix"],
            "description": "Vague troubleshooting"
        },
        {
            "query": "help with secrets",
            "intent": "security",
            "relevant_terms": ["secret", "config", "environment", "api", "key"],
            "description": "Secrets management"
        },
        {
            "query": "make code better",
            "intent": "optimization",
            "relevant_terms": ["quality", "optimize", "test", "performance", "improve", "refactor", "clean", "maintain"],
            "description": "Code improvement"
        }
    ]

    def __init__(self):
        self.memory_index: Optional[UnifiedMemoryIndex] = None

    def setup(self):
        self.memory_index = UnifiedMemoryIndex()
        print("Memory index initialized")

    def grade_result(self, query_spec: dict, content: str) -> Result:
        """Grade a single result for relevance"""
        content_lower = content.lower()
        relevant_terms = query_spec.get("relevant_terms", [])
        
        # Count matches
        matches = sum(1 for term in relevant_terms if term in content_lower)
        
        if not relevant_terms:
            return Result(content[:100], RelevanceGrade.MARGINAL, "No terms to match")
        
        match_ratio = matches / len(relevant_terms)
        
        if match_ratio >= 0.4:
            grade = RelevanceGrade.HIGHLY_RELEVANT
            reasoning = f"Strong: {matches}/{len(relevant_terms)} terms"
        elif match_ratio >= 0.25:
            grade = RelevanceGrade.RELEVANT
            reasoning = f"Good: {matches}/{len(relevant_terms)} terms"
        elif match_ratio >= 0.1:
            grade = RelevanceGrade.MARGINAL
            reasoning = f"Weak: {matches}/{len(relevant_terms)} terms"
        else:
            grade = RelevanceGrade.IRRELEVANT
            reasoning = f"Poor: {matches}/{len(relevant_terms)} terms"
        
        return Result(content[:100], grade, reasoning)

    def test_query(self, query_spec: dict, use_enhancement: bool = True, limit: int = 5) -> dict:
        """Test a single query and return metrics"""
        query = query_spec["query"]
        
        results = self.memory_index.retrieve(
            query=query,
            limit=limit,
            use_structured_enhancement=use_enhancement
        )
        
        graded = []
        for result in results:
            content = result.content if hasattr(result, 'content') else str(result)
            grade_result = self.grade_result(query_spec, content)
            graded.append(grade_result)
        
        # Calculate precision
        relevant_count = sum(1 for g in graded if g.grade.value >= RelevanceGrade.RELEVANT.value)
        precision = relevant_count / len(graded) if graded else 0
        
        # Count highly relevant
        highly_relevant = sum(1 for g in graded if g.grade == RelevanceGrade.HIGHLY_RELEVANT)
        
        return {
            "query": query,
            "description": query_spec["description"],
            "precision": precision,
            "highly_relevant": highly_relevant,
            "total_results": len(graded),
            "grades": graded
        }

    def run_test(self):
        """Run complete relevancy test"""
        print("\n" + "="*70)
        print("RELEVANCY TEST V2 - Testing Against Actual ACE Content")
        print("="*70)
        
        self.setup()
        
        # Test with enhancement
        print("\n[1] WITH Structured Enhancement")
        print("-"*50)
        
        enhanced_results = []
        for i, spec in enumerate(self.TEST_QUERIES, 1):
            result = self.test_query(spec, use_enhancement=True)
            enhanced_results.append(result)
            
            status = "+" if result["precision"] >= 0.6 else "~" if result["precision"] >= 0.4 else "-"
            print(f"[{status}] {result['query'][:40]:<40} | P: {result['precision']:.0%} | HR: {result['highly_relevant']}")
        
        # Test without enhancement
        print("\n[2] WITHOUT Enhancement (baseline)")
        print("-"*50)
        
        baseline_results = []
        for i, spec in enumerate(self.TEST_QUERIES, 1):
            result = self.test_query(spec, use_enhancement=False)
            baseline_results.append(result)
            
            status = "+" if result["precision"] >= 0.6 else "~" if result["precision"] >= 0.4 else "-"
            print(f"[{status}] {result['query'][:40]:<40} | P: {result['precision']:.0%} | HR: {result['highly_relevant']}")
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        enhanced_avg = sum(r["precision"] for r in enhanced_results) / len(enhanced_results)
        baseline_avg = sum(r["precision"] for r in baseline_results) / len(baseline_results)
        
        enhanced_hr = sum(r["highly_relevant"] for r in enhanced_results)
        baseline_hr = sum(r["highly_relevant"] for r in baseline_results)
        
        enhanced_good = sum(1 for r in enhanced_results if r["precision"] >= 0.6)
        baseline_good = sum(1 for r in baseline_results if r["precision"] >= 0.6)
        
        print(f"\n{'Metric':<30} | {'Enhanced':>12} | {'Baseline':>12} | {'Delta':>10}")
        print("-"*70)
        print(f"{'Average Precision':<30} | {enhanced_avg:>11.1%} | {baseline_avg:>11.1%} | {enhanced_avg - baseline_avg:>+9.1%}")
        print(f"{'Total Highly Relevant':<30} | {enhanced_hr:>12} | {baseline_hr:>12} | {enhanced_hr - baseline_hr:>+10}")
        print(f"{'Queries with 60%+ precision':<30} | {enhanced_good:>12} | {baseline_good:>12} | {enhanced_good - baseline_good:>+10}")
        
        # Improvement analysis
        print("\n" + "-"*70)
        print("PER-QUERY IMPROVEMENT")
        print("-"*70)
        
        improved = 0
        degraded = 0
        same = 0
        
        for e, b in zip(enhanced_results, baseline_results):
            delta = e["precision"] - b["precision"]
            if delta > 0.05:
                improved += 1
                status = "UP"
            elif delta < -0.05:
                degraded += 1
                status = "DOWN"
            else:
                same += 1
                status = "SAME"
            print(f"  {e['query'][:35]:<35} | {b['precision']:.0%} -> {e['precision']:.0%} | {status}")
        
        print(f"\nImproved: {improved} | Same: {same} | Degraded: {degraded}")
        
        # Final verdict
        print("\n" + "="*70)
        print("VERDICT")
        print("="*70)
        
        if enhanced_avg >= 0.80:
            verdict = "EXCELLENT - High relevance quality"
            code = 0
        elif enhanced_avg >= 0.60:
            verdict = "GOOD - Acceptable relevance"
            code = 0
        elif enhanced_avg >= 0.40:
            verdict = "FAIR - Needs improvement"
            code = 0
        else:
            verdict = "POOR - Significant issues"
            code = 1
        
        print(f"\n{verdict}")
        print(f"Enhanced Average Precision: {enhanced_avg:.1%}")
        print(f"Enhancement Impact: {'+' if enhanced_avg > baseline_avg else ''}{(enhanced_avg - baseline_avg)*100:.1f}pp")
        
        return code


def main():
    tester = RelevancyTesterV2()
    return tester.run_test()


if __name__ == "__main__":
    sys.exit(main())
