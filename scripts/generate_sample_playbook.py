"""Generate a sample playbook for retrieval benchmarking."""
import json
from ace.playbook import Playbook, Bullet

# Create bullet content based on ID (realistic examples)
BULLET_TEMPLATES = {
    # Debugging
    "debug_timeout": ("debugging", "For timeout errors, check network latency, database query duration, and external API response times. Use distributed tracing to identify bottlenecks."),
    "check_logs": ("debugging", "Always start by checking application logs, error logs, and system logs. Look for stack traces, error messages, and timing patterns."),
    "debug_memory_leak": ("debugging", "Use memory profilers to track heap growth over time. Look for unreleased references, event listeners, and large object retention."),
    "reproduce_issue": ("debugging", "Create a minimal reproduction case that isolates the bug. This helps verify the fix and prevents regressions."),
    "binary_search_debug": ("debugging", "When debugging complex issues, use binary search: disable half the code, test, repeat until you isolate the problematic section."),

    # Security
    "incident_response": ("security", "Security incident response: isolate affected systems, preserve evidence, analyze logs, contain the breach, and notify stakeholders."),
    "forensics": ("security", "Digital forensics: create disk images, analyze file access timestamps, check network logs, and preserve chain of custody."),
    "security_xss": ("security", "Prevent XSS by sanitizing all user input, using Content Security Policy headers, and escaping output in templates."),
    "input_validation": ("security", "Validate all input on the server side. Never trust client-side validation. Use allowlists instead of denylists."),
    "auth_code_flow": ("security", "For OAuth2, use Authorization Code flow with PKCE. Never store tokens in localStorage; use httpOnly cookies."),
    "least_privilege": ("security", "Apply principle of least privilege: grant minimum permissions needed. Regularly audit and revoke unnecessary access."),

    # Performance optimization
    "optimize_query": ("performance", "Optimize slow database queries by adding indexes, analyzing query plans, and denormalizing when necessary."),
    "add_indexes": ("performance", "Add database indexes on foreign keys, frequently queried columns, and WHERE clause fields. Monitor index usage."),
    "analyze_query_plan": ("performance", "Use EXPLAIN ANALYZE to understand query execution plans. Look for table scans, missing indexes, and inefficient joins."),
    "cache_strategy": ("performance", "Implement caching at multiple levels: CDN, application cache (Redis), and database query cache. Set appropriate TTLs."),
    "cdn_assets": ("performance", "Serve static assets via CDN with proper cache headers. Use content hashing for cache busting."),
    "lazy_loading": ("performance", "Implement lazy loading for images, components, and data. Load only what's visible or needed immediately."),
    "bottleneck_focus": ("performance", "Focus optimization efforts on actual bottlenecks identified through profiling, not assumed slow areas."),

    # Code quality
    "refactor_code": ("code_quality", "Refactor code when you touch it: improve naming, extract functions, remove duplication. Leave code better than you found it."),
    "code_review_checklist": ("code_quality", "Code review checklist: correctness, tests, security, performance, readability, and documentation."),
    "naming_conventions": ("code_quality", "Use clear, descriptive names. Classes/types are nouns (User, Order), functions are verbs (getUserById, createOrder)."),
    "single_responsibility": ("code_quality", "Apply Single Responsibility Principle: each function/class should have one reason to change."),
    "dry_principle": ("code_quality", "Don't Repeat Yourself: extract common code into reusable functions. Each piece of knowledge should have a single representation."),

    # Testing
    "write_tests": ("testing", "Write tests before fixing bugs. The test should fail initially, then pass after the fix. This prevents regressions."),
    "test_pyramid": ("testing", "Follow test pyramid: many unit tests, fewer integration tests, minimal E2E tests. Unit tests are fast and precise."),
    "automated_tests": ("testing", "Automate all tests in CI/CD. Tests should run on every commit and block merges if failing."),
    "edge_cases": ("testing", "Test edge cases: null/empty inputs, boundary values, concurrent access, network failures, and invalid data."),
    "test_coverage": ("testing", "Aim for high test coverage of critical paths, but don't chase 100%. Focus on important logic and edge cases."),

    # Problem solving
    "break_down_problem": ("reasoning", "Break complex problems into smaller subproblems. Solve each independently, then combine solutions."),
    "use_examples": ("reasoning", "Start with concrete examples before generalizing. Work through 2-3 examples by hand to understand the pattern."),
    "think_aloud": ("reasoning", "When stuck, explain the problem aloud or in writing. Teaching forces you to clarify your thinking."),
    "rubber_duck": ("reasoning", "Rubber duck debugging: explain your code line-by-line to an inanimate object. Often reveals the bug."),

    # Process/workflow
    "review_pr": ("workflow", "PR reviews: check for correctness, tests, documentation, breaking changes, and backward compatibility."),
    "git_commit_msgs": ("workflow", "Write clear git commit messages: start with verb (Fix, Add, Update), describe what and why, reference tickets."),
    "feature_flags": ("workflow", "Use feature flags to deploy code without activating features. Enables gradual rollout and easy rollback."),
    "monitoring_alerts": ("workflow", "Set up monitoring and alerts for critical metrics: error rate, latency, throughput, and resource usage."),
    "rollback_plan": ("workflow", "Always have a rollback plan before deploying. Automate rollbacks for quick recovery."),

    # Architecture
    "acid_properties": ("architecture", "Database ACID properties: Atomicity (all-or-nothing), Consistency (valid states), Isolation (concurrent safety), Durability (persisted)."),
    "eventual_consistency": ("architecture", "In distributed systems, embrace eventual consistency when strong consistency is too expensive. Design for idempotency."),
    "circuit_breaker": ("architecture", "Implement circuit breaker pattern for external services. Fail fast instead of cascading failures."),
    "retry_strategy": ("architecture", "For transient failures, use exponential backoff with jitter. Set max retries and timeout limits."),
    "api_versioning": ("architecture", "Version APIs from day one. Use URL versioning (/v1/, /v2/) or header-based versioning."),
    "backward_compat": ("architecture", "Maintain backward compatibility: additive changes only, deprecate gracefully, support N-1 versions."),

    # Communication
    "post_mortem": ("communication", "After incidents, write blameless post-mortems: what happened, root cause, action items, timeline."),
    "documentation": ("communication", "Document why, not just what. Explain design decisions, tradeoffs, and constraints for future maintainers."),
    "escalation_path": ("communication", "Know when to escalate: if blocked >2 hours, if security issue, if customer-impacting, or if architectural decision needed."),
}

def generate_bullet_content(bullet_id: str) -> tuple[str, str]:
    """Generate realistic content for a bullet based on its ID."""
    if bullet_id in BULLET_TEMPLATES:
        return BULLET_TEMPLATES[bullet_id]

    # Infer category and generate generic content
    category = "general"
    if "debug" in bullet_id or "error" in bullet_id:
        category = "debugging"
    elif "security" in bullet_id or "auth" in bullet_id:
        category = "security"
    elif "optimize" in bullet_id or "performance" in bullet_id or "cache" in bullet_id:
        category = "performance"
    elif "test" in bullet_id:
        category = "testing"
    elif "refactor" in bullet_id or "code" in bullet_id:
        category = "code_quality"
    elif "api" in bullet_id or "service" in bullet_id:
        category = "architecture"

    # Generate generic text based on ID
    text = f"Strategy for {bullet_id.replace('_', ' ')}: implement best practices and follow established patterns."
    return (category, text)

def main():
    # Load test datasets to extract all bullet IDs
    with open('benchmarks/data/representative.json', 'r') as f:
        representative = json.load(f)

    with open('benchmarks/data/adversarial.json', 'r') as f:
        adversarial = json.load(f)

    # Collect all unique bullet IDs
    bullet_ids = set()
    for item in representative + adversarial:
        bullet_ids.update(item.get('relevant_bullet_ids', []))
        bullet_ids.update(item.get('irrelevant_bullet_ids', []))

    print(f"Generating playbook with {len(bullet_ids)} bullets...")

    # Create playbook
    playbook = Playbook()

    for bullet_id in sorted(bullet_ids):
        category, text = generate_bullet_content(bullet_id)
        # add_bullet signature: section, content, bullet_id, metadata
        playbook.add_bullet(
            section=category,
            content=text,
            bullet_id=bullet_id,
            metadata={"helpful_count": 5, "harmful_count": 0}
        )

    # Save playbook
    output_path = 'benchmarks/data/sample_playbook.json'
    with open(output_path, 'w') as f:
        json.dump(playbook.to_dict(), f, indent=2)

    print(f"Playbook saved to {output_path}")
    print(f"Categories: {set(b.category for b in playbook.bullets())}")
    print(f"Total bullets: {len(playbook.bullets())}")

if __name__ == "__main__":
    main()
