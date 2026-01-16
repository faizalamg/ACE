"""
Demonstration of confidence decay feature for ACE bullets.

This example shows how to use the time-based decay system to prevent
stale knowledge from dominating retrieval in the ACE framework.
"""

from datetime import datetime, timedelta, timezone

from ace.playbook import Playbook


def main():
    print("=" * 70)
    print("ACE Confidence Decay Demonstration")
    print("=" * 70)
    print()

    # Create a playbook with several bullets
    playbook = Playbook()

    # Add bullets with different validation states
    print("Creating bullets with different validation ages...")
    print()

    # Bullet 1: Fresh strategy (just validated)
    fresh = playbook.add_bullet(
        section="API Design",
        content="Use versioned endpoints with /v1/ prefix for backward compatibility"
    )
    fresh.helpful = 10
    fresh.harmful = 1
    fresh.validate()
    print(f"[OK] Created fresh bullet: {fresh.id}")
    print(f"  Base score: {fresh.helpful - fresh.harmful}")
    print(f"  Validated: Just now")
    print()

    # Bullet 2: Moderately stale (2 weeks old)
    moderate = playbook.add_bullet(
        section="API Design",
        content="Always return 200 OK with error details in response body"
    )
    moderate.helpful = 10
    moderate.harmful = 1
    two_weeks_ago = datetime.now(timezone.utc) - timedelta(weeks=2)
    moderate.last_validated = two_weeks_ago.isoformat()
    print(f"[OK] Created moderate bullet: {moderate.id}")
    print(f"  Base score: {moderate.helpful - moderate.harmful}")
    print(f"  Validated: 2 weeks ago")
    print()

    # Bullet 3: Very stale (8 weeks old)
    stale = playbook.add_bullet(
        section="API Design",
        content="Use XML for all API responses for maximum compatibility"
    )
    stale.helpful = 10
    stale.harmful = 1
    eight_weeks_ago = datetime.now(timezone.utc) - timedelta(weeks=8)
    stale.last_validated = eight_weeks_ago.isoformat()
    print(f"[OK] Created stale bullet: {stale.id}")
    print(f"  Base score: {stale.helpful - stale.harmful}")
    print(f"  Validated: 8 weeks ago")
    print()

    # Bullet 4: Never validated (high base score but no validation)
    never = playbook.add_bullet(
        section="API Design",
        content="Implement GraphQL for all new APIs"
    )
    never.helpful = 15
    never.harmful = 3
    print(f"[OK] Created never-validated bullet: {never.id}")
    print(f"  Base score: {never.helpful - never.harmful}")
    print(f"  Validated: Never")
    print()

    # Show effective scores with decay
    print("=" * 70)
    print("Effective Scores (with 5% weekly decay)")
    print("=" * 70)
    print()

    bullets = [
        ("Fresh (just validated)", fresh),
        ("Moderate (2 weeks old)", moderate),
        ("Stale (8 weeks old)", stale),
        ("Never validated", never),
    ]

    for label, bullet in bullets:
        base_score = bullet.helpful - bullet.harmful
        effective = bullet.effective_score(decay_rate=0.95)
        decay_percent = ((base_score - effective) / base_score * 100) if base_score > 0 else 0

        print(f"{label}:")
        print(f"  Base score:      {base_score:>6.2f}")
        print(f"  Effective score: {effective:>6.2f} (-{decay_percent:.1f}%)")
        print()

    # Demonstrate sorting by effective score
    print("=" * 70)
    print("Bullets Sorted by Effective Score (Best First)")
    print("=" * 70)
    print()

    sorted_bullets = sorted(
        playbook.bullets(),
        key=lambda b: b.effective_score(),
        reverse=True
    )

    for i, bullet in enumerate(sorted_bullets, 1):
        effective = bullet.effective_score()
        print(f"{i}. {bullet.content[:50]}...")
        print(f"   Effective score: {effective:.2f}")
        print()

    # Demonstrate revalidation
    print("=" * 70)
    print("Revalidation Demo: Refreshing Stale Knowledge")
    print("=" * 70)
    print()

    print(f"Stale bullet before revalidation:")
    print(f"  Effective score: {stale.effective_score():.2f}")
    print()

    print("Revalidating bullet (simulating successful use)...")
    stale.validate()
    print()

    print(f"Stale bullet after revalidation:")
    print(f"  Effective score: {stale.effective_score():.2f}")
    print(f"  Last validated: {stale.last_validated}")
    print()

    # Demonstrate custom decay rates
    print("=" * 70)
    print("Custom Decay Rates Comparison")
    print("=" * 70)
    print()

    # Create a test bullet
    test = playbook.add_bullet(section="Test", content="Test bullet")
    test.helpful = 10
    test.harmful = 0
    four_weeks_ago = datetime.now(timezone.utc) - timedelta(weeks=4)
    test.last_validated = four_weeks_ago.isoformat()

    decay_rates = [
        (0.90, "Aggressive (10% weekly)"),
        (0.95, "Standard (5% weekly)"),
        (0.98, "Conservative (2% weekly)"),
    ]

    print("Base score: 10.0 (4 weeks since validation)")
    print()

    for rate, label in decay_rates:
        effective = test.effective_score(decay_rate=rate)
        decay_percent = (10.0 - effective) / 10.0 * 100
        print(f"{label}:")
        print(f"  Effective score: {effective:>6.2f} (-{decay_percent:.1f}%)")
        print()

    # Save playbook with validated timestamps
    print("=" * 70)
    print("Persistence: Saving Playbook with Validation Timestamps")
    print("=" * 70)
    print()

    output_path = "examples/confidence_decay_playbook.json"
    playbook.save_to_file(output_path)
    print(f"[OK] Playbook saved to {output_path}")
    print()

    # Load and verify
    loaded = Playbook.from_json_file(output_path)
    loaded_bullet = loaded.get_bullet(fresh.id)
    print(f"[OK] Loaded playbook from disk")
    print(f"  Bullet {loaded_bullet.id} last_validated: {loaded_bullet.last_validated}")
    print()

    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("1. Bullets decay exponentially based on time since validation")
    print("2. Fresh bullets maintain high effective scores")
    print("3. Stale bullets get lower priority in retrieval")
    print("4. validate() resets decay when strategies prove useful")
    print("5. Never-validated bullets don't decay (cold start)")
    print("6. Validation timestamps persist in JSON serialization")
    print()


if __name__ == "__main__":
    main()
