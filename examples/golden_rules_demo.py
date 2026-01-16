"""
Golden Rules Auto-Promotion Demo

Demonstrates the automatic promotion and demotion of high-performing bullets
to/from the golden_rules section based on feedback metrics.
"""

import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from ace.playbook import Playbook
from ace.config import ELFConfig


def main():
    """Demonstrate golden rules auto-promotion feature."""
    print("=" * 70)
    print("Golden Rules Auto-Promotion Demo")
    print("=" * 70)

    # Create playbook
    playbook = Playbook()

    # Configure golden rules feature
    config = ELFConfig(
        enable_golden_rules=True,
        golden_rule_helpful_threshold=10,  # Need 10+ helpful to promote
        golden_rule_max_harmful=0,  # No harmful feedback allowed
        golden_rule_demotion_harmful_threshold=3,  # Demote at 3+ harmful
    )

    print("\nConfiguration:")
    print(f"  Promotion threshold: {config.golden_rule_helpful_threshold} helpful")
    print(f"  Max harmful for promotion: {config.golden_rule_max_harmful}")
    print(f"  Demotion threshold: {config.golden_rule_demotion_harmful_threshold} harmful")

    # Add various bullets with different performance metrics
    print("\n" + "=" * 70)
    print("Adding bullets with different performance metrics...")
    print("=" * 70)

    # High performer - will be promoted
    b1 = playbook.add_bullet(
        "strategies",
        "Always validate user input before processing",
        metadata={"helpful": 15, "harmful": 0},
    )
    print(f"\n✓ Added: {b1.content}")
    print(f"  Section: {b1.section}, helpful={b1.helpful}, harmful={b1.harmful}")

    # Another high performer
    b2 = playbook.add_bullet(
        "tactics",
        "Use try-except blocks for error handling",
        metadata={"helpful": 12, "harmful": 0},
    )
    print(f"\n✓ Added: {b2.content}")
    print(f"  Section: {b2.section}, helpful={b2.helpful}, harmful={b2.harmful}")

    # Exactly at threshold
    b3 = playbook.add_bullet(
        "strategies",
        "Log all security events",
        metadata={"helpful": 10, "harmful": 0},
    )
    print(f"\n✓ Added: {b3.content}")
    print(f"  Section: {b3.section}, helpful={b3.helpful}, harmful={b3.harmful}")

    # Below threshold - won't promote
    b4 = playbook.add_bullet(
        "strategies",
        "Write unit tests",
        metadata={"helpful": 8, "harmful": 0},
    )
    print(f"\n✓ Added: {b4.content}")
    print(f"  Section: {b4.section}, helpful={b4.helpful}, harmful={b4.harmful}")

    # Has harmful feedback - won't promote
    b5 = playbook.add_bullet(
        "strategies",
        "Use global variables for state",
        metadata={"helpful": 12, "harmful": 2},
    )
    print(f"\n✓ Added: {b5.content}")
    print(f"  Section: {b5.section}, helpful={b5.helpful}, harmful={b5.harmful}")

    # Check initial state
    print("\n" + "=" * 70)
    print("Initial Playbook State")
    print("=" * 70)
    print(f"\nSections: {', '.join(playbook.list_sections())}")
    print(f"Total bullets: {len(playbook.bullets())}")

    # Promote qualifying bullets
    print("\n" + "=" * 70)
    print("Checking for promotions...")
    print("=" * 70)

    promoted = playbook.check_and_promote_golden_rules(config)

    if promoted:
        print(f"\n✓ Promoted {len(promoted)} bullets to golden_rules:")
        for bullet_id in promoted:
            bullet = playbook.get_bullet(bullet_id)
            print(f"  - [{bullet_id}] {bullet.content}")
    else:
        print("\n✗ No bullets qualified for promotion")

    # Show golden rules
    print("\n" + "=" * 70)
    print("Golden Rules Section")
    print("=" * 70)

    if "golden_rules" in playbook._sections:
        for bullet_id in playbook._sections["golden_rules"]:
            bullet = playbook.get_bullet(bullet_id)
            print(f"\n★ {bullet.content}")
            print(f"  ID: {bullet_id}")
            print(f"  Stats: helpful={bullet.helpful}, harmful={bullet.harmful}")
    else:
        print("\n(No golden rules yet)")

    # Simulate accumulating harmful feedback
    print("\n" + "=" * 70)
    print("Simulating harmful feedback accumulation...")
    print("=" * 70)

    # Pick one golden rule and add harmful feedback
    if promoted:
        victim_id = promoted[0]
        victim = playbook.get_bullet(victim_id)
        print(f"\nAdding harmful feedback to: {victim.content}")
        victim.harmful = 5
        print(f"  New stats: helpful={victim.helpful}, harmful={victim.harmful}")

        # Check for demotion
        print("\n" + "=" * 70)
        print("Checking for demotions...")
        print("=" * 70)

        demoted = playbook.demote_from_golden_rules(config)

        if demoted:
            print(f"\n✓ Demoted {len(demoted)} bullets from golden_rules:")
            for bullet_id in demoted:
                bullet = playbook.get_bullet(bullet_id)
                print(f"  - [{bullet_id}] {bullet.content}")
                print(f"    New section: {bullet.section}")
        else:
            print("\n✗ No bullets qualified for demotion")

    # Final state
    print("\n" + "=" * 70)
    print("Final Playbook State")
    print("=" * 70)

    stats = playbook.stats()
    print(f"\nSections: {stats['sections']}")
    print(f"Total bullets: {stats['bullets']}")
    print(f"\nTag totals:")
    print(f"  Helpful: {stats['tags']['helpful']}")
    print(f"  Harmful: {stats['tags']['harmful']}")
    print(f"  Neutral: {stats['tags']['neutral']}")

    print("\n" + "=" * 70)
    print("Section breakdown:")
    print("=" * 70)
    for section in sorted(playbook.list_sections()):
        bullet_ids = playbook._sections[section]
        print(f"\n{section.upper()} ({len(bullet_ids)} bullets):")
        for bullet_id in bullet_ids:
            bullet = playbook.get_bullet(bullet_id)
            print(f"  - [{bullet_id}] {bullet.content}")
            print(f"    Stats: helpful={bullet.helpful}, harmful={bullet.harmful}")

    # Demonstrate full lifecycle
    print("\n" + "=" * 70)
    print("Full Lifecycle Demo")
    print("=" * 70)

    # Create a bullet and watch it evolve
    b_lifecycle = playbook.add_bullet(
        "experiments",
        "Use database connection pooling",
        metadata={"helpful": 0, "harmful": 0},
    )
    print(f"\n1. Created: {b_lifecycle.content}")
    print(f"   Section: {b_lifecycle.section}")
    print(f"   Stats: helpful={b_lifecycle.helpful}, harmful={b_lifecycle.harmful}")

    # Accumulate positive feedback
    b_lifecycle.helpful = 10
    print(f"\n2. After positive feedback:")
    print(f"   Stats: helpful={b_lifecycle.helpful}, harmful={b_lifecycle.harmful}")

    promoted = playbook.check_and_promote_golden_rules(config)
    print(f"\n3. After promotion check:")
    print(f"   Promoted: {len(promoted) > 0}")
    print(f"   Section: {b_lifecycle.section}")

    # Accumulate harmful feedback
    b_lifecycle.harmful = 3
    print(f"\n4. After harmful feedback:")
    print(f"   Stats: helpful={b_lifecycle.helpful}, harmful={b_lifecycle.harmful}")

    demoted = playbook.demote_from_golden_rules(config)
    print(f"\n5. After demotion check:")
    print(f"   Demoted: {len(demoted) > 0}")
    print(f"   Section: {b_lifecycle.section}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
