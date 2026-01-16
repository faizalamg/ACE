# Golden Rules Quick Start

## 5-Minute Setup

### 1. Enable the Feature

Add to `.env` file:
```bash
ACE_GOLDEN_RULES=True
ACE_GOLDEN_THRESHOLD=10
ACE_GOLDEN_MAX_HARMFUL=0
ACE_GOLDEN_DEMOTION_HARMFUL=3
```

### 2. Basic Usage

```python
from ace.playbook import Playbook
from ace.config import get_elf_config

# Create playbook
playbook = Playbook()

# Add bullets (simulating feedback)
b1 = playbook.add_bullet("strategies", "Validate input", metadata={"helpful": 15, "harmful": 0})
b2 = playbook.add_bullet("strategies", "Log errors", metadata={"helpful": 8, "harmful": 0})

# Check for promotions
promoted = playbook.check_and_promote_golden_rules()
print(f"Promoted: {len(promoted)} bullets")  # Output: 1 (only b1 qualifies)

# Check for demotions
demoted = playbook.demote_from_golden_rules()
print(f"Demoted: {len(demoted)} bullets")  # Output: 0
```

### 3. Integration with Training

```python
from ace import OfflineAdapter

# After training
adapter.run(samples, environment, epochs=3)

# Promote high performers
promoted = playbook.check_and_promote_golden_rules()
print(f"Found {len(promoted)} golden rules")

# Save
playbook.save_to_file("trained.json")
```

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `check_and_promote_golden_rules(config=None)` | Promote qualifying bullets | List of promoted IDs |
| `demote_from_golden_rules(config=None)` | Demote problematic bullets | List of demoted IDs |

## Default Thresholds

| Threshold | Default | Meaning |
|-----------|---------|---------|
| Promotion helpful | 10 | Need 10+ helpful to promote |
| Promotion max harmful | 0 | Zero harmful feedback required |
| Demotion harmful | 3 | Demote at 3+ harmful |

## Example Lifecycle

```python
# 1. Create bullet
bullet = playbook.add_bullet("strategies", "Strategy X", metadata={"helpful": 0})

# 2. Accumulate feedback (in real usage, this happens during execution)
bullet.helpful = 10  # Meets promotion threshold

# 3. Promote
playbook.check_and_promote_golden_rules()
assert bullet.section == "golden_rules"

# 4. Later, accumulate harmful feedback
bullet.harmful = 3  # Meets demotion threshold

# 5. Demote
playbook.demote_from_golden_rules()
assert bullet.section == "deprecated"
```

## Run the Demo

```bash
python examples/golden_rules_demo.py
```

## Run Tests

```bash
python -m pytest tests/test_golden_rules.py -v
```

## Troubleshooting

**Bullets not promoting?**
```python
# Check config
from ace.config import get_elf_config
config = get_elf_config()
print(f"Enabled: {config.enable_golden_rules}")
print(f"Thresholds: helpful>={config.golden_rule_helpful_threshold}, harmful<={config.golden_rule_max_harmful}")

# Check bullet stats
for b in playbook.bullets():
    print(f"{b.id}: helpful={b.helpful}, harmful={b.harmful}, section={b.section}")
```

## Full Documentation

See [GOLDEN_RULES.md](GOLDEN_RULES.md) for complete documentation.
