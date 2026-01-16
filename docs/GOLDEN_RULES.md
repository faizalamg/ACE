# Golden Rules Auto-Promotion

## Overview

The **Golden Rules Auto-Promotion** feature automatically promotes high-performing bullets to golden status and demotes underperforming ones. This ELF-inspired mechanism ensures that proven strategies are prioritized while problematic ones are retired.

**Two Implementations:**
- **Qdrant-Native (Recommended)**: `UnifiedMemoryIndex` with `is_golden` payload field
- **Legacy JSON Playbook**: Section-based promotion (deprecated)

## Key Concepts

### Promotion
Bullets that accumulate sufficient **helpful** feedback (and minimal **harmful** feedback) are automatically promoted to the `golden_rules` section, signaling their proven value across multiple executions.

### Demotion
Bullets in the `golden_rules` section that later accumulate harmful feedback are demoted to a `deprecated` section to prevent continued use of strategies that have become problematic.

## Configuration

Golden rules behavior is controlled via `ELFConfig` in `ace/config.py`:

```python
from ace.config import ELFConfig

config = ELFConfig(
    enable_golden_rules=True,               # Feature toggle
    golden_rule_helpful_threshold=10,       # Min helpful to promote
    golden_rule_max_harmful=0,              # Max harmful allowed for promotion
    golden_rule_demotion_harmful_threshold=3  # Harmful count triggers demotion
)
```

### Environment Variables

Override defaults via `.env` or environment:

| Variable | Default | Description |
|----------|---------|-------------|
| `ACE_GOLDEN_RULES` | `True` | Enable/disable feature |
| `ACE_GOLDEN_THRESHOLD` | `10` | Min helpful count for promotion |
| `ACE_GOLDEN_MAX_HARMFUL` | `0` | Max harmful count for promotion |
| `ACE_GOLDEN_DEMOTION_HARMFUL` | `3` | Harmful count triggers demotion |

## API Reference

### Qdrant-Native Methods (Recommended)

These methods operate directly on Qdrant payloads via `UnifiedMemoryIndex`:

#### `tag_bullet(bullet_id, tag, increment=1)`

Tag a bullet as helpful or harmful. Auto-checks golden status after update.

**Parameters:**
- `bullet_id` (str): Original bullet ID
- `tag` (str): "helpful" or "harmful"
- `increment` (int): Amount to increment (default: 1)

**Returns:**
- `bool`: True if update succeeded

**Example:**
```python
from ace.unified_memory import UnifiedMemoryIndex

index = UnifiedMemoryIndex()

# Tag as helpful - also validates (resets decay timer)
index.tag_bullet("bullet-001", "helpful")

# Tag as harmful - may trigger demotion if harmful >= 3
index.tag_bullet("bullet-002", "harmful")
```

#### `validate_bullet(bullet_id)`

Mark a bullet as recently validated. Resets confidence decay timer.

**Parameters:**
- `bullet_id` (str): Original bullet ID

**Returns:**
- `bool`: True if update succeeded

**Example:**
```python
# Reset decay timer for a bullet that proved useful
index.validate_bullet("bullet-001")
```

#### `get_golden_rules(limit=50)`

Retrieve all golden rules from Qdrant.

**Parameters:**
- `limit` (int): Maximum rules to return (default: 50)

**Returns:**
- `List[UnifiedBullet]`: Golden rule bullets

**Example:**
```python
golden = index.get_golden_rules()
print(f"Found {len(golden)} golden rules")
for bullet in golden:
    print(f"  {bullet.content}")
```

#### `promote_golden_rules()`

Scan and promote eligible bullets to golden status.

**Returns:**
- `int`: Number of bullets promoted

**Example:**
```python
promoted = index.promote_golden_rules()
print(f"Promoted {promoted} bullets to golden status")
```

#### `demote_golden_rules()`

Scan and demote golden bullets exceeding harmful threshold.

**Returns:**
- `int`: Number of bullets demoted

**Example:**
```python
demoted = index.demote_golden_rules()
print(f"Demoted {demoted} bullets from golden status")
```

#### `retrieve_with_decay(query, namespace=None, limit=10, threshold=0.3)`

Retrieve with confidence decay applied to ranking. Favors recently validated bullets.

**Parameters:**
- `query` (str): Search query
- `namespace` (Optional): Namespace filter
- `limit` (int): Maximum results
- `threshold` (float): Minimum score threshold

**Returns:**
- `List[UnifiedBullet]`: Re-ranked by decayed effectiveness

**Example:**
```python
# Get results with decay-adjusted ranking
results = index.retrieve_with_decay("authentication patterns")
```

---

### UnifiedBullet ELF Methods

These methods operate on individual `UnifiedBullet` instances:

#### `effective_score_with_decay()`

Compute effectiveness score with confidence decay.

**Returns:**
- `float`: Score between 0.0 and 1.0, decayed based on time since validation

**Example:**
```python
bullet = UnifiedBullet(...)
decayed = bullet.effective_score_with_decay()
# Fresh bullet: ~0.8 (raw effectiveness)
# 10 weeks old: ~0.48 (with decay)
```

#### `validate()`

Mark this bullet as recently validated. Resets decay timer.

**Example:**
```python
bullet.validate()
# bullet.last_validated is now current timestamp
```

#### `check_golden_status()`

Check if this bullet qualifies for golden rule status.

**Returns:**
- `bool`: True if bullet qualifies

**Example:**
```python
if bullet.check_golden_status():
    print("Eligible for golden promotion!")
```

#### `check_demotion_status()`

Check if this golden bullet should be demoted.

**Returns:**
- `bool`: True if bullet should be demoted

---

### Legacy Playbook Methods (Deprecated)

#### `check_and_promote_golden_rules(config=None)`

Check all bullets and promote qualifying ones to `golden_rules` section.

**Parameters:**
- `config` (Optional[ELFConfig]): Configuration (uses global config if not provided)

**Returns:**
- `List[str]`: List of promoted bullet IDs

**Example:**
```python
from ace.playbook import Playbook
from ace.config import get_elf_config

playbook = Playbook()
# Add bullets and accumulate feedback...

promoted = playbook.check_and_promote_golden_rules()
print(f"Promoted {len(promoted)} bullets to golden_rules")
```

#### `demote_from_golden_rules(config=None)`

Check `golden_rules` section and demote bullets that no longer qualify.

**Parameters:**
- `config` (Optional[ELFConfig]): Configuration (uses global config if not provided)

**Returns:**
- `List[str]`: List of demoted bullet IDs

**Example:**
```python
demoted = playbook.demote_from_golden_rules()
print(f"Demoted {len(demoted)} bullets from golden_rules")
```

## Promotion Criteria

A bullet qualifies for promotion when:

1. **Helpful threshold met**: `bullet.helpful >= config.golden_rule_helpful_threshold`
2. **Minimal harmful feedback**: `bullet.harmful <= config.golden_rule_max_harmful`
3. **Not already in golden_rules**: Prevents re-promotion

**Default thresholds:**
- Helpful: ≥ 10
- Harmful: ≤ 0 (zero tolerance for errors)

## Demotion Criteria

A `golden_rules` bullet is demoted when:

1. **Harmful threshold exceeded**: `bullet.harmful >= config.golden_rule_demotion_harmful_threshold`

**Default threshold:**
- Harmful: ≥ 3 (allows some tolerance before deprecation)

## Lifecycle Example

```python
from ace.playbook import Playbook
from ace.config import ELFConfig

# Setup
playbook = Playbook()
config = ELFConfig(
    enable_golden_rules=True,
    golden_rule_helpful_threshold=10,
    golden_rule_max_harmful=0,
    golden_rule_demotion_harmful_threshold=3
)

# 1. Create bullet
bullet = playbook.add_bullet(
    "strategies",
    "Always validate input",
    metadata={"helpful": 0, "harmful": 0}
)
print(f"Section: {bullet.section}")  # strategies

# 2. Accumulate positive feedback
bullet.helpful = 10

# 3. Check for promotion
promoted = playbook.check_and_promote_golden_rules(config)
print(f"Section: {bullet.section}")  # golden_rules

# 4. Accumulate harmful feedback
bullet.harmful = 3

# 5. Check for demotion
demoted = playbook.demote_from_golden_rules(config)
print(f"Section: {bullet.section}")  # deprecated
```

## Section Cleanup

Empty sections are automatically removed after promotion/demotion:

```python
# Before: playbook has "strategies" section with 1 bullet
promoted = playbook.check_and_promote_golden_rules(config)

# After: "strategies" section removed (empty), "golden_rules" created
assert "strategies" not in playbook._sections
assert "golden_rules" in playbook._sections
```

## Timestamp Updates

Bullets' `updated_at` timestamp is automatically updated on promotion/demotion:

```python
original_time = bullet.updated_at
playbook.check_and_promote_golden_rules(config)
assert bullet.updated_at != original_time  # Timestamp changed
```

## Integration with Adaptation Loop

Golden rules auto-promotion integrates seamlessly with the standard ACE adaptation loop:

```python
from ace import OfflineAdapter, Playbook, Generator, Reflector, Curator
from ace.config import get_elf_config

# Setup
playbook = Playbook()
adapter = OfflineAdapter(playbook, generator, reflector, curator)

# Run training
adapter.run(training_samples, environment, epochs=3)

# Check for promotions after training
config = get_elf_config()
promoted = playbook.check_and_promote_golden_rules(config)
print(f"Promoted {len(promoted)} high-performing strategies")

# Save playbook with golden rules
playbook.save_to_file("trained_with_golden_rules.json")
```

## Best Practices

### 1. Promotion Frequency
Check for promotions after:
- Each training epoch (offline adaptation)
- Batch of successful executions (online adaptation)
- Periodic intervals (e.g., daily/weekly)

### 2. Threshold Tuning
Adjust thresholds based on:
- **Task complexity**: Higher thresholds for simple tasks
- **Execution volume**: Lower thresholds for low-volume systems
- **Risk tolerance**: Stricter (higher helpful, lower harmful) for critical systems

### 3. Monitoring Golden Rules
Track golden rules metrics:

```python
golden_bullets = [
    b for b in playbook.bullets()
    if b.section == "golden_rules"
]

print(f"Golden rules count: {len(golden_bullets)}")
for bullet in golden_bullets:
    ratio = bullet.helpful / (bullet.helpful + bullet.harmful)
    print(f"  {bullet.content}: {ratio:.2%} success rate")
```

### 4. Deprecated Cleanup
Periodically review and remove deprecated bullets:

```python
deprecated_bullets = [
    b for b in playbook.bullets()
    if b.section == "deprecated"
]

# Manual review or automated cleanup
for bullet in deprecated_bullets:
    if should_permanently_remove(bullet):
        playbook.remove_bullet(bullet.id)
```

## Testing

Comprehensive test suite in `tests/test_golden_rules.py` covers:

- ✓ Basic promotion/demotion
- ✓ Threshold boundary conditions
- ✓ Multiple bullet handling
- ✓ Section cleanup
- ✓ Timestamp updates
- ✓ Feature disable/enable
- ✓ Full lifecycle scenarios

Run tests:
```bash
python -m pytest tests/test_golden_rules.py -v
```

## Demo

See `examples/golden_rules_demo.py` for a complete demonstration:

```bash
python examples/golden_rules_demo.py
```

## Troubleshooting

### Bullets not promoting

**Check:**
1. Feature enabled: `config.enable_golden_rules == True`
2. Thresholds met: `bullet.helpful >= threshold and bullet.harmful <= max_harmful`
3. Not already promoted: `bullet.section != "golden_rules"`

**Debug:**
```python
from ace.config import get_elf_config

config = get_elf_config()
print(f"Feature enabled: {config.enable_golden_rules}")
print(f"Thresholds: helpful>={config.golden_rule_helpful_threshold}, harmful<={config.golden_rule_max_harmful}")

for bullet in playbook.bullets():
    qualifies = (
        bullet.helpful >= config.golden_rule_helpful_threshold and
        bullet.harmful <= config.golden_rule_max_harmful
    )
    print(f"{bullet.id}: helpful={bullet.helpful}, harmful={bullet.harmful}, qualifies={qualifies}")
```

### Unexpected demotions

**Check:**
1. Harmful threshold: `bullet.harmful < config.golden_rule_demotion_harmful_threshold`
2. Golden section exists: `"golden_rules" in playbook._sections`

**Debug:**
```python
config = get_elf_config()
print(f"Demotion threshold: harmful>={config.golden_rule_demotion_harmful_threshold}")

golden_bullets = [b for b in playbook.bullets() if b.section == "golden_rules"]
for bullet in golden_bullets:
    at_risk = bullet.harmful >= config.golden_rule_demotion_harmful_threshold
    print(f"{bullet.id}: harmful={bullet.harmful}, at_risk={at_risk}")
```

## Performance Considerations

- **Complexity**: O(n) where n = number of bullets
- **Frequency**: Run after batch operations, not every single execution
- **Memory**: Minimal overhead (just metadata updates)

## Future Enhancements

Potential improvements:

1. **Gradient-based promotion**: Require sustained performance (e.g., 10 helpful in a row)
2. **Time-decay**: Demote bullets that haven't been used recently
3. **Confidence scoring**: Promote based on effectiveness ratio (helpful/(helpful+harmful))
4. **Section hierarchy**: Multi-tier golden rules (gold/silver/bronze)
5. **Automatic threshold tuning**: Adjust thresholds based on playbook statistics

## See Also

- [ELF Configuration](../ace/config.py) - Configuration options
- [Playbook API](../ace/playbook.py) - Core playbook implementation
- [Adaptation Loop](../ace/adaptation.py) - Integration with training
- [Test Suite](../tests/test_golden_rules.py) - Comprehensive tests
- [Demo Script](../examples/golden_rules_demo.py) - Working example
