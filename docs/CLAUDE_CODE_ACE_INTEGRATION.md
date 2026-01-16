# Claude Code ACE Integration

## Overview

This guide documents the integration of ACE (Agentic Context Engine) intelligent learning into Claude Code hooks, enabling automatic extraction of patterns and lessons from code editing operations.

## Architecture

### Hook Workflow

```
PostToolUse (Write/Edit Operation)
    ↓
ace_learn_from_edit.py
    ↓
Load Compressed Playbook → Decompress
    ↓
ACE Pipeline (Generator → Reflector → Curator)
    ↓
Apply Delta Operations → Prune if >250 bullets
    ↓
Save Uncompressed → Compress → Store Both Formats
```

### Components

1. **Learning Hook** (`ace_learn_from_edit.py`, 8277 bytes)
   - Triggers on Write/Edit tool operations
   - Extracts code content and tool metadata
   - Runs full ACE pipeline for intelligent pattern extraction
   - Applies delta operations to playbook
   - Manages growth (MAX_BULLETS=250)
   - Compresses on save

2. **Session Start Hook** (`ace_session_start.py`, 2885 bytes)
   - Loads learned patterns into context at session start
   - Preferentially loads compressed format
   - Decompresses to temporary file for ACE
   - Provides seamless pattern retrieval

3. **Compression Layer** (`ace_playbook_compressor.py`, 11820 bytes)
   - Bidirectional structural compression (58% token reduction)
   - Preserves full semantic content (zero degradation)
   - Transparent to ACE framework
   - Automatic compression/decompression

4. **Maintenance Script** (`ace_playbook_maintenance.py`, 4297 bytes)
   - Standalone deep cleaning utility
   - Rating-based pruning with recency decay
   - Manual execution for playbook optimization

## ACE Pipeline

### Three-Phase Learning Process

#### Phase 1: Generator
```python
generator_output = generator.generate(
    question="What patterns can be learned from this code?",
    context=content[:2000],  # Code snippet for analysis
    playbook=playbook
)
```
- **Duration**: ~7 seconds
- **Input**: Code content, existing playbook
- **Output**: Initial pattern candidates

#### Phase 2: Reflector
```python
reflector_output = reflector.reflect(
    question="What patterns can be learned from this code?",
    generator_output=generator_output,
    playbook=playbook,
    ground_truth=None,  # No ground truth in hook context
    feedback="Learn from successful tool execution"
)
```
- **Duration**: ~8 seconds
- **Input**: Generator output, playbook context
- **Output**: Refined insights with playbook context

#### Phase 3: Curator
```python
curator_output = curator.curate(
    reflection=reflector_output,
    playbook=playbook,
    question_context=content[:1000],
    progress=f"Learned from {tool_name} operation"
)

delta = curator_output.delta  # DeltaBatch with operations
```
- **Duration**: ~7 seconds
- **Input**: Reflector output, playbook, context
- **Output**: Delta operations (ADD, UPDATE, TAG, REMOVE)

**Total Learning Cycle**: ~22 seconds (3 LLM calls via Z.AI)

## Playbook Structure

### Uncompressed Format (`ace_playbook.json`)
```json
{
  "bullets": {
    "validation_principles-00007": {
      "content": "Prioritize accuracy over simplicity...",
      "helpful": 2,
      "harmful": 0,
      "tags": ["validation", "accuracy"],
      "created_at": "2025-01-15T17:45:23Z",
      "last_modified": "2025-01-15T17:45:23Z"
    }
  },
  "sections": {
    "validation_principles": ["validation_principles-00007"]
  },
  "next_id": 8
}
```

### Compressed Format (`ace_playbook_compressed.json`)
```json
{
  "b": {
    "vp-7": {
      "c": "Prioritize accuracy over simplicity...",
      "h": 2,
      "t": ["validation", "accuracy"],
      "ca": "2025-01-15T17:45:23Z",
      "lm": "2025-01-15T17:45:23Z"
    }
  },
  "s": {
    "vp": ["vp-7"]
  },
  "nid": 8
}
```

**Key Differences**:
- Structural keys compressed (`bullets` → `b`, `content` → `c`)
- IDs shortened (`validation_principles-00007` → `vp-7`)
- Defaults omitted (`harmful: 0` removed)
- Full content preserved (no semantic compression)

## Compression Strategy

### Design Principles

1. **Structural Compression Only**
   - Compress JSON structure, NOT semantic content
   - Preserve full learning patterns for ACE effectiveness
   - Rejected Chinese storage (translation overhead > savings)

2. **Zero Degradation Guarantee**
   - 100% round-trip integrity
   - Full semantic content preservation
   - No loss of learning quality

3. **Transparent Integration**
   - ACE framework sees uncompressed data
   - Compression layer external to ACE
   - No ACE code modifications required

### Compression Mappings

| Original Key | Compressed | Type |
|-------------|-----------|------|
| `bullets` | `b` | Object |
| `content` | `c` | String |
| `helpful` | `h` | Integer |
| `harmful` | `m` | Integer |
| `tags` | `t` | Array |
| `sections` | `s` | Object |
| `created_at` | `ca` | ISO String |
| `last_modified` | `lm` | ISO String |
| `next_id` | `nid` | Integer |

### ID Compression Algorithm

```python
# Original: "validation_principles-00007"
# Compressed: "vp-7"

def compress_id(bullet_id: str) -> str:
    section, num = bullet_id.rsplit("-", 1)
    prefix = "".join([word[0] for word in section.split("_")])
    return f"{prefix}-{int(num)}"

def decompress_id(short_id: str, section_name: str) -> str:
    num = short_id.split("-")[1]
    return f"{section_name}-{int(num):05d}"
```

### Performance Metrics

| Metric | Current (9 bullets) | Projected (250 bullets) |
|--------|-------------------|------------------------|
| Uncompressed Size | 4,616 bytes | ~128,000 bytes |
| Compressed Size | 1,949 bytes | ~54,000 bytes |
| Token Count (Uncompressed) | ~1,154 tokens | ~32,000 tokens |
| Token Count (Compressed) | ~488 tokens | ~13,500 tokens |
| **Reduction** | **57.8%** | **57.8%** |
| Context Window (Uncompressed) | 0.6% | 16.0% |
| Context Window (Compressed) | 0.2% | 6.8% |

**Token Savings**: 18,500 tokens at capacity (59% reduction in context usage)

## Growth Management

### Configuration

```python
MAX_BULLETS = 250              # Hard cap on total bullets
MAX_NEUTRAL_BULLETS = 50       # Limit neutral patterns
MIN_HELPFUL_BULLETS = 50       # Preserve minimum valuable patterns
HARMFUL_GRACE_PERIOD_DAYS = 7  # Keep harmful for analysis
```

### Pruning Strategy

#### 1. Rating-Based Scoring
```python
def calculate_score(bullet: dict) -> float:
    """Score combines quality rating with recency decay"""
    age_days = (datetime.now() - bullet.created_at).days
    return bullet.rating × (1.0 / (age_days + 1))
```

**Examples**:
- High-quality recent: `rating=3, age=1 day → score=1.5`
- Low-quality old: `rating=1, age=30 days → score=0.032`

#### 2. Pruning Order
1. Remove oldest neutral bullets if count > `MAX_NEUTRAL_BULLETS`
2. Remove harmful bullets after 7-day grace period
3. Remove lowest-scored helpful bullets (NOT oldest) if total > `MAX_BULLETS`
4. Never remove below `MIN_HELPFUL_BULLETS` threshold

#### 3. Automatic Execution
Pruning triggers automatically in `ace_learn_from_edit.py` after applying delta operations.

## LiteLLM Z.AI Integration

### Configuration

```python
import os

# Map Z.AI auth token to LiteLLM expected key
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_AUTH_TOKEN")

# Model name with provider prefix for LiteLLM routing
model_name = "anthropic/claude-3-sonnet-20240229"
```

### Authentication
- **Z.AI Token**: `3b1cc2ff006243e393260f017a228ebd.h26K4ZBWkIsKSSjm`
- **Environment Variable**: `ANTHROPIC_AUTH_TOKEN` → `ANTHROPIC_API_KEY` mapping
- **Base URL**: Automatically uses `ANTHROPIC_BASE_URL` from environment

### Model Routing
- **Correct**: `anthropic/claude-3-sonnet-20240229` (provider prefix required)
- **Incorrect**: `glm-4.6` (LiteLLM fails to recognize provider)

## API Reference

### ACE Framework APIs

#### Generator
```python
from ace import ACELiteLLM

generator = ACELiteLLM.create_generator(model=model_name)

output = generator.generate(
    question: str,           # Question to generate response for
    context: str,            # Code context (recommended: 2000 chars)
    playbook: Playbook       # Existing playbook with learned patterns
) -> GeneratorOutput
```

#### Reflector
```python
reflector = ACELiteLLM.create_reflector(model=model_name)

output = reflector.reflect(
    question: str,                     # Same question as generator
    generator_output: GeneratorOutput, # Output from generator
    playbook: Playbook,                # Existing playbook
    ground_truth: Optional[str],       # Optional ground truth (None in hooks)
    feedback: str                      # Feedback on execution
) -> ReflectorOutput
```

#### Curator
```python
curator = ACELiteLLM.create_curator(model=model_name)

output = curator.curate(
    reflection: ReflectorOutput,  # Output from reflector
    playbook: Playbook,           # Current playbook
    question_context: str,        # Additional context (recommended: 1000 chars)
    progress: str                 # Progress description
) -> CuratorOutput

# Access delta operations
delta = output.delta              # DeltaBatch object
operations = delta.operations     # List[DeltaOperation]
```

### DeltaOperation Structure

```python
class DeltaOperation:
    type: str              # "ADD", "UPDATE", "TAG", or "REMOVE"
    section: str           # Section name (e.g., "validation_principles")
    content: Optional[str] # Pattern content for ADD/UPDATE
    bullet_id: Optional[str] # Target bullet ID for UPDATE/TAG/REMOVE
    metadata: dict         # Additional metadata (tags, ratings, etc.)
```

### PlaybookCompressor API

```python
from ace_playbook_compressor import PlaybookCompressor

compressor = PlaybookCompressor()

# Compress playbook file
stats = compressor.compress_file(
    input_path="ace_playbook.json",
    output_path="ace_playbook_compressed.json"
)
print(f"Reduction: {stats['reduction_percent']}%")

# Decompress playbook file
compressor.decompress_file(
    input_path="ace_playbook_compressed.json",
    output_path="ace_playbook.json"
)
```

## Common Issues and Solutions

### Issue 1: Delta Import Error
**Error**: `cannot import name 'Delta' from 'ace.delta'`

**Solution**: Delta class doesn't exist. Use `curator_output.delta` instead:
```python
# ❌ Wrong
from ace.delta import Delta

# ✅ Correct
curator_output = curator.curate(...)
delta = curator_output.delta
```

### Issue 2: Generator Missing Context
**Error**: `missing 1 required keyword-only argument: 'context'`

**Solution**: Always provide context parameter:
```python
# ❌ Wrong
generator.generate(question=q, playbook=p)

# ✅ Correct
generator.generate(question=q, context=content[:2000], playbook=p)
```

### Issue 3: LiteLLM Provider Recognition
**Error**: `LLM Provider NOT provided. You passed model=glm-4.6`

**Solution**: Use provider prefix for model name:
```python
# ❌ Wrong
model_name = "glm-4.6"

# ✅ Correct
model_name = "anthropic/claude-3-sonnet-20240229"
```

### Issue 4: Authentication Failure
**Error**: Authentication failed with `ANTHROPIC_AUTH_TOKEN`

**Solution**: Map Z.AI token to LiteLLM expected key:
```python
# ✅ Required mapping
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_AUTH_TOKEN")
```

### Issue 5: Curator Missing Parameters
**Error**: `missing 2 required keyword-only arguments: 'question_context' and 'progress'`

**Solution**: Provide both required parameters:
```python
# ❌ Wrong
curator.curate(reflection=r, playbook=p)

# ✅ Correct
curator.curate(
    reflection=r,
    playbook=p,
    question_context=content[:1000],
    progress="Learned from Write operation"
)
```

### Issue 6: CuratorOutput Structure
**Error**: `'CuratorOutput' object has no attribute 'operations'`

**Solution**: Operations are nested inside `.delta` attribute:
```python
# ❌ Wrong
operations = curator_output.operations

# ✅ Correct
delta = curator_output.delta
operations = delta.operations
```

### Issue 7: DeltaOperation Attribute
**Error**: `'DeltaOperation' object has no attribute 'operation_type'`

**Solution**: Attribute is named `.type` (not `.operation_type.value`):
```python
# ❌ Wrong
if op.operation_type.value == "ADD":

# ✅ Correct
if op.type == "ADD":
```

## Performance Optimization

### Token Efficiency
- **Compression**: 58% reduction in storage and context usage
- **Pruning**: Maintains 250 bullet cap with quality preservation
- **At Capacity**: 13.5K tokens (6.8% of context) vs 32K uncompressed (16%)

### Learning Cycle Timing
- **Generator**: ~7 seconds
- **Reflector**: ~8 seconds
- **Curator**: ~7 seconds
- **Total**: ~22 seconds per learning operation
- **Acceptable**: Only triggers on Write/Edit operations

### Scalability
- **Current**: 9 bullets, 488 tokens compressed
- **Projected**: 250 bullets, 13.5K tokens compressed
- **Sustainable**: Automatic pruning prevents unbounded growth

## Best Practices

1. **Content Length**: Provide sufficient context (2000 chars for Generator, 1000 for Curator)
2. **Compression**: Let hooks handle compression automatically (transparent to users)
3. **Pruning**: Run maintenance script periodically for deep cleaning
4. **Monitoring**: Check compression logs for size reductions
5. **Testing**: Validate playbook integrity after manual modifications

## Testing Validation

### Email Validator Test
Successfully learned 3 intelligent patterns from email validation code:

1. "Prioritize accuracy over simplicity when implementing validation logic, especially for well-defined formats like email addresses"
2. "Implement comprehensive validation beyond basic format checks, including RFC-compliant regex patterns that handle edge cases"
3. "Consider edge cases like internationalized domain names (IDN) and quoted strings in local parts when designing validators"

**Verification**: Full semantic content preserved, 57.8% token reduction, zero degradation.

## File Locations

- **Learning Hook**: `C:\Users\Erwin\.claude\hooks\ace_learn_from_edit.py`
- **Session Start**: `C:\Users\Erwin\.claude\hooks\ace_session_start.py`
- **Compressor**: `C:\Users\Erwin\.claude\hooks\ace_playbook_compressor.py`
- **Maintenance**: `C:\Users\Erwin\.claude\hooks\ace_playbook_maintenance.py`
- **Playbook (Uncompressed)**: `C:\Users\Erwin\.claude\ace_playbook.json`
- **Playbook (Compressed)**: `C:\Users\Erwin\.claude\ace_playbook_compressed.json`

## Future Enhancements

1. **Async Learning**: Move ACE pipeline to background thread
2. **Incremental Updates**: Delta-only compression for faster saves
3. **Quality Metrics**: Track pattern effectiveness over time
4. **Auto-tuning**: Dynamic MAX_BULLETS based on context window usage
5. **Cross-session Analytics**: Aggregate learning patterns across sessions
