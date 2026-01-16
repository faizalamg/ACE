# ACE Intelligent Learning User Guide

## What is ACE Intelligent Learning?

ACE (Agentic Context Engine) intelligent learning automatically extracts patterns and lessons from your code as you edit it in Claude Code. Instead of generic "learned from code" bullets, you get sophisticated insights like:

> "Prioritize accuracy over simplicity when implementing validation logic, especially for well-defined formats like email addresses"

These patterns are stored in a compressed playbook and automatically loaded into your context window at the start of each session, making Claude Code smarter over time based on YOUR actual coding patterns.

## How It Works

### Automatic Learning Cycle

Every time you use **Write** or **Edit** operations in Claude Code, the learning hook:

1. **Captures** the code you wrote/edited
2. **Analyzes** it through ACE's three-phase pipeline:
   - **Generator**: Identifies potential patterns
   - **Reflector**: Refines patterns with playbook context
   - **Curator**: Decides what to add/update/remove
3. **Stores** validated patterns in compressed format
4. **Loads** patterns into context at next session start

**Time**: ~22 seconds per learning cycle (happens in background)

### What Gets Learned?

ACE extracts high-value patterns from your code:

- **Validation Principles**: "Implement comprehensive validation beyond basic format checks..."
- **Error Handling**: "Consider edge cases like internationalized domain names..."
- **Performance Optimizations**: "Use efficient algorithms for large datasets..."
- **Security Best Practices**: "Always sanitize user input before database queries..."
- **Code Organization**: "Separate business logic from presentation layer..."

### What Doesn't Get Learned?

- Obvious patterns already widely known
- Low-quality code snippets
- Temporary debugging code
- Simple variable assignments

ACE rates patterns as **helpful**, **neutral**, or **harmful** and automatically prunes low-value patterns to maintain quality.

## Token Efficiency

### The Growth Problem (Solved)

Without management, learning would eventually consume your entire context window. ACE solves this with:

1. **Compression** (58% token reduction)
   - 9 bullets: 1,154 tokens → 488 tokens (667 tokens saved)
   - 250 bullets: 32K tokens → 13.5K tokens (18.5K tokens saved)

2. **Smart Pruning** (250 bullet cap)
   - Keeps high-quality recent patterns
   - Removes low-quality old patterns
   - Maintains minimum valuable patterns

### Context Window Impact

| Bullets | Compressed Tokens | % of Context Window |
|---------|------------------|-------------------|
| 10 | ~540 tokens | 0.3% |
| 50 | ~2,700 tokens | 1.4% |
| 100 | ~5,400 tokens | 2.7% |
| 250 | ~13,500 tokens | **6.8%** |

**Sustainable**: At maximum capacity (250 bullets), ACE uses only 6.8% of your 200K context window.

## Configuration Options

### Growth Management

```python
# ace_learn_from_edit.py

MAX_BULLETS = 250              # Hard cap on total patterns
MAX_NEUTRAL_BULLETS = 50       # Limit neutral patterns
MIN_HELPFUL_BULLETS = 50       # Preserve minimum valuable patterns
HARMFUL_GRACE_PERIOD_DAYS = 7  # Keep harmful for analysis
```

**When to adjust**:
- Increase `MAX_BULLETS` if you want more pattern retention (higher context usage)
- Decrease `MAX_NEUTRAL_BULLETS` to be more selective about neutral patterns
- Increase `MIN_HELPFUL_BULLETS` to preserve more high-quality patterns

### Pattern Scoring

Patterns are scored using: **rating × (1.0 / (age_days + 1))**

**Examples**:
- High-quality recent: `rating=3, age=1 day → score=1.5`
- Low-quality old: `rating=1, age=30 days → score=0.032`

Lowest-scored patterns are pruned first when capacity is reached.

## Monitoring Your Playbook

### Check Current State

```powershell
# View uncompressed playbook (full format)
cat C:\Users\Erwin\.claude\ace_playbook.json

# View compressed playbook (storage format)
cat C:\Users\Erwin\.claude\ace_playbook_compressed.json

# Check file sizes
Get-ChildItem C:\Users\Erwin\.claude\ace_playbook*.json | Select Name, Length
```

### Compression Logs

Learning hook logs compression stats on every save:

```
[ACE Compression] Saved uncompressed: 4616 bytes
[ACE Compression] Saved compressed: 1949 bytes (57.8% reduction)
[ACE Compression] Token savings: 667 tokens (1154 → 488)
```

## Troubleshooting

### Issue: Learning Cycle Too Slow

**Symptom**: Write/Edit operations take 22+ seconds

**Solutions**:
1. **Accept it**: Learning only happens on Write/Edit, not every query
2. **Disable temporarily**: Comment out PostToolUse hook registration
3. **Use faster model**: Switch to lighter Z.AI model (tradeoff: lower quality)

### Issue: No Patterns Being Learned

**Symptom**: Playbook stays empty after multiple Write operations

**Checks**:
1. Verify hook is registered: Check `C:\Users\Erwin\.claude\hooks\ace_learn_from_edit.py` exists
2. Check logs: Look for ACE pipeline execution messages
3. Verify Z.AI token: Ensure `ANTHROPIC_AUTH_TOKEN` environment variable is set
4. Test manually: Run hook with simple code example

### Issue: Playbook Growing Too Large

**Symptom**: Context window usage increasing over time

**Solutions**:
1. **Automatic**: Pruning happens automatically at MAX_BULLETS=250
2. **Manual**: Run maintenance script:
   ```powershell
   python C:\Users\Erwin\.claude\hooks\ace_playbook_maintenance.py
   ```
3. **Adjust config**: Lower MAX_BULLETS in `ace_learn_from_edit.py`

### Issue: Compression Not Working

**Symptom**: `ace_playbook_compressed.json` missing or same size as uncompressed

**Checks**:
1. Verify compressor exists: Check `ace_playbook_compressor.py` file
2. Check logs: Look for "[ACE Compression]" messages
3. Test manually:
   ```python
   from ace_playbook_compressor import PlaybookCompressor
   compressor = PlaybookCompressor()
   stats = compressor.compress_file("ace_playbook.json", "ace_playbook_compressed.json")
   print(f"Reduction: {stats['reduction_percent']}%")
   ```

### Issue: Patterns Lost After Session

**Symptom**: Previously learned patterns not appearing in context

**Checks**:
1. Verify SessionStart hook: Check `ace_session_start.py` exists
2. Check compressed file: Ensure `ace_playbook_compressed.json` exists
3. Verify decompression: Look for temporary decompressed file creation in logs
4. Manual load: Check playbook content manually:
   ```python
   from ace_playbook_compressor import PlaybookCompressor
   compressor = PlaybookCompressor()
   compressor.decompress_file("ace_playbook_compressed.json", "temp_playbook.json")
   ```

### Issue: Authentication Failure

**Symptom**: "Authentication failed" errors during learning

**Solution**: Verify Z.AI token environment variable:
```powershell
$env:ANTHROPIC_AUTH_TOKEN  # Should show your token
```

If missing, add to PowerShell profile:
```powershell
$env:ANTHROPIC_AUTH_TOKEN = "3b1cc2ff006243e393260f017a228ebd.h26K4ZBWkIsKSSjm"
```

## Best Practices

### 1. Let It Learn Naturally

Don't force feed code to ACE. Write normally and let it extract patterns from actual work.

### 2. Review Learned Patterns

Periodically check `ace_playbook.json` to see what patterns were learned. Remove unhelpful ones manually if needed.

### 3. Monitor Token Usage

Keep an eye on compression logs to ensure token efficiency is maintained.

### 4. Run Maintenance Periodically

Every few weeks, run the maintenance script to deep clean the playbook:
```powershell
python C:\Users\Erwin\.claude\hooks\ace_playbook_maintenance.py
```

### 5. Adjust Thresholds as Needed

If you find patterns aging out too quickly, increase `MAX_BULLETS`. If context usage is too high, decrease it.

## Advanced Usage

### Manual Playbook Editing

You can manually edit `ace_playbook.json` to:
- Remove specific patterns
- Update pattern ratings
- Add custom tags
- Adjust section categorization

**Important**: After manual edits, run compression to update compressed file:
```powershell
python -c "from ace_playbook_compressor import PlaybookCompressor; PlaybookCompressor().compress_file('ace_playbook.json', 'ace_playbook_compressed.json')"
```

### Custom Section Categories

Add new section categories in playbook:
```json
{
  "bullets": {
    "custom_category-00001": {
      "content": "Your custom pattern...",
      "helpful": 2,
      "harmful": 0,
      "tags": ["custom"],
      "created_at": "2025-01-15T18:00:00Z",
      "last_modified": "2025-01-15T18:00:00Z"
    }
  },
  "sections": {
    "custom_category": ["custom_category-00001"]
  },
  "next_id": 2
}
```

### Disable Learning Temporarily

Comment out hook registration in learning hook file:
```python
# @tool_callback("PostToolUse", ["Write", "Edit"])
def learn_from_edit_hook(tool_name: str, tool_input: dict, tool_output: dict):
    # Learning code here...
```

### Export Patterns for Sharing

Share learned patterns with team:
```powershell
# Export uncompressed for readability
Copy-Item ace_playbook.json -Destination ace_patterns_export.json

# Or export compressed for efficiency
Copy-Item ace_playbook_compressed.json -Destination ace_patterns_export_compressed.json
```

## Understanding Pattern Quality

### Helpful Patterns (rating ≥ 2)

These patterns are **preserved** during pruning:
- "Prioritize accuracy over simplicity when implementing validation logic..."
- "Implement comprehensive validation beyond basic format checks..."
- "Consider edge cases like internationalized domain names..."

### Neutral Patterns (rating = 1)

These are **pruned aggressively** (max 50 kept):
- "Use descriptive variable names"
- "Add comments for complex logic"
- "Follow consistent indentation"

### Harmful Patterns (rating ≤ 0)

These are **removed after 7 days** grace period:
- "Skip error handling for simple functions" (harmful advice)
- "Hardcode credentials for quick testing" (security risk)

## Performance Expectations

### Learning Cycle Timing

- **Quick Edit**: ~22 seconds (Generator 7s + Reflector 8s + Curator 7s)
- **Large File**: ~25-30 seconds (longer context processing)
- **First Session**: ~30 seconds (includes playbook initialization)

### Token Budget Impact

- **Per Session**: ~488 tokens loaded (compressed playbook at 9 bullets)
- **At Capacity**: ~13.5K tokens loaded (compressed playbook at 250 bullets)
- **Savings**: 59% reduction vs uncompressed (18.5K tokens saved at capacity)

### Storage Impact

- **Uncompressed**: ~4.6KB per 9 bullets (~128KB at 250 bullets)
- **Compressed**: ~1.9KB per 9 bullets (~54KB at 250 bullets)
- **Total**: Both files stored (~182KB at capacity)

## Getting Help

### Log Locations

Check Claude Code logs for ACE-related messages:
- Look for `[ACE Learning]` prefix (learning pipeline)
- Look for `[ACE Compression]` prefix (compression operations)
- Look for `[ACE Pruning]` prefix (growth management)

### Diagnostic Commands

```powershell
# Check playbook stats
python -c "import json; p=json.load(open('ace_playbook.json')); print(f'Bullets: {len(p[\"bullets\"])}')"

# Verify compression ratio
python -c "from ace_playbook_compressor import PlaybookCompressor; print(PlaybookCompressor().compress_file('ace_playbook.json', 'temp.json'))"

# List all patterns by section
python -c "import json; p=json.load(open('ace_playbook.json')); [print(f'{s}: {len(ids)}') for s,ids in p['sections'].items()]"
```

### Common Questions

**Q: Can I disable ACE learning permanently?**  
A: Yes, remove or rename `ace_learn_from_edit.py` hook file.

**Q: Does learning work offline?**  
A: No, ACE requires Z.AI API access for Generator/Reflector/Curator.

**Q: Can I share my playbook with others?**  
A: Yes, copy `ace_playbook.json` or `ace_playbook_compressed.json` to share learned patterns.

**Q: What happens if playbook gets corrupted?**  
A: Delete both playbook files and restart Claude Code. ACE will create fresh playbook.

**Q: How do I reset and start fresh?**  
A: Delete `ace_playbook.json` and `ace_playbook_compressed.json`, restart session.

**Q: Can I learn from other people's code?**  
A: Yes, use Write/Edit operations on any code to trigger learning.

## Related Documentation

- [Technical Integration Guide](CLAUDE_CODE_ACE_INTEGRATION.md) - Detailed architecture and API reference
- [ACE Framework Docs](COMPLETE_GUIDE_TO_ACE.md) - Full ACE framework documentation
- [Integration Guide](INTEGRATION_GUIDE.md) - Framework-agnostic integration patterns

## Restored Intelligence Features (v0.7.0)

The following intelligence features were restored/enhanced in the unified memory architecture:

### 1. Memory Deduplication System

**Problem Solved**: Duplicate memories were being stored when similar feedback was given multiple times.

**How It Works**:
- Before storing a new memory, the system checks semantic similarity (cosine > 0.92)
- If a similar memory exists, it's **reinforced** instead of duplicated
- Reinforcement increments `reinforcement_count`, updates `severity`, adds timestamp

**API Change (BREAKING)**:
```python
# OLD: index_bullet() returned bool
result = index.index_bullet(bullet)  # True/False

# NEW: index_bullet() returns Dict
result = index.index_bullet(bullet)
# {
#     "stored": True,
#     "action": "new" | "reinforced",
#     "similarity": 0.0-1.0,
#     "existing_id": None | str,
#     "reinforcement_count": int
# }
```

**Configuration**:
```python
index.index_bullet(bullet, dedup_threshold=0.92, enable_dedup=True)
```

### 2. Unified Memory Architecture

**Problem Solved**: Two separate "ACE" systems caused confusion about which memories were being used.

**Before**: Personal Memory Bank + ACE Playbook (disconnected)
**After**: Single `ace_unified` collection with namespace separation

| Namespace | Content | Source |
|-----------|---------|--------|
| `user_prefs` | Preferences, communication styles | User feedback |
| `task_strategies` | Code patterns, error fixes | Task execution |
| `project_specific` | Project-specific context | Project hooks |

**Performance Gains**:
- 50% faster stores (1 write instead of 2)
- 50% faster searches (1 retrieval instead of merge)
- 57% code reduction in memory module

### 3. Hybrid Search (BM25 + Dense + RRF)

**Problem Solved**: Keyword-only or semantic-only search missed relevant memories.

**How It Works**:
1. **Dense vectors**: Semantic similarity via embeddings
2. **BM25 sparse vectors**: Keyword/term matching
3. **RRF fusion**: Reciprocal Rank Fusion combines both results

**Technical Details**:
- Dense: `nomic-embed-text-v1.5` (768 dimensions)
- Sparse: BM25 with k1=1.5, b=0.75
- 3x oversampling during prefetch for better recall

### 4. Embedding Model Fix

**Problem Solved**: `snowflake-arctic-embed-m-v1.5` returned identical embeddings for long text.

**Fix Applied**:
- Switched to `nomic-embed-text-v1.5` model
- Re-embedded all 2010+ memories
- Fixed in: `ace/unified_memory.py`, `ace/qdrant_retrieval.py`, `ace/async_retrieval.py`

### 5. SmartBulletIndex Integration

**Problem Solved**: ACE Framework's sophisticated retrieval wasn't used for personal memories.

**How It Works**:
- `SmartBulletIndex` now accepts `unified_index` parameter
- Namespace-aware scoring and filtering
- Dynamic weighting based on bullet type

```python
from ace.retrieval import SmartBulletIndex
from ace.unified_memory import UnifiedMemoryIndex, UnifiedNamespace

unified = UnifiedMemoryIndex()
index = SmartBulletIndex(unified_index=unified)

# Retrieve with namespace filter
results = index.retrieve(
    query="how to handle errors",
    namespace=UnifiedNamespace.TASK_STRATEGIES,
    limit=10
)
```

### 6. Curator Unified Storage

**Problem Solved**: Curator-learned patterns weren't stored in unified memory.

**How It Works**:
- Curator now accepts `unified_index` and `store_to_unified` parameters
- ADD operations automatically stored to unified memory
- Namespace: `task_strategies`, Source: `task_execution`

```python
from ace.roles import Curator

curator = Curator(
    llm=client,
    unified_index=unified_index,
    store_to_unified=True  # Default: True
)
```

### Migration Guide

**Prerequisites**:
1. Run memory migration before updating:
   ```bash
   python scripts/migrate_memories_to_unified.py --verify
   ```

2. Optional: Deduplicate existing memories:
   ```bash
   python scripts/deduplicate_memories.py --dry-run  # Preview
   python scripts/deduplicate_memories.py            # Execute
   ```

**Rollback** (if needed):
```bash
python scripts/rollback_unified_migration.py --check   # Check status
python scripts/rollback_unified_migration.py --rollback  # Execute rollback
```

### Verification Commands

```bash
# Check unified collection stats
curl http://localhost:6333/collections/ace_unified

# Check memory count by namespace
python -c "
from ace.unified_memory import UnifiedMemoryIndex, UnifiedNamespace
idx = UnifiedMemoryIndex()
print(f'Total: {idx.count()}')
print(f'User Prefs: {idx.count(UnifiedNamespace.USER_PREFS)}')
print(f'Task Strategies: {idx.count(UnifiedNamespace.TASK_STRATEGIES)}')
"

# Test retrieval
python -c "
from ace.unified_memory import UnifiedMemoryIndex
idx = UnifiedMemoryIndex()
results = idx.retrieve('error handling', limit=5)
for r in results:
    print(f'{r.namespace}: {r.content[:50]}...')
"
```

---

## Feedback and Improvements

Learned patterns and compression efficiency improve over time. Monitor your playbook growth and adjust thresholds as needed for optimal performance.

**Key Metrics to Watch**:
- Bullet count (target: stay under 250)
- Compression ratio (target: >50% reduction)
- Token usage (target: <15K tokens at capacity)
- Pattern quality (target: >80% helpful patterns)
- Deduplication rate (target: <5% duplicates)
