# ACE Unified Memory Rollback Guide

## Overview

The rollback script provides safety mechanisms to revert the ACE unified memory migration if issues are detected. It allows you to restore the system to its pre-migration state by deleting the new unified collection while preserving the old memory collections.

## Prerequisites

- Python 3.11+
- `httpx` package installed (`pip install httpx`)
- Qdrant server running (default: `http://localhost:6333`)
- Access to both old and new collections

## Collections

| Collection Name | Purpose | Status After Rollback |
|-----------------|---------|----------------------|
| `ace_memories_hybrid` | Old memory storage | PRESERVED |
| `ace_unified` | New unified storage | DELETED |

## Usage

### 1. Check Rollback Feasibility

Before executing a rollback, verify that it's possible:

```bash
python scripts/rollback_unified_migration.py --check
```

**Output:**
- ✅ **Feasible**: Old collection exists, can safely rollback
- ❌ **Not Feasible**: Old collection deleted, cannot rollback

**Rollback is feasible when:**
- Old collection (`ace_memories_hybrid`) still exists
- Qdrant server is accessible

### 2. Show Migration Status

View the current state of both collections:

```bash
python scripts/rollback_unified_migration.py --status
```

**Output Example:**
```
======================================================================
ACE Unified Memory Migration Status
======================================================================

Old Collection: ace_memories_hybrid
  Status: green
  Points: 2100
  Vectors: 0

New Collection: ace_unified
  Status: green
  Points: 2100
  Vectors: 4200

======================================================================
Migration State: COMPLETED (both collections exist)
Action: Can rollback to delete new collection
======================================================================
```

**Migration States:**
- **COMPLETED**: Both collections exist (can rollback)
- **NOT RUN or ROLLED BACK**: Only old collection exists (no rollback needed)
- **OLD DATA LOST**: Only new collection exists (⚠️  cannot rollback!)
- **NO COLLECTIONS**: Neither exists (clean state)

### 3. Execute Rollback

Rollback the migration by deleting the unified collection:

```bash
# With confirmation prompt (recommended)
python scripts/rollback_unified_migration.py --rollback

# Without confirmation (use with caution!)
python scripts/rollback_unified_migration.py --rollback --no-confirm
```

**Process:**
1. Checks rollback feasibility
2. Shows what will be deleted/preserved
3. Asks for confirmation (unless `--no-confirm`)
4. Deletes unified collection
5. Verifies old collection still exists
6. Reports success/failure

**Output Example:**
```
======================================================================
ROLLBACK PLAN
======================================================================
Will DELETE: ace_unified (2100 points)
Will KEEP:   ace_memories_hybrid (2100 points)
======================================================================

Are you sure you want to rollback? This will DELETE the unified collection. (yes/no): yes

[2025-12-11 16:20:00] [WARN] Deleting unified collection 'ace_unified'...
[2025-12-11 16:20:01] [INFO] Collection ace_unified deleted successfully

======================================================================
ROLLBACK SUCCESSFUL
======================================================================
✓ Deleted unified collection: ace_unified
✓ Preserved old collection: ace_memories_hybrid (2100 points)

System restored to pre-migration state.
======================================================================
```

## Safety Features

### 1. **Pre-flight Checks**
- Verifies old collection exists before rollback
- Checks Qdrant connectivity
- Validates collection states

### 2. **Confirmation Prompts**
- Requires explicit "yes" confirmation (unless `--no-confirm`)
- Shows exactly what will be deleted
- No accidental deletions

### 3. **Verification**
- Confirms deletion succeeded
- Verifies old collection still intact
- Reports final state

### 4. **Logging**
- All operations logged with timestamps
- Clear success/error messages
- Audit trail of actions

### 5. **Exit Codes**
- `0` = Success
- `1` = Failure (old collection lost, deletion failed, etc.)

## Error Handling

### "Old collection does NOT exist"

**Cause**: The old collection was already deleted
**Action**: Cannot rollback - old data is permanently lost
**Prevention**: Never delete old collection until unified collection is verified

### "Failed to delete unified collection"

**Cause**: Permission issues, Qdrant offline, or network error
**Action**: Check Qdrant logs, verify connectivity
**Recovery**: Retry rollback after fixing issue

### "Qdrant server not accessible"

**Cause**: Qdrant not running or wrong URL
**Action**: Start Qdrant server or check `--qdrant-url` parameter
**Default URL**: `http://localhost:6333`

## Common Scenarios

### Scenario 1: Migration Failed, Need to Retry

```bash
# Check that rollback is possible
python scripts/rollback_unified_migration.py --check

# View current state
python scripts/rollback_unified_migration.py --status

# Rollback to pre-migration state
python scripts/rollback_unified_migration.py --rollback

# Re-run migration with fixes
python scripts/migrate_memories_to_unified.py
```

### Scenario 2: Migration Succeeded, But Want to Revert

```bash
# Check collections
python scripts/rollback_unified_migration.py --status

# Rollback (confirms before delete)
python scripts/rollback_unified_migration.py --rollback
```

### Scenario 3: Verify Migration Success

```bash
# Check that both collections exist
python scripts/rollback_unified_migration.py --status

# Verify old collection intact (feasibility check)
python scripts/rollback_unified_migration.py --check
```

## Advanced Usage

### Custom Qdrant URL

```bash
# Non-default Qdrant server
python scripts/rollback_unified_migration.py --status \
  --qdrant-url http://remote-server:6333
```

### Environment Variables

```bash
# Set Qdrant URL via environment
export QDRANT_URL="http://remote-server:6333"

python scripts/rollback_unified_migration.py --status
```

### Automated Rollback (CI/CD)

```bash
# Non-interactive rollback (no confirmation)
python scripts/rollback_unified_migration.py --rollback --no-confirm

# Check exit code
if [ $? -eq 0 ]; then
  echo "Rollback successful"
else
  echo "Rollback failed"
  exit 1
fi
```

## Best Practices

### ✅ DO

- **Always check feasibility** before rollback
- **Use status command** to understand current state
- **Keep old collection** until unified collection is verified
- **Test rollback** in development environment first
- **Monitor exit codes** in automated scripts

### ❌ DON'T

- **Don't delete old collection** immediately after migration
- **Don't use `--no-confirm`** unless in automated environment
- **Don't assume rollback works** without checking feasibility
- **Don't ignore error messages** - investigate root cause

## Troubleshooting

### Problem: Cannot connect to Qdrant

**Check:**
```bash
# Verify Qdrant is running
curl http://localhost:6333/healthz

# Check collections
curl http://localhost:6333/collections
```

**Fix:**
- Start Qdrant server
- Check firewall rules
- Verify URL is correct

### Problem: Old collection shows 0 points

**Check:**
```bash
python scripts/rollback_unified_migration.py --status
```

**Investigate:**
- Was old collection actually populated?
- Did migration accidentally delete source data?
- Check Qdrant logs for errors

**Action:**
- If old collection is empty, rollback won't help
- Restore from backup if available

### Problem: Rollback succeeds but unified collection still exists

**Rare edge case** - Qdrant might have caching issues

**Fix:**
```bash
# Restart Qdrant
docker restart qdrant  # if using Docker

# Or manually delete via API
curl -X DELETE http://localhost:6333/collections/ace_unified

# Verify deletion
python scripts/rollback_unified_migration.py --status
```

## Integration with Migration Workflow

```bash
# Complete migration workflow with safety checks

# 1. Pre-migration backup (optional)
python scripts/backup_collections.py --collection ace_memories_hybrid

# 2. Dry-run migration
python scripts/migrate_memories_to_unified.py --dry-run

# 3. Run migration
python scripts/migrate_memories_to_unified.py --verify

# 4. Verify success
python scripts/rollback_unified_migration.py --status

# 5. If issues found, rollback
python scripts/rollback_unified_migration.py --rollback

# 6. If successful, optionally delete old collection (after testing!)
# (NOT recommended until unified collection is proven stable)
```

## Related Documentation

- **Migration Guide**: `docs/PROJECT_UNIFIED_MEMORY_ARCHITECTURE.md`
- **Migration Scripts**:
  - `scripts/migrate_memories_to_unified.py`
  - `scripts/migrate_playbook_to_unified.py`
- **Tests**: `tests/test_rollback_script.py`

## Support

If you encounter issues with rollback:

1. Run `--status` to see current state
2. Check Qdrant logs
3. Verify Qdrant connectivity
4. Review error messages carefully
5. Open GitHub issue with status output

## License

Same as ACE Framework (see LICENSE file)
