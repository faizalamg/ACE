# Rollback Script for ACE Unified Memory Migration

## Quick Reference

```bash
# Check if rollback is possible
python scripts/rollback_unified_migration.py --check

# Show current migration status
python scripts/rollback_unified_migration.py --status

# Execute rollback (with confirmation)
python scripts/rollback_unified_migration.py --rollback

# Execute rollback without confirmation (dangerous!)
python scripts/rollback_unified_migration.py --rollback --no-confirm
```

## What It Does

**Safely reverts unified memory migration by:**
1. Verifying old collection (`ace_memories_hybrid`) still exists
2. Deleting new unified collection (`ace_unified`)
3. Preserving original data
4. Logging all operations

## Safety Features

- ✅ **Confirmation prompts** before destructive operations
- ✅ **Pre-flight checks** to verify rollback is possible
- ✅ **Verification** that old collection remains intact
- ✅ **Detailed logging** with timestamps
- ✅ **Clear exit codes** (0 = success, 1 = failure)

## When to Use

### ✅ **Use Rollback When:**
- Migration produced incorrect results
- Unified collection has data corruption
- Need to re-run migration with fixes
- Want to test migration in isolation
- Discovered bugs in migrated data

### ❌ **DON'T Use Rollback When:**
- Old collection was already deleted (rollback impossible)
- Unified collection is in production use
- No backup exists

## Example Workflow

```bash
# 1. Check current status
python scripts/rollback_unified_migration.py --status

# Output shows:
# - Old Collection: ace_memories_hybrid (2100 points)
# - New Collection: ace_unified (2100 points)
# - Migration State: COMPLETED

# 2. Verify rollback is feasible
python scripts/rollback_unified_migration.py --check

# Output:
# [SUCCESS] Rollback is FEASIBLE - old collection is intact

# 3. Execute rollback
python scripts/rollback_unified_migration.py --rollback

# Prompts for confirmation:
# Are you sure? (yes/no): yes

# Output:
# ✓ Deleted unified collection: ace_unified
# ✓ Preserved old collection: ace_memories_hybrid (2100 points)
```

## Exit Codes

| Code | Meaning | Action |
|------|---------|--------|
| `0` | Success | Rollback completed or not needed |
| `1` | Failure | Old collection lost, deletion failed, or Qdrant offline |

## Collections

| Name | Type | Rollback Action |
|------|------|-----------------|
| `ace_memories_hybrid` | Old storage | **PRESERVED** - Never deleted |
| `ace_unified` | New storage | **DELETED** - Removed completely |

## Output Examples

### Successful Rollback

```
======================================================================
ROLLBACK PLAN
======================================================================
Will DELETE: ace_unified (2100 points)
Will KEEP:   ace_memories_hybrid (2100 points)
======================================================================

[2025-12-11 16:20:00] [WARN] Deleting unified collection 'ace_unified'...
[2025-12-11 16:20:01] [SUCCESS] Rollback completed successfully

======================================================================
ROLLBACK SUCCESSFUL
======================================================================
✓ Deleted unified collection: ace_unified
✓ Preserved old collection: ace_memories_hybrid (2100 points)

System restored to pre-migration state.
======================================================================
```

### Rollback Not Feasible

```
[2025-12-11 16:20:00] [INFO] Checking rollback feasibility...
[2025-12-11 16:20:00] [ERROR] Old collection 'ace_memories_hybrid' does NOT exist
[2025-12-11 16:20:00] [ERROR] Cannot rollback - old data is lost
[2025-12-11 16:20:00] [ERROR] Rollback is NOT feasible - aborting
```

### Migration Not Run

```
======================================================================
ACE Unified Memory Migration Status
======================================================================

Old Collection: ace_memories_hybrid
  Status: green
  Points: 2100

New Collection: ace_unified
  Status: DOES NOT EXIST (Migration not run or rolled back)

======================================================================
Migration State: NOT RUN or ROLLED BACK
Action: No rollback needed
======================================================================
```

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| Old collection does NOT exist | Already deleted | Cannot rollback - restore from backup |
| Failed to delete unified collection | Permissions/network | Check Qdrant logs, retry |
| Qdrant server not accessible | Not running | Start Qdrant, verify URL |

## Advanced Options

### Custom Qdrant URL

```bash
python scripts/rollback_unified_migration.py --status \
  --qdrant-url http://remote-server:6333
```

### Environment Variables

```bash
export QDRANT_URL="http://custom-server:6333"
python scripts/rollback_unified_migration.py --check
```

### CI/CD Integration

```bash
#!/bin/bash
# Automated rollback in CI/CD pipeline

if python scripts/rollback_unified_migration.py --rollback --no-confirm; then
  echo "Rollback successful"
else
  echo "Rollback failed - manual intervention required"
  exit 1
fi
```

## Testing

```bash
# Run test suite
python -m pytest tests/test_rollback_script.py -v

# Tests verify:
# - Collection info retrieval
# - Feasibility checks
# - CLI commands work
# - Business logic
```

## Best Practices

1. **Always check feasibility** before rollback
2. **Use `--status`** to understand current state
3. **Never delete old collection** until unified is verified in production
4. **Test in dev environment** first
5. **Monitor exit codes** in automation
6. **Keep confirmation prompts** in interactive use

## Related Scripts

- `scripts/migrate_memories_to_unified.py` - Migrate memories
- `scripts/migrate_playbook_to_unified.py` - Migrate playbooks
- `tests/test_rollback_script.py` - Test suite

## Documentation

See `docs/ROLLBACK_GUIDE.md` for comprehensive documentation.

## Implementation

**Architecture:**
- Uses `httpx` for Qdrant HTTP API
- No dependencies on `qdrant_client` (lightweight)
- Direct REST API calls for collection operations
- Clear separation of concerns (get info, delete, check feasibility)

**Key Functions:**
- `get_collection_info()` - Retrieve collection metadata
- `delete_collection()` - Remove collection safely
- `check_rollback_feasibility()` - Verify rollback is possible
- `show_status()` - Display current state
- `execute_rollback()` - Full rollback workflow

**Safety Mechanisms:**
1. Pre-flight checks (old collection exists)
2. User confirmation (unless `--no-confirm`)
3. Post-rollback verification (old collection intact)
4. Detailed logging (timestamps, levels)
5. Exit codes (0 = success, 1 = failure)

## Maintenance

**When to Update:**
- Collection names change
- New safety checks needed
- Additional verification required
- Qdrant API changes

**Testing Requirements:**
- All CLI commands work
- Exit codes correct
- Logging clear
- Edge cases handled

## License

Same as ACE Framework (see LICENSE file)
