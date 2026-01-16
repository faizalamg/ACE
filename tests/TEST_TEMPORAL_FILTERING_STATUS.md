# Temporal Filtering Tests - TDD RED Phase Complete ✅

## Status: RED Phase ✅ (All tests failing as expected)

**Created**: 2025-12-13
**Test File**: `tests/test_temporal_filtering.py`
**Target Production File**: `ace/retrieval.py` (SmartBulletIndex.retrieve())

---

## Test Results

**Total Tests**: 13
**Status**: All failing with `TypeError: SmartBulletIndex.retrieve() got an unexpected keyword argument`

This is **EXPECTED** - the parameters don't exist yet in production code.

---

## Test Coverage

### 1. Basic Temporal Filters (3 tests)
- ✅ `test_filter_by_created_after` - Filter bullets created after timestamp
- ✅ `test_filter_by_created_before` - Filter bullets created before timestamp
- ✅ `test_filter_by_updated_after` - Filter bullets updated after timestamp

### 2. Combined Temporal Filters (2 tests)
- ✅ `test_combine_created_after_and_before` - Range filtering (after AND before)
- ✅ `test_combine_temporal_and_task_type_filters` - Temporal + task_type
- ✅ `test_combine_temporal_and_domain_filters` - Temporal + domain

### 3. Edge Cases (5 tests)
- ✅ `test_temporal_filter_with_none_value` - None values should be ignored
- ✅ `test_temporal_filter_with_future_date` - Future dates edge case
- ✅ `test_updated_after_boundary_condition` - Exact timestamp boundary (>= semantics)
- ✅ `test_temporal_filter_with_timezone_handling` - Timezone-aware datetimes
- ✅ `test_match_reasons_include_temporal_filter` - Match reasons include temporal info

### 4. Integration with Existing Features (2 tests)
- ✅ `test_temporal_filter_preserves_scoring` - Scoring logic preserved
- ✅ `test_temporal_filter_with_limit` - Works with limit parameter

### 5. Missing Test: created_before_boundary_condition
- ⚠️ **Gap Identified**: No test for `created_before` exact boundary condition
- Recommendation: Add in GREEN phase for symmetry with `updated_after_boundary_condition`

---

## New Parameters Required in SmartBulletIndex.retrieve()

```python
def retrieve(
    self,
    # ... existing parameters ...
    created_after: Optional[datetime] = None,   # NEW
    created_before: Optional[datetime] = None,  # NEW
    updated_after: Optional[datetime] = None,   # NEW
) -> List[ScoredBullet]:
```

### Parameter Semantics

| Parameter | Filter Condition | Boundary |
|-----------|------------------|----------|
| `created_after` | `bullet.created_at >= created_after` | Inclusive (>=) |
| `created_before` | `bullet.created_at < created_before` | Exclusive (<) |
| `updated_after` | `bullet.updated_at >= updated_after` | Inclusive (>=) |

**Rationale**:
- `>=` for "after" allows filtering "from this point onward"
- `<` for "before" allows clean range queries (after X and before Y)

---

## Test Data Setup

### Bullet Timeline (in setUp())

```
Timeline (days ago):
│
├─ 10 days ago: bullet1 created
│
├─ 5 days ago:  bullet2 created, bullet1 updated
│
├─ 2 days ago:  bullet3 created, bullet2 updated
│
├─ 1 day ago:   bullet3 updated
│
└─ Today:       bullet4 created/updated
```

### Bullet Characteristics

| Bullet | Created | Updated | Section | Task Type | Domains |
|--------|---------|---------|---------|-----------|---------|
| bullet1 | -10d | -5d | debugging | debugging | - |
| bullet2 | -5d | -2d | debugging | debugging | - |
| bullet3 | -2d | -1d | testing | testing | - |
| bullet4 | 0d | 0d | optimization | optimization | - |
| domain_bullet | -3d | -3d | python | coding | python |

---

## Expected Implementation Strategy (GREEN Phase)

### 1. Add Parameters to Method Signature
```python
def retrieve(
    self,
    query: Optional[str] = None,
    # ... existing parameters ...
    created_after: Optional[datetime] = None,
    created_before: Optional[datetime] = None,
    updated_after: Optional[datetime] = None,
) -> List[ScoredBullet]:
```

### 2. Parse Bullet Timestamps
```python
from datetime import datetime

def _parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Parse ISO 8601 timestamp string to datetime."""
    try:
        return datetime.fromisoformat(timestamp_str)
    except (ValueError, TypeError):
        return None
```

### 3. Apply Filters in Bullet Loop
```python
# Inside the bullet processing loop:

# Temporal filtering
if created_after is not None:
    bullet_created = _parse_timestamp(bullet.created_at)
    if bullet_created is None or bullet_created < created_after:
        continue  # Skip bullet

if created_before is not None:
    bullet_created = _parse_timestamp(bullet.created_at)
    if bullet_created is None or bullet_created >= created_before:
        continue  # Skip bullet

if updated_after is not None:
    bullet_updated = _parse_timestamp(bullet.updated_at)
    if bullet_updated is None or bullet_updated < updated_after:
        continue  # Skip bullet
```

### 4. Update Match Reasons
```python
if created_after or created_before or updated_after:
    match_reasons.append(f"temporal_filter_applied")
```

---

## Potential Issues to Address in GREEN Phase

### 1. Timezone Handling
**Issue**: Bullets use UTC ISO timestamps, but filter parameters could be any timezone.

**Solution**:
```python
def _normalize_to_utc(dt: datetime) -> datetime:
    """Normalize datetime to UTC for comparison."""
    if dt.tzinfo is None:
        # Assume naive datetimes are UTC
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
```

### 2. Invalid Timestamps
**Issue**: What if `bullet.created_at` is corrupted/invalid?

**Solution**: Skip bullets with unparseable timestamps (defensive approach)

### 3. Performance Impact
**Issue**: Parsing timestamps for every bullet could be slow.

**Solution**:
- Parse timestamp once per bullet
- Cache parsed values if needed
- Consider database-level filtering in future (requires migration)

### 4. UnifiedBullet Support
**Issue**: UnifiedBullet may have different timestamp field names.

**Check Required**: Verify UnifiedBullet has `created_at` and `updated_at` fields.

---

## Validation Checklist for GREEN Phase

Before marking GREEN phase complete, verify:

- [ ] All 13 tests pass
- [ ] Timezone-aware datetime handling works
- [ ] None values are properly ignored
- [ ] Boundary conditions use correct operators (>=, <)
- [ ] Invalid timestamps don't crash (defensive parsing)
- [ ] Match reasons include temporal filter info
- [ ] Scoring and sorting still work correctly
- [ ] Existing tests still pass (no regressions)
- [ ] Works with UnifiedBullet (if applicable)
- [ ] Documentation updated (docstring for new params)

---

## Next Steps (GREEN Phase)

1. **Implement temporal filtering logic** in `ace/retrieval.py`
2. **Run tests**: `python -m unittest tests.test_temporal_filtering -v`
3. **Iterate until all tests pass**
4. **Verify no regressions**: `python -m unittest discover -s tests`
5. **Update docstrings** for new parameters
6. **Optional**: Add `created_before_boundary_condition` test for completeness

---

## Dependencies

### External Libraries Used in Tests
- `unittest` (standard library)
- `datetime` (standard library)
- `unittest.mock` (standard library)
- `pytz` (for timezone test - **may need install**)

### Production Code Dependencies
- No new dependencies required
- Uses existing `datetime` handling

---

## Notes

- **TDD Philosophy**: Tests written FIRST to define behavior
- **RED Phase Complete**: All tests fail with expected TypeError
- **No Production Code Modified**: Only test file created
- **Test Quality**: Comprehensive coverage of edge cases and integration scenarios
- **Documentation**: Extensive docstrings explaining expected behavior

---

## Commands to Re-run Tests

```bash
# Run temporal filtering tests only
python -m unittest tests.test_temporal_filtering -v

# Run specific test
python -m unittest tests.test_temporal_filtering.TestTemporalFiltering.test_filter_by_created_after

# Run all tests (after GREEN phase)
python -m unittest discover -s tests
```

---

**TDD Status**: 🔴 RED (failing tests) → Ready for GREEN phase implementation
