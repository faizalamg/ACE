---
phase: "1"
plan: "2"
title: "Implement .ace.json Config Loading"
one_liner: "Workspace config file detection with upward search and caching"
date: "2026-01-21"
type: "completed"
wave: 1
autonomous: true
depends_on:
  - "01-01"
completed_at: "2026-01-21T20:00:00Z"
duration_minutes: 5
tags: ["config", "workspace", "json", "detection"]
---

# Phase 1 Plan 2: Implement .ace.json Config Loading Summary

## Overview

Implemented `.ace/.ace.json` config file loading with automatic workspace detection (upward search), field merging, and caching. Config file values now override environment variable defaults.

## Tech Stack Added

- None (used existing: `pathlib`, `json`, `logging`)

## Tech Patterns Established

- **Upward directory search**: Search from current directory upward until config found
- **Priority chain**: Config file > Environment variable > Default value
- **Module-level caching**: Config file loaded once, cached for performance

## Key Files Modified

| File | Changes |
|------|---------|
| `ace/config.py` | Added config file loading functions and updated ACEConfig |

## Key Functions Added

### `_get_workspace_root()`
- Searches upward from `Path.cwd()` until `.ace/.ace.json` found
- Returns path to workspace root
- Raises `FileNotFoundError` if config not found

### `_load_ace_config()`
- Loads and parses `.ace/.ace.json`
- Returns empty dict if file not found (graceful degradation)
- Logs warnings on parse errors

### `_get_ace_config_field(field, default)`
- Returns config value with priority: config > env var > default
- Uses module-level cache `_ace_config_file_cache`
- Maps field names to `ACE_{FIELD_UPPER}` environment variables

### `update_ace_config_field(field, value)`
- Updates or adds a field to `.ace/.ace.json`
- Invalidates cache on update
- Raises `FileNotFoundError` if config doesn't exist

## Changes to ACEConfig

```python
# Before: Only env var
code_embedding_model: str = field(
    default_factory=lambda: _get_env("ACE_CODE_EMBEDDING_MODEL", "voyage")
)

# After: Config file first, then env var
code_embedding_model: str = field(
    default_factory=lambda: _get_ace_config_field(
        "code_embedding_model",
        _get_env("ACE_CODE_EMBEDDING_PROVIDER", "voyage")
    )
)
```

## Valid Config Fields

- `code_embedding_model`: `str` ("voyage" | "jina" | "nomic" | "local")
- `workspace_name`: `str` (used by ace_onboard)
- `workspace_path`: `str` (used by ace_onboard)
- `collection_name`: `str` (for Qdrant collection naming)
- `onboarded_at`: `str` (timestamp for ace_onboard)

## Verification Results

All tests passed:

| Test | Description | Result |
|------|-------------|--------|
| Workspace root detection | Finds `.ace/.ace.json` from any subdirectory | PASS |
| Config loading | Returns dict with config values | PASS |
| Field retrieval | Returns config file value, then env var, then default | PASS |
| ACEConfig integration | `code_embedding_model` reads from config file first | PASS |
| Priority chain | Config file overrides env var | PASS |
| Config update | `update_ace_config_field()` writes changes | PASS |
| Cache invalidation | Updates are reflected immediately | PASS |

## Deviations from Plan

None - plan executed exactly as written.

## Backward Compatibility

- Existing behavior (env vars only) preserved when no config file exists
- No changes to existing config classes other than `ACEConfig`
- Existing code using env vars continues to work

## Next Phase Readiness

- CONFIG-02 requirement met: Config file values override env vars
- Ready for PLAN 01-03: Implement Provider Switching Logic
