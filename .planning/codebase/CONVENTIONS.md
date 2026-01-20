# Coding Conventions

**Analysis Date:** 2026-01-19

## Code Style

### Formatting
- **Formatter:** Black 23.0+
- **Line Length:** 88 characters (Black default)
- **Quotes:** Double quotes for strings
- **Imports:** Sorted with isort compatibility

### Type Hints
- **Enforcement:** mypy 1.0+ with strict mode
- **Style:** Full type annotations on public APIs
- **Optional:** `Optional[T]` for nullable types
- **Generics:** Use `List`, `Dict`, `Tuple` from typing

```python
# Example pattern
def retrieve(
    self,
    query: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[ScoredBullet]:
```

## Naming Patterns

### Classes
```python
# Dataclasses for data structures
@dataclass
class GeneratorOutput:
    reasoning: str
    final_answer: str
    bullet_ids: List[str]
    raw: Dict[str, Any]

# Classes for behavior
class Generator:
    def __init__(self, llm: LLMClient) -> None:
    def generate(self, *, question: str, ...) -> GeneratorOutput:
```

### Functions
```python
# Public methods - keyword-only args for clarity
def generate(self, *, question: str, context: Optional[str]) -> Output:

# Private methods - underscore prefix
def _apply_operation(self, operation: DeltaOperation) -> None:

# Factory/accessor pattern
def get_elf_config() -> ELFConfig:
```

## Error Handling

### Pattern: Fast-fail with context
```python
try:
    data = json.loads(response.text)
except json.JSONDecodeError as exc:
    raise ValueError(f"LLM response is not valid JSON: {exc}") from exc
```

### Pattern: Graceful degradation for optional features
```python
try:
    from .observability import OpikIntegration
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OpikIntegration = None
    OBSERVABILITY_AVAILABLE = False
```

### Pattern: Retry with exponential backoff
```python
for attempt in range(self.max_retries):
    try:
        return self._attempt_operation()
    except ValueError as err:
        if attempt + 1 >= self.max_retries:
            raise
        prompt = base_prompt + self.retry_prompt
```

## Documentation

### Docstrings
```python
def retrieve(
    self,
    query: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[ScoredBullet]:
    """Retrieve bullets matching the given criteria.

    Args:
        query: Natural language query for trigger pattern matching
        limit: Maximum number of results to return

    Returns:
        List of ScoredBullet objects, sorted by relevance score descending.
    """
```

### Module Docstrings
```python
"""Smart retrieval system for purpose-aware bullet retrieval.

This module provides intelligent retrieval of bullets from a playbook
using semantic scaffolding metadata for purpose-aware, multi-dimensional filtering.

ELF-Inspired Features (when enabled via config):
- Confidence Decay: Older knowledge scores lower over time
- Golden Rules: High-performing strategies get score boost
"""
```

## Patterns

### Lazy Loading
```python
@property
def analyzer(self):
    """Lazy-load CodeAnalyzer on first use."""
    if self._analyzer is None:
        from ace.code_analysis import CodeAnalyzer
        self._analyzer = CodeAnalyzer()
    return self._analyzer
```

### Feature Flags
```python
from .features import has_opik, has_litellm

if has_litellm():
    from .llm_providers import LiteLLMClient
```

### Configuration Hierarchy
```python
# 1. Environment variable (highest priority)
# 2. Config file
# 3. Default value
def get_config_value(env_var: str, default: Any) -> Any:
    return os.getenv(env_var, default)
```

---

*Conventions analysis: 2026-01-19*
