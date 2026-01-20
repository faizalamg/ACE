# Testing

**Analysis Date:** 2026-01-19

## Framework

**Test Runner:** pytest 7.0+
**Async Support:** pytest-asyncio 0.21+
**Coverage:** pytest-cov 4.0+

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── integrations/            # Framework integration tests
├── test_ace.py              # Basic smoke tests
├── test_playbook.py         # Playbook operations
├── test_roles.py            # Generator/Reflector/Curator
├── test_retrieval.py        # SmartBulletIndex
├── test_unified_memory.py   # Unified memory operations
├── test_code_*.py           # Code intelligence tests
└── test_*.py                # ~120+ test files
```

## Test Philosophy

**CRITICAL: NO MOCKING**

From `conftest.py`:
```python
"""NOTE: All tests use REAL implementations. Tests requiring LLM will be
skipped if no API key is available. NO MOCKING/FAKING/STUBBING."""
```

Tests use real implementations with conditional skipping:
```python
@pytest.fixture
def real_llm_client():
    if not LLM_AVAILABLE:
        pytest.skip("No LLM API key available")
    return LiteLLMClient(model=config.model, ...)
```

## Fixtures (conftest.py)

### Core Fixtures
```python
@pytest.fixture
def empty_playbook():
    """Provides an empty Playbook instance."""
    return Playbook()

@pytest.fixture
def sample_playbook():
    """Playbook with sample bullets (general, math, reasoning sections)."""

@pytest.fixture
def real_llm_client():
    """REAL LLM client - skips if no API key."""
```

### JSON Response Fixtures
```python
@pytest.fixture
def generator_valid_json():
    """Valid JSON response for Generator tests."""

@pytest.fixture
def reflector_valid_json():
    """Valid JSON response for Reflector tests."""

@pytest.fixture
def curator_add_operation_json():
    """Valid JSON for Curator ADD operation tests."""
```

## Custom Markers

```python
# Register in conftest.py
@pytest.mark.slow           # Long-running tests
@pytest.mark.integration    # Integration tests
@pytest.mark.unit           # Unit tests
@pytest.mark.requires_llm   # Needs real LLM API
@pytest.mark.requires_qdrant  # Needs Qdrant running
```

**Usage:**
```bash
# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"

# Run only if Qdrant available
pytest -m requires_qdrant
```

## Coverage

**Target:** Not specified, but comprehensive test suite

**Running with coverage:**
```bash
pytest --cov=ace --cov-report=html
```

**Key test areas:**
- Playbook CRUD and delta operations
- Role components (Generator, Reflector, Curator)
- Retrieval and ranking algorithms
- Code chunking and indexing
- Unified memory operations
- MCP server tool handlers

## Test Patterns

### Testing Dataclasses
```python
def test_bullet_creation():
    bullet = Bullet(id="test-1", section="general", content="test")
    assert bullet.helpful == 0
    assert bullet.harmful == 0
```

### Testing with Real LLM
```python
def test_generator_with_real_llm(real_llm_client, sample_playbook):
    generator = Generator(real_llm_client)
    output = generator.generate(
        question="What is 2+2?",
        context="Simple math",
        playbook=sample_playbook,
    )
    assert "4" in output.final_answer
```

### Testing Async Operations
```python
@pytest.mark.asyncio
async def test_async_retrieval():
    index = AsyncSmartBulletIndex()
    results = await index.retrieve(query="test")
    assert isinstance(results, list)
```

---

*Testing analysis: 2026-01-19*
