"""Tests for ACE Code-Aware Bullet Enrichment module (Phase 2B).

This module tests the CodeAwareEnricher class which enriches bullets
with code-specific metadata for improved code-specific queries.

TDD: Tests written first, implementation follows.
"""

import pytest


# Sample code fixtures for testing
PYTHON_CODE_SAMPLE = '''
def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)


class DataProcessor:
    """Process data with various transformations."""

    def __init__(self, data: list):
        self.data = data

    def transform(self, func):
        """Apply transformation function to data."""
        return [func(x) for x in self.data]
'''

TYPESCRIPT_CODE_SAMPLE = '''
interface ApiResponse {
    data: any;
    status: number;
}

async function fetchData(url: string): Promise<ApiResponse> {
    const response = await fetch(url);
    return response.json();
}

class UserService {
    async getUser(id: number): Promise<User> {
        return await fetchData(`/api/users/${id}`);
    }
}
'''

MARKDOWN_WITH_CODE = '''
# How to handle errors

When debugging, always check the logs first.

```python
def handle_error(error: Exception) -> None:
    logger.error(f"Error occurred: {error}")
    raise CustomException(error)
```

This pattern ensures errors are logged before re-raising.

```typescript
function validateInput(input: string): boolean {
    return input.length > 0;
}
```
'''


class TestCodeAwareEnricherInit:
    """Tests for CodeAwareEnricher initialization."""

    def test_enricher_initialization(self):
        """Test that CodeAwareEnricher initializes correctly."""
        from ace.code_enrichment import CodeAwareEnricher

        enricher = CodeAwareEnricher()
        assert enricher is not None

    def test_enricher_with_custom_analyzer(self):
        """Test CodeAwareEnricher with provided analyzer."""
        from ace.code_enrichment import CodeAwareEnricher
        from unittest.mock import MagicMock

        mock_analyzer = MagicMock()
        enricher = CodeAwareEnricher(code_analyzer=mock_analyzer)
        assert enricher._analyzer is mock_analyzer


class TestEnrichBulletWithCode:
    """Tests for enriching bullets with code context."""

    def test_enrich_bullet_with_code_context(self):
        """Test enriching a bullet with code context."""
        from ace.code_enrichment import CodeAwareEnricher
        from ace.playbook import Bullet

        enricher = CodeAwareEnricher()
        bullet = Bullet(
            id="test-001",
            section="debugging",
            content="Use fibonacci for sequence calculations"
        )

        enriched = enricher.enrich_bullet_with_code(bullet, PYTHON_CODE_SAMPLE)

        assert enriched is not None
        assert enriched.id == "test-001"
        # Should have code-derived trigger patterns
        assert any("fibonacci" in t.lower() for t in enriched.trigger_patterns)

    def test_enriched_bullet_has_domains(self):
        """Test that enriched bullet has detected language in domains."""
        from ace.code_enrichment import CodeAwareEnricher
        from ace.playbook import Bullet

        enricher = CodeAwareEnricher()
        bullet = Bullet(
            id="test-002",
            section="coding",
            content="Data processing pattern"
        )

        enriched = enricher.enrich_bullet_with_code(bullet, PYTHON_CODE_SAMPLE)

        assert "python" in enriched.domains or "code" in enriched.domains

    def test_enriched_bullet_has_task_types(self):
        """Test that enriched bullet has code-related task types."""
        from ace.code_enrichment import CodeAwareEnricher
        from ace.playbook import Bullet

        enricher = CodeAwareEnricher()
        bullet = Bullet(
            id="test-003",
            section="debugging",
            content="Error handling pattern"
        )

        enriched = enricher.enrich_bullet_with_code(bullet, PYTHON_CODE_SAMPLE)

        # Should have code-related task type
        assert len(enriched.task_types) > 0

    def test_enriched_bullet_preserves_original_content(self):
        """Test that original bullet content is preserved."""
        from ace.code_enrichment import CodeAwareEnricher
        from ace.playbook import Bullet

        enricher = CodeAwareEnricher()
        original_content = "Important debugging strategy"
        bullet = Bullet(
            id="test-004",
            section="strategies",
            content=original_content
        )

        enriched = enricher.enrich_bullet_with_code(bullet, PYTHON_CODE_SAMPLE)

        assert enriched.content == original_content


class TestGenerateCodeTriggers:
    """Tests for generating trigger patterns from code."""

    def test_generate_code_triggers_from_python(self):
        """Test generating triggers from Python code."""
        from ace.code_enrichment import CodeAwareEnricher

        enricher = CodeAwareEnricher()
        triggers = enricher.generate_code_triggers(PYTHON_CODE_SAMPLE, "python")

        assert isinstance(triggers, list)
        assert len(triggers) > 0
        # Should include function/class names
        trigger_lower = [t.lower() for t in triggers]
        assert any("fibonacci" in t for t in trigger_lower)
        assert any("dataprocessor" in t or "data_processor" in t or "processor" in t for t in trigger_lower)

    def test_generate_code_triggers_from_typescript(self):
        """Test generating triggers from TypeScript code."""
        from ace.code_enrichment import CodeAwareEnricher

        enricher = CodeAwareEnricher()
        triggers = enricher.generate_code_triggers(TYPESCRIPT_CODE_SAMPLE, "typescript")

        assert isinstance(triggers, list)
        trigger_lower = [t.lower() for t in triggers]
        assert any("fetchdata" in t or "fetch_data" in t or "fetch" in t for t in trigger_lower)
        assert any("userservice" in t or "user_service" in t or "user" in t for t in trigger_lower)

    def test_generate_code_triggers_includes_key_terms(self):
        """Test that triggers include key programming terms."""
        from ace.code_enrichment import CodeAwareEnricher

        enricher = CodeAwareEnricher()
        triggers = enricher.generate_code_triggers(PYTHON_CODE_SAMPLE, "python")

        # Should extract meaningful code terms
        assert len(triggers) >= 2


class TestDetectCodeLanguage:
    """Tests for detecting programming language in text."""

    def test_detect_python_language(self):
        """Test detecting Python from code patterns."""
        from ace.code_enrichment import CodeAwareEnricher

        enricher = CodeAwareEnricher()
        language = enricher.detect_code_language(PYTHON_CODE_SAMPLE)

        assert language == "python"

    def test_detect_typescript_language(self):
        """Test detecting TypeScript from code patterns."""
        from ace.code_enrichment import CodeAwareEnricher

        enricher = CodeAwareEnricher()
        language = enricher.detect_code_language(TYPESCRIPT_CODE_SAMPLE)

        assert language in ["typescript", "javascript"]

    def test_detect_language_returns_none_for_plain_text(self):
        """Test that plain text returns None."""
        from ace.code_enrichment import CodeAwareEnricher

        enricher = CodeAwareEnricher()
        plain_text = "This is just regular text without any code."
        language = enricher.detect_code_language(plain_text)

        assert language is None

    def test_detect_language_from_markers(self):
        """Test detecting language from common markers."""
        from ace.code_enrichment import CodeAwareEnricher

        enricher = CodeAwareEnricher()

        # Python markers
        assert enricher.detect_code_language("def foo(): pass") == "python"
        assert enricher.detect_code_language("import os") == "python"

        # JavaScript/TypeScript markers
        js_code = "const x = () => { return 1; }"
        assert enricher.detect_code_language(js_code) in ["javascript", "typescript"]


class TestExtractCodeBlocks:
    """Tests for extracting code blocks from markdown-style content."""

    def test_extract_code_blocks_from_markdown(self):
        """Test extracting code blocks from markdown."""
        from ace.code_enrichment import CodeAwareEnricher

        enricher = CodeAwareEnricher()
        blocks = enricher.extract_code_blocks(MARKDOWN_WITH_CODE)

        assert isinstance(blocks, list)
        assert len(blocks) >= 2

        # Each block should be (language, code) tuple
        for block in blocks:
            assert isinstance(block, tuple)
            assert len(block) == 2

    def test_extract_code_blocks_with_language_hints(self):
        """Test that language hints are extracted from code blocks."""
        from ace.code_enrichment import CodeAwareEnricher

        enricher = CodeAwareEnricher()
        blocks = enricher.extract_code_blocks(MARKDOWN_WITH_CODE)

        languages = [block[0] for block in blocks]
        assert "python" in languages
        assert "typescript" in languages

    def test_extract_code_blocks_empty_input(self):
        """Test extracting from text without code blocks."""
        from ace.code_enrichment import CodeAwareEnricher

        enricher = CodeAwareEnricher()
        blocks = enricher.extract_code_blocks("No code here, just text.")

        assert blocks == []

    def test_extract_code_blocks_without_language(self):
        """Test extracting code blocks without language specification."""
        from ace.code_enrichment import CodeAwareEnricher

        enricher = CodeAwareEnricher()
        text = """
Here's some code:

```
function test() { return true; }
```
"""
        blocks = enricher.extract_code_blocks(text)

        assert len(blocks) >= 1
        # Should have empty or detected language
        lang, code = blocks[0]
        assert "function" in code


class TestCodeEnrichmentIntegration:
    """Integration tests for code enrichment with playbook."""

    def test_enrichment_produces_valid_enriched_bullet(self):
        """Test that enrichment produces a valid EnrichedBullet."""
        from ace.code_enrichment import CodeAwareEnricher
        from ace.playbook import Bullet, EnrichedBullet

        enricher = CodeAwareEnricher()
        bullet = Bullet(
            id="integration-001",
            section="patterns",
            content="Recursive pattern for calculations"
        )

        enriched = enricher.enrich_bullet_with_code(bullet, PYTHON_CODE_SAMPLE)

        assert isinstance(enriched, EnrichedBullet)
        assert enriched.id == bullet.id
        assert enriched.section == bullet.section
        assert enriched.content == bullet.content

    def test_enrichment_with_existing_enriched_bullet(self):
        """Test enriching an already EnrichedBullet."""
        from ace.code_enrichment import CodeAwareEnricher
        from ace.playbook import EnrichedBullet

        enricher = CodeAwareEnricher()
        bullet = EnrichedBullet(
            id="enriched-001",
            section="patterns",
            content="Existing strategy",
            task_types=["reasoning"],
            trigger_patterns=["existing_pattern"]
        )

        enriched = enricher.enrich_bullet_with_code(bullet, PYTHON_CODE_SAMPLE)

        # Should preserve existing patterns and add new ones
        assert "existing_pattern" in enriched.trigger_patterns
        assert len(enriched.trigger_patterns) > 1


class TestEmbeddingTextGeneration:
    """Tests for generating optimized embedding text."""

    def test_embedding_text_includes_code_terms(self):
        """Test that embedding text includes code-derived terms."""
        from ace.code_enrichment import CodeAwareEnricher
        from ace.playbook import Bullet

        enricher = CodeAwareEnricher()
        bullet = Bullet(
            id="embed-001",
            section="algorithms",
            content="Fibonacci calculation"
        )

        enriched = enricher.enrich_bullet_with_code(bullet, PYTHON_CODE_SAMPLE)

        # Should have embedding_text set
        assert enriched.embedding_text is not None
        # Should include relevant terms
        assert "fibonacci" in enriched.embedding_text.lower() or \
               "calculation" in enriched.embedding_text.lower()
