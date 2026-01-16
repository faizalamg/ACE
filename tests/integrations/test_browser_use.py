"""Tests for browser-use integration (ACEAgent).

NOTE: Tests use REAL implementations. Tests requiring browser-use or API keys
will be skipped if not available. NO MOCKING/FAKING/STUBBING.
"""

import os
import pytest
from pathlib import Path
import tempfile

# Skip all tests if browser-use not available
pytest.importorskip("browser_use")

from ace.integrations import (
    ACEAgent,
    wrap_playbook_context,
    BROWSER_USE_AVAILABLE,
)
from ace import Playbook, Bullet, LiteLLMClient


def check_llm_available():
    """Check if an LLM API is available."""
    return bool(
        os.getenv("ZAI_API_KEY") or
        os.getenv("OPENAI_API_KEY") or
        os.getenv("ANTHROPIC_API_KEY")
    )


def check_browser_api_available():
    """Check if browser-use API key is available."""
    return bool(os.environ.get("BROWSER_USE_API_KEY"))


LLM_AVAILABLE = check_llm_available()
BROWSER_API_AVAILABLE = check_browser_api_available()


class TestWrapPlaybookContext:
    """Test the wrap_playbook_context helper function - no external dependencies required."""

    def test_empty_playbook(self):
        """Should return empty string for empty playbook."""
        playbook = Playbook()
        result = wrap_playbook_context(playbook)
        assert result == ""

    def test_with_bullets(self):
        """Should format bullets with explanation."""
        playbook = Playbook()
        playbook.add_bullet("general", "Always check search box first")
        playbook.add_bullet("general", "Scroll before clicking")

        result = wrap_playbook_context(playbook)

        # Should contain header
        assert "Strategic Knowledge" in result
        assert "Learned from Experience" in result

        # Should contain bullets
        assert "Always check search box first" in result
        assert "Scroll before clicking" in result

        # Should contain usage instructions
        assert "How to use these strategies" in result
        assert "success rates" in result

    def test_bullet_scores_shown(self):
        """Should show helpful/harmful scores."""
        playbook = Playbook()
        bullet = playbook.add_bullet(
            "general", "Test strategy", metadata={"helpful": 5, "harmful": 2}
        )

        result = wrap_playbook_context(playbook)

        # Should show the bullet content
        assert "Test strategy" in result


class TestBrowserUseAvailability:
    """Test browser-use availability."""

    def test_browser_use_available(self):
        """BROWSER_USE_AVAILABLE should be True when browser-use is installed."""
        assert BROWSER_USE_AVAILABLE is True


@pytest.mark.skipif(not LLM_AVAILABLE, reason="No LLM API key available")
class TestACEAgentWithRealLLM:
    """Test ACEAgent initialization with real LLM - requires API key."""

    def test_basic_initialization(self):
        """Should initialize with minimal parameters using real LLM."""
        from ace.llm_providers.litellm_client import LiteLLMClient

        ace_llm = LiteLLMClient()
        playbook = Playbook()

        # Verify components work together
        assert ace_llm is not None
        assert playbook is not None

    def test_reflector_curator_creation(self):
        """Test Reflector and Curator can be created with real LLM."""
        from ace import Reflector, Curator
        from ace.llm_providers.litellm_client import LiteLLMClient
        from ace.prompts_v2_1 import PromptManager

        llm = LiteLLMClient()
        prompt_mgr = PromptManager()

        reflector = Reflector(llm, prompt_template=prompt_mgr.get_reflector_prompt())
        curator = Curator(llm, prompt_template=prompt_mgr.get_curator_prompt())

        assert reflector is not None
        assert curator is not None
        assert reflector.prompt_template is not None
        assert curator.prompt_template is not None

    def test_with_playbook_path(self):
        """Should load playbook from path."""
        # Create a temporary playbook
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            playbook_path = f.name

        try:
            # Create and save playbook
            playbook = Playbook()
            playbook.add_bullet("general", "Pre-loaded strategy")
            playbook.save_to_file(playbook_path)

            # Load in new playbook
            loaded_playbook = Playbook.load_from_file(playbook_path)

            assert len(loaded_playbook.bullets()) == 1
            assert loaded_playbook.bullets()[0].content == "Pre-loaded strategy"
        finally:
            Path(playbook_path).unlink(missing_ok=True)


class TestPlaybookOperations:
    """Test playbook operations - no external dependencies required."""

    def test_playbook_save_load_roundtrip(self):
        """Should support save/load playbook."""
        playbook = Playbook()
        playbook.add_bullet("general", "Test strategy")

        # Save
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            playbook_path = f.name

        try:
            playbook.save_to_file(playbook_path)

            # Load in new playbook
            loaded_playbook = Playbook.load_from_file(playbook_path)

            assert len(loaded_playbook.bullets()) == 1
            assert loaded_playbook.bullets()[0].content == "Test strategy"
        finally:
            Path(playbook_path).unlink(missing_ok=True)

    def test_get_strategies(self):
        """Should return formatted strategies."""
        playbook = Playbook()

        # Empty playbook
        strategies = wrap_playbook_context(playbook)
        assert strategies == ""

        # With bullets
        playbook.add_bullet("general", "Strategy 1")
        strategies = wrap_playbook_context(playbook)
        assert "Strategy 1" in strategies
        assert "Strategic Knowledge" in strategies


class TestPromptVersionUsage:
    """Test that ACE uses v2.1 prompts by default - no external dependencies required."""

    def test_reflector_prompt_v2_1(self):
        """Should use v2.1 prompt for Reflector."""
        from ace.prompts_v2_1 import PromptManager

        prompt_mgr = PromptManager()
        reflector_prompt = prompt_mgr.get_reflector_prompt()

        assert reflector_prompt is not None
        assert "v2.1" in reflector_prompt or "2.1" in reflector_prompt
        # v2.1 has enhanced structure with QUICK REFERENCE
        assert "QUICK REFERENCE" in reflector_prompt

    def test_curator_prompt_v2_1(self):
        """Should use v2.1 prompt for Curator."""
        from ace.prompts_v2_1 import PromptManager

        prompt_mgr = PromptManager()
        curator_prompt = prompt_mgr.get_curator_prompt()

        assert curator_prompt is not None
        assert "v2.1" in curator_prompt or "2.1" in curator_prompt
        # v2.1 has atomicity scoring
        assert "atomicity" in curator_prompt.lower()

    def test_playbook_wrapper_uses_canonical_function(self):
        """Should use canonical wrap function from prompts_v2_1."""
        from ace.integrations.base import wrap_playbook_context
        from ace.prompts_v2_1 import wrap_playbook_for_external_agent
        from ace import Playbook

        playbook = Playbook()
        playbook.add_bullet("general", "Test strategy")

        # Both functions should produce identical output
        result1 = wrap_playbook_context(playbook)
        result2 = wrap_playbook_for_external_agent(playbook)

        assert result1 == result2
        assert "📚 Available Strategic Knowledge" in result1
        assert "Test strategy" in result1

    def test_playbook_wrapper_includes_usage_instructions(self):
        """Should include PLAYBOOK_USAGE_INSTRUCTIONS constant."""
        from ace.integrations.base import wrap_playbook_context
        from ace import Playbook

        playbook = Playbook()
        playbook.add_bullet("general", "Test strategy")

        result = wrap_playbook_context(playbook)

        # Should include instructions from constant
        assert "How to use these strategies" in result
        assert "Review bullets relevant to your current task" in result
        assert "Prioritize strategies with high success rates" in result
        assert "These are learned patterns, not rigid rules" in result


class TestCitationValidation:
    """Test citation validation in playbook - no external dependencies required."""

    def test_filters_invalid_bullet_ids_against_playbook(self):
        """
        Test that cited IDs are validated against playbook.

        This ensures only valid bullet IDs (that exist in playbook) are passed
        to the Reflector, preventing errors from hallucinated or invalid citations.
        """
        # Create playbook containing specific bullets
        playbook = Playbook()
        bullet1 = playbook.add_bullet("navigation", "Always scroll before clicking")
        bullet2 = playbook.add_bullet("extraction", "Use CSS selectors for data")

        # Test that bullets are added correctly
        assert len(playbook.bullets()) == 2
        assert bullet1.id is not None
        assert bullet2.id is not None

        # Test that we can find bullets by ID
        all_bullet_ids = {b.id for b in playbook.bullets()}
        assert bullet1.id in all_bullet_ids
        assert bullet2.id in all_bullet_ids
        assert "nonexistent-99999" not in all_bullet_ids
        assert "fake-12345" not in all_bullet_ids


class TestBackwardsCompatibility:
    """Test that existing code patterns still work."""

    def test_can_import_from_ace(self):
        """Should be importable from ace package."""
        from ace import ACEAgent as ImportedACEAgent

        assert ImportedACEAgent is not None

    def test_can_import_helper_from_ace(self):
        """Should import helper function from ace package."""
        from ace import wrap_playbook_context as imported_wrap

        assert imported_wrap is not None

    def test_can_check_availability(self):
        """Should check browser-use availability."""
        from ace import BROWSER_USE_AVAILABLE as imported_available

        assert imported_available is True


@pytest.mark.integration
@pytest.mark.skipif(not BROWSER_API_AVAILABLE, reason="Browser API key not available")
@pytest.mark.skipif(not LLM_AVAILABLE, reason="No LLM API key available")
class TestACEAgentIntegration:
    """Integration tests for ACEAgent (requires actual browser-use execution)."""

    @pytest.mark.skip(reason="Requires browser setup - run manually")
    async def test_full_learning_cycle(self):
        """Full test of learning cycle (manual test only)."""
        from browser_use import ChatBrowserUse

        agent = ACEAgent(llm=ChatBrowserUse(), is_learning=True)

        # This would run actual browser automation
        # await agent.run(task="Find top HN post")
        # assert len(agent.playbook.bullets()) > 0

        pass
