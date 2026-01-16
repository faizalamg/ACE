"""Tests for LangChain integration (ACELangChain).

NOTE: Tests use REAL implementations. Tests requiring LangChain or API keys
will be skipped if not available. NO MOCKING/FAKING/STUBBING.
"""

import os
import pytest
from pathlib import Path
import tempfile

# Skip all tests if langchain not available
pytest.importorskip("langchain_core")

from ace.integrations import ACELangChain, LANGCHAIN_AVAILABLE
from ace import Playbook, Bullet, LiteLLMClient


def check_llm_available():
    """Check if an LLM API is available."""
    return bool(
        os.getenv("ZAI_API_KEY") or
        os.getenv("OPENAI_API_KEY") or
        os.getenv("ANTHROPIC_API_KEY")
    )


LLM_AVAILABLE = check_llm_available()


class TestLangChainAvailability:
    """Test LangChain availability flag."""

    def test_langchain_available(self):
        """LANGCHAIN_AVAILABLE should be True when langchain-core is installed."""
        assert LANGCHAIN_AVAILABLE is True


@pytest.mark.skipif(not LLM_AVAILABLE, reason="No LLM API key available")
class TestACELangChainInitialization:
    """Test ACELangChain initialization with real LLM."""

    def test_basic_initialization(self):
        """Should initialize with minimal parameters using real runnable."""
        from langchain_core.runnables import RunnableLambda

        # Create a simple real runnable that just returns input
        runnable = RunnableLambda(lambda x: f"Response to: {x}")

        agent = ACELangChain(runnable=runnable)

        assert agent.runnable is runnable
        assert agent.is_learning is True  # Default
        assert agent.playbook is not None
        assert agent.reflector is not None
        assert agent.curator is not None
        assert agent.output_parser is not None

    def test_with_playbook_path(self):
        """Should load existing playbook from path."""
        from langchain_core.runnables import RunnableLambda

        runnable = RunnableLambda(lambda x: f"Response: {x}")

        # Create temp playbook
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            playbook_path = f.name

        try:
            # Create and save playbook first
            playbook = Playbook()
            playbook.save_to_file(playbook_path)

            agent = ACELangChain(runnable=runnable, playbook_path=playbook_path)
            assert agent.playbook is not None
        finally:
            Path(playbook_path).unlink()

    def test_with_learning_disabled(self):
        """Should respect is_learning parameter."""
        from langchain_core.runnables import RunnableLambda

        runnable = RunnableLambda(lambda x: x)
        agent = ACELangChain(runnable=runnable, is_learning=False)

        assert agent.is_learning is False

    def test_with_custom_ace_model(self):
        """Should accept custom ace_model parameter."""
        from langchain_core.runnables import RunnableLambda

        runnable = RunnableLambda(lambda x: x)
        agent = ACELangChain(runnable=runnable, ace_model="gpt-4o-mini")

        assert agent.llm is not None
        assert agent.llm.model == "gpt-4o-mini"

    def test_with_custom_output_parser(self):
        """Should accept custom output_parser."""
        from langchain_core.runnables import RunnableLambda

        runnable = RunnableLambda(lambda x: x)

        def custom_parser(result):
            return f"custom: {result}"

        agent = ACELangChain(runnable=runnable, output_parser=custom_parser)

        assert agent.output_parser is custom_parser
        assert agent.output_parser("test") == "custom: test"


@pytest.mark.skipif(not LLM_AVAILABLE, reason="No LLM API key available")
class TestContextInjection:
    """Test _inject_context method with real components."""

    def test_empty_playbook_returns_unchanged(self):
        """Should return input unchanged when playbook is empty."""
        from langchain_core.runnables import RunnableLambda

        runnable = RunnableLambda(lambda x: x)
        agent = ACELangChain(runnable=runnable)

        # String input
        result = agent._inject_context("test input")
        assert result == "test input"

        # Dict input
        result = agent._inject_context({"input": "test"})
        assert result == {"input": "test"}

    def test_string_input_appends_context(self):
        """Should append playbook context to string input."""
        from langchain_core.runnables import RunnableLambda

        runnable = RunnableLambda(lambda x: x)
        agent = ACELangChain(runnable=runnable)

        # Add a bullet
        agent.playbook.add_bullet("general", "Test strategy")

        result = agent._inject_context("What is ACE?")

        assert isinstance(result, str)
        assert "What is ACE?" in result
        assert "Test strategy" in result

    def test_dict_with_input_key_enhances_input_field(self):
        """Should enhance 'input' field in dict."""
        from langchain_core.runnables import RunnableLambda

        runnable = RunnableLambda(lambda x: x)
        agent = ACELangChain(runnable=runnable)

        # Add a bullet
        agent.playbook.add_bullet("general", "Test strategy")

        original = {"input": "Question", "other": "data"}
        result = agent._inject_context(original)

        assert isinstance(result, dict)
        assert "other" in result
        assert result["other"] == "data"
        assert "Question" in result["input"]
        assert "Test strategy" in result["input"]

    def test_dict_without_input_key_adds_playbook_context(self):
        """Should add playbook_context key to dict."""
        from langchain_core.runnables import RunnableLambda

        runnable = RunnableLambda(lambda x: x)
        agent = ACELangChain(runnable=runnable)

        # Add a bullet
        agent.playbook.add_bullet("general", "Test strategy")

        original = {"question": "What?", "data": "value"}
        result = agent._inject_context(original)

        assert isinstance(result, dict)
        assert "question" in result
        assert result["question"] == "What?"
        assert "playbook_context" in result
        assert "Test strategy" in result["playbook_context"]


class TestOutputParser:
    """Test _default_output_parser method - no external dependencies required."""

    def test_string_input_returns_as_is(self):
        """Should return string unchanged."""
        result = ACELangChain._default_output_parser("simple string")
        assert result == "simple string"

    def test_dict_with_output_key(self):
        """Should extract common output keys from dict."""
        result = ACELangChain._default_output_parser({"output": "the answer"})
        assert result == "the answer"

        result = ACELangChain._default_output_parser({"answer": "the answer"})
        assert result == "the answer"

        result = ACELangChain._default_output_parser({"result": "the answer"})
        assert result == "the answer"

    def test_dict_without_common_keys(self):
        """Should convert entire dict to string if no common keys."""
        input_dict = {"custom": "value", "data": 123}
        result = ACELangChain._default_output_parser(input_dict)
        assert "custom" in result
        assert "value" in result

    def test_other_types_convert_to_string(self):
        """Should convert other types to string."""
        result = ACELangChain._default_output_parser(42)
        assert result == "42"

        result = ACELangChain._default_output_parser([1, 2, 3])
        assert "1" in result and "2" in result


@pytest.mark.skipif(not LLM_AVAILABLE, reason="No LLM API key available")
class TestInvokeMethod:
    """Test invoke() method with real components."""

    def test_invoke_calls_runnable(self):
        """Should call runnable.invoke with enhanced input."""
        from langchain_core.runnables import RunnableLambda

        # Create a real runnable that tracks calls
        call_args = []
        def track_invoke(x):
            call_args.append(x)
            return f"response to {x}"

        runnable = RunnableLambda(track_invoke)
        agent = ACELangChain(runnable=runnable, is_learning=False)

        result = agent.invoke("test input")

        assert len(call_args) == 1
        assert "test input" in call_args[0]
        assert "response to" in result

    def test_invoke_with_dict_input(self):
        """Should handle dict input."""
        from langchain_core.runnables import RunnableLambda

        runnable = RunnableLambda(lambda x: {"output": f"answer to {x}"})
        agent = ACELangChain(runnable=runnable, is_learning=False)

        result = agent.invoke({"input": "question"})

        assert isinstance(result, dict)
        assert "output" in result


@pytest.mark.skipif(not LLM_AVAILABLE, reason="No LLM API key available")
class TestAsyncInvokeMethod:
    """Test ainvoke() method with real components."""

    @pytest.mark.asyncio
    async def test_ainvoke_calls_runnable(self):
        """Should call runnable.ainvoke with enhanced input."""
        from langchain_core.runnables import RunnableLambda

        runnable = RunnableLambda(lambda x: f"response to {x}")
        agent = ACELangChain(runnable=runnable, is_learning=False)

        result = await agent.ainvoke("test input")

        assert "response to" in result


@pytest.mark.skipif(not LLM_AVAILABLE, reason="No LLM API key available")
class TestLearningControl:
    """Test learning enable/disable methods with real components."""

    def test_enable_disable_learning(self):
        """Should toggle learning flag."""
        from langchain_core.runnables import RunnableLambda

        runnable = RunnableLambda(lambda x: x)
        agent = ACELangChain(runnable=runnable, is_learning=True)

        assert agent.is_learning is True

        agent.disable_learning()
        assert agent.is_learning is False

        agent.enable_learning()
        assert agent.is_learning is True


@pytest.mark.skipif(not LLM_AVAILABLE, reason="No LLM API key available")
class TestPlaybookOperations:
    """Test playbook save/load methods with real components."""

    def test_save_playbook(self):
        """Should save playbook to file."""
        from langchain_core.runnables import RunnableLambda

        runnable = RunnableLambda(lambda x: x)
        agent = ACELangChain(runnable=runnable)

        agent.playbook.add_bullet("general", "Test bullet")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            agent.save_playbook(temp_path)

            # Verify file exists and is valid JSON
            loaded_playbook = Playbook.load_from_file(temp_path)
            assert len(loaded_playbook.bullets()) == 1
            assert loaded_playbook.bullets()[0].content == "Test bullet"
        finally:
            Path(temp_path).unlink()

    def test_load_playbook(self):
        """Should load playbook from file."""
        from langchain_core.runnables import RunnableLambda

        runnable = RunnableLambda(lambda x: x)
        agent = ACELangChain(runnable=runnable)

        # Create and save a playbook
        temp_playbook = Playbook()
        temp_playbook.add_bullet("general", "Loaded bullet")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            temp_playbook.save_to_file(temp_path)

            # Load it
            agent.load_playbook(temp_path)

            assert len(agent.playbook.bullets()) == 1
            assert agent.playbook.bullets()[0].content == "Loaded bullet"
        finally:
            Path(temp_path).unlink()


@pytest.mark.skipif(not LLM_AVAILABLE, reason="No LLM API key available")
class TestReprMethod:
    """Test __repr__ method with real components."""

    def test_repr_includes_key_info(self):
        """Should include runnable type, strategies count, and learning status."""
        from langchain_core.runnables import RunnableLambda

        runnable = RunnableLambda(lambda x: x)
        agent = ACELangChain(runnable=runnable, is_learning=True)
        agent.playbook.add_bullet("general", "Test")

        repr_str = repr(agent)

        assert "RunnableLambda" in repr_str
        assert "strategies=1" in repr_str
        assert "enabled" in repr_str

    def test_repr_with_learning_disabled(self):
        """Should show disabled in repr when learning is off."""
        from langchain_core.runnables import RunnableLambda

        runnable = RunnableLambda(lambda x: x)
        agent = ACELangChain(runnable=runnable, is_learning=False)

        repr_str = repr(agent)

        assert "disabled" in repr_str


@pytest.mark.skipif(not LLM_AVAILABLE, reason="No LLM API key available")
class TestErrorHandling:
    """Test error handling in invoke with real components."""

    def test_invoke_propagates_runnable_errors(self):
        """Should propagate errors from runnable execution."""
        from langchain_core.runnables import RunnableLambda

        def raise_error(x):
            raise ValueError("Test error")

        runnable = RunnableLambda(raise_error)
        agent = ACELangChain(runnable=runnable, is_learning=False)

        with pytest.raises(ValueError, match="Test error"):
            agent.invoke("test")

    def test_learning_errors_dont_crash(self):
        """Should continue execution even if learning fails."""
        from langchain_core.runnables import RunnableLambda

        runnable = RunnableLambda(lambda x: "response")
        agent = ACELangChain(runnable=runnable, is_learning=True)

        # Add a bullet so learning will be triggered
        agent.playbook.add_bullet("general", "Test strategy")

        # Should not raise, should return result
        # Learning may fail silently, but invoke should complete
        result = agent.invoke("test")
        assert result == "response"


class TestBackwardsCompatibility:
    """Test imports and backward compatibility - no external dependencies required."""

    def test_can_import_from_ace(self):
        """Should be able to import ACELangChain from ace package."""
        from ace import ACELangChain as ImportedACELangChain

        assert ImportedACELangChain is not None

    def test_can_import_from_integrations(self):
        """Should be able to import from ace.integrations."""
        from ace.integrations import ACELangChain as ImportedACELangChain

        assert ImportedACELangChain is not None

    def test_can_check_availability(self):
        """Should be able to check LANGCHAIN_AVAILABLE flag."""
        from ace import LANGCHAIN_AVAILABLE as flag

        assert flag is True
