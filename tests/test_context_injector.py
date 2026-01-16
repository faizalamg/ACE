"""Tests for context injection module.

TDD RED phase - tests written before implementation.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestContextInjectorConfig:
    """Test configuration-based enable/disable of context injection."""
    
    def test_injector_disabled_by_default(self):
        """Context injection should be disabled by default."""
        import os
        os.environ.pop("ACE_ENABLE_CONTEXT_INJECTION", None)
        from ace.context_injector import ContextInjector
        injector = ContextInjector()
        assert injector.is_enabled() is False
    
    def test_injector_respects_config_toggle(self):
        """Injector should check config before processing."""
        with patch.dict("os.environ", {"ACE_ENABLE_CONTEXT_INJECTION": "true"}):
            from ace.context_injector import ContextInjector
            injector = ContextInjector()
            assert injector.is_enabled() is True
        
        with patch.dict("os.environ", {"ACE_ENABLE_CONTEXT_INJECTION": "false"}):
            from ace.context_injector import ContextInjector
            injector = ContextInjector()
            assert injector.is_enabled() is False
    
    def test_injector_returns_original_when_disabled(self):
        """When disabled, injector should return original prompt unchanged."""
        with patch.dict("os.environ", {"ACE_ENABLE_CONTEXT_INJECTION": "false"}):
            from ace.context_injector import ContextInjector
            injector = ContextInjector()
            
            prompt = "How does the authentication work?"
            result = injector.inject(prompt)
            
            assert result == prompt


class TestContextInjectorBasic:
    """Test basic context injection functionality."""
    
    def test_inject_adds_context_section(self):
        """Should add a context section when relevant memories exist."""
        with patch.dict("os.environ", {"ACE_ENABLE_CONTEXT_INJECTION": "true"}):
            from ace.context_injector import ContextInjector
            
            # Mock the retrieval
            mock_memories = [
                MagicMock(content="Use JWT for authentication", category="SECURITY"),
                MagicMock(content="Hash passwords with bcrypt", category="SECURITY"),
            ]
            
            with patch.object(ContextInjector, '_retrieve_context', return_value=mock_memories):
                injector = ContextInjector()
                
                prompt = "How should I implement login?"
                result = injector.inject(prompt)
                
                # Should contain context section
                assert "Context:" in result or "context:" in result.lower()
                assert "JWT" in result or "authentication" in result.lower()
    
    def test_inject_preserves_original_prompt(self):
        """Injected result should still contain the original prompt."""
        with patch.dict("os.environ", {"ACE_ENABLE_CONTEXT_INJECTION": "true"}):
            from ace.context_injector import ContextInjector
            
            mock_memories = [MagicMock(content="Test memory")]
            
            with patch.object(ContextInjector, '_retrieve_context', return_value=mock_memories):
                injector = ContextInjector()
                
                prompt = "Explain the caching strategy"
                result = injector.inject(prompt)
                
                assert prompt in result
    
    def test_inject_no_context_returns_original(self):
        """When no relevant context found, return original prompt."""
        with patch.dict("os.environ", {"ACE_ENABLE_CONTEXT_INJECTION": "true"}):
            from ace.context_injector import ContextInjector
            
            with patch.object(ContextInjector, '_retrieve_context', return_value=[]):
                injector = ContextInjector()
                
                prompt = "Some random question"
                result = injector.inject(prompt)
                
                assert result == prompt


class TestContextInjectorConfigOptions:
    """Test configurable injection parameters."""
    
    def test_max_context_items_configurable(self):
        """max_context_items should be configurable via environment."""
        with patch.dict("os.environ", {
            "ACE_ENABLE_CONTEXT_INJECTION": "true",
            "ACE_CONTEXT_MAX_ITEMS": "3"
        }):
            from ace.context_injector import ContextInjector
            injector = ContextInjector()
            assert injector.max_items == 3
    
    def test_context_format_configurable(self):
        """Context format should be configurable."""
        with patch.dict("os.environ", {
            "ACE_ENABLE_CONTEXT_INJECTION": "true",
            "ACE_CONTEXT_FORMAT": "markdown"
        }):
            from ace.context_injector import ContextInjector
            injector = ContextInjector()
            assert injector.format == "markdown"


class TestContextInjectorAsync:
    """Test async context injection."""
    
    @pytest.mark.asyncio
    async def test_async_inject(self):
        """Should support async injection for integration with async pipelines."""
        with patch.dict("os.environ", {"ACE_ENABLE_CONTEXT_INJECTION": "true"}):
            from ace.context_injector import ContextInjector
            
            mock_memories = [MagicMock(content="Async pattern")]
            
            with patch.object(ContextInjector, '_retrieve_context', return_value=mock_memories):
                injector = ContextInjector()
                
                prompt = "How to use async/await?"
                result = await injector.inject_async(prompt)
                
                assert prompt in result


class TestContextInjectorFormatting:
    """Test different context formatting options."""
    
    def test_plain_format(self):
        """Plain format should use simple text."""
        with patch.dict("os.environ", {
            "ACE_ENABLE_CONTEXT_INJECTION": "true",
            "ACE_CONTEXT_FORMAT": "plain"
        }):
            from ace.context_injector import ContextInjector
            
            mock_memories = [MagicMock(content="Memory 1", category="CAT1")]
            
            with patch.object(ContextInjector, '_retrieve_context', return_value=mock_memories):
                injector = ContextInjector()
                result = injector.inject("Test prompt")
                
                # Should not have markdown formatting
                assert "```" not in result
    
    def test_markdown_format(self):
        """Markdown format should use markdown headers/lists."""
        with patch.dict("os.environ", {
            "ACE_ENABLE_CONTEXT_INJECTION": "true",
            "ACE_CONTEXT_FORMAT": "markdown"
        }):
            from ace.context_injector import ContextInjector
            
            mock_memories = [MagicMock(content="Memory 1", category="CAT1")]
            
            with patch.object(ContextInjector, '_retrieve_context', return_value=mock_memories):
                injector = ContextInjector()
                result = injector.inject("Test prompt")
                
                # Should have markdown formatting
                assert "#" in result or "-" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
