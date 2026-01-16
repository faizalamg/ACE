"""Context injection module for enriching prompts with relevant memories.

This module provides automatic context injection that retrieves relevant
memories/knowledge from ACE and prepends them to prompts.

Configuration:
    ACE_ENABLE_CONTEXT_INJECTION: Enable/disable context injection (default: false)
    ACE_CONTEXT_MAX_ITEMS: Maximum context items to inject (default: 5)
    ACE_CONTEXT_FORMAT: Output format - "plain" or "markdown" (default: plain)

When disabled, returns the original prompt unchanged.
"""

from __future__ import annotations

import os
import logging
from typing import List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ace.unified_memory import MemoryResult

logger = logging.getLogger(__name__)


class ContextInjector:
    """Context injector for enriching prompts with relevant memories.
    
    Retrieves relevant context from ACE unified memory and prepends it to prompts.
    Disabled by default - enable via ACE_ENABLE_CONTEXT_INJECTION=true.
    
    Configuration via environment variables:
        ACE_ENABLE_CONTEXT_INJECTION: "true" to enable, anything else to disable
        ACE_CONTEXT_MAX_ITEMS: Maximum items to retrieve (default: 5)
        ACE_CONTEXT_FORMAT: "plain" or "markdown" (default: plain)
    
    Workspace Isolation:
        For project_specific namespace, retrieval is scoped to current workspace.
        Pass workspace_id to retrieve() for strict workspace separation.
    """
    
    def __init__(self, memory_index: Optional[Any] = None, workspace_id: Optional[str] = None):
        """Initialize injector with config from environment.
        
        Args:
            memory_index: Optional UnifiedMemoryIndex instance. If not provided,
                         will create one lazily when needed.
            workspace_id: Optional workspace identifier for project_specific memory isolation.
                         If not provided, project_specific memories won't be filtered by workspace.
        """
        self._enabled = os.environ.get("ACE_ENABLE_CONTEXT_INJECTION", "false").lower() == "true"
        self._max_items = int(os.environ.get("ACE_CONTEXT_MAX_ITEMS", "5"))
        self._format = os.environ.get("ACE_CONTEXT_FORMAT", "plain").lower()
        self._memory_index = memory_index
        self._workspace_id = workspace_id
    
    def is_enabled(self) -> bool:
        """Check if context injection is enabled."""
        return self._enabled
    
    @property
    def max_items(self) -> int:
        """Maximum context items to retrieve."""
        return self._max_items
    
    @property
    def format(self) -> str:
        """Output format (plain or markdown)."""
        return self._format
    
    def _get_memory_index(self) -> Any:
        """Get or create the memory index."""
        if self._memory_index is None:
            try:
                from ace.unified_memory import UnifiedMemoryIndex
                self._memory_index = UnifiedMemoryIndex()
            except ImportError:
                logger.warning("UnifiedMemoryIndex not available")
                return None
        return self._memory_index
    
    def _retrieve_context(self, query: str) -> List[Any]:
        """Retrieve relevant context for a query.
        
        Args:
            query: The prompt/query to find context for
        
        Returns:
            List of memory results
        """
        index = self._get_memory_index()
        if index is None:
            return []
        
        try:
            # Pass workspace_id for project_specific namespace isolation
            results = index.retrieve(
                query,
                limit=self._max_items,
                workspace_id=self._workspace_id,
            )
            return results
        except Exception as e:
            logger.warning(f"Context retrieval failed: {e}")
            return []
    
    def _format_context(self, memories: List[Any]) -> str:
        """Format memories into context string.
        
        Args:
            memories: List of memory results
        
        Returns:
            Formatted context string
        """
        if not memories:
            return ""
        
        if self._format == "markdown":
            return self._format_markdown(memories)
        else:
            return self._format_plain(memories)
    
    def _format_plain(self, memories: List[Any]) -> str:
        """Format memories as plain text."""
        lines = ["Relevant Context:"]
        for i, mem in enumerate(memories, 1):
            content = getattr(mem, 'content', str(mem))
            category = getattr(mem, 'category', None)
            if category:
                lines.append(f"{i}. [{category}] {content}")
            else:
                lines.append(f"{i}. {content}")
        return "\n".join(lines)
    
    def _format_markdown(self, memories: List[Any]) -> str:
        """Format memories as markdown."""
        lines = ["## Relevant Context"]
        for mem in memories:
            content = getattr(mem, 'content', str(mem))
            category = getattr(mem, 'category', None)
            if category:
                lines.append(f"- **[{category}]** {content}")
            else:
                lines.append(f"- {content}")
        return "\n".join(lines)
    
    def inject(self, prompt: str) -> str:
        """Inject relevant context into a prompt.
        
        Args:
            prompt: Original prompt/query
        
        Returns:
            Prompt with context prepended (or original if disabled/no context)
        """
        if not self._enabled:
            return prompt
        
        memories = self._retrieve_context(prompt)
        if not memories:
            return prompt
        
        context = self._format_context(memories)
        return f"{context}\n\n{prompt}"
    
    async def inject_async(self, prompt: str) -> str:
        """Async version of inject for integration with async pipelines.
        
        Args:
            prompt: Original prompt/query
        
        Returns:
            Prompt with context prepended (or original if disabled/no context)
        """
        # For now, just wraps sync version
        # TODO: Implement truly async retrieval when async retrieval is available
        return self.inject(prompt)


# Convenience function for direct use
def inject_context(prompt: str, workspace_id: Optional[str] = None) -> str:
    """Inject context into a prompt using the default ContextInjector instance.
    
    Args:
        prompt: Original prompt/query
        workspace_id: Optional workspace ID for project_specific namespace isolation
    
    Returns:
        Prompt with context prepended (or original if disabled/no context)
    """
    return ContextInjector(workspace_id=workspace_id).inject(prompt)


async def inject_context_async(prompt: str, workspace_id: Optional[str] = None) -> str:
    """Async version of inject_context.
    
    Args:
        prompt: Original prompt/query
        workspace_id: Optional workspace ID for project_specific namespace isolation
    
    Returns:
        Prompt with context prepended (or original if disabled/no context)
    """
    return await ContextInjector(workspace_id=workspace_id).inject_async(prompt)
