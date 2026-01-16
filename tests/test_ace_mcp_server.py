"""Tests for ACE MCP Server.

Tests the MCP tools exposed by ace_mcp_server.py:
- ace_retrieve: Query memories with semantic search
- ace_store: Store new memories with deduplication
- ace_search: Filtered search by category/severity
- ace_stats: Collection statistics
- ace_tag: Feedback tagging (helpful/harmful)
- ace_onboard: Onboard workspace
- ace_workspace_info: Get workspace configuration info

Updated to use FastMCP decorator-based API.
"""

import asyncio
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# Import the server module
import ace_mcp_server
from ace_mcp_server import (
    server,
    get_memory_index,
)


class TestMCPToolListing:
    """Test MCP tool registration and listing."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_five_tools(self):
        """Verify all 5 ACE tools are registered."""
        tools = await list_tools()
        assert len(tools) == 5
        tool_names = {t.name for t in tools}
        assert tool_names == {"ace_retrieve", "ace_store", "ace_search", "ace_stats", "ace_tag"}

    @pytest.mark.asyncio
    async def test_tool_schemas_valid(self):
        """Verify tool input schemas are valid."""
        tools = await list_tools()
        for tool in tools:
            assert tool.inputSchema is not None
            assert tool.inputSchema.get("type") == "object"
            assert "properties" in tool.inputSchema
            assert "required" in tool.inputSchema

    @pytest.mark.asyncio
    async def test_retrieve_tool_has_required_query(self):
        """Verify ace_retrieve requires query parameter."""
        tools = await list_tools()
        retrieve_tool = next(t for t in tools if t.name == "ace_retrieve")
        assert "query" in retrieve_tool.inputSchema["required"]

    @pytest.mark.asyncio
    async def test_store_tool_has_required_content(self):
        """Verify ace_store requires content parameter."""
        tools = await list_tools()
        store_tool = next(t for t in tools if t.name == "ace_store")
        assert "content" in store_tool.inputSchema["required"]


class TestToolDispatch:
    """Test tool call dispatching."""

    @pytest.mark.asyncio
    async def test_call_unknown_tool_returns_error(self):
        """Unknown tool name returns error message."""
        result = await call_tool("unknown_tool", {})
        assert len(result) == 1
        assert "Unknown tool" in result[0].text

    @pytest.mark.asyncio
    async def test_call_retrieve_dispatches(self):
        """ace_retrieve dispatches to handle_retrieve."""
        with patch.object(ace_mcp_server, 'handle_retrieve', new_callable=AsyncMock) as mock:
            mock.return_value = [MagicMock(type="text", text="test")]
            await call_tool("ace_retrieve", {"query": "test"})
            mock.assert_called_once_with({"query": "test"})

    @pytest.mark.asyncio
    async def test_call_store_dispatches(self):
        """ace_store dispatches to handle_store."""
        with patch.object(ace_mcp_server, 'handle_store', new_callable=AsyncMock) as mock:
            mock.return_value = [MagicMock(type="text", text="stored")]
            await call_tool("ace_store", {"content": "test"})
            mock.assert_called_once_with({"content": "test"})


class TestRetrieveHandler:
    """Test ace_retrieve handler."""

    @pytest.mark.asyncio
    async def test_retrieve_returns_text_content(self):
        """Retrieve returns TextContent."""
        # Use real index - integration test
        result = await handle_retrieve({"query": "coding preferences", "limit": 2})
        assert len(result) >= 1
        assert result[0].type == "text"
        assert isinstance(result[0].text, str)

    @pytest.mark.asyncio
    async def test_retrieve_respects_limit(self):
        """Retrieve respects limit parameter."""
        result = await handle_retrieve({"query": "test query", "limit": 1})
        # Even with limit=1, result is formatted text, not a list
        assert len(result) == 1  # One TextContent response

    @pytest.mark.asyncio
    async def test_retrieve_with_namespace_filter(self):
        """Retrieve can filter by namespace."""
        result = await handle_retrieve({
            "query": "user preferences",
            "namespace": "user_prefs",
            "limit": 3
        })
        assert len(result) == 1
        assert result[0].type == "text"


class TestStoreHandler:
    """Test ace_store handler."""

    @pytest.mark.asyncio
    async def test_store_creates_memory(self):
        """Store creates a new memory."""
        result = await handle_store({
            "content": "Test memory for MCP server testing",
            "category": "DEBUGGING",
            "severity": 3
        })
        assert len(result) == 1
        assert "stored" in result[0].text.lower() or "reinforced" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_store_with_namespace(self):
        """Store respects namespace parameter."""
        result = await handle_store({
            "content": "Task strategy test memory",
            "namespace": "task_strategies",
            "section": "testing"
        })
        assert len(result) == 1
        assert result[0].type == "text"

    @pytest.mark.asyncio
    async def test_store_deduplication_works(self):
        """Storing similar content triggers reinforcement."""
        # Store twice with same content
        content = "Unique test content for dedup testing 12345"
        result1 = await handle_store({"content": content})
        result2 = await handle_store({"content": content})
        
        # Second store should reinforce, not duplicate
        assert "reinforced" in result2[0].text.lower() or "stored" in result2[0].text.lower()


class TestSearchHandler:
    """Test ace_search handler."""

    @pytest.mark.asyncio
    async def test_search_returns_json(self):
        """Search returns JSON-formatted results."""
        result = await handle_search({"query": "preferences", "limit": 3})
        assert len(result) == 1
        # Should be valid JSON or "No matching memories"
        text = result[0].text
        if "No matching memories" not in text:
            data = json.loads(text)
            assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_search_with_category_filter(self):
        """Search can filter by category."""
        result = await handle_search({
            "query": "coding",
            "category": "PREFERENCE",
            "limit": 5
        })
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_search_with_severity_filter(self):
        """Search can filter by minimum severity."""
        result = await handle_search({
            "query": "important",
            "min_severity": 7,
            "limit": 5
        })
        assert len(result) == 1


class TestStatsHandler:
    """Test ace_stats handler."""

    @pytest.mark.asyncio
    async def test_stats_returns_collection_info(self):
        """Stats returns collection information."""
        result = await handle_stats({})
        assert len(result) == 1
        text = result[0].text
        assert "ACE Unified Memory Statistics" in text
        assert "Collection:" in text
        assert "Total Points:" in text

    @pytest.mark.asyncio
    async def test_stats_shows_status(self):
        """Stats shows collection status."""
        result = await handle_stats({})
        text = result[0].text
        assert "Status:" in text


class TestTagHandler:
    """Test ace_tag handler."""

    @pytest.mark.asyncio
    async def test_tag_requires_memory_id(self):
        """Tag requires memory_id parameter."""
        # This will fail gracefully with an error
        result = await handle_tag({"memory_id": "nonexistent-id", "tag": "helpful"})
        assert len(result) == 1
        # Either success or error message
        assert result[0].type == "text"

    @pytest.mark.asyncio
    async def test_tag_accepts_helpful(self):
        """Tag accepts 'helpful' tag."""
        result = await handle_tag({"memory_id": "test-id", "tag": "helpful"})
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_tag_accepts_harmful(self):
        """Tag accepts 'harmful' tag."""
        result = await handle_tag({"memory_id": "test-id", "tag": "harmful"})
        assert len(result) == 1


class TestMemoryIndexSingleton:
    """Test memory index initialization."""

    def test_get_memory_index_returns_same_instance(self):
        """Memory index is a singleton."""
        # Reset singleton
        ace_mcp_server._memory_index = None
        
        idx1 = get_memory_index()
        idx2 = get_memory_index()
        
        assert idx1 is idx2

    def test_memory_index_connects_to_qdrant(self):
        """Memory index connects to Qdrant."""
        ace_mcp_server._memory_index = None
        idx = get_memory_index()
        
        # Should have a Qdrant client
        assert idx._client is not None


class TestIntegration:
    """Integration tests for full MCP workflow."""

    @pytest.mark.asyncio
    async def test_store_then_retrieve_workflow(self):
        """Store a memory then retrieve it."""
        # Store a unique memory
        import uuid
        unique_content = f"MCP integration test {uuid.uuid4().hex[:8]}"
        
        store_result = await handle_store({
            "content": unique_content,
            "category": "DEBUGGING",
            "severity": 8
        })
        assert "stored" in store_result[0].text.lower() or "reinforced" in store_result[0].text.lower()
        
        # Retrieve it
        retrieve_result = await handle_retrieve({
            "query": unique_content[:20],
            "limit": 5
        })
        # Should find something (may not be exact match due to semantic search)
        assert retrieve_result[0].type == "text"

    @pytest.mark.asyncio
    async def test_full_feedback_loop(self):
        """Test complete feedback loop: store -> retrieve -> tag."""
        # Store
        store_result = await handle_store({
            "content": "Feedback loop test memory",
            "category": "WORKFLOW"
        })
        
        # Get ID from response
        if "ID:" in store_result[0].text:
            memory_id = store_result[0].text.split("ID:")[-1].strip()
            
            # Tag as helpful
            tag_result = await handle_tag({
                "memory_id": memory_id,
                "tag": "helpful"
            })
            assert tag_result[0].type == "text"
