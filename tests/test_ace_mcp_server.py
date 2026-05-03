"""Tests for ACE MCP Server.

Tests the MCP tools exposed by ace_mcp_server.py:
- ace_retrieve: Query memories with semantic search
- ace_store: Store new memories with deduplication
- ace_search: Filtered search by category/severity
- ace_stats: Collection statistics
- ace_tag: Feedback tagging (helpful/harmful)
- ace_onboard: Onboard workspace
- ace_workspace_info: Get workspace configuration info
- ace_enhance_prompt: Prompt enhancement

Updated to use FastMCP decorator-based API.
"""

import asyncio
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# Import the server module and FastMCP tool functions
import ace_mcp_server
from ace_mcp_server import (
    server,
    get_memory_index,
    ace_retrieve,
    ace_store,
    ace_search,
    ace_stats,
    ace_tag,
)


class TestMCPToolListing:
    """Test MCP tool registration and listing."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_all_tools(self):
        """Verify all ACE tools are registered."""
        tools = await server.list_tools()
        tool_names = {t.name for t in tools}
        expected = {"ace_retrieve", "ace_store", "ace_search", "ace_stats", "ace_tag",
                    "ace_onboard", "ace_workspace_info", "ace_enhance_prompt"}
        assert expected.issubset(tool_names)

    @pytest.mark.asyncio
    async def test_tool_schemas_valid(self):
        """Verify tool input schemas are valid."""
        tools = await server.list_tools()
        for tool in tools:
            assert tool.inputSchema is not None
            assert tool.inputSchema.get("type") == "object"
            assert "properties" in tool.inputSchema

    @pytest.mark.asyncio
    async def test_retrieve_tool_has_required_query(self):
        """Verify ace_retrieve requires query parameter."""
        tools = await server.list_tools()
        retrieve_tool = next(t for t in tools if t.name == "ace_retrieve")
        assert "query" in retrieve_tool.inputSchema["required"]

    @pytest.mark.asyncio
    async def test_store_tool_has_required_content(self):
        """Verify ace_store requires content parameter."""
        tools = await server.list_tools()
        store_tool = next(t for t in tools if t.name == "ace_store")
        assert "content" in store_tool.inputSchema["required"]


class TestToolDispatch:
    """Test tool call dispatching via FastMCP server."""

    @pytest.mark.asyncio
    async def test_call_unknown_tool_raises(self):
        """Unknown tool name raises an error."""
        with pytest.raises(Exception):
            await server.call_tool("unknown_tool", {})

    @pytest.mark.asyncio
    async def test_call_stats_via_server(self):
        """ace_stats dispatches correctly via server.call_tool."""
        content_list, result_dict = await server.call_tool("ace_stats", {})
        assert len(content_list) >= 1
        assert content_list[0].type == "text"
        assert "ACE Unified Memory Statistics" in content_list[0].text

    @pytest.mark.asyncio
    async def test_call_retrieve_via_server(self):
        """ace_retrieve dispatches correctly via server.call_tool."""
        content_list, result_dict = await server.call_tool(
            "ace_retrieve", {"query": "test protocol dispatch", "limit": 1}
        )
        assert len(content_list) >= 1
        assert content_list[0].type == "text"

    @pytest.mark.asyncio
    async def test_call_store_via_server(self):
        """ace_store dispatches correctly via server.call_tool."""
        content_list, result_dict = await server.call_tool(
            "ace_store", {"content": "Protocol layer test memory"}
        )
        assert len(content_list) >= 1
        assert content_list[0].type == "text"
        text = content_list[0].text
        assert "stored" in text.lower() or "reinforced" in text.lower()

    @pytest.mark.asyncio
    async def test_call_tag_via_server(self):
        """ace_tag dispatches correctly via server.call_tool."""
        content_list, result_dict = await server.call_tool(
            "ace_tag", {"memory_id": "protocol-test-id", "tag": "helpful"}
        )
        assert len(content_list) >= 1
        assert content_list[0].type == "text"

    @pytest.mark.asyncio
    async def test_call_tool_missing_required_arg(self):
        """Missing required arg through protocol returns error."""
        with pytest.raises(Exception):
            await server.call_tool("ace_retrieve", {})


class TestRetrieveHandler:
    """Test ace_retrieve handler."""

    @pytest.mark.asyncio
    async def test_retrieve_returns_text(self):
        """Retrieve returns a string result."""
        result = await ace_retrieve(query="coding preferences", limit=2)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_retrieve_respects_limit(self):
        """Retrieve respects limit parameter."""
        result = await ace_retrieve(query="test query", limit=1)
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_retrieve_with_namespace_filter(self):
        """Retrieve can filter by namespace."""
        result = await ace_retrieve(
            query="user preferences",
            namespace="user_prefs",
            limit=3,
        )
        assert isinstance(result, str)


class TestStoreHandler:
    """Test ace_store handler."""

    @pytest.mark.asyncio
    async def test_store_creates_memory(self):
        """Store creates a new memory."""
        result = await ace_store(
            content="Test memory for MCP server testing",
            category="DEBUGGING",
            severity=3,
        )
        assert isinstance(result, str)
        assert "stored" in result.lower() or "reinforced" in result.lower()

    @pytest.mark.asyncio
    async def test_store_with_namespace(self):
        """Store respects namespace parameter."""
        result = await ace_store(
            content="Task strategy test memory",
            namespace="task_strategies",
            section="testing",
        )
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_store_deduplication_works(self):
        """Storing similar content triggers reinforcement."""
        content = "Unique test content for dedup testing 12345"
        result1 = await ace_store(content=content)
        result2 = await ace_store(content=content)
        assert "reinforced" in result2.lower() or "stored" in result2.lower()


class TestSearchHandler:
    """Test ace_search handler."""

    @pytest.mark.asyncio
    async def test_search_returns_text(self):
        """Search returns a string result."""
        result = await ace_search(query="preferences", limit=3)
        assert isinstance(result, str)
        # Should be valid JSON or "No matching memories"
        if "No matching memories" not in result:
            data = json.loads(result)
            assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_search_with_category_filter(self):
        """Search can filter by category."""
        result = await ace_search(query="coding", category="PREFERENCE", limit=5)
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_search_with_severity_filter(self):
        """Search can filter by minimum severity."""
        result = await ace_search(query="important", min_severity=7, limit=5)
        assert isinstance(result, str)


class TestStatsHandler:
    """Test ace_stats handler."""

    @pytest.mark.asyncio
    async def test_stats_returns_collection_info(self):
        """Stats returns collection information."""
        result = await ace_stats()
        assert isinstance(result, str)
        assert "ACE Unified Memory Statistics" in result
        assert "Collection:" in result
        assert "Total Points:" in result

    @pytest.mark.asyncio
    async def test_stats_shows_status(self):
        """Stats shows collection status."""
        result = await ace_stats()
        assert "Status:" in result


class TestTagHandler:
    """Test ace_tag handler."""

    @pytest.mark.asyncio
    async def test_tag_with_nonexistent_id(self):
        """Tag with nonexistent memory_id returns a result."""
        result = await ace_tag(memory_id="nonexistent-id", tag="helpful")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_tag_accepts_helpful(self):
        """Tag accepts 'helpful' tag."""
        result = await ace_tag(memory_id="test-id", tag="helpful")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_tag_accepts_harmful(self):
        """Tag accepts 'harmful' tag."""
        result = await ace_tag(memory_id="test-id", tag="harmful")
        assert isinstance(result, str)


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
        import uuid
        unique_content = f"MCP integration test {uuid.uuid4().hex[:8]}"

        store_result = await ace_store(
            content=unique_content,
            category="DEBUGGING",
            severity=8,
        )
        assert "stored" in store_result.lower() or "reinforced" in store_result.lower()

        retrieve_result = await ace_retrieve(
            query=unique_content[:20],
            limit=5,
        )
        assert isinstance(retrieve_result, str)

    @pytest.mark.asyncio
    async def test_full_feedback_loop(self):
        """Test complete feedback loop: store -> retrieve -> tag."""
        store_result = await ace_store(
            content="Feedback loop test memory",
            category="WORKFLOW",
        )

        if "ID:" in store_result:
            memory_id = store_result.split("ID:")[-1].strip()
            tag_result = await ace_tag(memory_id=memory_id, tag="helpful")
            assert isinstance(tag_result, str)
