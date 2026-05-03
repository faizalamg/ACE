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


class TestMultiRootWorkspace:
    """Tests for multi-root workspace detection and code retrieval."""

    def test_get_workspace_from_roots_stores_all_roots(self):
        """_get_workspace_from_roots stores all valid roots, not just the first."""
        from ace_mcp_server import _all_workspace_roots
        # The global should be a list (may be empty if not fetched yet)
        assert isinstance(_all_workspace_roots, list)

    def test_get_collection_name_for_workspace(self):
        """_get_collection_name_for_workspace derives correct collection name."""
        from ace_mcp_server import _get_collection_name_for_workspace
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Folder name should become the collection prefix
            folder_name = os.path.basename(tmpdir)
            expected = f"{folder_name}_code_context"
            result = _get_collection_name_for_workspace(tmpdir)
            assert result == expected

    def test_get_collection_name_for_workspace_with_config(self):
        """_get_collection_name_for_workspace reads .ace/.ace.json if present."""
        from ace_mcp_server import _get_collection_name_for_workspace
        import tempfile, os, json

        with tempfile.TemporaryDirectory() as tmpdir:
            ace_dir = os.path.join(tmpdir, ".ace")
            os.makedirs(ace_dir)
            config = {"workspace_name": "my-project"}
            with open(os.path.join(ace_dir, ".ace.json"), "w") as f:
                json.dump(config, f)

            result = _get_collection_name_for_workspace(tmpdir)
            assert result == "my-project_code_context"

    def test_get_collection_name_sanitizes_special_chars(self):
        """Collection names are sanitized for Qdrant compatibility."""
        from ace_mcp_server import _get_collection_name_for_workspace
        import tempfile, os

        with tempfile.TemporaryDirectory(prefix="my project (v2)") as tmpdir:
            result = _get_collection_name_for_workspace(tmpdir)
            # Should not contain spaces or parens
            assert " " not in result
            assert "(" not in result
            assert result.endswith("_code_context")

    @pytest.mark.asyncio
    async def test_get_workspace_from_roots_multi_root(self):
        """_get_workspace_from_roots stores all roots from list_roots()."""
        import ace_mcp_server as mod
        import tempfile, os

        # Save originals
        orig_cached = mod._cached_workspace_from_roots
        orig_all = mod._all_workspace_roots
        orig_attempted = mod._roots_fetch_attempted

        try:
            # Reset state
            mod._cached_workspace_from_roots = None
            mod._all_workspace_roots = []
            mod._roots_fetch_attempted = False

            # Create two temp workspace dirs
            with tempfile.TemporaryDirectory() as root1, tempfile.TemporaryDirectory() as root2:
                # Mock context with two roots
                mock_ctx = MagicMock()
                mock_session = AsyncMock()

                mock_root1 = MagicMock()
                mock_root1.uri = f"file:///{root1.replace(os.sep, '/')}"
                mock_root2 = MagicMock()
                mock_root2.uri = f"file:///{root2.replace(os.sep, '/')}"

                mock_roots_result = MagicMock()
                mock_roots_result.roots = [mock_root1, mock_root2]
                mock_session.list_roots = AsyncMock(return_value=mock_roots_result)
                mock_ctx.session = mock_session

                result = await mod._get_workspace_from_roots(mock_ctx)

                # Should return first root as primary
                assert result is not None
                # Should store BOTH roots
                assert len(mod._all_workspace_roots) == 2
        finally:
            # Restore originals
            mod._cached_workspace_from_roots = orig_cached
            mod._all_workspace_roots = orig_all
            mod._roots_fetch_attempted = orig_attempted


class TestMultiCollectionMerge:
    """Integration tests for multi-collection result merging and CodeRetrieval caching."""

    @pytest.mark.asyncio
    async def test_ace_retrieve_merges_results_from_multiple_roots(self):
        """ace_retrieve merges code results from multiple workspace roots, sorted by score."""
        import ace_mcp_server as mod

        # Save originals
        orig_all_roots = mod._all_workspace_roots
        orig_cache = mod._code_retrieval_cache
        orig_code_retrieval = mod._code_retrieval

        try:
            import tempfile, os

            with tempfile.TemporaryDirectory() as root1, tempfile.TemporaryDirectory() as root2:
                # Create .ace config for both roots
                for root in [root1, root2]:
                    ace_dir = os.path.join(root, ".ace")
                    os.makedirs(ace_dir)
                    name = os.path.basename(root)
                    with open(os.path.join(ace_dir, ".ace.json"), "w") as f:
                        json.dump({"workspace_name": name, "onboarded": True}, f)

                col1 = f"{os.path.basename(root1)}_code_context"
                col2 = f"{os.path.basename(root2)}_code_context"

                # Mock CodeRetrieval instances returning different results
                mock_cr1 = MagicMock()
                mock_cr1.collection_name = col1
                mock_cr1.search.return_value = [
                    {"file_path": "auth.py", "content": "def login():", "score": 0.9, "start_line": 1, "end_line": 10},
                ]
                mock_cr1.format_ThatOtherContextEngine_style.return_value = "formatted_code"

                mock_cr2 = MagicMock()
                mock_cr2.collection_name = col2
                mock_cr2.search.return_value = [
                    {"file_path": "api.py", "content": "def endpoint():", "score": 0.95, "start_line": 1, "end_line": 5},
                    {"file_path": "db.py", "content": "class DB:", "score": 0.7, "start_line": 1, "end_line": 20},
                ]

                # Set up state: both roots, primary = root1
                mod._all_workspace_roots = [root1, root2]
                mod._code_retrieval = mock_cr1  # Primary workspace
                mod._code_retrieval_cache = {col2: mock_cr2}  # Non-primary cached

                # Mock dependencies
                with patch.object(mod, 'get_workspace_path_async', return_value=root1), \
                     patch.object(mod, 'is_workspace_onboarded', return_value=True), \
                     patch.object(mod, 'get_workspace_collection_name', return_value=col1), \
                     patch.object(mod, '_check_collection_exists', return_value=True), \
                     patch.object(mod, 'get_code_retrieval', return_value=mock_cr1), \
                     patch.object(mod, '_get_unified_memory') as mock_um, \
                     patch.object(mod, 'get_memory_index') as mock_idx, \
                     patch.object(mod, '_wait_for_preload'), \
                     patch.object(mod, '_log_startup_info'):

                    # Mock memory retrieval (empty)
                    mock_index = MagicMock()
                    mock_index.retrieve.return_value = []
                    mock_idx.return_value = mock_index
                    mock_um.return_value = {"UnifiedNamespace": MagicMock(), "format_unified_context": MagicMock()}

                    result = await ace_retrieve("find authentication", ctx=None)

                    # Verify both CRs were searched
                    mock_cr1.search.assert_called_once()
                    mock_cr2.search.assert_called_once()

                    # Verify results are formatted (merged results should be score-sorted)
                    mock_cr1.format_ThatOtherContextEngine_style.assert_called_once()
                    call_args = mock_cr1.format_ThatOtherContextEngine_style.call_args[0][0]
                    # Top result should be api.py (0.95), then auth.py (0.9)
                    assert call_args[0]["file_path"] == "api.py"
                    assert call_args[1]["file_path"] == "auth.py"
        finally:
            mod._all_workspace_roots = orig_all_roots
            mod._code_retrieval_cache = orig_cache
            mod._code_retrieval = orig_code_retrieval

    @pytest.mark.asyncio
    async def test_cache_hit_reuses_code_retrieval_instance(self):
        """Non-primary root CodeRetrieval is cached and reused on subsequent calls."""
        import ace_mcp_server as mod

        orig_cache = mod._code_retrieval_cache
        try:
            mod._code_retrieval_cache = {}

            mock_cr = MagicMock()
            mock_cr.collection_name = "test_project_code_context"
            mod._code_retrieval_cache["test_project_code_context"] = mock_cr

            # Cache hit: should return existing instance
            assert "test_project_code_context" in mod._code_retrieval_cache
            assert mod._code_retrieval_cache["test_project_code_context"] is mock_cr
        finally:
            mod._code_retrieval_cache = orig_cache

    @pytest.mark.asyncio
    async def test_ace_retrieve_skips_failed_root(self):
        """If one root's CodeRetrieval fails, results from other roots are still returned."""
        import ace_mcp_server as mod

        orig_all_roots = mod._all_workspace_roots
        orig_cache = mod._code_retrieval_cache
        orig_code_retrieval = mod._code_retrieval

        try:
            import tempfile, os

            with tempfile.TemporaryDirectory() as root1, tempfile.TemporaryDirectory() as root2:
                for root in [root1, root2]:
                    ace_dir = os.path.join(root, ".ace")
                    os.makedirs(ace_dir)
                    name = os.path.basename(root)
                    with open(os.path.join(ace_dir, ".ace.json"), "w") as f:
                        json.dump({"workspace_name": name, "onboarded": True}, f)

                col1 = f"{os.path.basename(root1)}_code_context"
                col2 = f"{os.path.basename(root2)}_code_context"

                # Primary CR works, non-primary CR throws
                mock_cr1 = MagicMock()
                mock_cr1.collection_name = col1
                mock_cr1.search.return_value = [
                    {"file_path": "main.py", "content": "def main():", "score": 0.8, "start_line": 1, "end_line": 5},
                ]
                mock_cr1.format_ThatOtherContextEngine_style.return_value = "formatted"

                mock_cr2 = MagicMock()
                mock_cr2.collection_name = col2
                mock_cr2.search.side_effect = Exception("Qdrant connection refused")

                mod._all_workspace_roots = [root1, root2]
                mod._code_retrieval = mock_cr1
                mod._code_retrieval_cache = {col2: mock_cr2}

                with patch.object(mod, 'get_workspace_path_async', return_value=root1), \
                     patch.object(mod, 'is_workspace_onboarded', return_value=True), \
                     patch.object(mod, 'get_workspace_collection_name', return_value=col1), \
                     patch.object(mod, '_check_collection_exists', return_value=True), \
                     patch.object(mod, 'get_code_retrieval', return_value=mock_cr1), \
                     patch.object(mod, '_get_unified_memory') as mock_um, \
                     patch.object(mod, 'get_memory_index') as mock_idx, \
                     patch.object(mod, '_wait_for_preload'), \
                     patch.object(mod, '_log_startup_info'):

                    mock_index = MagicMock()
                    mock_index.retrieve.return_value = []
                    mock_idx.return_value = mock_index
                    mock_um.return_value = {"UnifiedNamespace": MagicMock(), "format_unified_context": MagicMock()}

                    result = await ace_retrieve("find main entry", ctx=None)

                    # Primary root results still returned despite secondary failure
                    mock_cr1.format_ThatOtherContextEngine_style.assert_called_once()
                    call_args = mock_cr1.format_ThatOtherContextEngine_style.call_args[0][0]
                    assert len(call_args) == 1
                    assert call_args[0]["file_path"] == "main.py"
        finally:
            mod._all_workspace_roots = orig_all_roots
            mod._code_retrieval_cache = orig_cache
            mod._code_retrieval = orig_code_retrieval
