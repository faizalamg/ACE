#!/usr/bin/env python3
"""ACE MCP Server - Model Context Protocol server for ACE unified memory.

This server exposes ACE's unified memory capabilities as MCP tools, enabling
any MCP-compatible client (VS Code Copilot, Claude Desktop, Cursor, etc.) to:
- Retrieve relevant context from ACE memory
- Store new memories/lessons learned
- Search memories by query or namespace

Usage:
    # Direct execution (stdio transport)
    python ace_mcp_server.py
    
    # With uvx (recommended for MCP clients)
    uvx --from . ace-mcp-server

Requirements:
    pip install mcp ace-framework
    # or: pip install fastmcp ace-framework

NOTE: Uses FastMCP with list_roots() capability to dynamically request
workspace path from MCP client (VS Code, Claude Desktop, etc.) at runtime.
No need to configure workspace path in mcp.json - the server asks the client!
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

# Add the ace package to path for development
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from mcp.server.fastmcp import FastMCP, Context
    from mcp.types import TextContent
except ImportError:
    print("Error: MCP/FastMCP package not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

from ace.unified_memory import (
    UnifiedMemoryIndex,
    UnifiedBullet,
    UnifiedNamespace,
    UnifiedSource,
    format_unified_context,
)
from ace.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ace_mcp_server")

# Lazy-loaded code retrieval for blended results
_code_retrieval = None
_workspace_path = os.environ.get("ACE_WORKSPACE_PATH", None)
_auto_index_done = False  # Track if auto-indexing was attempted
_workspace_collection_name: Optional[str] = None  # Computed collection name for current workspace
_workspace_onboarded = False  # Track if workspace has been onboarded
_onboarding_pending = False  # Track if onboarding is waiting for user input

# Workspace path captured from MCP initialize message (Option B fallback)
_mcp_client_workspace: Optional[str] = None

# Cache for workspace path obtained from MCP list_roots() capability
_cached_workspace_from_roots: Optional[str] = None
_roots_fetch_attempted = False

# ACE workspace configuration file
_ACE_CONFIG_DIR = ".ace"
_ACE_CONFIG_FILE = ".ace.json"


# Paths that should NOT be treated as workspace roots (user home, temp dirs, etc.)
_INVALID_WORKSPACE_PATHS = set()


def _get_invalid_workspace_paths() -> set:
    """Get set of paths that should not be auto-indexed as workspaces."""
    global _INVALID_WORKSPACE_PATHS
    if not _INVALID_WORKSPACE_PATHS:
        # User home directory - never auto-index
        home = os.path.expanduser("~")
        if home:
            _INVALID_WORKSPACE_PATHS.add(os.path.normpath(home).lower())
        
        # Common system paths
        for env_var in ["TEMP", "TMP", "USERPROFILE", "SYSTEMROOT", "WINDIR"]:
            path = os.environ.get(env_var)
            if path:
                _INVALID_WORKSPACE_PATHS.add(os.path.normpath(path).lower())
        
        # Drive roots
        for letter in "CDEFGH":
            _INVALID_WORKSPACE_PATHS.add(f"{letter.lower()}:\\")
            _INVALID_WORKSPACE_PATHS.add(f"{letter.lower()}:/")
    
    return _INVALID_WORKSPACE_PATHS


def _is_valid_workspace_path(path: str) -> bool:
    """Check if a path is valid for use as a workspace (not home, temp, etc.)."""
    if not path:
        return False
    normalized = os.path.normpath(path).lower()
    invalid_paths = _get_invalid_workspace_paths()
    return normalized not in invalid_paths


def find_workspace_root(start_path: Optional[str] = None) -> Optional[str]:
    """Find the workspace root by looking for .ace/.ace.json or common markers.

    Searches upward from start_path (or cwd) for:
    1. .ace/.ace.json file (ACE workspace marker)
    2. .git directory (git repo root)
    3. package.json, pyproject.toml, Cargo.toml, go.mod (project markers)

    Returns the absolute path to the workspace root, or None if not found.
    Excludes invalid paths like user home, temp directories, and drive roots.
    """
    start = start_path or os.getcwd()
    current = os.path.abspath(start)

    # Search upward, but stop at filesystem root
    while current and current != os.path.dirname(current):
        # Skip invalid workspace paths (home, temp, etc.)
        if not _is_valid_workspace_path(current):
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
            continue

        # Check for ACE workspace marker
        ace_config = os.path.join(current, _ACE_CONFIG_DIR, _ACE_CONFIG_FILE)
        if os.path.isfile(ace_config):
            logger.debug(f"Found workspace root via .ace/.ace.json: {current}")
            return current

        # Check for common project markers
        markers = [
            os.path.join(current, ".git"),
            os.path.join(current, "package.json"),
            os.path.join(current, "pyproject.toml"),
            os.path.join(current, "Cargo.toml"),
            os.path.join(current, "go.mod"),
            os.path.join(current, ".hg"),
            os.path.join(current, ".svn"),
        ]
        if any(os.path.exists(m) for m in markers):
            logger.debug(f"Found workspace root via project marker: {current}")
            return current

        # Move up one directory
        parent = os.path.dirname(current)
        if parent == current:  # Reached root
            break
        current = parent

    return None


def get_workspace_config(workspace_path: str) -> Optional[Dict[str, Any]]:
    """Load ACE workspace configuration from .ace/.ace.json.

    Returns None if config doesn't exist (workspace not onboarded).
    """
    config_path = os.path.join(workspace_path, _ACE_CONFIG_DIR, _ACE_CONFIG_FILE)
    if not os.path.isfile(config_path):
        return None

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load workspace config from {config_path}: {e}")
        return None


def save_workspace_config(workspace_path: str, workspace_name: str) -> bool:
    """Save workspace configuration to .ace/.ace.json.

    Creates the .ace directory if it doesn't exist.
    """
    ace_dir = os.path.join(workspace_path, _ACE_CONFIG_DIR)
    config_path = os.path.join(ace_dir, _ACE_CONFIG_FILE)

    try:
        os.makedirs(ace_dir, exist_ok=True)

        config = {
            "workspace_name": workspace_name,
            "workspace_path": workspace_path,
            "collection_name": f"{workspace_name}_code_context",
            "onboarded_at": __import__('datetime').datetime.now().isoformat(),
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved workspace config to {config_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save workspace config: {e}")
        return False


def get_workspace_path() -> Optional[str]:
    """Get the workspace path (sync version).

    Priority:
    1. Cached workspace from MCP list_roots() (set by async version)
    2. ACE_WORKSPACE_PATH environment variable (set by mcp.json config)
    3. MCP client workspace (captured from initialize message - Option B)
    4. Find workspace root via .ace/.ace.json or project markers
    5. Current working directory (fallback)

    Returns None if no workspace is available.
    """
    global _mcp_client_workspace, _cached_workspace_from_roots

    # 1. Check cached workspace from list_roots() - set by async version
    if _cached_workspace_from_roots:
        logger.debug(f"Using cached workspace from list_roots: {_cached_workspace_from_roots}")
        return _cached_workspace_from_roots

    # 2. Always check for explicit environment overrides at call-time.
    env_path = os.environ.get("ACE_WORKSPACE_PATH") or os.environ.get("MCP_WORKSPACE_FOLDER")
    if env_path:
        try:
            env_path = os.path.abspath(env_path)
            if _is_valid_workspace_path(env_path):
                logger.debug(f"Using workspace path from env: {env_path}")
                return env_path
        except Exception:
            logger.debug(f"Invalid ACE_WORKSPACE_PATH value: {env_path}")

    # 3. Check for workspace captured from MCP initialize message (Option B)
    if _mcp_client_workspace:
        logger.debug(f"Using workspace path from MCP initialize: {_mcp_client_workspace}")
        return _mcp_client_workspace

    # 4. Try to find workspace root from current directory
    workspace_root = find_workspace_root()
    if workspace_root:
        return workspace_root

    # 5. Fallback to current working directory (only if valid)
    cwd = os.getcwd()
    if _is_valid_workspace_path(cwd):
        logger.debug(f"Falling back to CWD as workspace path: {cwd}")
        return cwd
    
    # 6. No valid workspace found
    logger.warning(f"No valid workspace detected (cwd={cwd} is not a valid workspace path)")
    return None


def get_workspace_collection_name() -> str:
    """Get a workspace-specific collection name.

    Reads from .ace/.ace.json if available, otherwise derives from workspace folder name.
    Falls back to default collection name if no workspace detected.

    Collection name format: {workspace_name}_code_context
    Example: "agentic-context-engine_code_context"
    """
    global _workspace_collection_name

    if _workspace_collection_name:
        return _workspace_collection_name

    # Check if ACE_CODE_COLLECTION is explicitly set (not default)
    env_collection = os.environ.get("ACE_CODE_COLLECTION", "")
    default_collection = "ace_code_context"

    # If user explicitly set a custom collection, use it
    if env_collection and env_collection != default_collection:
        _workspace_collection_name = env_collection
        return _workspace_collection_name

    # Try to read from workspace config
    workspace = get_workspace_path()
    if workspace:
        config = get_workspace_config(workspace)
        if config and "workspace_name" in config:
            workspace_name = config["workspace_name"]
            _workspace_collection_name = f"{workspace_name}_code_context"
            return _workspace_collection_name

    # Derive from workspace folder name
    if workspace:
        workspace_name = os.path.basename(os.path.normpath(workspace))
        # Sanitize for Qdrant (alphanumeric, underscore, hyphen only)
        workspace_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in workspace_name)
        _workspace_collection_name = f"{workspace_name}_code_context"
    else:
        _workspace_collection_name = default_collection

    return _workspace_collection_name


def is_workspace_onboarded(workspace_path: str) -> bool:
    """Check if workspace has been onboarded (has .ace/.ace.json)."""
    config = get_workspace_config(workspace_path)
    return config is not None


def onboard_workspace(workspace_path: str, workspace_name: str) -> bool:
    """Onboard a workspace by creating .ace/.ace.json configuration."""
    global _workspace_onboarded, _workspace_collection_name

    if save_workspace_config(workspace_path, workspace_name):
        _workspace_onboarded = True
        # Reset collection name cache to use new config
        _workspace_collection_name = None
        return True
    return False


def _check_collection_exists(qdrant_url: str, collection_name: str) -> bool:
    """Check if a Qdrant collection exists and has points."""
    try:
        import httpx
        response = httpx.get(
            f"{qdrant_url}/collections/{collection_name}",
            timeout=5.0
        )
        if response.status_code == 200:
            info = response.json().get("result", {})
            points_count = info.get("points_count", 0)
            return points_count > 0
        return False
    except Exception as e:
        logger.debug(f"Collection check failed: {e}")
        return False


def _auto_index_workspace(workspace_path: str) -> tuple[bool, Optional[str]]:
    """Auto-index the workspace if collection doesn't exist.

    Checks for .ace/.ace.json to determine if workspace is onboarded.
    If not onboarded, returns onboarding prompt instead of indexing.

    Returns:
        (success: bool, message: Optional[str])
        - If onboarding needed: (False, onboarding_message)
        - If already indexed: (True, None)
        - If indexed successfully: (True, None)
        - If failed: (False, error_message)
    """
    global _auto_index_done

    if _auto_index_done:
        return True, None

    _auto_index_done = True  # Mark as attempted regardless of outcome

    # Check if workspace is onboarded
    if not is_workspace_onboarded(workspace_path):
        workspace_name = os.path.basename(os.path.normpath(workspace_path))
        onboarding_msg = (
            f"**Workspace Not Onboarded**\n\n"
            f"Detected workspace: `{workspace_path}`\n\n"
            f"Please run the **ace_onboard** tool to set up this workspace:\n"
            f"- Use workspace name: `{workspace_name}`\n\n"
            f"This will:\n"
            f"1. Create `.ace/.ace.json` configuration file\n"
            f"2. Index your code for workspace-specific retrieval\n"
            f"3. Enable memory features for this project"
        )
        return False, onboarding_msg

    try:
        from ace.code_indexer import CodeIndexer
        from ace.config import get_config

        config = get_config()
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        collection_name = get_workspace_collection_name()

        # Check if collection exists and has points
        if _check_collection_exists(qdrant_url, collection_name):
            logger.info(f"Code index already exists for workspace: {workspace_path} (collection: {collection_name})")
            return True, None

        # Collection doesn't exist or is empty - create it
        logger.info(f"Auto-indexing workspace: {workspace_path} -> collection: {collection_name}")

        indexer = CodeIndexer(
            workspace_path=workspace_path,
            qdrant_url=qdrant_url,
            collection_name=collection_name,
        )

        stats = indexer.index_workspace()
        logger.info(f"Auto-indexed {stats.get('files_indexed', 0)} files, "
                   f"{stats.get('chunks_indexed', 0)} chunks")
        return True, None

    except Exception as e:
        logger.warning(f"Auto-indexing failed: {e}")
        return False, f"Auto-indexing failed: {e}"


def get_code_retrieval():
    """Get or create CodeRetrieval instance with auto-indexing support.

    On first call, checks if workspace is onboarded and code index exists.
    If not onboarded, returns None and sets onboarding pending flag.

    Uses workspace-specific collection name to avoid cross-project contamination.
    """
    global _code_retrieval, _onboarding_pending
    if _code_retrieval is None:
        try:
            # Auto-index if workspace path is configured
            workspace = get_workspace_path()
            logger.info(f"get_code_retrieval: workspace={workspace}, cwd={os.getcwd()}")

            if workspace:
                success, msg = _auto_index_workspace(workspace)
                logger.info(f"_auto_index_workspace result: success={success}, msg={msg}")
                if not success and msg:
                    # Onboarding needed - set pending flag
                    _onboarding_pending = True
                    return None

            from ace.code_retrieval import CodeRetrieval

            # Use workspace-specific collection name
            collection_name = get_workspace_collection_name()
            logger.info(f"Creating CodeRetrieval with collection: {collection_name}")
            _code_retrieval = CodeRetrieval(collection_name=collection_name)

            logger.info(f"Code retrieval initialized (workspace: {workspace or 'not set'}, collection: {collection_name})")
        except Exception as e:
            logger.exception(f"Failed to initialize code retrieval: {e}")
    return _code_retrieval


def is_onboarding_pending() -> bool:
    """Check if workspace onboarding is pending (user needs to run ace_onboard)."""
    global _onboarding_pending
    return _onboarding_pending


logger = logging.getLogger("ace-mcp")

# Initialize FastMCP server with list_roots capability
server = FastMCP("ace")


async def _get_workspace_from_roots(ctx: Context) -> Optional[str]:
    """Get workspace path from MCP client via list_roots().
    
    This is the proper MCP protocol way to get workspace information:
    - Server requests roots from client at runtime
    - No need for env vars or config files
    - Works with any MCP-compliant client (VS Code, Claude Desktop, etc.)
    """
    global _cached_workspace_from_roots, _roots_fetch_attempted
    
    # Return cached value if we already have it
    if _cached_workspace_from_roots:
        return _cached_workspace_from_roots
    
    # Only try once per session to avoid repeated failures
    if _roots_fetch_attempted:
        return None
    
    _roots_fetch_attempted = True
    
    try:
        # Request roots from MCP client
        if ctx.session:
            logger.info("Requesting workspace roots from MCP client via list_roots()...")
            roots_result = await ctx.session.list_roots()
            
            if roots_result and roots_result.roots:
                for root in roots_result.roots:
                    # Extract path from file:// URI
                    workspace_path = _extract_workspace_from_uri(str(root.uri))
                    if workspace_path and _is_valid_workspace_path(workspace_path):
                        _cached_workspace_from_roots = workspace_path
                        logger.info(f"Got workspace from MCP roots: {workspace_path}")
                        return workspace_path
                    else:
                        logger.debug(f"Skipping invalid root: {root.uri}")
            else:
                logger.debug("list_roots() returned no roots")
        else:
            logger.debug("No session available for list_roots()")
            
    except Exception as e:
        logger.warning(f"list_roots() failed: {e}")
    
    return None


def _extract_workspace_from_uri(uri: str) -> Optional[str]:
    """Extract filesystem path from a URI (file:// or plain path)."""
    if not uri:
        return None
    try:
        if uri.startswith("file://"):
            # Handle file:// URIs
            from urllib.parse import urlparse, unquote
            parsed = urlparse(uri)
            # On Windows, path starts with /C:/... so strip leading /
            path = unquote(parsed.path)
            if sys.platform == "win32" and path.startswith("/") and len(path) > 2 and path[2] == ":":
                path = path[1:]  # Remove leading /
            return os.path.abspath(path)
        else:
            # Plain path
            return os.path.abspath(uri)
    except Exception as e:
        logger.debug(f"Failed to extract workspace from URI '{uri}': {e}")
        return None


async def get_workspace_path_async(ctx: Optional[Context] = None) -> Optional[str]:
    """Get the workspace path (async version that uses list_roots).

    Priority:
    1. MCP list_roots() - proper protocol way (requires Context)
    2. ACE_WORKSPACE_PATH environment variable (fallback)
    3. Find workspace root via .ace/.ace.json or project markers
    4. Current working directory (fallback)

    IMPORTANT: This function caches the workspace in _cached_workspace_from_roots
    so that sync code (like get_code_retrieval) can access it.

    Returns None if no workspace is available.
    """
    global _mcp_client_workspace, _cached_workspace_from_roots

    # Return cached value if already set
    if _cached_workspace_from_roots:
        return _cached_workspace_from_roots

    # 1. Try MCP list_roots() first if we have context
    if ctx:
        workspace = await _get_workspace_from_roots(ctx)
        if workspace:
            # Already cached by _get_workspace_from_roots
            return workspace

    # 2. Check for explicit environment overrides
    env_path = os.environ.get("ACE_WORKSPACE_PATH") or os.environ.get("MCP_WORKSPACE_FOLDER")
    if env_path:
        try:
            env_path = os.path.abspath(env_path)
            if _is_valid_workspace_path(env_path):
                logger.debug(f"Using workspace path from env: {env_path}")
                _cached_workspace_from_roots = env_path  # Cache it!
                return env_path
        except Exception:
            logger.debug(f"Invalid ACE_WORKSPACE_PATH value: {env_path}")

    # 3. Check for workspace captured from MCP initialize message (legacy Option B)
    if _mcp_client_workspace:
        logger.debug(f"Using workspace path from MCP initialize: {_mcp_client_workspace}")
        _cached_workspace_from_roots = _mcp_client_workspace  # Cache it!
        return _mcp_client_workspace

    # 4. Try to find workspace root from current directory
    workspace_root = find_workspace_root()
    if workspace_root:
        logger.info(f"Found workspace root via project markers: {workspace_root}")
        _cached_workspace_from_roots = workspace_root  # Cache it!
        return workspace_root

    # 5. Fallback to current working directory (only if valid)
    cwd = os.getcwd()
    if _is_valid_workspace_path(cwd):
        logger.debug(f"Falling back to CWD as workspace path: {cwd}")
        _cached_workspace_from_roots = cwd  # Cache it!
        return cwd
    
    # 6. No valid workspace found
    logger.warning(f"No valid workspace detected (cwd={cwd} is not a valid workspace path)")
    return None


# Log startup info on first tool call
_startup_logged = False


def _log_startup_info():
    """Log startup information once per process."""
    global _startup_logged
    if _startup_logged:
        return
    _startup_logged = True
    
    logger.info("=" * 60)
    logger.info("ACE MCP Server Starting (FastMCP with list_roots)")
    logger.info(f"  PID: {os.getpid()}")
    logger.info(f"  CWD: {os.getcwd()}")
    logger.info(f"  ACE_WORKSPACE_PATH env: {os.environ.get('ACE_WORKSPACE_PATH', '(not set)')}")
    logger.info("=" * 60)


# Global unified memory index (lazy initialization)
_memory_index: Optional[UnifiedMemoryIndex] = None


def get_memory_index() -> UnifiedMemoryIndex:
    """Get or create the unified memory index singleton."""
    global _memory_index
    if _memory_index is None:
        config = get_config()
        _memory_index = UnifiedMemoryIndex(
            collection_name=config.qdrant.unified_collection,
            qdrant_url=config.qdrant.url,
        )
        logger.info(f"Initialized UnifiedMemoryIndex: {config.qdrant.unified_collection}")
    return _memory_index


# ============================================================================
# FastMCP Tool Definitions
# ============================================================================

@server.tool()
async def ace_retrieve(
    query: str,
    namespace: str = "all",
    limit: int = 5,
    ctx: Context = None,
) -> str:
    """Retrieve relevant context from ACE unified memory based on a query.
    
    Use this tool FIRST when you need to:
    - Understand user preferences or past decisions
    - Find relevant coding patterns or lessons learned
    - Get context about the current task or project
    - Look up remembered corrections or directives

    Returns formatted context with relevance scores and severity indicators.
    
    Args:
        query: Natural language query describing what context you need
        namespace: Filter by namespace: user_prefs, task_strategies, project_specific, or all
        limit: Maximum number of results to return (1-20)
    """
    _log_startup_info()
    
    logger.info(f"ace_retrieve called: query='{query}', namespace={namespace}, limit={limit}")
    results_parts = []

    # Get workspace via list_roots() or fallback
    workspace = await get_workspace_path_async(ctx)
    logger.info(f"ace_retrieve: workspace={workspace}")

    # Check if onboarding is needed - AUTO-ONBOARD
    if workspace and not is_workspace_onboarded(workspace):
        workspace_name = os.path.basename(os.path.normpath(workspace))
        
        # Auto-onboard with default workspace name
        onboard_result = await ace_onboard(workspace_name=workspace_name, ctx=ctx)
        
        if "Onboarding Complete" in onboard_result or "already onboarded" in onboard_result:
            results_parts.append(onboard_result)
        else:
            results_parts.append(onboard_result)

    # 1. Code retrieval (only if onboarded)
    # IMPORTANT: We must ensure the cached workspace is set before getting code_retrieval
    # since get_code_retrieval() uses get_workspace_path() which reads _cached_workspace_from_roots
    global _code_retrieval
    expected_collection = get_workspace_collection_name()
    
    # Reset code_retrieval if collection doesn't match (workspace changed)
    if _code_retrieval is not None and _code_retrieval.collection_name != expected_collection:
        logger.info(f"Workspace changed - resetting code_retrieval (was {_code_retrieval.collection_name}, need {expected_collection})")
        _code_retrieval = None
    
    code_retrieval = get_code_retrieval()
    logger.info(f"code_retrieval instance: {code_retrieval is not None}, collection: {code_retrieval.collection_name if code_retrieval else 'N/A'}")
    if code_retrieval:
        try:
            code_results = await asyncio.to_thread(code_retrieval.search, query, limit)
            logger.info(f"Code search returned {len(code_results)} results")
            if code_results:
                # Use Auggie-compatible format directly (no wrapper header)
                # Output starts with "The following code sections were retrieved:"
                formatted_code = code_retrieval.format_ThatOtherContextEngine_style(code_results)
                results_parts.append(formatted_code)
        except Exception as e:
            logger.exception(f"Code retrieval failed: {e}")
    else:
        if workspace:
            logger.warning("Code retrieval not available")

    # 2. Memory retrieval (always available)
    index = get_memory_index()
    
    ns = None
    if namespace != "all":
        ns = UnifiedNamespace(namespace)
    
    # Get workspace_id for project_specific namespace isolation
    workspace_id_for_retrieval = None
    if namespace == "project_specific" or namespace == "all":
        # For project_specific or "all" namespaces, we need workspace_id
        if workspace:
            config = get_workspace_config(workspace)
            if config and "workspace_name" in config:
                workspace_id_for_retrieval = config["workspace_name"]
            else:
                workspace_id_for_retrieval = os.path.basename(os.path.normpath(workspace))
                workspace_id_for_retrieval = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in workspace_id_for_retrieval)
    
    memory_results = await asyncio.to_thread(
        index.retrieve,
        query=query,
        limit=limit,
        namespace=ns,
        auto_detect_preset=True,
        use_cross_encoder=True,
        workspace_id=workspace_id_for_retrieval,
    )

    if memory_results:
        formatted_memories = format_unified_context(memory_results)
        results_parts.append(formatted_memories)

    if not results_parts:
        return "No relevant memories or code found."

    return "\n\n".join(results_parts)


@server.tool()
async def ace_store(
    content: str,
    namespace: str = "user_prefs",
    section: str = "general",
    severity: int = 5,
    category: str = "PREFERENCE",
) -> str:
    """Store a new memory/lesson in ACE unified memory.

    ACE now supports BOTH generalizable AND project-specific memories:
    - Generalizable (user_prefs, task_strategies): Cross-workspace patterns
    - Project-specific: Workspace-scoped, automatically tagged with workspace_id

    The system auto-classifies content:
    - If content is project-specific, it's stored in project_specific namespace
    - If content is generalizable, it's stored in user_prefs/task_strategies
    - You can also explicitly specify namespace="project_specific"

    Use this tool when the user:
    - Expresses a preference or directive ("I prefer...", "Always...", "Never...")
    - Provides a correction or teaches you something
    - Shares workflow patterns or coding standards
    - Gives feedback that should be remembered for future sessions
    - Has project-specific knowledge to store (file paths, class names, etc.)

    The memory will be automatically deduplicated against existing memories.

    Args:
        content: The lesson, preference, or pattern to remember
        namespace: Category: user_prefs, task_strategies, project_specific
        section: Sub-category like 'communication', 'architecture', 'testing'
        severity: Importance 1-10 (10=critical directive, 5=normal, 1=minor)
        category: Type: PREFERENCE, CORRECTION, DIRECTIVE, WORKFLOW, ARCHITECTURE, DEBUGGING, SECURITY
    """
    import uuid

    _log_startup_info()

    # Classify content to determine recommended namespace
    from ace.memory_generalizability import should_store_in_ace, classify_memory_generalizability

    should_store, reason, extracted_principle, recommended_namespace = should_store_in_ace(content)
    classification = classify_memory_generalizability(content)

    # Respect user's explicit namespace choice, but allow auto-switch if classification is HIGH confidence project-specific
    final_namespace = namespace
    if namespace not in ("project_specific",) and recommended_namespace == "project_specific":
        # Only auto-switch if classification is confident (confidence > 0.5)
        if classification.confidence > 0.5:
            final_namespace = "project_specific"
        else:
            # Low confidence classification - respect user's choice
            final_namespace = namespace
    elif namespace == "project_specific":
        # User explicitly requested project_specific
        final_namespace = "project_specific"

    # For project_specific namespace, get workspace_id for isolation
    workspace_id = None
    if final_namespace == "project_specific":
        workspace = get_workspace_path()
        if workspace:
            config = get_workspace_config(workspace)
            if config and "workspace_name" in config:
                workspace_id = config["workspace_name"]
            else:
                # Derive from folder name
                workspace_id = os.path.basename(os.path.normpath(workspace))
                workspace_id = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in workspace_id)

        if not workspace_id:
            return "ERROR: Cannot store project-specific memory without workspace context. Please ensure you're in a valid workspace."

    # If an extracted principle is different from content, use that (for generalizable content)
    final_content = content
    if final_namespace != "project_specific" and extracted_principle and extracted_principle != content:
        final_content = extracted_principle

    index = get_memory_index()

    # Build bullet with optional workspace_id
    bullet_kwargs = {
        "id": str(uuid.uuid4()),
        "content": final_content,
        "section": section,
        "namespace": UnifiedNamespace(final_namespace),
        "source": UnifiedSource.USER_FEEDBACK,
        "severity": severity,
        "category": category,
    }

    # Add workspace_id to payload for project_specific memories
    if workspace_id:
        bullet_kwargs["workspace_id"] = workspace_id

    bullet = UnifiedBullet(**bullet_kwargs)

    result = await asyncio.to_thread(index.index_bullet, bullet)

    if result.get("action") == "reinforced":
        message = f"Memory reinforced (similar exists). Reinforcement count: {result.get('reinforcement_count', 1)}. Similarity: {result.get('similarity', 0):.2f}"
    else:
        message = f"Memory stored successfully. ID: {bullet.id}"

    # Add namespace info
    if final_namespace == "project_specific":
        message += f"\n\nNamespace: project_specific (workspace: {workspace_id})"
    else:
        message += f"\n\nNamespace: {final_namespace} (cross-workspace)"

    # Add note about principle extraction
    if final_namespace != "project_specific" and extracted_principle and extracted_principle != content:
        message += f"\n\nNote: Extracted general principle:\n  Original: {content[:100]}...\n  Stored: {extracted_principle}"

    return message


@server.tool()
async def ace_search(
    query: str,
    category: str = "all",
    min_severity: int = 1,
    limit: int = 10,
) -> str:
    """Search ACE memories with filters. Use for targeted queries.
    
    Use this tool when you need:
    - Memories from a specific category (DIRECTIVE, CORRECTION, etc.)
    - High-severity memories only
    - Namespace-specific search
    
    Args:
        query: Search query
        category: Filter by category: PREFERENCE, CORRECTION, DIRECTIVE, WORKFLOW, ARCHITECTURE, DEBUGGING, SECURITY, or all
        min_severity: Minimum severity level (1-10)
        limit: Maximum results (1-50)
    """
    _log_startup_info()
    
    index = get_memory_index()
    
    results = await asyncio.to_thread(
        index.retrieve,
        query=query,
        limit=limit * 3,
    )
    
    filtered = []
    for r in results:
        if category != "all" and r.category != category:
            continue
        if r.severity < min_severity:
            continue
        filtered.append(r)
        if len(filtered) >= limit:
            break
    
    if not filtered:
        return "No matching memories found."
    
    output = []
    for r in filtered:
        output.append({
            "id": r.id,
            "content": r.content,
            "category": r.category,
            "severity": r.severity,
            "namespace": r.namespace.value if hasattr(r.namespace, 'value') else str(r.namespace),
            "helpful": r.helpful_count,
            "harmful": r.harmful_count,
        })
    
    return json.dumps(output, indent=2)


@server.tool()
async def ace_stats() -> str:
    """Get statistics about ACE unified memory collection.
    
    Use this tool to understand memory state, namespace distribution, and health.
    """
    _log_startup_info()
    
    index = get_memory_index()
    
    try:
        collection_info = await asyncio.to_thread(
            index._client.get_collection, index.collection_name
        )
        points_count = collection_info.points_count
        
        return f"""ACE Unified Memory Statistics
=============================
Collection: {index.collection_name}
Total Points: {points_count:,}
Status: {collection_info.status.name if hasattr(collection_info.status, 'name') else collection_info.status}
Vectors Config: {collection_info.config.params.vectors}

Memory types tracked:
- User preferences (directives, communication style)
- Task strategies (coding patterns, debugging approaches)
- Project-specific (architecture, codebase patterns)
- Corrections and lessons learned
"""
    except Exception as e:
        return f"Error getting stats: {e}"


@server.tool()
async def ace_tag(memory_id: str, tag: str) -> str:
    """Tag a memory as helpful or harmful to improve future retrieval.

    Use this tool when memory retrieval quality feedback is available.
    
    Args:
        memory_id: ID of the memory to tag
        tag: Whether the memory was helpful or harmful
    """
    _log_startup_info()
    
    index = get_memory_index()

    try:
        await asyncio.to_thread(index.tag_bullet, memory_id, tag)
        return f"Memory {memory_id} tagged as {tag}"
    except Exception as e:
        return f"Error tagging memory: {e}"


@server.tool()
async def ace_onboard(
    workspace_name: str = "",
    ctx: Context = None,
) -> str:
    """Onboard the current workspace to ACE.

    This tool should be called when a workspace is detected but not yet onboarded
    (no .ace/.ace.json file exists).

    Onboarding will:
    1. Create .ace/.ace.json configuration file with workspace name
    2. Index all code in the workspace for semantic search
    3. Enable workspace-specific code retrieval (isolated from other projects)

    After onboarding, use ace_retrieve to search both code and memories.
    
    Args:
        workspace_name: Name for this workspace (used for collection naming). Defaults to folder name if not provided.
    """
    global _workspace_collection_name, _auto_index_done, _code_retrieval, _onboarding_pending

    _log_startup_info()

    workspace_name_arg = workspace_name.strip() if workspace_name else ""

    # Get workspace via list_roots() or fallback
    workspace_path = await get_workspace_path_async(ctx)
    if not workspace_path:
        return "Error: Could not detect workspace path. Make sure you have a folder open in your MCP client (VS Code, Claude Desktop, Cursor), or set ACE_WORKSPACE_PATH environment variable."

    # Check if already onboarded
    if is_workspace_onboarded(workspace_path):
        config = get_workspace_config(workspace_path)
        existing_name = config.get("workspace_name", "unknown") if config else "unknown"
        return (
            f"Workspace already onboarded as '{existing_name}'.\n\n"
            f"Workspace: {workspace_path}\n"
            f"Config: .ace/.ace.json\n\n"
            f"To re-onboard with a different name, delete .ace/.ace.json first."
        )

    # Determine workspace name
    if workspace_name_arg:
        ws_name = workspace_name_arg
    else:
        ws_name = os.path.basename(os.path.normpath(workspace_path))
        ws_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in ws_name)

    # Sanitize
    ws_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in ws_name)

    try:
        # Save workspace config
        if not save_workspace_config(workspace_path, ws_name):
            return "Error: Failed to create .ace/.ace.json file."

        # Reset caches
        _workspace_collection_name = None
        _auto_index_done = False
        _code_retrieval = None
        _onboarding_pending = False

        # Get collection name and index
        collection_name = get_workspace_collection_name()
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")

        # Index the workspace
        from ace.code_indexer import CodeIndexer

        output_parts = [
            f"**Onboarding Workspace**\n",
            f"Workspace: `{workspace_path}`",
            f"Name: `{ws_name}`",
            f"Collection: `{collection_name}`",
            f"\nCreating code index...\n"
        ]

        indexer = CodeIndexer(
            workspace_path=workspace_path,
            qdrant_url=qdrant_url,
            collection_name=collection_name,
        )

        stats = await asyncio.to_thread(indexer.index_workspace)
        output_parts.append(
            f"\n**Onboarding Complete!**\n"
            f"- Indexed {stats.get('files_indexed', 0)} files\n"
            f"- Created {stats.get('chunks_indexed', 0)} code chunks\n"
            f"- Collection: `{collection_name}`\n\n"
            f"You can now use **ace_retrieve** to search your code and memories."
        )

        return "\n".join(output_parts)

    except Exception as e:
        logger.exception(f"Onboarding failed: {e}")
        return f"Error during onboarding: {e}"


@server.tool()
async def ace_workspace_info(ctx: Context = None) -> str:
    """Get information about the current workspace ACE configuration.

    Shows workspace path, onboarding status, collection name, and configuration details.
    """
    _log_startup_info()

    workspace_path = await get_workspace_path_async(ctx)

    if not workspace_path:
        return "No workspace detected. Make sure you have a folder open in your MCP client (VS Code, Claude Desktop, Cursor), or set ACE_WORKSPACE_PATH environment variable."

    config = get_workspace_config(workspace_path)
    collection_name = get_workspace_collection_name()

    is_onboarded = config is not None

    output = f"""**ACE Workspace Information**

**Workspace Path:**
`{workspace_path}`

**Status:** {'Onboarded' if is_onboarded else 'Not Onboarded'}

**Collection Name:**
`{collection_name}`

"""

    if is_onboarded:
        output += f"""**Configuration:**
- Name: `{config.get('workspace_name', 'N/A')}`
- Onboarded: `{config.get('onboarded_at', 'N/A')}`
- Config File: `.ace/.ace.json`

**Code Retrieval:**
Workspace-specific indexing is enabled. Code from this workspace is isolated from other projects.

**Next Steps:**
Use `ace_retrieve` to search both code and memories from this workspace.
"""
    else:
        folder_name = os.path.basename(os.path.normpath(workspace_path))
        output += f"""**To onboard this workspace:**

Run `ace_onboard` with:
- `workspace_name`: `{folder_name}` (or choose a custom name)

This will:
1. Create `.ace/.ace.json` configuration
2. Index your code for semantic search
3. Enable workspace-specific code retrieval
"""

    return output


# ============================================================================
# Prompt Enhancement Tool
# ============================================================================

# Cached enhancement prompt - loaded lazily on first use
_enhancement_prompt_cache: Optional[str] = None


def _load_enhancement_prompt() -> str:
    """Load the enhancement system prompt from ace/prompts/enhance_prompt.md.
    
    Uses caching to avoid reading the file on every request.
    """
    global _enhancement_prompt_cache
    
    if _enhancement_prompt_cache is not None:
        return _enhancement_prompt_cache
    
    prompt_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ace", "prompts", "enhance_prompt.md"
    )
    
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            _enhancement_prompt_cache = f.read()
        logger.info(f"Loaded enhancement prompt from {prompt_path}")
        return _enhancement_prompt_cache
    except FileNotFoundError:
        logger.error(f"Enhancement prompt not found at {prompt_path}")
        raise FileNotFoundError(
            f"Enhancement prompt not found at {prompt_path}. "
            "Please ensure ace/prompts/enhance_prompt.md exists."
        )


def _get_recent_git_commits(workspace_path: str, max_commits: int = 5) -> str:
    """Get recent git commits from the workspace.
    
    Args:
        workspace_path: Path to git repository
        max_commits: Maximum number of commits to retrieve
        
    Returns:
        Formatted string of recent commits or empty string if not available
    """
    import subprocess
    
    try:
        result = subprocess.run(
            ["git", "log", f"-{max_commits}", "--oneline", "--format=%h %s (%cr)"],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.debug(f"Git commit retrieval failed: {e}")
    return ""


def _get_git_status(workspace_path: str) -> str:
    """Get current git status (modified/staged files).
    
    Args:
        workspace_path: Path to git repository
        
    Returns:
        Formatted string of git status or empty string if not available
    """
    import subprocess
    
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.debug(f"Git status retrieval failed: {e}")
    return ""


@server.tool()
async def ace_enhance_prompt(
    prompt: str,
    include_memories: bool = True,
    include_git_commits: bool = False,
    include_git_status: bool = False,
    open_files: str = "",
    chat_history: str = "",
    custom_context: str = "",
    workspace_path: str = "",
    provider: str = "zai",
    model: str = "",
    max_tokens: int = 8000,
    temperature: float = 0.3,
    ctx: Context = None,
) -> str:
    """Enhance a user prompt into a detailed, structured, actionable prompt.
    
    This tool transforms vague or ambiguous user requests into comprehensive,
    precisely structured prompts optimized for AI agent execution. The enhanced
    prompt is returned for user review - it does NOT auto-submit.
    
    Workflow:
    1. User types a prompt (e.g., "add oauth support")
    2. Call this tool to enhance it (optionally with context)
    3. Review the enhanced prompt (structured with OBJECTIVE, CONTEXT, REQUIREMENTS, etc.)
    4. Modify if needed
    5. Submit the enhanced version to your AI agent
    
    Context Enrichment Options:
    - include_memories: Adds relevant learnings/patterns from ACE memory
    - include_git_commits: Adds recent commit history for project context
    - include_git_status: Adds current modified/staged files
    - open_files: Pass currently open file paths/content for context
    - chat_history: Pass recent conversation messages for continuity
    - custom_context: Any additional context you want to include
    
    Args:
        prompt: The original user prompt to enhance
        include_memories: If True, retrieves relevant ACE memories (default True)
        include_git_commits: If True, includes recent git commits (default False)
        include_git_status: If True, includes current git status (default False)
        open_files: String containing open file paths/content (optional)
        chat_history: String containing recent chat messages (optional)
        custom_context: Any additional context string (optional)
        workspace_path: Path to workspace for git operations (optional, uses cwd if empty)
        provider: LLM provider - "zai" (default GLM), "openai", "anthropic", "lmstudio"
        model: Specific model to use (empty = provider default)
        max_tokens: Maximum tokens for enhancement response (default 8000)
        temperature: LLM temperature (default 0.3 for focused enhancement)
    
    Returns:
        The enhanced, structured prompt ready for review and submission
    """
    _log_startup_info()
    
    logger.info(f"ace_enhance_prompt called: prompt_len={len(prompt)}, provider={provider}, model={model}")
    
    # Load the enhancement system prompt
    try:
        enhancement_system_prompt = _load_enhancement_prompt()
    except FileNotFoundError as e:
        return f"Error: {e}"
    
    # ========================================================================
    # Gather all context elements
    # ========================================================================
    context_parts = []
    
    # 1. ACE Memory Context
    if include_memories:
        try:
            index = get_memory_index()
            memory_results = await asyncio.to_thread(
                index.retrieve,
                query=prompt,
                limit=5,
                namespace=None,  # Search all namespaces
                auto_detect_preset=True,
                use_cross_encoder=True,
            )
            if memory_results:
                memory_text = format_unified_context(memory_results)
                context_parts.append(f"**Relevant Learnings from ACE Memory:**\n{memory_text}")
                logger.info(f"Retrieved {len(memory_results)} memories for context enrichment")
        except Exception as e:
            logger.warning(f"Memory retrieval failed (continuing without): {e}")
    
    # 2. Git Commits Context
    effective_workspace = workspace_path or os.getcwd()
    if include_git_commits:
        git_commits = _get_recent_git_commits(effective_workspace)
        if git_commits:
            context_parts.append(f"**Recent Git Commits:**\n```\n{git_commits}\n```")
            logger.info("Added git commits to context")
    
    # 3. Git Status Context
    if include_git_status:
        git_status = _get_git_status(effective_workspace)
        if git_status:
            context_parts.append(f"**Current Git Status (modified/staged files):**\n```\n{git_status}\n```")
            logger.info("Added git status to context")
    
    # 4. Open Files Context
    if open_files and open_files.strip():
        context_parts.append(f"**Currently Open Files:**\n{open_files.strip()}")
        logger.info("Added open files to context")
    
    # 5. Chat History Context
    if chat_history and chat_history.strip():
        # Truncate if too long
        max_history_len = 2000
        history_text = chat_history.strip()
        if len(history_text) > max_history_len:
            history_text = history_text[-max_history_len:] + "\n... (truncated older messages)"
        context_parts.append(f"**Recent Chat History:**\n{history_text}")
        logger.info("Added chat history to context")
    
    # 6. Custom Context
    if custom_context and custom_context.strip():
        context_parts.append(f"**Additional Context:**\n{custom_context.strip()}")
        logger.info("Added custom context")
    
    # ========================================================================
    # Build the final prompts
    # ========================================================================
    
    # Build the final system prompt with user input placeholder replaced
    # The enhancement prompt has {{userInput}} at the end
    system_prompt = enhancement_system_prompt.replace("{{userInput}}", prompt)
    
    # Build user message with all context
    if context_parts:
        context_block = "\n\n---\n\n".join(context_parts)
        user_message = f"""**Context for Enhancement:**

{context_block}

---

**User's Original Request:**
{prompt}"""
    else:
        user_message = prompt
    
    # Get LLM configuration
    config = get_config()
    llm_config = config.llm
    
    # Determine API credentials based on provider
    if provider == "zai" or provider == "glm":
        api_key = llm_config.api_key
        api_base = llm_config.api_base
        actual_model = model if model else llm_config.model  # Default: glm-4.7
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        actual_model = model if model else "gpt-4o"
    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        api_base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        actual_model = model if model else "claude-sonnet-4-20250514"
    elif provider == "lmstudio":
        api_key = "not-needed"
        api_base = llm_config.local_llm_url
        actual_model = model if model else llm_config.local_llm_model
    else:
        return f"Error: Unknown provider '{provider}'. Use: zai, openai, anthropic, lmstudio"
    
    if not api_key and provider != "lmstudio":
        return f"Error: No API key configured for provider '{provider}'"
    
    # Use LiteLLM client for the LLM call
    try:
        from ace.llm_providers import LiteLLMClient, LiteLLMConfig
        
        # Create client with individual parameters (not config object as first arg)
        client = LiteLLMClient(
            model=actual_model,
            api_key=api_key,
            api_base=api_base,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        logger.info(f"Calling {provider}/{actual_model} for prompt enhancement...")
        
        # Make the LLM call
        response = await asyncio.to_thread(
            client.complete,
            prompt=user_message,
            system=system_prompt,
            timeout=120.0,  # Long timeout for complex enhancement
        )
        
        enhanced_prompt = response.text.strip()
        
        # Log success
        logger.info(f"Enhancement complete: {len(enhanced_prompt)} chars")
        
        return enhanced_prompt
        
    except ImportError:
        return "Error: LiteLLM client not available. Run: pip install litellm"
    except Exception as e:
        logger.exception(f"Enhancement failed: {e}")
        return f"Error enhancing prompt: {e}"


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the MCP server."""
    logger.info("Starting ACE MCP Server (FastMCP with list_roots)...")
    server.run()


if __name__ == "__main__":
    main()
