#!/usr/bin/env python3
"""Debug why code retrieval isn't working in MCP server."""

import os

print("=== ENV CHECK ===")
print(f"ACE_WORKSPACE_PATH: {os.environ.get('ACE_WORKSPACE_PATH', 'NOT SET')}")
print(f"QDRANT_URL: {os.environ.get('QDRANT_URL', 'NOT SET')}")
print(f"ACE_CODE_COLLECTION: {os.environ.get('ACE_CODE_COLLECTION', 'NOT SET')}")

print("\n=== QDRANT COLLECTIONS ===")
from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333")
collections = [c.name for c in client.get_collections().collections]
print(f"Collections: {collections}")

code_collection = os.environ.get("ACE_CODE_COLLECTION", "ace_code_context")
if code_collection in collections:
    info = client.get_collection(code_collection)
    print(f"Code collection '{code_collection}': {info.points_count} points")
else:
    print(f"Code collection '{code_collection}' NOT FOUND!")

print("\n=== DIRECT CODE RETRIEVAL TEST ===")
from ace.code_retrieval import CodeRetrieval
try:
    cr = CodeRetrieval()
    results = cr.search("retrieve code", limit=3)
    print(f"Code search returned {len(results)} results")
    if results:
        print(f"First result: {results[0].get('file_path', '?')}")
except Exception as e:
    print(f"Code retrieval FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n=== MCP SERVER HANDLE_RETRIEVE TEST ===")
import asyncio
from ace_mcp_server import handle_retrieve, get_code_retrieval

# Check if get_code_retrieval returns something
print(f"get_code_retrieval() returns: {get_code_retrieval()}")

async def test():
    result = await handle_retrieve({"query": "code search retrieval", "limit": 3})
    text = result[0].text
    # Check for Auggie-compatible format (starts with "The following code sections")
    has_code = "The following code sections were retrieved:" in text
    has_memory = "User Preferences" in text or "Project Context" in text
    print(f"Has Code: {has_code}")
    print(f"Has Memory: {has_memory}")
    print(f"BLENDED: {has_code and has_memory}")
    if not has_code:
        print("\n=== FULL RESPONSE (no code found) ===")
        print(text[:2000])
    return has_code and has_memory

asyncio.run(test())
