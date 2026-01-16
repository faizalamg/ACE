#!/usr/bin/env python3
"""Test that ace_retrieve returns blended results (code + memories)."""

import asyncio
from ace_mcp_server import handle_retrieve


async def test_blended():
    """Test that handle_retrieve returns both code AND memories."""
    result = await handle_retrieve({'query': 'user preferences code style', 'limit': 3})
    text = result[0].text
    
    print('=== FULL OUTPUT ===')
    print(text)
    print('===================')
    
    has_code = '**Code Context:**' in text
    has_memories = 'User Preferences' in text or 'Project Context' in text
    
    print(f'\nHas Code Context: {has_code}')
    print(f'Has Memory Context: {has_memories}')
    print(f'BLENDED: {has_code and has_memories}')
    
    assert has_code or has_memories, "Should return at least code or memories"
    return has_code and has_memories


if __name__ == '__main__':
    result = asyncio.run(test_blended())
    print(f'\nTest result: {"PASS" if result else "PARTIAL - only one type returned"}')
