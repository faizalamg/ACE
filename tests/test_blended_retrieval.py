"""Test ace_retrieve blended results (code + memory)."""
import os
os.environ['VOYAGE_API_KEY'] = os.environ.get('VOYAGE_API_KEY', 'pa-3OeAEQVzfUwm9e0UKyW3L1JST8HJ4WT2sReaIv8GdDr')
os.environ['ACE_WORKSPACE_PATH'] = os.environ.get('ACE_WORKSPACE_PATH', os.getcwd())

from ace.code_retrieval import CodeRetrieval
from ace.unified_memory import UnifiedMemoryIndex, format_unified_context

print("=" * 60)
print("TESTING BLENDED RETRIEVAL (CODE + MEMORY)")
print("=" * 60)

query = "UnifiedMemoryIndex retrieve implementation"
limit = 5

# 1. Code retrieval
print("\n--- CODE RETRIEVAL ---")
cr = CodeRetrieval()
code_results = cr.search(query, limit=limit)
if code_results:
    formatted_code = cr.format_ThatOtherContextEngine_style(code_results)
    print(f"Found {len(code_results)} code results")
    print(formatted_code[:1500])
else:
    print("No code results")

# 2. Memory retrieval  
print("\n--- MEMORY RETRIEVAL ---")
from ace.config import get_config
config = get_config()
index = UnifiedMemoryIndex(
    collection_name=config.qdrant.unified_collection,
    qdrant_url=config.qdrant.url,
)
memory_results = index.retrieve(
    query=query,
    limit=limit,
    auto_detect_preset=True,
    use_cross_encoder=True,
)
if memory_results:
    formatted_memories = format_unified_context(memory_results)
    print(f"Found {len(memory_results)} memory results")
    print(formatted_memories)
else:
    print("No memory results")

# 3. Combined output (what ace_retrieve should return)
print("\n" + "=" * 60)
print("COMBINED OUTPUT (ace_retrieve style)")
print("=" * 60)

combined_parts = []
if code_results:
    combined_parts.append(formatted_code)
if memory_results:
    combined_parts.append(formatted_memories)

combined = "\n\n".join(combined_parts) if combined_parts else "No results"
print(combined[:3000])
