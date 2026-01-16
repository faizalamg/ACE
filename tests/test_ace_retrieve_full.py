"""Full test of ACE retrieve with code context + memories."""
import os

# Set environment before imports
os.environ['VOYAGE_API_KEY'] = os.environ.get('VOYAGE_API_KEY', 'pa-3OeAEQVzfUwm9e0UKyW3L1JST8HJ4WT2sReaIv8GdDr')
os.environ['ACE_WORKSPACE_PATH'] = os.environ.get('ACE_WORKSPACE_PATH', os.getcwd())

from ace.code_retrieval import CodeRetrieval
from ace.unified_memory import UnifiedMemoryIndex
from ace.config import get_config

def test_code_retrieval():
    """Test code retrieval with Voyage embeddings."""
    print("=" * 60)
    print("TEST 1: CODE RETRIEVAL")
    print("=" * 60)
    
    cr = CodeRetrieval()
    results = cr.search('UnifiedMemoryIndex class', limit=3)
    
    print(f"Found {len(results)} code results\n")
    for i, r in enumerate(results):
        score = r['score']
        path = r['file_path']
        lines = f"{r['start_line']}-{r['end_line']}"
        print(f"Result {i+1} (score: {score:.4f}): {path} lines {lines}")
        content = r['content'][:200].replace('\n', ' ')
        print(f"   Preview: {content}...")
        print()
    
    # Format in ThatOtherContextEngine style
    formatted = cr.format_ThatOtherContextEngine_style(results)
    print("=== ThatOtherContextEngine-STYLE FORMAT (first 1000 chars) ===")
    print(formatted[:1000])
    print()
    return results


def test_memory_retrieval():
    """Test memory retrieval."""
    print("=" * 60)
    print("TEST 2: MEMORY RETRIEVAL")
    print("=" * 60)
    
    config = get_config()
    index = UnifiedMemoryIndex(
        collection_name=config.qdrant.unified_collection,
        qdrant_url=config.qdrant.url,
    )
    
    results = index.retrieve(
        query="UnifiedMemoryIndex class retrieval",
        limit=3,
        auto_detect_preset=True,
        use_cross_encoder=True,
    )
    
    print(f"Found {len(results)} memory results\n")
    for r in results[:3]:
        cat = r.category.name if hasattr(r.category, 'name') else str(r.category)
        print(f"[{cat}] (sev={r.severity}) {r.content[:100]}...")
        print()
    
    return results


def test_blended_retrieval():
    """Test blended code + memory retrieval (what ace_retrieve does)."""
    print("=" * 60)
    print("TEST 3: BLENDED RETRIEVAL (ace_retrieve simulation)")
    print("=" * 60)
    
    query = "UnifiedMemoryIndex class retrieval implementation"
    
    # 1. Code retrieval
    cr = CodeRetrieval()
    code_results = cr.search(query, limit=3)
    
    # 2. Memory retrieval
    config = get_config()
    index = UnifiedMemoryIndex(
        collection_name=config.qdrant.unified_collection,
        qdrant_url=config.qdrant.url,
    )
    memory_results = index.retrieve(
        query=query,
        limit=3,
        auto_detect_preset=True,
        use_cross_encoder=True,
    )
    
    # 3. Format combined output
    parts = []
    
    if code_results:
        formatted_code = cr.format_ThatOtherContextEngine_style(code_results)
        parts.append("**Code Context:**\n" + formatted_code)
    
    if memory_results:
        from ace_mcp_server import format_unified_context
        formatted_memories = format_unified_context(memory_results)
        parts.append(formatted_memories)
    
    combined = "\n\n".join(parts)
    
    print("=== COMBINED OUTPUT ===")
    print(combined[:2000])
    print()
    
    return combined


def test_workspace_isolation():
    """Verify workspace isolation."""
    print("=" * 60)
    print("TEST 4: WORKSPACE ISOLATION")
    print("=" * 60)
    
    from qdrant_client import QdrantClient
    client = QdrantClient(url="http://localhost:6333")
    
    # Get unique file paths
    results = client.scroll(
        collection_name="ace_code_context",
        limit=500,
        with_payload=True,
        with_vectors=False
    )
    
    unique_paths = set()
    for point in results[0]:
        path = point.payload.get("file_path", "")
        # Get root directory or filename
        root = path.split("/")[0] if "/" in path else path
        unique_paths.add(root)
    
    ace_expected = {"ace", "tests", "scripts", "examples", "docs", "benchmarks", "rag_training", ".ace"}
    
    print(f"Total unique root paths: {len(unique_paths)}")
    print(f"Root directories: {sorted(unique_paths)}")
    
    # Check for unexpected directories (excluding root .py/.md files)
    unexpected = {p for p in unique_paths if not p.endswith(('.py', '.md')) and p not in ace_expected}
    
    if unexpected:
        print(f"\n[WARN] Unexpected directories: {unexpected}")
    else:
        print("\n[PASS] All indexed code is from ACE workspace only")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ACE RETRIEVE FULL VERIFICATION")
    print("=" * 60 + "\n")
    
    test_code_retrieval()
    test_memory_retrieval()
    test_blended_retrieval()
    test_workspace_isolation()
    
    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
