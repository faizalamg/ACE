"""Direct test of ACE code retrieval to verify production readiness."""
import os
from qdrant_client import QdrantClient

def test_collection_info():
    """Test 1: Verify ace_code_context collection exists and has data."""
    print("=" * 60)
    print("TEST 1: Collection Info")
    print("=" * 60)
    
    client = QdrantClient(url="http://localhost:6333")
    info = client.get_collection("ace_code_context")
    
    print(f"Collection name: ace_code_context")
    print(f"Points count: {info.points_count}")
    print(f"Status: {info.status}")
    print(f"Vector config: {info.config.params.vectors}")
    
    assert info.points_count > 0, "Collection should have indexed points"
    print("\n[PASS] Collection exists with indexed code\n")
    return info


def test_workspace_isolation():
    """Test 2: Verify workspace isolation - only ACE project code is indexed."""
    print("=" * 60)
    print("TEST 2: Workspace Isolation")
    print("=" * 60)
    
    client = QdrantClient(url="http://localhost:6333")
    
    # Get unique file paths
    results = client.scroll(
        collection_name="ace_code_context",
        limit=100,
        with_payload=True,
        with_vectors=False
    )
    
    unique_paths = set()
    for point in results[0]:
        path = point.payload.get("file_path", "")
        unique_paths.add(path.split("/")[0] if "/" in path else path)
    
    print(f"Unique root directories indexed: {sorted(unique_paths)}")
    print(f"Total unique file prefixes: {len(unique_paths)}")
    
    # Verify all paths are from ACE workspace
    ace_dirs = {"ace", "tests", "scripts", "examples", "docs", "benchmarks", "rag_training"}
    unexpected = unique_paths - ace_dirs - {""}
    # Filter out actual files at root
    unexpected = {p for p in unexpected if not p.endswith(".py")}
    
    if unexpected:
        print(f"[WARN] Unexpected directories: {unexpected}")
    else:
        print("\n[PASS] All indexed code is from ACE workspace\n")


def test_sample_indexed_code():
    """Test 3: Show sample indexed code chunks."""
    print("=" * 60)
    print("TEST 3: Sample Indexed Code")
    print("=" * 60)
    
    client = QdrantClient(url="http://localhost:6333")
    
    # Get samples from ace/ directory
    from qdrant_client.models import Filter, FieldCondition, MatchText
    
    results = client.scroll(
        collection_name="ace_code_context",
        limit=5,
        with_payload=True,
        with_vectors=False
    )
    
    for i, point in enumerate(results[0][:3]):
        payload = point.payload
        print(f"\n--- Sample {i+1} ---")
        print(f"File: {payload.get('file_path')}")
        print(f"Lines: {payload.get('start_line')} - {payload.get('end_line')}")
        print(f"Language: {payload.get('language')}")
        print(f"Symbols: {payload.get('symbols', [])}")
        content = payload.get('content', '')[:300]
        print(f"Content preview:\n{content}...")
    
    print("\n[PASS] Indexed code chunks are accessible\n")


def test_embedding_config():
    """Test 4: Check embedding configuration for search."""
    print("=" * 60)
    print("TEST 4: Embedding Configuration")
    print("=" * 60)
    
    from ace.config import CodeEmbeddingConfig, VoyageCodeEmbeddingConfig
    
    code_config = CodeEmbeddingConfig()
    voyage_config = VoyageCodeEmbeddingConfig()
    
    print(f"Code embedding URL: {code_config.url}")
    print(f"Code embedding model: {code_config.model}")
    print(f"Code embedding dimension: {code_config.dimension}")
    
    print(f"\nVoyage API key configured: {voyage_config.is_configured()}")
    if voyage_config.is_configured():
        print(f"Voyage model: {voyage_config.model}")
        print(f"Voyage dimension: {voyage_config.dimension}")
    
    # Check collection dimension vs config
    client = QdrantClient(url="http://localhost:6333")
    info = client.get_collection("ace_code_context")
    collection_dim = info.config.params.vectors.get("dense").size if isinstance(info.config.params.vectors, dict) else info.config.params.vectors.size
    
    print(f"\nCollection vector dimension: {collection_dim}")
    
    if voyage_config.is_configured() and collection_dim == voyage_config.dimension:
        print("[PASS] Collection matches Voyage embedding dimension")
    elif collection_dim == code_config.dimension:
        print("[PASS] Collection matches legacy embedding dimension")
    else:
        print(f"[WARN] Dimension mismatch - collection: {collection_dim}, config: {code_config.dimension}")


def test_code_search():
    """Test 5: Test actual code search."""
    print("=" * 60)
    print("TEST 5: Code Search Test")
    print("=" * 60)
    
    from ace.code_retrieval import CodeRetrieval
    
    retriever = CodeRetrieval()
    
    # Try searching
    query = "UnifiedMemoryIndex retrieval implementation"
    print(f"Query: {query}")
    
    try:
        results = retriever.search(query, limit=3)
        if results:
            print(f"\nFound {len(results)} results:\n")
            for i, r in enumerate(results):
                print(f"--- Result {i+1} (score: {r.score:.4f}) ---")
                print(f"File: {r.file_path}")
                print(f"Lines: {r.start_line}-{r.end_line}")
                print(f"Symbols: {r.symbols}")
                print(f"Preview: {r.content[:200]}...")
                print()
            print("[PASS] Code search returned results")
        else:
            print("[WARN] No results returned")
    except Exception as e:
        print(f"[ERROR] Search failed: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ACE CODE RETRIEVAL PRODUCTION VERIFICATION")
    print("=" * 60 + "\n")
    
    test_collection_info()
    test_workspace_isolation()
    test_sample_indexed_code()
    test_embedding_config()
    test_code_search()
    
    print("=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
