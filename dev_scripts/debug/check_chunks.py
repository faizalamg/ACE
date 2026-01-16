"""Check indexed chunks for a specific file."""
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

c = QdrantClient('http://localhost:6333')
r = c.scroll(
    'ace_code_context', 
    limit=50,
    scroll_filter=Filter(
        must=[FieldCondition(key='file_path', match=MatchValue(value='ace/unified_memory.py'))]
    ),
    with_payload=True
)

print(f"Found {len(r[0])} chunks for ace/unified_memory.py")
print()

for p in sorted(r[0], key=lambda x: x.payload['start_line'])[:20]:
    print(f"{p.payload['start_line']}-{p.payload['end_line']}: {p.payload.get('symbols', [])}")
