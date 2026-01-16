"""Check config.py chunks."""
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

c = QdrantClient('http://localhost:6333')
r = c.scroll(
    'ace_code_context', 
    limit=50,
    scroll_filter=Filter(must=[FieldCondition(key='file_path', match=MatchValue(value='ace/config.py'))]),
    with_payload=True
)

sorted_chunks = sorted(r[0], key=lambda x: x.payload['start_line'])
print(f"Total chunks for ace/config.py: {len(sorted_chunks)}")
print()

for p in sorted_chunks[:20]:
    content = p.payload['content'][:150].replace('\n', ' ')
    print(f"{p.payload['start_line']}-{p.payload['end_line']}: {content}...")
