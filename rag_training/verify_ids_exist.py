"""Verify test suite memory IDs exist in Qdrant collection."""
import json
from pathlib import Path
import httpx

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "ace_memories_hybrid"

# Load test suite
test_suite_path = Path("rag_training/test_suite/enhanced_test_suite.json")
with open(test_suite_path, 'r') as f:
    test_suite = json.load(f)

# Extract memory IDs
memory_ids = [tc['memory_id'] for tc in test_suite['test_cases']]

print(f"Checking if {len(memory_ids)} memory IDs exist in Qdrant...")

# Query Qdrant for these specific IDs
client = httpx.Client(timeout=60.0)

found_ids = []
missing_ids = []

for i, memory_id in enumerate(memory_ids, 1):
    if i % 10 == 0:
        print(f"  Checked {i}/{len(memory_ids)}...")

    try:
        resp = client.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/{memory_id}")
        if resp.status_code == 200:
            result = resp.json()
            if result.get('result'):
                found_ids.append(memory_id)
            else:
                missing_ids.append(memory_id)
        else:
            missing_ids.append(memory_id)
    except Exception as e:
        print(f"  Error checking ID {memory_id}: {e}")
        missing_ids.append(memory_id)

client.close()

print(f"\nResults:")
print(f"  Found: {len(found_ids)}/{len(memory_ids)} ({len(found_ids)/len(memory_ids)*100:.1f}%)")
print(f"  Missing: {len(missing_ids)}/{len(memory_ids)} ({len(missing_ids)/len(memory_ids)*100:.1f}%)")

if missing_ids:
    print(f"\nMissing IDs (first 20):")
    for mid in missing_ids[:20]:
        print(f"  {mid}")

print(f"\nConclusion:")
if len(found_ids) == len(memory_ids):
    print(f"  ALL test suite IDs exist in Qdrant - data integrity OK")
    print(f"  Problem must be in retrieval logic or query generation")
elif len(found_ids) == 0:
    print(f"  NO test suite IDs found in Qdrant - CRITICAL DATA MISMATCH")
    print(f"  Test suite was generated for a different dataset!")
else:
    print(f"  PARTIAL match - {len(found_ids)}/{len(memory_ids)} found")
    print(f"  Test suite may be stale or collection was modified")
