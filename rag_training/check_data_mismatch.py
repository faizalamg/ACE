"""Quick script to check test suite vs Qdrant collection data."""
import json
from pathlib import Path

# Load test suite
test_suite_path = Path("rag_training/test_suite/enhanced_test_suite.json")
with open(test_suite_path, 'r') as f:
    test_suite = json.load(f)

# Extract memory IDs
memory_ids = set()
for tc in test_suite['test_cases']:
    memory_ids.add(tc['memory_id'])

print(f"Test Suite Analysis:")
print(f"  Total test cases: {len(test_suite['test_cases'])}")
print(f"  Unique memory IDs: {len(memory_ids)}")
print(f"  Total queries: {test_suite['metadata']['generation_stats']['total_queries_generated']}")

print(f"\nQdrant Collection:")
print(f"  Points: 820")
print(f"  Indexed vectors: 1969")

print(f"\nMismatch Analysis:")
print(f"  Expected points (test suite): {len(memory_ids)}")
print(f"  Actual points (Qdrant): 820")
print(f"  Difference: {820 - len(memory_ids)}")

print(f"\nSample memory IDs from test suite:")
for i, mid in enumerate(list(memory_ids)[:10], 1):
    print(f"  {i}. {mid}")

print(f"\nNext step: Query Qdrant to see if these IDs exist in collection...")
