"""Step 2: sanity-check the memory bank and the labelled query set."""
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

memories = json.loads((DATA_DIR / "memories.json").read_text(encoding="utf-8"))
queries = json.loads((DATA_DIR / "test_queries.json").read_text(encoding="utf-8"))

memory_ids = {m["id"] for m in memories}
print(f"Memories: {len(memories)}")
print(f"Queries:  {len(queries)}")

errors = []
seen_query_ids = set()
for q in queries:
    if q["id"] in seen_query_ids:
        errors.append(f"{q['id']}: duplicate query id")
    seen_query_ids.add(q["id"])

    unknown = [i for i in q["relevant_ids"] if i not in memory_ids]
    if unknown:
        errors.append(f"{q['id']}: unknown memory ids {unknown}")

    if q["category"] == "negative" and q["relevant_ids"]:
        errors.append(f"{q['id']}: negative query should have no relevant ids")
    if q["category"] != "negative" and not q["relevant_ids"]:
        errors.append(f"{q['id']}: non-negative query has no relevant ids")

print("\nQueries per category:")
for category, count in sorted(Counter(q["category"] for q in queries).items()):
    print(f"  {category:10s} {count}")

covered = {i for q in queries for i in q["relevant_ids"]}
print(f"\nMemories referenced by at least one query: {len(covered)}/{len(memories)}")

print("\nResult:", "OK" if not errors else "FOUND PROBLEMS")
for err in errors:
    print("  -", err)