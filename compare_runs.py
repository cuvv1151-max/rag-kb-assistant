"""Compare two cached LLM judgment runs: repeatability of one setup, or
agreement between two models / prompt variants.

Usage:
    python compare_runs.py cache/llm_judgments_run1.json cache/llm_judgments.json
"""
import json
import sys
from collections import Counter

path_a, path_b = sys.argv[1], sys.argv[2]
run_a = json.load(open(path_a, encoding="utf-8"))
run_b = json.load(open(path_b, encoding="utf-8"))

# Match entries on (query, candidate ids), the last two parts of the cache key,
# so runs with different models or prompt variants can be compared.
def by_query(run: dict) -> dict:
    return {tuple(key.split("|")[-2:]): scores for key, scores in run.items()}


run_a, run_b = by_query(run_a), by_query(run_b)
shared_keys = sorted(set(run_a) & set(run_b))
print(f"Queries in run A: {len(run_a)}, run B: {len(run_b)}, compared: {len(shared_keys)}")

pairs = 0
exact = 0
top_level_flips = 0  # memories that are score 3 in one run but not the other
diffs = Counter()
changed = []
for key in shared_keys:
    query = key[0]
    for mid, score_a in run_a[key].items():
        score_b = run_b[key].get(mid)
        if score_b is None:
            continue
        pairs += 1
        diffs[abs(score_a - score_b)] += 1
        if score_a == score_b:
            exact += 1
        else:
            changed.append((query, mid, score_a, score_b))
        if (score_a == 3) != (score_b == 3):
            top_level_flips += 1

print(f"\n(query, memory) pairs compared: {pairs}")
print(f"Identical score:        {exact} ({exact / pairs:.1%})")
for gap in sorted(diffs):
    if gap:
        print(f"Differ by {gap} point(s):    {diffs[gap]}")
print(f"Crossed the score-3 line: {top_level_flips}")

if changed:
    print("\nChanged judgments (query / memory id: run A -> run B):")
    for query, mid, a, b in changed:
        print(f"  {query} / {mid}: {a} -> {b}")
