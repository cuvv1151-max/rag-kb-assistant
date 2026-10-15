"""Summarise cache/usage_log.jsonl per prompt variant: tokens, DeepSeek prefix-cache
hits, latency and estimated cost. Every number in the report should trace back here."""
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

LOG_PATH = Path(__file__).parent / "cache" / "usage_log.jsonl"
# deepseek-flash, peak-hour price in CNY per million tokens (checked 2026-10-15).
PRICE = {"cache_hit": 0.04, "cache_miss": 2.0, "output": 8.0}

records = [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines() if line]
by_variant = defaultdict(list)
for r in records:
    by_variant[r["variant"]].append(r)


def cost_per_10k(prompt: float, hit: float, completion: float) -> float:
    """CNY per 10,000 messages for the given average tokens per call."""
    per_call = (hit * PRICE["cache_hit"] + (prompt - hit) * PRICE["cache_miss"]
                + completion * PRICE["output"]) / 1_000_000
    return per_call * 10_000


for variant, rows in by_variant.items():
    n = len(rows)
    prompt = statistics.mean(r["prompt_tokens"] for r in rows)
    hit = statistics.mean(r["prompt_cache_hit_tokens"] for r in rows)
    completion = statistics.mean(r["completion_tokens"] for r in rows)
    latencies = sorted(r["seconds"] for r in rows)
    served = Counter(r["served_model"] for r in rows)
    hit_pattern = Counter(r["prompt_cache_hit_tokens"] for r in rows)

    print(f"=== {variant} ({n} calls, served by {dict(served)}) ===")
    print(f"  prompt tokens      {prompt:7.1f}   of which cache hits {hit:6.1f}")
    print(f"  cache-hit tokens per call -> number of calls: {dict(sorted(hit_pattern.items()))}")
    print(f"  completion tokens  {completion:7.1f}")
    print(f"  latency            median {statistics.median(latencies):.2f} s, "
          f"slowest {latencies[-1]:.2f} s")
    print(f"  cost per 10k msgs  {cost_per_10k(prompt, hit, completion):6.2f} CNY as measured, "
          f"{cost_per_10k(prompt, 0, completion):6.2f} CNY with no cache hits\n")
