"""Step 5b: sweep a minimum-score threshold and measure the trade-off between
keeping correct memories and filtering out irrelevant ones."""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from retrievers import (
    KeywordRetriever,
    LLMIdsCachedRetriever,
    LLMIdsRetriever,
    LLMRetriever,
    RerankRetriever,
    VectorRetriever,
)

DATA_DIR = Path(__file__).parent / "data"
TOP_K = 3

# Score scales differ per retriever, so each gets its own threshold grid.
SETTINGS = {
    "keyword": (KeywordRetriever, [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]),
    "vector": (VectorRetriever, [0.40, 0.45, 0.50, 0.525, 0.55, 0.575, 0.60, 0.65]),
    "rerank": (RerankRetriever, [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70]),
    # LLM scores are 0-3 divided by 3, so only four distinct levels exist.
    "llm": (LLMRetriever, [0.0, 0.33, 0.66, 1.0]),
    # The ids-only variants are binary: selected (1.0) or not (0.0).
    "llm_ids": (LLMIdsRetriever, [0.0, 1.0]),
    "llm_ids_cached": (LLMIdsCachedRetriever, [0.0, 1.0]),
}


def main(name: str) -> None:
    retriever_cls, thresholds = SETTINGS[name]
    memories = json.loads((DATA_DIR / "memories.json").read_text(encoding="utf-8"))
    queries = json.loads((DATA_DIR / "test_queries.json").read_text(encoding="utf-8"))
    retriever = retriever_cls(memories)

    # Retrieve once; every threshold is applied to the same cached results.
    cached = [(q, retriever.retrieve(q["query"], k=TOP_K)) for q in queries]
    positives = [(q, r) for q, r in cached if q["relevant_ids"]]
    negatives = [(q, r) for q, r in cached if not q["relevant_ids"]]
    total_relevant = sum(len(q["relevant_ids"]) for q, _ in positives)

    print(f"=== Threshold sweep: {name}, top-{TOP_K} ===")
    print(f"{len(positives)} positive queries ({total_relevant} relevant memories), "
          f"{len(negatives)} negative queries\n")
    print(f"{'threshold':>9s} {'recall':>7s} {'neg_clean':>9s} {'noise/query':>11s}")

    for t in thresholds:
        # recall: share of relevant memories still in top-k after filtering
        kept = sum(
            1 for q, results in positives
            for mid, score in results
            if score >= t and mid in q["relevant_ids"]
        )
        # neg_clean: share of negative queries for which nothing survives
        clean = sum(1 for _, results in negatives if all(s < t for _, s in results))
        # noise: irrelevant memories that would be injected, averaged over all queries
        noise = sum(
            1 for q, results in cached
            for mid, score in results
            if score >= t and mid not in q["relevant_ids"]
        )
        print(f"{t:9.3f} {kept / total_relevant:7.2f} "
              f"{clean / len(negatives):9.2f} {noise / len(cached):11.2f}")

    # Which kind of negative is hardest to filter? Show the top-1 score per probe group.
    print("\nTop-1 score of negative queries, by probe type:")
    by_probe: dict[str, list[float]] = {}
    for q, results in negatives:
        top = results[0][1] if results else 0.0
        by_probe.setdefault(q.get("probe", "unlabelled"), []).append(top)
    for probe, scores in sorted(by_probe.items()):
        print(f"  {probe:14s} n={len(scores)}  "
              f"mean {sum(scores) / len(scores):.3f}  max {max(scores):.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "vector")
