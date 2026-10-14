"""Step 3: evaluate a retriever on the labelled query set."""
import json
from collections import defaultdict
from pathlib import Path

from retrievers import KeywordRetriever

DATA_DIR = Path(__file__).parent / "data"
K_VALUES = (1, 3, 5)


def load_data():
    memories = json.loads((DATA_DIR / "memories.json").read_text(encoding="utf-8"))
    queries = json.loads((DATA_DIR / "test_queries.json").read_text(encoding="utf-8"))
    return memories, queries


def recall_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    """Fraction of the relevant memories that appear in the top-k results."""
    hits = set(retrieved_ids[:k]) & set(relevant_ids)
    return len(hits) / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[int], relevant_ids: list[int]) -> float:
    """1 / rank of the first relevant result; 0 if none was retrieved."""
    for rank, mid in enumerate(retrieved_ids, start=1):
        if mid in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate(retriever, queries: list[dict], memories: list[dict]) -> None:
    content = {m["id"]: m["content"] for m in memories}
    per_category = defaultdict(lambda: defaultdict(list))
    negatives = []
    misses = []

    for q in queries:
        results = retriever.retrieve(q["query"], k=max(K_VALUES))
        retrieved_ids = [mid for mid, _ in results]

        # Recall is undefined when nothing is relevant; report negatives separately.
        if not q["relevant_ids"]:
            negatives.append((q, results))
            continue

        for bucket in (q["category"], "ALL"):
            for k in K_VALUES:
                per_category[bucket][f"R@{k}"].append(
                    recall_at_k(retrieved_ids, q["relevant_ids"], k)
                )
            per_category[bucket]["MRR"].append(
                reciprocal_rank(retrieved_ids, q["relevant_ids"])
            )

        if recall_at_k(retrieved_ids, q["relevant_ids"], max(K_VALUES)) < 1.0:
            misses.append((q, retrieved_ids))

    columns = [f"R@{k}" for k in K_VALUES] + ["MRR"]
    print(f"\n=== Retriever: {retriever.name} ===")
    print(f"{'category':10s} {'n':>3s} " + " ".join(f"{c:>6s}" for c in columns))
    for bucket in ["lexical", "semantic", "multi", "ALL"]:
        if bucket not in per_category:
            continue
        metrics = per_category[bucket]
        n = len(metrics["MRR"])
        row = " ".join(f"{sum(metrics[c]) / n:6.2f}" for c in columns)
        print(f"{bucket:10s} {n:3d} {row}")

    print(f"\nNegative queries (ideally nothing is returned):")
    for q, results in negatives:
        top = ", ".join(f"{mid}({score:.2f})" for mid, score in results[:3]) or "nothing"
        print(f"  {q['id']} {q['query']} -> {top}")

    print(f"\nQueries with relevant memories missing from top-{max(K_VALUES)}:")
    for q, retrieved_ids in misses:
        missing = [mid for mid in q["relevant_ids"] if mid not in retrieved_ids]
        print(f"  {q['id']} [{q['category']}] {q['query']}")
        for mid in missing:
            print(f"      missed {mid}: {content[mid]}")


if __name__ == "__main__":
    memories, queries = load_data()
    evaluate(KeywordRetriever(memories), queries, memories)