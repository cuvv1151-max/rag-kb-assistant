"""Print the top-3 results with scores for every query, to inspect ranking errors
and to compare the scores of correct hits against those of negative queries."""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from retrievers import (
    KeywordRetriever,
    LLMGroupedRetriever,
    LLMIdsCachedRetriever,
    LLMIdsRetriever,
    LLMRetriever,
    LLMScoreListRetriever,
    RerankRetriever,
    VectorRetriever,
)

DATA_DIR = Path(__file__).parent / "data"
RETRIEVERS = {
    "keyword": KeywordRetriever,
    "vector": VectorRetriever,
    "rerank": RerankRetriever,
    "llm": LLMRetriever,
    "llm_ids": LLMIdsRetriever,
    "llm_ids_cached": LLMIdsCachedRetriever,
    "llm_grouped": LLMGroupedRetriever,
    "llm_score_list": LLMScoreListRetriever,
}

name = sys.argv[1] if len(sys.argv) > 1 else "vector"
memories = json.loads((DATA_DIR / "memories.json").read_text(encoding="utf-8"))
queries = json.loads((DATA_DIR / "test_queries.json").read_text(encoding="utf-8"))
content = {m["id"]: m["content"] for m in memories}
retriever = RETRIEVERS[name](memories)

hit_scores, negative_scores = [], []
for q in queries:
    results = retriever.retrieve(q["query"], k=3)
    print(f"\n{q['id']} [{q['category']}] {q['query']}  relevant={q['relevant_ids']}")
    for rank, (mid, score) in enumerate(results, start=1):
        mark = "OK " if mid in q["relevant_ids"] else "   "
        print(f"  {mark}#{rank} {score:.3f}  {mid}: {content[mid]}")
    if q["relevant_ids"]:
        hit_scores += [s for mid, s in results if mid in q["relevant_ids"]]
    elif results:
        negative_scores.append(results[0][1])

print("\n=== Score ranges ===")
if hit_scores:
    print(f"Correct hits in top-3:    min {min(hit_scores):.3f}  max {max(hit_scores):.3f}")
if negative_scores:
    print(f"Top-1 of negative queries: min {min(negative_scores):.3f}  max {max(negative_scores):.3f}")
