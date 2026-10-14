"""Memory retrievers. Each retriever returns a ranked list of (memory_id, score)."""
import re

# Characters kept for keyword matching: CJK ideographs, ASCII letters and digits.
_NON_WORD = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]+")


def char_bigrams(text: str) -> set[str]:
    """Split text on punctuation, then return the set of 2-character windows.

    Chinese has no spaces, so overlapping character bigrams are a common
    dictionary-free approximation of 'words'.
    """
    grams: set[str] = set()
    for segment in _NON_WORD.split(text.lower()):
        grams.update(segment[i : i + 2] for i in range(len(segment) - 1))
    return grams


def jaccard(a: set[str], b: set[str]) -> float:
    """Overlap of two sets divided by their union (0 = disjoint, 1 = identical)."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class KeywordRetriever:
    """Baseline: rank memories by bigram Jaccard overlap with the query.

    Mirrors the core idea of a keyword-overlap memory recall: no model,
    zero cost, but blind to synonyms and paraphrases.
    """

    name = "keyword"

    def __init__(self, memories: list[dict]):
        self.memories = memories
        # Pre-compute keywords once; tags are included as extra keywords.
        self.index = {
            m["id"]: char_bigrams(m["content"]) | set(m.get("tags", []))
            for m in memories
        }

    def retrieve(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        query_grams = char_bigrams(query)
        scored = [(mid, jaccard(query_grams, grams)) for mid, grams in self.index.items()]
        # Drop zero-score memories: no overlap means no evidence of relevance.
        scored = [item for item in scored if item[1] > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]