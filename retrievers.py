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

    

class VectorRetriever:
    """Rank memories by cosine similarity of sentence embeddings.

    Each memory is encoded once into a vector and stored in a Chroma
    collection. A query is encoded the same way, and Chroma returns the
    memories whose vectors point in the most similar direction.
    """

    name = "vector"

    def __init__(
        self,
        memories: list[dict],
        model_name: str = "BAAI/bge-small-zh-v1.5",
        db_path: str = "chroma_db",
    ):
        # Imported here so the keyword baseline runs without these packages.
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        client = chromadb.PersistentClient(path=db_path)

        # Rebuild the collection on every run so it never holds stale vectors.
        collection_name = "memories"
        if collection_name in [c.name for c in client.list_collections()]:
            client.delete_collection(collection_name)
        self.collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # rank by cosine distance
        )

        texts = [m["content"] for m in memories]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        self.collection.add(
            ids=[str(m["id"]) for m in memories],
            embeddings=embeddings.tolist(),
            documents=texts,
        )

    def retrieve(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        query_vec = self.model.encode([query], normalize_embeddings=True)
        result = self.collection.query(query_embeddings=query_vec.tolist(), n_results=k)
        ids = result["ids"][0]
        distances = result["distances"][0]
        # Chroma returns cosine distance = 1 - cosine similarity; convert back.
        return [(int(mid), 1.0 - dist) for mid, dist in zip(ids, distances)]