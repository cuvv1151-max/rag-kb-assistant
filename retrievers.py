"""Memory retrievers. Each retriever returns a ranked list of (memory_id, score)."""
import json
import re
import time
from pathlib import Path

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


class RerankRetriever:
    """Two-stage retrieval: vector search proposes candidates, a cross-encoder
    re-scores each (query, memory) pair and re-orders them.

    A bi-encoder (the vector stage) encodes query and memory separately and
    only compares the two vectors. A cross-encoder reads both texts together,
    so it can judge whether the memory actually answers the query. It is much
    slower, which is why it only re-scores a short candidate list.
    """

    name = "rerank"

    def __init__(
        self,
        memories: list[dict],
        reranker_name: str = "BAAI/bge-reranker-base",
        candidates: int = 10,
    ):
        import torch
        from sentence_transformers import CrossEncoder

        self.vector = VectorRetriever(memories)
        self.reranker = CrossEncoder(reranker_name)
        self.activation = torch.nn.Sigmoid()  # map raw logits to a 0-1 relevance score
        self.candidates = candidates
        self.content = {m["id"]: m["content"] for m in memories}

    def retrieve(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        # Stage 1: cheap recall-oriented candidate generation.
        candidate_ids = [mid for mid, _ in self.vector.retrieve(query, k=self.candidates)]
        # Stage 2: precise re-scoring of each (query, memory) pair.
        pairs = [(query, self.content[mid]) for mid in candidate_ids]
        scores = self.reranker.predict(pairs, activation_fn=self.activation)
        ranked = sorted(zip(candidate_ids, scores), key=lambda item: item[1], reverse=True)
        return [(mid, float(score)) for mid, score in ranked[:k]]


class LLMRetriever:
    """Two-stage retrieval with an LLM judge: vector search proposes candidates,
    then DeepSeek rates how useful each candidate is for replying to the message.

    Unlike a cross-encoder, the LLM brings world knowledge (slang, synonyms,
    implied meaning) and follows a task definition we write ourselves.
    Responses are cached on disk so repeated evaluation runs cost nothing,
    and every real API call is logged for the cost / latency analysis.

    Variant A0 (this class): rate every candidate 0-3, message first in the prompt.
    """

    name = "llm"
    MAX_SCORE = 3

    # deepseek-flash, peak-hour price in CNY per million tokens (checked 2026-10-15).
    PRICE_PER_MILLION = {"cache_hit": 0.04, "cache_miss": 2.0, "output": 8.0}

    SYSTEM_PROMPT = (
        "You select long-term memories that help an AI companion reply to the "
        "user's latest chat message. Answer in JSON only."
    )

    USER_PROMPT = """The user just said:
"{query}"

Candidate memories about the user:
{candidates}

Rate how useful each memory is for writing a natural, caring reply to this message:
3 = directly relevant; the reply would miss something important without it
2 = clearly related; worth bringing up
1 = loosely related; probably not worth bringing up
0 = unrelated
Judge by meaning, including implied meaning, synonyms and everyday slang,
not by shared words.

Return JSON in exactly this form, with one entry for every candidate:
{{"ratings": [{{"id": <memory id>, "score": <0-3>}}]}}"""

    def __init__(
        self,
        memories: list[dict],
        candidates: int = 10,
        usage_log_path: str = "cache/usage_log.jsonl",
    ):
        import os

        from openai import OpenAI

        self.vector = VectorRetriever(memories)
        self.content = {m["id"]: m["content"] for m in memories}
        self.candidates = candidates
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
        )

        # One cache file per (model, variant), so runs never mix and can be compared.
        self.cache_path = Path(f"cache/judgments_{self.model}_{self.name}.json")
        self.cache = (
            json.loads(self.cache_path.read_text(encoding="utf-8"))
            if self.cache_path.exists()
            else {}
        )
        self.usage_log_path = Path(usage_log_path)
        # Usage statistics for this run.
        self.api_calls = 0
        self.cache_hits = 0
        self.prompt_tokens = 0
        self.prompt_cache_hit_tokens = 0
        self.completion_tokens = 0
        self.total_seconds = 0.0

    # ----- prompt and parsing: overridden by the optimised variants -----

    def build_messages(self, query: str, listing: str) -> list[dict]:
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.USER_PROMPT.format(query=query, candidates=listing)},
        ]

    def parse_scores(self, reply: dict, candidate_ids: list[int]) -> dict[int, int]:
        scores = {mid: 0 for mid in candidate_ids}
        for item in reply["ratings"]:
            mid, score = int(item["id"]), int(item["score"])
            if mid in scores:  # ignore ids the model invented
                scores[mid] = max(0, min(self.MAX_SCORE, score))
        return scores

    # ----- shared machinery -----

    def _judge(self, query: str, candidate_ids: list[int]) -> dict[int, int]:
        """Return {memory_id: score} for the candidates, using the cache when possible."""
        # Present candidates in id order: deterministic, and it does not leak the
        # vector ranking to the judge. The cache key uses the same order.
        candidate_ids = sorted(candidate_ids)
        cache_key = f"{self.model}|{self.name}|{query}|{','.join(map(str, candidate_ids))}"
        if cache_key in self.cache:
            self.cache_hits += 1
            return {int(k): v for k, v in self.cache[cache_key].items()}

        listing = "\n".join(f"[{mid}] {self.content[mid]}" for mid in candidate_ids)
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.build_messages(query, listing),
            response_format={"type": "json_object"},
            temperature=0,
            # deepseek-flash thinks by default; reasoning tokens are billed and slow.
            extra_body={"thinking": {"type": "disabled"}},
        )
        seconds = time.perf_counter() - start
        usage = response.usage
        hit_tokens = getattr(usage, "prompt_cache_hit_tokens", None) or 0
        self.api_calls += 1
        self.total_seconds += seconds
        self.prompt_tokens += usage.prompt_tokens
        self.prompt_cache_hit_tokens += hit_tokens
        self.completion_tokens += usage.completion_tokens
        self._log_usage(query, response.model, usage, hit_tokens, seconds)

        try:
            scores = self.parse_scores(json.loads(response.choices[0].message.content), candidate_ids)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Do not cache a failed reply, so the next run retries this query.
            print(f"  [warn] could not parse LLM output for: {query}")
            return {mid: 0 for mid in candidate_ids}

        self.cache[cache_key] = scores
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return scores

    def _log_usage(self, query, served_model, usage, hit_tokens, seconds) -> None:
        """Append one line per real API call, so cost figures survive the terminal."""
        record = {
            "variant": self.name,
            "requested_model": self.model,
            "served_model": served_model,
            "query": query,
            "prompt_tokens": usage.prompt_tokens,
            "prompt_cache_hit_tokens": hit_tokens,
            "completion_tokens": usage.completion_tokens,
            "seconds": round(seconds, 3),
        }
        self.usage_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.usage_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def retrieve(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        candidate_ids = [mid for mid, _ in self.vector.retrieve(query, k=self.candidates)]
        scores = self._judge(query, candidate_ids)
        # Sort by LLM score; ties keep the vector order (Python's sort is stable).
        ranked = sorted(candidate_ids, key=lambda mid: scores[mid], reverse=True)
        return [(mid, scores[mid] / self.MAX_SCORE) for mid in ranked[:k]]

    def report(self) -> None:
        """Print API usage and an estimated cost for the calls made in this run."""
        print(f"\nLLM usage ({self.name}, {self.model}): "
              f"{self.api_calls} API calls, {self.cache_hits} cache hits")
        if not self.api_calls:
            return
        n = self.api_calls
        miss_tokens = self.prompt_tokens - self.prompt_cache_hit_tokens
        price = self.PRICE_PER_MILLION
        cost = (self.prompt_cache_hit_tokens * price["cache_hit"]
                + miss_tokens * price["cache_miss"]
                + self.completion_tokens * price["output"]) / 1_000_000
        print(f"  prompt tokens:     {self.prompt_tokens / n:.0f} per call "
              f"({self.prompt_cache_hit_tokens / n:.0f} served from DeepSeek's prefix cache)")
        print(f"  completion tokens: {self.completion_tokens / n:.0f} per call")
        print(f"  latency:           {self.total_seconds / n:.2f} s per call")
        print(f"  est. cost:         {cost / n * 10_000:.2f} CNY per 10k messages (peak-hour price)")


class LLMIdsRetriever(LLMRetriever):
    """Variant A1: return only the ids of directly relevant memories.

    The product only uses score-3 memories, so rating all ten candidates spends
    output tokens (the most expensive kind) on information that is thrown away.
    Prompt order is unchanged from A0, so only the output format differs.
    """

    name = "llm_ids"

    INSTRUCTIONS = """Select only the memories that are directly relevant: without them,
a natural, caring reply to this message would miss something important.
Leave out memories that are merely related.
Judge by meaning, including implied meaning, synonyms and everyday slang,
not by shared words.

Return JSON in exactly this form (an empty list if none qualify):
{"relevant_ids": [<memory id>, ...]}"""

    def build_messages(self, query: str, listing: str) -> list[dict]:
        user = (f'The user just said:\n"{query}"\n\n'
                f"Candidate memories about the user:\n{listing}\n\n{self.INSTRUCTIONS}")
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    def parse_scores(self, reply: dict, candidate_ids: list[int]) -> dict[int, int]:
        selected = {int(mid) for mid in reply["relevant_ids"]}
        # Selected memories get the top score; ids the model invented are ignored.
        return {mid: (self.MAX_SCORE if mid in selected else 0) for mid in candidate_ids}


class LLMIdsCachedRetriever(LLMIdsRetriever):
    """Variant A2: same wording as A1, but all fixed instructions come first.

    DeepSeek caches identical prompt prefixes and bills cache hits at about
    1/50 of the normal input price. Putting the fixed text in the system
    message, and the per-message content last, makes that prefix reusable.
    """

    name = "llm_ids_cached"

    def build_messages(self, query: str, listing: str) -> list[dict]:
        system = f"{self.SYSTEM_PROMPT}\n\n{self.INSTRUCTIONS}"
        user = (f"Candidate memories about the user:\n{listing}\n\n"
                f'The user just said:\n"{query}"')
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]


class LLMGroupedRetriever(LLMRetriever):
    """Variant A3: keep A0's 0-3 rubric and prompt order, but return only the
    non-zero scores, grouped by score.

    A1 cut output tokens but dropped the graded rubric, and the judge became
    more permissive (score-2 memories were selected too). A3 tests whether the
    rubric, not the output format, is what keeps the judge strict.
    """

    name = "llm_grouped"

    USER_PROMPT = """The user just said:
"{query}"

Candidate memories about the user:
{candidates}

Rate how useful each memory is for writing a natural, caring reply to this message:
3 = directly relevant; the reply would miss something important without it
2 = clearly related; worth bringing up
1 = loosely related; probably not worth bringing up
0 = unrelated
Judge by meaning, including implied meaning, synonyms and everyday slang,
not by shared words.

Return JSON that groups memory ids by score. Leave out memories that score 0,
and use an empty list for a score that no memory received:
{{"3": [<memory id>, ...], "2": [...], "1": [...]}}"""

    def parse_scores(self, reply: dict, candidate_ids: list[int]) -> dict[int, int]:
        scores = {mid: 0 for mid in candidate_ids}
        for level in (1, 2, 3):
            for mid in reply.get(str(level), []):
                mid = int(mid)
                if mid in scores:  # ignore ids the model invented
                    scores[mid] = max(scores[mid], level)
        return scores


class LLMScoreListRetriever(LLMRetriever):
    """Variant A4: keep A0's rubric, prompt order and one explicit score for
    every candidate, but return the scores as a bare list in candidate order.

    A3 omitted zero scores and its scores drifted upward. A4 tests whether
    explicitly scoring every candidate (including the zeros) is what keeps the
    judge calibrated, while still cutting the verbose JSON of A0.
    """

    name = "llm_score_list"

    USER_PROMPT = """The user just said:
"{query}"

Candidate memories about the user:
{candidates}

Rate how useful each memory is for writing a natural, caring reply to this message:
3 = directly relevant; the reply would miss something important without it
2 = clearly related; worth bringing up
1 = loosely related; probably not worth bringing up
0 = unrelated
Judge by meaning, including implied meaning, synonyms and everyday slang,
not by shared words.

Return JSON with exactly one score per candidate, in the same order as the list above:
{{"scores": [<score of the first candidate>, <score of the second candidate>, ...]}}"""

    def parse_scores(self, reply: dict, candidate_ids: list[int]) -> dict[int, int]:
        scores = [int(s) for s in reply["scores"]]
        if len(scores) != len(candidate_ids):
            # A misaligned list cannot be mapped back safely; treat it as a failed reply.
            raise ValueError(f"expected {len(candidate_ids)} scores, got {len(scores)}")
        return {mid: max(0, min(self.MAX_SCORE, s)) for mid, s in zip(candidate_ids, scores)}
