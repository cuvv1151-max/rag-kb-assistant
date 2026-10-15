# Long-Term Memory Recall for an AI Companion: Keyword vs. Vector vs. Reranker vs. LLM Judge

## 1. Summary

An AI companion has to recall what the user told it before. The baseline recall
in our product scored memories by keyword overlap. This project built a labelled
test set and compared four retrieval methods on it, then wired the best one into
an end-to-end reply pipeline with LangChain and DeepSeek.

Main results:

1. **Keyword recall cannot handle paraphrase.** On the 13 queries that share no
   words with their target memory, it found none of them (Recall@5 = 0.00).
   Vector search found all 13 (Recall@5 = 1.00), but it cannot tell relevant
   from irrelevant: on unrelated messages it still returned memories with
   similar scores, so no single similarity threshold worked.
2. **An LLM judge on top of vector search was the only method that got both
   recall and precision.** Vector search proposes 10 candidates, and DeepSeek
   rates each one 0-3. Keeping only score-3 memories gave recall 0.92 and kept
   no irrelevant memory for any of the 31 messages, at about 0.0016 CNY and
   1.2 s per message.
3. **Two plausible upgrades made things worse, and the reasons are useful.**
   An off-the-shelf cross-encoder reranker lowered semantic Recall@1 from 0.62
   to 0.38, because it fails whenever the link needs world knowledge or slang.
   Four cheaper output formats for the LLM judge cut cost by 41-61%, but each
   one changed the judge's behaviour: three made it more permissive, and one
   broke parsing on about 30% of calls. Output format is not a neutral detail.

All numbers come from a small, synthetic test set (25 memories, 31 messages).
Section 10 lists the limitations; the results show directions, not production
guarantees.

## 2. Problem

The companion stores short facts about the user (job, pets, preferences,
recent events). For every new message it must pick the few memories that make
the reply personal, and leave out the rest. Two failure modes matter:

- **Missed memory.** The user says "my little master scratched the sofa again"
  (Chinese slang for a cat), and the reply ignores that the user has a cat
  called Qiuqiu.
- **Noise memory.** The user mentions the rain, and the reply brings up work
  stress for no reason. Repeated, this makes the companion feel scripted.

The keyword baseline only matches shared words, so it is exposed to the first
failure. This project measures how much semantic retrieval helps, and what it
costs in noise, money and latency.

## 3. Dataset

All data is synthetic and describes a fictional user.

- **Memory bank** (`data/memories.json`): 25 memories, each with a type
  (profile, preference, event, emotion, boundary) and tags.
- **Query set** (`data/test_queries.json`): 31 user messages, each labelled with
  the ids of the memories a good reply needs.

| Category | n | What it tests |
|---|---|---|
| lexical | 4 | Message and memory share words |
| semantic | 13 | Same meaning, **no shared two-character word** with the target memory |
| multi | 3 | Two memories are needed |
| negative | 11 | No memory is needed |

The 20 positive queries reference 24 relevant memories in total, covering 22 of
the 25 memories. Negative queries carry a `probe` label: `unrelated` (5),
`work-adjacent` (3: work topics with no matching memory) and `near-topic` (3:
close to a memory, e.g. football when the user plays basketball).

**How it was built.** Queries and labels were drafted with the help of an AI
assistant and reviewed by the author. The "no shared word" rule for semantic
queries was checked with a script; two drafts failed it and were rewritten (one
shared the common word 最近 "recently" with its target). `validate_dataset.py`
checks ids and category rules.

**Built-in bias.** Because semantic queries were filtered to share no words with
their target, the keyword baseline scores 0 on them by construction. That score
shows the size of the gap, not an independent measurement.

## 4. Methods

| Retriever | How it ranks memories |
|---|---|
| `keyword` | Character-bigram Jaccard overlap between message and memory (plus tags). Same idea as the product's existing recall. |
| `vector` | `BAAI/bge-small-zh-v1.5` sentence embeddings (normalised), cosine similarity in Chroma. Runs locally on CPU. |
| `rerank` | Vector top-10, re-scored by the cross-encoder `BAAI/bge-reranker-base` (sigmoid score 0-1). |
| `llm` | Vector top-10, rated 0-3 by DeepSeek in one call per message (Section 7 gives the prompt variants). |

**LLM judge settings.** Model `deepseek-flash`, thinking mode disabled,
temperature 0, JSON output. Candidates are listed in id order, so the vector
ranking is not leaked to the judge. Responses are cached on disk
(`cache/judgments_<model>_<variant>.json`), and every real API call is logged to
`cache/usage_log.jsonl`.

A model check (`check_model.py`) showed that the legacy name `deepseek-chat`
is served by `deepseek-flash` with no reasoning tokens, so earlier runs under
that name are comparable. Called without settings, `deepseek-flash` thinks by
default: it spent 29 reasoning tokens on a one-word reply. Reasoning tokens are
billed as output, so the code disables thinking on every call.

**Metrics.**

- **Recall@k and MRR** over the 20 positive queries, averaged per query.
- **Threshold sweep** (`threshold_sweep.py`), on the top 3 results, for a given
  minimum score:
  - `recall`: share of the 24 relevant memories that survive (pooled over memories)
  - `neg_clean`: share of the 11 negative messages for which nothing survives
  - `noise/query`: irrelevant memories that survive, averaged over all 31 messages

Because `recall` is pooled over memories and Recall@k is averaged per query, the
two can differ slightly for the same method.

## 5. Results

### Ranking quality (20 positive queries)

| Retriever | Category | R@1 | R@3 | R@5 | MRR |
|---|---|---|---|---|---|
| keyword | lexical (4) | 0.88 | 1.00 | 1.00 | 1.00 |
| | semantic (13) | 0.00 | 0.00 | 0.00 | 0.00 |
| | multi (3) | 0.33 | 0.33 | 0.33 | 0.67 |
| | **all (20)** | **0.23** | **0.25** | **0.25** | **0.30** |
| vector | lexical | 0.62 | 1.00 | 1.00 | 0.88 |
| | semantic | 0.62 | 0.92 | 1.00 | 0.79 |
| | multi | 0.50 | 0.83 | 0.83 | 1.00 |
| | **all** | **0.60** | **0.93** | **0.97** | **0.84** |
| rerank | lexical | 0.88 | 1.00 | 1.00 | 1.00 |
| | semantic | 0.38 | 0.69 | 1.00 | 0.60 |
| | multi | 0.33 | 0.83 | 0.83 | 0.78 |
| | **all** | **0.47** | **0.78** | **0.97** | **0.71** |
| llm (A0) | lexical | 0.88 | 1.00 | 1.00 | 1.00 |
| | semantic | 1.00 | 1.00 | 1.00 | 1.00 |
| | multi | 0.50 | 1.00 | 1.00 | 1.00 |
| | **all** | **0.90** | **1.00** | **1.00** | **1.00** |

Vector search fixes semantic recall but loses some lexical precision: on "which
city do I work in", the work-stress memory outranked the memory that names the
city. Only the LLM judge ranks well in every category.

### Filtering quality (threshold sweep, top 3)

| Retriever | Threshold | recall | neg_clean | noise/query |
|---|---|---|---|---|
| keyword | > 0 | 0.29 | 0.45 | 0.61 |
| keyword | 0.06 | 0.17 | 0.82 | 0.06 |
| vector | 0.45 | 0.92 | 0.18 | 1.55 |
| vector | 0.50 | 0.83 | 0.64 | 0.94 |
| vector | 0.55 | 0.67 | 0.82 | 0.29 |
| vector | 0.60 | 0.33 | 1.00 | 0.06 |
| rerank | 0.01 | 0.71 | 0.36 | 1.32 |
| rerank | 0.10 | 0.50 | 0.91 | 0.29 |
| llm (A0) | score >= 2 | 1.00 | 0.27 | 0.94 |
| **llm (A0)** | **score = 3** | **0.92** | **1.00** | **0.00** |

No similarity threshold on vector or rerank scores reaches both high recall and
clean negatives. For vector search, correct hits scored 0.458-0.719 and the top
result of negative messages scored 0.382-0.596; the ranges overlap. The LLM
judge at score 3 is the only setting that keeps most relevant memories and adds
no noise. Its two misses (basketball as a hobby for "my ankle still hurts", and
"pays attention to fitness" for a gym message) were rated 2, and both labels
are arguable.

## 6. Findings

### 6.1 "Hub" memories and the work cluster

Some memories are general and emotional, and vector search pulls them in for
almost any message. In the vector top 3 across 23 queries, "works overtime
under heavy pressure" (memory 13) appeared 12 times, 10 of them wrongly, and took
first place from the correct memory 4 times (including on "it's raining and I
forgot my umbrella").

The negative probes widened the picture. The highest-scoring negatives were
work-adjacent ("a new intern joined our team" 0.596, "a convenience store opened
below the office" 0.551), both pulled to "preparing to change jobs". The user's
work memories (job, project launch, job change, overtime) form a cluster that
attracts any work small talk. For a companion, which hears a lot of work small
talk, this is the noise failure that matters most.

### 6.2 Why the reranker hurt: it cannot make a leap

The cross-encoder improved lexical ranking (R@1 0.62 to 0.88) but lowered
semantic R@1 from 0.62 to 0.38. A first hypothesis, that it handles questions
but not statements, was contradicted by the data: it succeeded on some
statements and failed on others. The pattern that fits every case is whether the
link needs an extra step of knowledge:

| Succeeded: memory paraphrases the message | Failed: link needs slang, synonyms or inference |
|---|---|
| can't sleep at night / insomnia (0.860) | 主子 "little master" = cat (score ~0) |
| can't get up in the morning / dislikes early rising (0.315) | ranked games, losing streak = League of Legends |
| empty home, no one to talk to / loneliness (0.208) | 芫荽 = 香菜, two words for coriander (0.001) |
| don't preach at me / dislikes being lectured (0.163) | black coffee = Americano |
| | craving spicy food = Chongqing hotpot (0.041) |
| | dreaming of "her" = ex-girlfriend (0.016) |

The reranker was good at rejecting unrelated memories (10 of 11 negatives had a
top score below 0.07). Its weakness is narrow, but it hits the messages where
personal memory matters most.

### 6.3 Boundary memories should not be retrieved

The LLM judge often rated "dislikes being lectured, don't give too much advice"
(memory 20) as relevant to unrelated messages, such as "how do I care for
succulents" or "should I learn to drive". The judge is right: this memory is a
rule about how to reply, so it applies to every reply. Style and boundary
memories should be included in every prompt rather than compete in retrieval.
The original product already weighted the boundary type highest, which points
the same way.

### 6.4 The LLM judge is repeatable where it matters

Two runs of the same judge (temperature 0) gave identical scores on 293 of 310
(message, memory) pairs (94.5%). All 17 differences were one point, all were
between 0 and 2, and none crossed the score-3 line that the product uses. Twelve
of the 17 involved the same general memories that act as hubs in vector search
(work stress, loneliness, lecturing, insomnia). A run under the new model name
(`deepseek-flash`) agreed with the `deepseek-chat` run on 306 of 310 pairs
(98.7%), again with no score-3 changes.

## 7. Prompt iteration: output format is not neutral

The A0 judge returns a JSON object with an id and a score for every candidate.
Output tokens cost four times as much as input tokens, so four cheaper formats
were tested. Each variant changed one thing.

| Variant | Change from A0 | Output tokens | Median latency | Cost / 10k msgs* | recall | neg_clean | noise/query |
|---|---|---|---|---|---|---|---|
| **A0** | id + score for every candidate | 126.0 | 1.22 s | 16.45 CNY | 0.92 | **1.00** | **0.00** |
| A1 | return only ids of directly relevant memories | 9.6 | 0.92 s | 6.48 CNY | 1.00 | 0.45 | 0.90 |
| A2 | A1, fixed instructions moved before the message | 9.1 | 0.89 s | 6.45 CNY | 1.00 | 0.73 | 0.68 |
| A3 | A0 rubric, ids grouped by score, zeros omitted | 38.0 | 1.01 s | 9.69 CNY | 1.00 | 0.82 | 0.61 |
| A4 | A0 rubric, bare list of scores in candidate order | 35.0 | 1.02 s | 9.23 CNY | (see below) | | |

\* deepseek-flash peak-hour price, no prompt-cache discount, so variants are
comparable. Quality columns use score 3 (A0, A3, A4) or "selected" (A1, A2).

- **A1 and A2** cut cost by about 61%, but the judge became permissive. Against
  A0, A1 changed 33 decisions at the score-3 line, and every change was upward:
  memories A0 rated 2 or lower were now selected. Moving the instructions (A2)
  changed decisions as well, not only cost.
- **A3** tested whether the 0-3 rubric was what kept A0 strict. It was not:
  with the identical rubric, omitting zeros shifted scores up by about one level
  (53.9% agreement with A0, 21 upward crossings of the score-3 line).
- **A4** kept an explicit score for every candidate but dropped the ids. On its
  first run, 9 of 31 replies could not be parsed (the code rejects a list of the
  wrong length rather than guess the mapping), and failures recurred on retry.
  The 24 parsed replies agreed with A0 on 80.8% of pairs, with 5 score-3
  changes in both directions. Because failed replies count as all-zero, A4's
  sweep numbers are not comparable and are left out.

**Conclusion: keep A0.** Writing an id and a score for every candidate,
including the zeros, is what keeps the judge calibrated. The savings from the
cheaper formats came with a real cost in noise or reliability.

These five variants were compared on the same 31 messages they were designed
on, so the ranking between them may not hold on new data.

## 8. Cost and latency

Prices are for `deepseek-flash` at peak hours, checked on 2026-10-15 (CNY per
million tokens): input 2.00 (cache miss) or 0.04 (cache hit), output 8.00.
Off-peak prices are half.

| | A0 judge, per message |
|---|---|
| Tokens | 318 input + 126 output |
| Cost | 0.0016 CNY (16.45 CNY per 10k messages) without cache discount; 13.94 CNY per 10k as measured |
| Latency | median 1.22 s, slowest 1.43 s (31 calls) |

Money is not the constraint: at 50 messages a day, retrieval costs about
0.08 CNY per user per day. **Latency is.** The judge adds about 1.2 s before
the reply can start, which users notice in a chat app.

**Prompt cache.** DeepSeek reports cached input tokens per call. Every A0, A3 and
A4 call had exactly 128 cached tokens; every A1 and A2 call had none, including
A2, whose prompt was rearranged so that all fixed text came first. The rule
behind this could not be determined from client-side measurements, and an
earlier guess (reuse of recently sent identical requests) does not explain why
A1 and A2 got no hits. For a prompt this short, cutting output tokens matters
more than caching.

## 9. End-to-end demo

`chat_demo.py` builds the reply with LangChain: a `ChatPromptTemplate` (persona,
selected memories, message), `ChatOpenAI` pointed at DeepSeek, and
`StrOutputParser`, joined as `prompt | llm | parser`. Each demo message is
answered three ways: no memories, keyword top 3, and the recommended pipeline
(LLM judge score-3 memories plus boundary memories, which are always included).
Full replies are in `results/demo_replies.md`.

What the five examples show:

- **The generator already understands slang.** Without any memory, it knew that
  "little master" means a cat and "losing streak" means gaming. Retrieval adds
  specifics, not understanding: the recommended pipeline called the cat by name
  (Qiuqiu, an orange tabby), mentioned the user's jungler role, and asked
  whether the user's mother had recovered from last month's hospital stay.
- **Keyword retrieval found nothing for three of the five messages**, so those
  replies were the same as with no memory.
- **Without memory, the model did not invent facts.** Asked which city the user
  works in, it said it did not remember instead of guessing.
- **One comparison is confounded.** The persona prompt already says "do not
  lecture" in every condition, so the demo cannot show what the boundary memory
  adds.

This is a qualitative illustration (five messages, one sample each at
temperature 0.7), not an evaluation.

## 10. Limitations

- **Small, synthetic data.** 25 memories and 31 messages; one query changes a
  lexical score by 0.25. Real users write differently and have far more memories.
- **An LLM drafted the labels, and an LLM is the judge.** The two may share
  intuitions about relevance, which would make the judge look better than it is.
  Independent human labels are needed to check this.
- **The judge prompt was written after seeing the reranker's failures** (it asks
  the judge to consider slang and synonyms), which favours the judge on this set.
- **Prompt variants were tuned and compared on the same set**, with no held-out
  data.
- **Semantic queries were filtered** to share no words with their target, so the
  keyword score of 0 on them is built in.
- **Two labels are arguable** (Section 5), and they decide the difference
  between recall 0.92 and 1.00 for the LLM judge.
- **The demo is confounded** on the boundary memory (Section 9).

## 11. Recommendations and future work

For the product:

1. Replace keyword recall with two stages: vector top 10, then the LLM judge
   with the A0 prompt, keeping score-3 memories only.
2. Put boundary and style memories into every prompt instead of retrieving them.
3. Do not filter vector results by a fixed similarity threshold, and do not use
   an off-the-shelf QA reranker for associative memory recall.
4. Treat any change to the judge prompt, including its output format, as a model
   change: re-run the labelled set and `compare_runs.py` before shipping.
5. Reduce the latency cost: call the judge only when vector results are
   ambiguous, and run retrieval in parallel with other reply preparation.

Future work:

- Have two or three people label the query set independently and measure their
  agreement with the current labels.
- Build a fresh, held-out query set (ideally from anonymised real chats) and
  re-run all methods on it.
- Hybrid retrieval: fuse keyword and vector rankings (reciprocal rank fusion).
- Test the boundary-memory recommendation directly, with the "do not lecture"
  line removed from the persona.
- Run a controlled experiment on DeepSeek prompt caching with a long fixed prefix.

## 12. Reproduce

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install langchain langchain-community langchain-openai langchain-text-splitters chromadb sentence-transformers python-dotenv openai pypdf
```

Create `.env` with `DEEPSEEK_API_KEY` and `DEEPSEEK_MODEL=deepseek-flash`, then:

```powershell
python validate_dataset.py            # dataset checks
python evaluate.py all                # keyword, vector, rerank
python evaluate.py llm                # LLM judge (A0); cached judgments are reused
python threshold_sweep.py vector      # or keyword / rerank / llm
python diagnose.py vector             # top-3 with scores for every query
python compare_runs.py <cache A> <cache B>
python summarize_usage.py             # tokens, latency and cost from the API log
python chat_demo.py                   # end-to-end replies with LangChain
```

Cached LLM judgments are committed, so the LLM results can be reproduced
without an API key.
