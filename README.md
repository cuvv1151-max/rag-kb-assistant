# Companion Memory Retrieval: Keyword vs. Vector vs. Reranker vs. LLM Judge

An evaluation of long-term memory recall for [OnlyChat](#context), a WeChat-based
AI companion. The companion must recall what the user told it before, such as a
cat's name, a recent job change or a dislike of being lectured, and must leave
irrelevant memories out of the reply.

Four retrieval methods were compared on a labelled test set, and the best one was
wired into an end-to-end reply pipeline with LangChain and DeepSeek.

**Full report:** [English](REPORT.md) · [中文](REPORT_zh.md)

## Results at a glance

| Method | R@3 (20 queries) | Relevant kept* | Unrelated messages kept clean* | Irrelevant memories per message* |
|---|---|---|---|---|
| Keyword overlap (original approach) | 0.25 | 0.29 | 0.45 | 0.61 |
| Vector search, bge-small-zh (threshold 0.50) | 0.93 | 0.83 | 0.64 | 0.94 |
| Cross-encoder reranker (threshold 0.10) | 0.78 | 0.50 | 0.91 | 0.29 |
| **Vector top-10 + DeepSeek judge (score 3 only)** | **1.00** | **0.92** | **1.00** | **0.00** |

\* Top 3 results after filtering; 24 relevant memories, 11 unrelated messages, 31 messages in total.

- **Keyword recall misses paraphrase and slang.** It found none of the 13 queries
  that share no words with their target memory (by design of the test set).
- **No similarity threshold separates relevant from irrelevant.** Vector scores of
  correct hits and of unrelated messages overlap.
- **An off-the-shelf reranker made semantic ranking worse** (R@1 0.62 to 0.38): it
  fails whenever the link needs slang, synonyms or common-sense inference.
- **Output format changes an LLM judge's behaviour.** Four cheaper formats cut cost
  by 41-61% but made the judge permissive or unreliable, so the original format
  was kept (about 0.0016 CNY and 1.2 s per message).

All data is synthetic, and the test set is small (25 memories, 31 messages).
The report's limitations section explains what these numbers can and cannot show.

## How it works

```
User message
   │
   ▼
Vector search (bge-small-zh-v1.5 + Chroma) ──► top 10 candidate memories
   │
   ▼
DeepSeek judge (deepseek-flash, thinking off) ──► score 0-3 per candidate
   │
   ▼
Keep score-3 memories + always-included boundary memories
   │
   ▼
LangChain: ChatPromptTemplate | ChatOpenAI (DeepSeek) | StrOutputParser ──► reply
```

## Repository layout

| Path | Purpose |
|---|---|
| `data/memories.json` | 25 synthetic memories about a fictional user |
| `data/test_queries.json` | 31 labelled messages: lexical, semantic, multi, negative |
| `retrievers.py` | Keyword, vector, reranker and LLM-judge retrievers (prompt variants A0-A4) |
| `evaluate.py` | Recall@k and MRR per category |
| `threshold_sweep.py` | Recall vs. filtering of unrelated messages at each score threshold |
| `diagnose.py` | Top-3 results with scores for every query |
| `compare_runs.py` | Agreement between two LLM judge runs or prompt variants |
| `summarize_usage.py` | Tokens, latency and cost from the API call log |
| `chat_demo.py` | End-to-end reply demo with LangChain |
| `check_model.py` | Which DeepSeek model serves a model name, and whether it is thinking |
| `validate_dataset.py` | Dataset checks |
| `cache/` | Cached LLM judgments and API usage log (results reproduce without an API key) |
| `results/demo_replies.md` | Replies from the demo, with and without memories |

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install langchain langchain-community langchain-openai langchain-text-splitters chromadb sentence-transformers python-dotenv openai pypdf
```

Create a `.env` file:

```
DEEPSEEK_API_KEY=your-key
DEEPSEEK_MODEL=deepseek-flash
```

Then:

```powershell
python evaluate.py all          # keyword, vector, reranker
python evaluate.py llm          # LLM judge, reuses cached judgments
python chat_demo.py             # end-to-end replies
```

The report's "Reproduce" section lists every script.

## Context

OnlyChat is a WeChat-based AI companion built by a small startup team, where I
work on prompts and the memory module. Its original memory recall scored memories
by keyword overlap. This repository is an independent evaluation of alternatives,
published with the team's agreement. It contains only synthetic data and my own
evaluation code, no product code or user data.
