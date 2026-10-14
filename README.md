# Companion Memory Retrieval: Keyword vs. Vector Search

An evaluation of long-term memory recall strategies for an AI companion chatbot.
The baseline is keyword overlap scoring; it is compared against embedding-based
vector retrieval and a hybrid of the two, on a hand-labelled query set.

## Motivation

Keyword matching misses memories that share meaning but not wording
(e.g. "can't sleep at night" vs. "often has insomnia"). This project measures
how much semantic retrieval closes that gap.

## Stack
- LLM: DeepSeek API (OpenAI-compatible)
- Embeddings: BAAI/bge-small-zh-v1.5 (local, CPU)
- Vector store: Chroma
- Framework: LangChain

## Data
- `data/memories.json` - mock long-term memory bank (fictional user)
- `data/test_queries.json` - labelled queries with relevant memory ids,
  grouped into lexical / semantic / multi / negative categories

## Setup
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install langchain langchain-community langchain-openai langchain-text-splitters chromadb sentence-transformers python-dotenv openai pypdf

Create a `.env` file with `DEEPSEEK_API_KEY` and `DEEPSEEK_MODEL`.

## Status
- [x] Step 1: Environment setup and API connectivity check
- [ ] Step 2: Memory bank and labelled query set
- [ ] Step 3: Keyword retrieval baseline
- [ ] Step 4: Vector retrieval (embeddings + Chroma)
- [ ] Step 5: Hybrid retrieval and evaluation report