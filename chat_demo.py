"""Step 3: end-to-end RAG demo. Retrieve memories, build the prompt with
LangChain, and let DeepSeek write the companion's reply.

Each demo message is answered three ways, so the effect of retrieval is visible:
  no_memory  - no memories at all
  keyword    - keyword baseline: top-3 memories with any word overlap
  llm_judge  - recommended pipeline: memories the LLM judge (variant A0) scored 3,
               plus boundary memories, which are always included
Replies are printed and saved to results/demo_replies.md.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env before any model import, so settings such as HF_HOME take effect.
load_dotenv()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from retrievers import KeywordRetriever, LLMRetriever

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_PATH = ROOT / "results" / "demo_replies.md"

# One lexical, two slang/semantic, one multi-memory and one negative message.
DEMO_QUERY_IDS = ["q01", "q05", "q15", "q20", "q31"]

SYSTEM_TEMPLATE = """You are an AI companion who chats with the user like a close friend on WeChat.
Reply in Chinese, in one to three short, casual sentences.
Respond to the user's feelings first. Do not lecture, and do not claim to be human.

What you know about the user:
{memories}

Use this knowledge only where it makes the reply more personal and natural.
Never list it, and never say that you are reading from memory."""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_TEMPLATE), ("human", "{message}")]
)


def make_llm() -> ChatOpenAI:
    """DeepSeek through LangChain's OpenAI-compatible chat model."""
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        base_url="https://api.deepseek.com",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.7,  # some variety, as in a real chat; replies differ between runs
        # No max_tokens: LangChain sends it as max_completion_tokens, which DeepSeek
        # may not recognise. Reply length is controlled by the system prompt instead.
        extra_body={"thinking": {"type": "disabled"}},  # no billed reasoning tokens
    )


def format_memories(selected: list[dict]) -> str:
    if not selected:
        return "(nothing relevant)"
    return "\n".join(f"- {m['content']}" for m in selected)


def main() -> None:
    memories = json.loads((DATA_DIR / "memories.json").read_text(encoding="utf-8"))
    queries = json.loads((DATA_DIR / "test_queries.json").read_text(encoding="utf-8"))
    by_id = {m["id"]: m for m in memories}
    demo = [q for q in queries if q["id"] in DEMO_QUERY_IDS]

    keyword = KeywordRetriever(memories)
    judge = LLMRetriever(memories)  # reads cached judgments; new messages call the API
    pinned = [m for m in memories if m["type"] == "boundary"]

    def select(strategy: str, message: str) -> list[dict]:
        if strategy == "no_memory":
            return []
        if strategy == "keyword":
            return [by_id[mid] for mid, _ in keyword.retrieve(message, k=3)]
        # llm_judge: keep only score-3 memories (normalised score 1.0), then add
        # boundary memories, which shape every reply rather than one topic.
        judged = [by_id[mid] for mid, score in judge.retrieve(message, k=judge.candidates)
                  if score >= 0.999]
        return judged + [m for m in pinned if m not in judged]

    chain = PROMPT | make_llm() | StrOutputParser()

    lines = ["# Demo replies", "",
             "Each message is answered with three memory strategies. "
             "Replies are sampled at temperature 0.7, so a re-run gives different wording.", ""]
    for q in demo:
        print(f"\n=== {q['id']} {q['query']} ===")
        lines += [f"## {q['id']}: {q['query']}", ""]
        for strategy in ("no_memory", "keyword", "llm_judge"):
            selected = select(strategy, q["query"])
            reply = chain.invoke(
                {"memories": format_memories(selected), "message": q["query"]}
            ).strip()
            used = ", ".join(str(m["id"]) for m in selected) or "none"
            print(f"[{strategy}] memories: {used}\n  {reply}")
            lines += [f"**{strategy}** (memories: {used})", "", f"> {reply}", ""]
            for m in selected:
                lines.append(f"- {m['id']}: {m['content']}")
            if selected:
                lines.append("")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    judge.report()
    print(f"\nSaved to {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
