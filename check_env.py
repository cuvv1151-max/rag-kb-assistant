"""Step 1: verify installed packages and DeepSeek API connectivity."""
import importlib
import os

from dotenv import load_dotenv

PACKAGES = [
    "langchain",
    "langchain_community",
    "langchain_openai",
    "langchain_text_splitters",
    "chromadb",
    "sentence_transformers",
    "torch",
    "openai",
]

print("=== Package versions ===")
for name in PACKAGES:
    try:
        module = importlib.import_module(name)
        print(f"{name:28s} {getattr(module, '__version__', 'unknown')}")
    except ImportError as exc:
        print(f"{name:28s} MISSING ({exc})")

print("\n=== DeepSeek API check ===")
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

if not api_key:
    print("DEEPSEEK_API_KEY not found in .env")
else:
    # Never print the key itself; only confirm it was loaded
    print(f"Key loaded (length {len(api_key)}), model = {model}")
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: connection ok"}],
        max_tokens=10,
    )
    print("Model reply:", response.choices[0].message.content)