"""Check which model actually serves each model name, and whether thinking mode
is on (reasoning tokens are billed as output, and they add latency)."""
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

# (label, model name, extra request body)
VARIANTS = [
    ("deepseek-chat (current .env)", "deepseek-chat", None),
    ("deepseek-flash, default", "deepseek-flash", None),
    ("deepseek-flash, thinking off", "deepseek-flash", {"thinking": {"type": "disabled"}}),
]

for label, model, extra_body in VARIANTS:
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            max_tokens=200,
            extra_body=extra_body,
        )
    except Exception as exc:  # report and continue with the next variant
        print(f"{label:30s} ERROR: {exc}")
        continue
    seconds = time.perf_counter() - start
    usage = response.usage
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) if details else None
    print(f"{label:30s} served by: {response.model}")
    print(f"{'':30s} completion tokens: {usage.completion_tokens}, "
          f"reasoning tokens: {reasoning}, latency: {seconds:.2f} s")