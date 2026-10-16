"""Shared helpers: load the policy database, talk to Gemini, time each call."""
import csv
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

POLICY_CSV = ROOT / "data" / "company_policies.csv"
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

NOT_FOUND = "I couldn't find a policy that covers this question."


def load_policies():
    with open(POLICY_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        r["id"] = i
    return rows


def policy_to_text(p):
    return f"{p['title']} ({p['category']}, owned by {p['department']}): {p['policy_text']}"


_client = None


def gemini():
    """Lazily create the Gemini client so the rules engine works without a key."""
    global _client
    if _client is None:
        from google import genai
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is missing. Put it in the .env file.")
        _client = genai.Client(api_key=key)
    return _client


def with_retry(fn, tries=5):
    """Free-tier keys hit 429 rate limits often; wait and retry."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if attempt < tries - 1 and ("429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg):
                wait = 15 * (attempt + 1)
                print(f"   rate limited, waiting {wait}s ...")
                time.sleep(wait)
                continue
            raise


ANSWER_RULES = (
    "You are a company policy assistant. Answer ONLY using the policies provided. "
    "Name the single most relevant policy by its exact title. "
    "If no provided policy answers the question, set policy to \"NONE\" and say the "
    "policy database does not cover it. Do not add details that are not in the policy text. "
    "Keep the answer to 1-2 sentences. "
    'Reply as JSON: {"answer": "...", "policy": "<exact title or NONE>"}'
)


def ask_gemini(context, question):
    """Send one question + context to Gemini. Returns answer, policy, tokens."""
    from google.genai import types

    prompt = f"POLICIES:\n{context}\n\nQUESTION: {question}"
    resp = with_retry(lambda: gemini().models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=ANSWER_RULES,
            response_mime_type="application/json",
            temperature=0,
        ),
    ))
    try:
        data = json.loads(resp.text)
    except (json.JSONDecodeError, TypeError):
        data = {"answer": (resp.text or "").strip(), "policy": "NONE"}
    u = resp.usage_metadata
    tokens = {
        "input": u.prompt_token_count or 0,
        "output": u.candidates_token_count or 0,
        "total": u.total_token_count or 0,
    }
    policy = (data.get("policy") or "NONE").strip()
    return data.get("answer", "").strip(), (None if policy.upper() == "NONE" else policy), tokens


def timed(fn, *args):
    start = time.perf_counter()
    out = fn(*args)
    return out, round((time.perf_counter() - start) * 1000, 1)
