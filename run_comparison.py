"""Run every test question through all three approaches and save docs/results.json.

    python run_comparison.py               # all three approaches (needs GEMINI_API_KEY)
    python run_comparison.py --rules-only  # quick test without any API key

Hallucination check: after each answer, Gemini (acting as a judge, with the
full policy database in front of it) decides whether every claim in the answer
is backed by the database. Judge tokens are NOT counted in the approach's
token use.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

from engines.common import CHAT_MODEL, EMBED_MODEL, ROOT, gemini, load_policies, policy_to_text, timed, with_retry
from engines.rules import RulesEngine

DELAY = float(os.getenv("RUN_DELAY_SECONDS", "4"))  # pause between API calls (free tier)

JUDGE_RULES = (
    "You check a policy assistant's answer against the company's policy database. "
    "Label it SUPPORTED if every factual claim in the answer is stated in the database, "
    "or if the answer correctly says the database does not cover the question. "
    "Label it UNSUPPORTED if it states any fact, number, permission or prohibition that "
    "is not in the database (even if it sounds reasonable). "
    'Reply as JSON: {"label": "SUPPORTED" or "UNSUPPORTED", "reason": "<one short sentence>"}'
)


def judge(question, answer, database):
    from google.genai import types

    prompt = f"POLICY DATABASE:\n{database}\n\nQUESTION: {question}\nANSWER: {answer}"
    resp = with_retry(lambda: gemini().models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=JUDGE_RULES, response_mime_type="application/json", temperature=0
        ),
    ))
    try:
        data = json.loads(resp.text)
        return data.get("label", "").upper() == "SUPPORTED", data.get("reason", "")
    except (json.JSONDecodeError, TypeError):
        return None, "Judge reply could not be read."


def main():
    rules_only = "--rules-only" in sys.argv
    questions = json.loads((ROOT / "data" / "test_questions.json").read_text())
    database = "\n".join(policy_to_text(p) for p in load_policies())

    engines = {"rules": RulesEngine()}
    if not rules_only:
        from engines.llm_full import LLMFullEngine
        from engines.llm_rag import LLMRagEngine
        engines["llm_full"] = LLMFullEngine()
        engines["llm_rag"] = LLMRagEngine()

    rows = []
    for n, q in enumerate(questions, 1):
        print(f"[{n}/{len(questions)}] {q['question']}")
        row = {**q, "results": {}}
        for key, engine in engines.items():
            out, ms = timed(engine.answer, q["question"])
            out["time_ms"] = ms
            out["policy_correct"] = (out["policy"] or None) == q["expected_policy"]
            if rules_only:
                out["supported"], out["judge_reason"] = None, "Not judged (rules-only run)."
            else:
                if key != "rules":
                    time.sleep(DELAY)
                out["supported"], out["judge_reason"] = judge(q["question"], out["answer"], database)
                time.sleep(DELAY)
            row["results"][key] = out
            flag = {True: "ok", False: "UNSUPPORTED", None: "-"}[out["supported"]]
            print(f"   {engine.name:28s} {ms:8.0f} ms  {out['tokens']['total']:6d} tok  "
                  f"policy={out['policy']!s:32s} {flag}")
        rows.append(row)

    summary = {}
    for key, engine in engines.items():
        res = [r["results"][key] for r in rows]
        judged = [x for x in res if x["supported"] is not None]
        summary[key] = {
            "name": engine.name,
            "avg_time_ms": round(sum(x["time_ms"] for x in res) / len(res), 1),
            "avg_tokens": round(sum(x["tokens"]["total"] for x in res) / len(res)),
            "policy_accuracy": round(sum(x["policy_correct"] for x in res) / len(res), 3),
            "unsupported_rate": (round(sum(not x["supported"] for x in judged) / len(judged), 3)
                                 if judged else None),
        }

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chat_model": CHAT_MODEL,
        "embed_model": EMBED_MODEL,
        "policy_count": len(load_policies()),
        "summary": summary,
        "questions": rows,
    }
    path = ROOT / "docs" / "results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved {path.relative_to(ROOT)}")
    for s in summary.values():
        print(f"  {s['name']:28s} avg {s['avg_time_ms']:7.0f} ms, {s['avg_tokens']:5d} tokens, "
              f"policy correct {s['policy_accuracy']:.0%}, unsupported {s['unsupported_rate']}")


if __name__ == "__main__":
    main()
