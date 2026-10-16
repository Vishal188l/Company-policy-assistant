"""Slack bot: answers policy questions with the vector-index approach.

Runs on your own computer using Slack Socket Mode, so no web hosting is needed.
Ways to call it in Slack:
  - mention it in a channel:  @PolicyBot how many vacation days do I get?
  - send it a direct message
  - slash command (optional):  /policy how many vacation days do I get?

    python bot.py
"""
import os
import re

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from engines.common import load_policies
from engines.llm_rag import LLMRagEngine

app = App(token=os.environ.get("SLACK_BOT_TOKEN"))
engine = LLMRagEngine()
POLICIES = {p["title"]: p for p in load_policies()}


def build_reply(question):
    out = engine.answer(question)
    lines = [f":books: *Answer:* {out['answer']}"]
    p = POLICIES.get(out["policy"] or "")
    if p:
        lines.append(f":page_facing_up: *Relevant policy:* {p['title']} "
                     f"_({p['category']} · {p['department']})_")
        lines.append(f"> {p['policy_text']}")
    else:
        lines.append(":grey_question: *Relevant policy:* none found in the policy database")
    return "\n".join(lines)


def clean(text):
    return re.sub(r"<@[A-Z0-9]+>", "", text or "").strip()


@app.event("app_mention")
def on_mention(event, say):
    question = clean(event.get("text"))
    thread = event.get("thread_ts") or event["ts"]
    if not question:
        say(text="Ask me a policy question, e.g. _how many vacation days do I get?_", thread_ts=thread)
        return
    say(text=build_reply(question), thread_ts=thread)


@app.event("message")
def on_dm(event, say):
    if event.get("channel_type") == "im" and not event.get("bot_id") and event.get("text"):
        say(text=build_reply(clean(event["text"])))


@app.command("/policy")
def on_slash(ack, command, respond):
    ack()
    question = (command.get("text") or "").strip()
    respond(response_type="in_channel",
            text=f"*Q:* {question}\n{build_reply(question)}" if question else "Usage: /policy <your question>")


if __name__ == "__main__":
    print("Policy bot is running. Press Ctrl+C to stop.")
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
