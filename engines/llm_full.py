"""Approach 2: LLM without a vector index.

There is no search step. All 98 policies are pasted into every prompt and
Gemini has to find the right one itself. Simple to build, but every single
question pays for the whole policy database in input tokens.
"""
from .common import ask_gemini, load_policies, policy_to_text


class LLMFullEngine:
    name = "LLM without vector index"

    def __init__(self):
        self.context = "\n".join(policy_to_text(p) for p in load_policies())

    def answer(self, question):
        text, policy, tokens = ask_gemini(self.context, question)
        return {"answer": text, "policy": policy, "tokens": tokens}
