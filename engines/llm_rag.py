"""Approach 3: LLM with a vector index (retrieval-augmented generation).

Once, up front: every policy is turned into an embedding (a list of numbers
describing its meaning) and saved to data/policy_index.npz. That file is the
vector index.

Per question: the question is embedded too, the TOP_K closest policies are
found by cosine similarity, and only those are sent to Gemini. Fewer tokens,
and the model only sees policies that are actually relevant.
"""
import numpy as np

from .common import EMBED_MODEL, ROOT, ask_gemini, gemini, load_policies, policy_to_text, with_retry

INDEX_FILE = ROOT / "data" / "policy_index.npz"
TOP_K = 3


def embed(texts, task):
    from google.genai import types

    res = with_retry(lambda: gemini().models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task),
    ))
    vecs = np.array([e.values for e in res.embeddings], dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def build_index():
    policies = load_policies()
    texts = [policy_to_text(p) for p in policies]
    vecs = []
    for i in range(0, len(texts), 50):  # small batches are kinder to free-tier limits
        vecs.append(embed(texts[i:i + 50], "RETRIEVAL_DOCUMENT"))
    np.savez(INDEX_FILE, vectors=np.vstack(vecs))
    print(f"Saved vector index for {len(texts)} policies to {INDEX_FILE.name}")


class LLMRagEngine:
    name = "LLM with vector index"

    def __init__(self):
        if not INDEX_FILE.exists():
            build_index()
        self.policies = load_policies()
        self.vectors = np.load(INDEX_FILE)["vectors"]

    def retrieve(self, question):
        q = embed([question], "RETRIEVAL_QUERY")[0]
        sims = self.vectors @ q
        top = np.argsort(-sims)[:TOP_K]
        return [(self.policies[i], float(sims[i])) for i in top]

    def answer(self, question):
        hits = self.retrieve(question)
        context = "\n".join(policy_to_text(p) for p, _ in hits)
        text, policy, tokens = ask_gemini(context, question)
        return {
            "answer": text,
            "policy": policy,
            "tokens": tokens,
            "retrieved": [{"title": p["title"], "similarity": round(s, 3)} for p, s in hits],
        }


if __name__ == "__main__":
    build_index()
