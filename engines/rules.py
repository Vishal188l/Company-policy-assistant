"""Approach 1: rules-based keyword search. No AI, no tokens.

Each question is split into keywords, which are scored against every policy's
title and text (rare words count more, title matches count double). The best
policy's text is returned word-for-word, so this approach can pick the wrong
policy but can never invent new facts.
"""
import math
import re
from collections import Counter

from .common import NOT_FOUND, load_policies

STOPWORDS = set("""
a an the is are am be been do does did can could may might must should would will
i me my we our you your he she it they them their this that these those of to in on
for at by with from about as or and if how what when where which who whom why
there here have has had any all some much many get got need want allowed allow
policy policies company employee employees work job please tell know rule rules
per many long often like just also than then so not no yes ok
""".split())

# Everyday words people use -> words that appear in the policies.
SYNONYMS = {
    "pto": "vacation", "holiday": "vacation", "holidays": "vacation", "leave": "leave",
    "wfh": "remote", "home": "remote", "telework": "remote",
    "sick": "sick", "ill": "illness", "doctor": "illness",
    "baby": "parental", "maternity": "parental", "paternity": "parental",
    "funeral": "bereavement", "died": "bereavement", "death": "bereavement",
    "clothes": "dress", "wear": "dress", "outfit": "dress",
    "receipt": "receipts", "trip": "travel", "flight": "travel", "hotel": "travel",
    "reimburse": "reimbursed", "refund": "reimbursed",
    "course": "training", "degree": "tuition", "university": "tuition",
    "vpn": "vpn", "hack": "security", "phishing": "security", "breach": "security",
    "phone": "mobile", "computer": "laptop",
    "journalist": "media", "press": "media", "reporter": "media",
    "car": "vehicle", "van": "vehicle",
    "abroad": "abroad", "overseas": "abroad", "country": "abroad",
    "expense": "expenses", "spend": "expenses", "purchase": "purchases", "buy": "purchases",
    "app": "software", "install": "installed", "program": "software",
}


def stem(word):
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def keywords(text):
    words = re.findall(r"[a-z0-9$]+", text.lower())
    out = []
    for w in words:
        if w in STOPWORDS:
            continue
        w = SYNONYMS.get(w, w)
        out.append(stem(w))
    return out


class RulesEngine:
    name = "Rules-based search"
    MIN_SCORE = 2.0  # below this we say "not found" instead of guessing

    def __init__(self):
        self.policies = load_policies()
        self.docs = []
        df = Counter()
        for p in self.policies:
            title = set(keywords(p["title"]))
            body = set(keywords(p["policy_text"] + " " + p["category"]))
            self.docs.append((title, body))
            df.update(title | body)
        n = len(self.policies)
        self.idf = {w: math.log(n / c) + 0.5 for w, c in df.items()}

    def score(self, q_words, doc):
        title, body = doc
        s = 0.0
        for w in q_words:
            if w in title:
                s += 2 * self.idf.get(w, 0)
            elif w in body:
                s += self.idf.get(w, 0)
        return s

    def answer(self, question):
        q = set(keywords(question))
        scored = sorted(
            ((self.score(q, d), i) for i, d in enumerate(self.docs)), reverse=True
        )
        best_score, best = scored[0]
        if best_score < self.MIN_SCORE:
            return {"answer": NOT_FOUND, "policy": None, "tokens": {"input": 0, "output": 0, "total": 0}}
        p = self.policies[best]
        return {
            "answer": p["policy_text"],
            "policy": p["title"],
            "tokens": {"input": 0, "output": 0, "total": 0},
        }
