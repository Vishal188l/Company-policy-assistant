# Company Policy Assistant

A bot that answers questions about 98 company policies and names the relevant policy, built three ways and compared.

| Approach | How it works |
|---|---|
| Rules-based search (`engines/rules.py`) | Keyword matching with synonyms and word weighting. Returns the best policy's text word-for-word. No AI. |
| LLM without vector index (`engines/llm_full.py`) | All 98 policies are pasted into every Gemini prompt. |
| LLM with vector index (`engines/llm_rag.py`) | Policies are embedded once into `data/policy_index.npz`. Each question retrieves the 3 closest policies, and only those go to Gemini. |

- **Website:** `docs/` (served by GitHub Pages), reads `docs/results.json`
- **Comparison run:** `run_comparison.py` answers `data/test_questions.json` with all three approaches and measures time, tokens, correct policy, and unsupported answers (checked by an LLM judge against the full database)
- **Slack bot:** `bot.py`, uses the vector-index approach, runs locally via Socket Mode

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp env.example .env         # then fill in your keys
python run_comparison.py    # writes docs/results.json
python bot.py               # starts the Slack bot
```

The Slack app can be created from `slack_manifest.yml` at api.slack.com/apps → Create New App → From a manifest.
