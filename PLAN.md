# Project Shonku — Roadmap

## Done (MVP v1)

**Core chat**
- FastAPI backend + Streamlit frontend, `deepagents` (LangGraph) agent, streaming responses ([backend/online_pipeline/main.py](backend/online_pipeline/main.py), [frontend/app.py](frontend/app.py))
- RAG via FAISS (local, file-based) + local sentence-transformers embeddings, no external key needed for retrieval ([backend/retrieval.py](backend/retrieval.py))
- Generation via OpenRouter, OpenAI-compatible ([backend/online_pipeline/agent.py](backend/online_pipeline/agent.py))

**Knowledge bases**
- Config-driven via `kb.yaml` per folder, live-discovered (no restart) ([backend/kb.py](backend/kb.py))
- Offline ingestion: incremental (hash-tracked, skips unchanged files), PDF/DOCX/MD/TXT, per-file chunk-id manifest for precise delete-on-change/removal ([backend/offline_pipeline/ingest.py](backend/offline_pipeline/ingest.py), [backend/offline_pipeline/loaders.py](backend/offline_pipeline/loaders.py))
- 3 seeded KBs: `product-docs`, `hr-policies`, `sec_10q` (23 real SEC 10-Q filings via [backend/offline_pipeline/download_sec_10q.py](backend/offline_pipeline/download_sec_10q.py))
- Backend split into `offline_pipeline/` (ingestion + data-download CLIs) and `online_pipeline/` (serving), with `kb.py`/`retrieval.py`/`config.py`/`logging_config.py` shared at `backend/` top level
- All data under `data/`: `kbs/` (ingested), `raw/` (scratch downloads), `faiss_indexes/` (generated) — no more repo-root scatter

**Agent behavior**
- Asks a clarifying follow-up only when *retrieved results* are genuinely ambiguous (multi-company/multi-period), not before searching — fixed a regression where it over-triggered on single-KB questions ([backend/online_pipeline/agent.py](backend/online_pipeline/agent.py) system prompt)
- Inline citations: retrieval keeps chunk metadata (`source`, `doc_title`, `chunk_index`) through to the tool output, model cites `(Source: filename)`, prompt hardened against citing sources not actually returned that turn (caught a real nano hallucination during testing) ([backend/retrieval.py](backend/retrieval.py), [backend/online_pipeline/agent.py](backend/online_pipeline/agent.py))
- Smart model router: heuristic (keyword/length, no LLM call) picks nano vs mini model per question via `wrap_model_call` middleware ([backend/online_pipeline/router.py](backend/online_pipeline/router.py), [backend/config.py](backend/config.py))

**Ops**
- Structured logging throughout (stdlib `logging`, `LOG_LEVEL` env var) — request lifecycle, retrieval timing, router decisions ([backend/logging_config.py](backend/logging_config.py))

## Next

**1. Answer quality**
- Query rewriting for multi-turn follow-ups (resolve pronouns/references using conversation history before hitting FAISS) — discussed, not built
- Evaluate/tune router heuristic accuracy (currently untested beyond a handful of manual cases)
- Retrieval recall gaps: exact-quarter queries sometimes miss the matching filing even when it exists (seen during citation testing) — worth a retrieval-quality pass

**2. Persistence**
- No database — FAISS is file-based (fine), but chat history is in-memory per Streamlit session only, lost on refresh
- No conversation/thread model, no per-conversation KB lock (history can technically carry across a KB switch un-scoped)

**3. Knowledge base admin**
- Still folder + `kb.yaml` + manual `ingest` run — no admin UI (explicitly deferred from day one, still true)
- No access control on which KBs a user can see/query

**4. Frontend**
- Streamlit is functional, not the Linear/VS Code/Notion-grade UI the product vision describes — would need a real frontend (Next.js/React) if that bar matters for this phase

**5. Platform shape**
- Still a single app (Chat) — the "AI application platform" framing from the vision doc has no second app or plugin surface yet, intentionally deferred

**6. Production readiness**
- No auth (explicitly deferred)
- No tests — zero test suite currently exists
- No rate limiting / abuse protection on `/chat`
- No deployment setup (Docker, CI, hosting target)
- Secrets are plain `.env` — fine for local dev, not for shipping

## Suggested order

Query rewriting (#1, sharpens follow-up handling) → persistence (#2, unlocks real multi-session use) → everything else is scope decisions, not obvious next steps — say which direction matters most.
