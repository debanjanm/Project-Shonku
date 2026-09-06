# backend

FastAPI backend. Four deepagents agents behind one `/chat` endpoint, shared
retrieval/persistence/memory underneath.

- `kb.py` — KB registry (`data/kbs/*/kb.yaml`, no restart needed) + CRUD for
  the admin UI (create/delete a KB, add/remove its documents)
- `retrieval.py` — hybrid FAISS+BM25 search (RRF merge), local
  sentence-transformers embeddings (`get_embeddings()` — reused by
  `memory/` too, no second embedding model)
- `db.py` — SQLite conversation/message persistence (`data/shonku.db`)
- `config.py` — router model config (`ROUTER_EASY_MODEL`/`ROUTER_HARD_MODEL`)
- `logging_config.py` — logging setup, `configure_logging(plain=True)` for
  CLI scripts vs the default for the server
- `agents/` — one subfolder per agent (`docqa`, `recommendation`,
  `story_developer`, `mystery_generator`), each with its own README
- `memory/` — cross-agent long-term memory, consumed by all four agents
  (see its own README)
- `offline_pipeline/` — file loaders (`loaders.py`), data-source downloaders
  (`download_sec_10q.py`, `download_arxiv.py`), retrieval eval harness
  (`eval_retrieval.py`) — run manually. `ingest.py`'s `ingest_kb()` is the
  exception — also called directly by `online_pipeline/main.py`'s `/kbs/{slug}/ingest`
  endpoint, not just the CLI
- `online_pipeline/main.py` — the FastAPI app itself: `/agents`, `/kbs` (+
  CRUD — create/delete a KB, list/upload/delete its documents, trigger
  ingest), `/conversations`, `/chat`, `/products/images/{filename}`

Tests: `pytest` from repo root runs `tests/` (mocked LLM calls, no network,
no API cost) — RRF merge/tokenize/period parsing, memory extraction/update
JSON parsing, `MemoryScope` filtering, `db.py` CRUD. `pytest.ini` scopes
default collection to `tests/` only — the live-LLM regression/eval scripts
next to each agent (`test_path_discipline.py`, `eval_*.py`) are meant to be
run explicitly (`python -m backend.agents.<agent>.<script>`), not by CI.
