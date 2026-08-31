# backend

FastAPI backend. Five deepagents agents behind one `/chat` endpoint, shared
retrieval/persistence/memory underneath.

- `kb.py` — KB registry, scans `data/kbs/*/kb.yaml`, no restart needed
- `retrieval.py` — hybrid FAISS+BM25 search (RRF merge), local
  sentence-transformers embeddings (`get_embeddings()` — reused by
  `memory/` too, no second embedding model)
- `db.py` — SQLite conversation/message persistence (`data/shonku.db`)
- `config.py` — router model config (`ROUTER_EASY_MODEL`/`ROUTER_HARD_MODEL`)
- `logging_config.py` — logging setup, `configure_logging(plain=True)` for
  CLI scripts vs the default for the server
- `agents/` — one subfolder per agent (`docqa`, `recommendation`,
  `story_developer`, `mystery_generator`), each with its own README
- `memory/` — cross-agent long-term memory, consumed by all five agents
  (see its own README)
- `offline_pipeline/` — ingestion CLI (`ingest.py`), file loaders
  (`loaders.py`), data-source downloaders (`download_sec_10q.py`,
  `download_arxiv.py`), retrieval eval harness (`eval_retrieval.py`) — run
  manually, not imported by the server
- `online_pipeline/main.py` — the FastAPI app itself: `/agents`, `/kbs`,
  `/datasets`, `/uploads`, `/conversations`, `/chat`, `/charts/{filename}`,
  `/products/images/{filename}`
