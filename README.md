# Project Shonku

AI application platform. Chat is the first app: users pick a curated
Knowledge Base (no file uploads) and chat with it, grounded via RAG.

MVP v1 stack: `deepagents` (LangGraph) agent, FAISS vector search, local
embeddings (sentence-transformers), OpenRouter for generation, FastAPI
backend, Streamlit frontend. No DB — FAISS is file-based, chat history is
in-memory per session.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENROUTER_API_KEY
```

## Run

```bash
python -m backend.offline_pipeline.ingest              # build/update FAISS indexes from data/kbs
uvicorn backend.online_pipeline.main:app --reload   # backend on :8000
streamlit run frontend/app.py         # frontend on :8501
```

## Ingestion (offline pipeline)

```bash
python -m backend.offline_pipeline.ingest                     # all KBs, incremental (skips unchanged files)
python -m backend.offline_pipeline.ingest --kb hr-policies     # just one KB (repeatable: --kb a --kb b)
python -m backend.offline_pipeline.ingest --rebuild            # wipe + fully re-embed the selected KB(s)
```

Re-embeds only new/changed files (tracked via content hash in
`data/faiss_indexes/<slug>/manifest.json`); removed source files have their
chunks deleted from the index automatically.

Supported source file types: `.md`, `.txt`, `.pdf`, `.docx`.

### Pulling in SEC 10-Q filings

```bash
export SEC_EDGAR_CONTACT_EMAIL=you@example.com   # required by SEC EDGAR
python -m backend.offline_pipeline.download_sec_10q
```

Downloads + converts filings to `data/raw/sec_10q/` (scratch, gitignored).
Copy the PDFs you want into `data/kbs/<slug>/` and ingest as usual — raw
downloads aren't picked up automatically.

## Adding a Knowledge Base

1. Create `data/kbs/<slug>/kb.yaml`:
   ```yaml
   name: My Knowledge Base
   description: What this KB covers.
   ```
2. Drop source files (`.md`/`.txt`/`.pdf`/`.docx`) into `data/kbs/<slug>/`.
3. Run `python -m backend.offline_pipeline.ingest --kb <slug>`.

No code changes or backend restart needed — `/kbs` picks up new KBs live.

## Project structure

```
backend/
  kb.py               # shared: KB registry, scans data/kbs/*/kb.yaml
  retrieval.py         # shared: FAISS load/save/search, local embeddings
  config.py             # shared: router model config
  logging_config.py      # shared: logging setup
  offline_pipeline/
    loaders.py               # file-type -> document loader dispatch
    ingest.py                  # incremental ingestion CLI
    download_sec_10q.py          # SEC EDGAR -> data/raw/sec_10q/ (scratch)
  online_pipeline/
    agent.py                # deepagents agent + KB retrieval tool
    router.py                # easy/hard model routing middleware
    main.py                   # FastAPI app (/kbs, /chat)
frontend/
  app.py        # Streamlit chat UI
data/
  kbs/<slug>/            # KB source files + kb.yaml (ingested)
  raw/                   # scratch downloads, not ingested directly (gitignored)
  faiss_indexes/<slug>/  # generated: FAISS index + manifest.json (gitignored)
```