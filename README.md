# Project Shonku

AI application platform, not a single chatbot. Users pick an **agent**
first, then that agent's **source** — a curated Knowledge Base for
Document Q&A, a built-in dataset / uploaded CSV for the Data Analyst,
nothing (fixed catalog) for Product Recommendations, or nothing (the
user's own message is the source) for Story Developer and Mystery
Generator.

MVP v1 stack: `deepagents` (LangGraph) agents, OpenRouter for generation,
FastAPI backend, Streamlit frontend. Document Q&A uses hybrid retrieval —
FAISS (local, file-based dense/semantic search) + BM25 (sparse/keyword),
merged via Reciprocal Rank Fusion — with local sentence-transformers
embeddings. Data Analyst uses pandas/SQL/matplotlib/seaborn tools across
4 subagents (analysis, SQL, viz, report-writing). Product Recommendations
wraps a CLIP + ChromaDB + LLM-reranking search pipeline as a single tool
so it fits the same agent shape as the others. Story Developer expands a
one-line idea into a title/synopsis/characters/scene. Mystery Generator
builds a crime case you have to solve — the solution lives in a hidden
per-conversation file the agent writes but never quotes back until you
guess or give up. Conversations for all five agents persist in a local
SQLite file (`data/shonku.db`) — a chat survives a page refresh or a
server restart, and each conversation is locked to the agent+source it
started with.

A cross-agent memory layer (`backend/memory/`) runs alongside every chat:
after each turn, an LLM extracts durable facts about the user (name,
preferences, goals) into ChromaDB (local embeddings) + an optional Neo4j
entity graph, and every agent's next reply — regardless of which agent —
gets those facts injected into its system prompt. Fully automatic, no UI
yet to browse or manage stored memories. See "Memory layer" below.

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
chunks deleted from the index automatically. The BM25 index (`bm25.pkl`,
same directory) has no incremental API and is always rebuilt in full —
cheap, and it means a plain ingest run backfills it for any KB that
predates hybrid search.

Supported source file types: `.md`, `.txt`, `.pdf`, `.docx`.

### Pulling in SEC 10-Q filings

```bash
export SEC_EDGAR_CONTACT_EMAIL=you@example.com   # required by SEC EDGAR
python -m backend.offline_pipeline.download_sec_10q
```

Downloads + converts filings to `data/raw/sec_10q/` (scratch, gitignored).
Copy the PDFs you want into `data/kbs/<slug>/` and ingest as usual — raw
downloads aren't picked up automatically.

### Pulling in arXiv papers

```bash
python -m backend.offline_pipeline.download_arxiv --topic "retrieval augmented generation" --count 10
```

Searches arXiv by topic, saves each paper as plain text (native/ar5iv HTML
stripped) or, if no full-text HTML is available, the raw PDF — to
`data/raw/arxiv/<topic>/` (scratch, gitignored). Copy what you want into
`data/kbs/<slug>/` and ingest as usual.

### Evaluating retrieval quality

```bash
python -m backend.offline_pipeline.eval_retrieval --fetch   # builds data/kbs/ragbench-eval/
python -m backend.offline_pipeline.ingest --kb ragbench-eval
python -m backend.offline_pipeline.eval_retrieval --run      # Recall@k + MRR against ground truth
```

Scores `backend.retrieval.search()` against Vectara's `open_ragbench` benchmark
(real queries + ground-truth relevant papers) — the same FAISS+BM25 hybrid
path every KB uses, not a separate simulation. Defaults to a 50-paper slice
(`MAX_PAPERS` in the script) for a fast/cheap run.

## Memory layer

Every agent has its own long-term memory of the user, ported from `AIF-Remembrane`'s
Mem0-architecture pipeline (Extract → Search → Update → Store), adapted to Shonku's
sync backend and local-first embeddings:

1. **Extract** — after each chat turn, an LLM (`LLM_MODEL` env var) pulls durable
   facts from the exchange (preferences, biographical details, corrections)
2. **Search** — embeds each fact (local `sentence-transformers/all-MiniLM-L6-v2`,
   same model `backend/retrieval.py` uses — no OpenRouter embedding cost) and finds
   related existing memories in ChromaDB, expanded via the Neo4j entity graph if
   connected
3. **Update** — an LLM decides ADD / UPDATE / DELETE / SKIP per fact against what's
   already stored, avoiding duplicate/stale memories
4. **Store** — writes to ChromaDB (`CHROMA_PERSIST_PATH`, scratch, gitignored) and,
   if configured, Neo4j AuraDB (entity relationships, enables graph-expanded recall)

At the start of every chat turn, the current agent's system prompt gets a
`## Relevant memories about this user` block injected — scoped per agent: a fact
learned in one Data Analyst conversation is recalled in any other Data Analyst
conversation, but never surfaces in Story Developer or any other agent. Each
memory is tagged with the conversation it came from too (for provenance), but
recall isn't locked to that single conversation — it spans every conversation
with the same agent, which is what makes it "long-term" at all. Storage runs as
a background thread after the response streams, so it adds no latency.

No login exists in Shonku (single-user by design), so memory is scoped to one
fixed user id, not per-account. The Neo4j graph is an enhancement, not a
requirement — if it's unreachable (wrong credentials, or an AuraDB free instance
that's paused/expired) the app logs a warning at startup and falls back to
vector-only memory; chat is unaffected either way. Needs `NEO4J_URI`,
`NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` in `.env` (free tier at
[neo4j.com/cloud/aura](https://neo4j.com/cloud/aura/)) to enable it.

Nothing to run manually — this is wired into `/chat` automatically. No browser UI
yet for viewing/deleting stored memories (see `PLAN.md`).

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
  retrieval.py         # shared: hybrid FAISS+BM25 search (RRF merge), local embeddings
  db.py                 # shared: SQLite conversation/message persistence
  config.py               # shared: router model config
  logging_config.py        # shared: logging setup
  memory/                    # shared: cross-agent memory (Extract->Search->Update->Store)
    manager.py                  # orchestrator, called from main.py before/after each chat turn
    chroma_store.py               # vector store (local embeddings, data/chroma/)
    neo4j_store.py                  # entity graph (Neo4j AuraDB, optional)
    pipeline.py                       # extraction + update-decision LLM calls, synthesis
    models.py                           # Memory / MemoryScope / ExtractedFact / etc.
  agents/
    docqa/
      agent.py              # deepagents agent + KB retrieval tool
      router.py              # easy/hard model routing middleware (has an eval:
                              #   python -m backend.agents.docqa.router)
    data_analyst/
      agent.py              # deepagents supervisor + 4 subagents (analysis,
                             #   SQL, viz, report-writing), no per-source rebuild
                             #   needed beyond source binding in the system prompt
      tools.py               # inspect/stats/correlations/outliers/SQL/python/charts
      sql_pipeline.py          # NL-to-SQL: schema context, few-shot, CoT, self-correct
    recommendation/
      agent.py              # deepagents agent, ONE tool wrapping the whole pipeline
      embedding_service.py   # CLIP (open_clip), text/image -> 512-d vectors
      vector_store.py         # ChromaDB wrapper (persist_dir -> data/products/chroma_db)
      llm_service.py           # LLM query expansion + reranking
      models.py                 # Product / SearchResponse / RankedProduct / QueryExpansion
    story_developer/
      agent.py              # deepagents agent, 3 subagents (writer, scene-director, editor),
                             #   stateless
    mystery_generator/
      agent.py              # deepagents agent, 4 subagents (planner, researcher, scenarist,
                             #   validator); FilesystemBackend per conversation
                             #   (data/mysteries/<conversation_id>/solution.md) holds the
                             #   hidden solution the agent writes but doesn't quote back
  offline_pipeline/
    loaders.py               # file-type -> document loader dispatch
    ingest.py                  # incremental ingestion CLI
    download_sec_10q.py          # SEC EDGAR -> data/raw/sec_10q/ (scratch)
    download_arxiv.py            # arXiv search -> data/raw/arxiv/<topic>/ (scratch)
  online_pipeline/
    main.py   # FastAPI app: /agents, /kbs, /datasets, /uploads, /conversations,
               # /chat, /charts/{filename}, /products/images/{filename}
frontend/
  app.py        # Streamlit UI: agent picker -> source picker -> chat
data/
  kbs/<slug>/            # KB source files + kb.yaml (ingested)
  raw/                   # scratch downloads, not ingested directly (gitignored)
  uploads/               # CSV uploads for the Data Analyst agent (gitignored)
  charts/                # generated chart PNGs, served via /charts/{filename} (gitignored)
  products/              # catalog.json + images/ + chroma_db/ (seeded, committed)
  mysteries/<conversation_id>/  # solution.md per mystery conversation (gitignored)
  chroma/                # generated: memory vector store (gitignored)
  faiss_indexes/<slug>/  # generated: FAISS index + bm25.pkl + manifest.json (gitignored)
  shonku.db              # generated: conversations + messages (gitignored)
```