# Project Shonku — Roadmap

## Done (MVP v1)

**Platform / multi-agent**
- Five agents, chosen first in the UI, then their source: **Document Q&A** (KB), **Data Analyst** (built-in dataset or uploaded CSV), **Product Recommendations** (fixed fashion catalog), **Story Developer** and **Mystery Generator** (both freeform — the user's own message is the source) — the "platform, not a single chatbot" vision now has real breadth ([backend/agents/](backend/agents/))
- Data Analyst ported from the sibling `AIB-DataAnalystAgent` repo: 4 deepagents subagents (data-analyst, viz-analyst, sql-analyst, report-writer), schema inspection, stats, correlations, outliers, NL-to-SQL with self-correction + few-shot memory, ad-hoc Python, chart generation
- Product Recommendations ported from `AIB-RecommendationAgent` (found live only in a git worktree, main branch was empty): LLM query expansion → CLIP embedding → ChromaDB vector search → LLM reranking with match explanations, wrapped as a single-tool deepagents agent (source repo was a fixed procedural pipeline, not agentic — wrapping it keeps every `agent_type` a uniform `.stream()`-compatible graph so `/chat`'s SSE loop needed zero new branching shape). Seeded 425-item catalog + working Chroma index copied wholesale, zero reseeding. Text-only for v1, image-upload search deferred
- Story Developer and Mystery Generator adapted from `AIF-CineBot` (which turned out to be two disconnected prototypes — a Google ADK film-pitch pipeline, foreign framework, not ported; and a bare deepagents notebook with no packaging). Story Developer: writer → scene-director → editor, stateless, trimmed from the ADK repo's 6-role team to just storyline/title/scenes as asked. Mystery Generator: planner → researcher → scenarist → validator (ported ~verbatim from the notebook — its prompts already produced a genuinely good case), but redesigned so the user has to *solve* it instead of being handed the answer — the full solution is written to a per-conversation file (`data/mysteries/<conversation_id>/solution.md`, `deepagents.backends.FilesystemBackend`) that the agent never quotes back except on a final accusation or give-up
- Backend restructured into `backend/agents/{docqa,data_analyst,recommendation,story_developer,mystery_generator}/` (mirrors the earlier offline/online_pipeline split); `backend/online_pipeline/main.py` is the single FastAPI entrypoint for all five
- Conversations schema generalized: `agent_type`/`source_type`/`source_ref` replace the old KB-only `kb_slug` column — the server-owned-history/source-lock design now covers all five agents uniformly, no separate memory system per agent
- New endpoints: `GET /agents`, `GET /datasets`, `POST /uploads`, `GET /charts/{filename}`, `GET /products/images/{filename}` (backend serves generated chart PNGs and product photos over HTTP, frontend never touches backend disk directly)
- Verified live: docqa and Data Analyst regression clean after each restructure; Data Analyst answered correctly on `iris` (real shape/columns), generated + served a real histogram PNG, CSV upload → analysis flow works end to end; Recommendation returned real, sensibly-reasoned products for a text query, served real product photos, and honestly reported zero results (no hallucination) for a query outside the sampled catalog's coverage; Story Developer produced a genuinely well-structured concept + scripted scene from a one-line idea; Mystery Generator's premise never named a culprit or motive, and its reveal matched the hidden solution file exactly on both a correct and an incorrect accusation — confirmed via API, direct file inspection, and a live browser session

**Caught a real bug in Mystery Generator during testing**: `/chat` builds a fresh deepagents graph every turn (no persistent process state), and the first turn's agent wrote the solution to a self-chosen path (`root/solution.md`) instead of the literal path the prompt asked for — the second turn's fresh agent instance then failed to find it and, instead of reporting the failure, fabricated a confident but wrong verdict (denied the actual culprit). Fixed by making the file path deterministic in the prompt and forcing the agent to `ls`/`read_file` before every judgment rather than trusting its own memory — reverified with a fresh conversation, correct and incorrect accusations both now resolve accurately against the real file.

**Memory layer**
- Per-agent long-term memory (`backend/memory/`), ported from `AIF-Remembrane` — a real, complete, tested implementation of the Mem0 architecture (Extract → Search → Update → Store) on ChromaDB + Neo4j AuraDB + OpenRouter. Adapted, not copy-pasted: Remembrane is async/FastAPI-standalone; this port is fully sync (matches `main.py`/`db.py`/`retrieval.py`) and reuses the existing free local `sentence-transformers/all-MiniLM-L6-v2` embeddings instead of adding a paid embedding dependency. `MemoryScope` carries `user_id` + `agent_id` (both filter retrieval) + `run_id`/conversation_id (stored as provenance only, not filtered) — a fact learned in one Data Analyst conversation is recalled in any other Data Analyst conversation, but never surfaces in Story Developer or any other agent. (First cut scoped memory globally across all agents — reverted after live-testing showed exactly that leak: a Document Q&A fact surfaced in a brand-new Story Developer conversation. Confirmed fixed: same fact now recalled in a second Document Q&A conversation but correctly absent from a Story Developer one, and Chroma metadata shows the right `agent_id` tag on each.)
- Wired into every agent: `/chat` fetches memory context before building the agent (`backend/online_pipeline/main.py`), all 5 `make_*_agent()` factories gained a `memory_context` param injected into their system prompt (4 of 5 previously had a static module-level `SYSTEM_PROMPT`, converted to build it per-call like `data_analyst` already did); storage runs as a background thread after the response streams, adding no latency
- Neo4j is an enhancement, not a hard requirement — `get_memory_manager()` catches a graph-store connection failure independently of the vector store and falls back to vector-only memory (`graph_enabled=False`, logged), so an unreachable/paused AuraDB instance degrades gracefully instead of breaking chat. Caught for real during testing: the first cut bundled Chroma+Neo4j construction together, so the Neo4j DNS failure (the `.env`-configured AuraDB instance no longer resolves — likely expired) silently killed vector memory too even though Chroma connected fine; fixed by making the graph store independently optional with `self._graph is not None` guards throughout `manager.py`
- Verified live end-to-end, vector-only (the AuraDB instance in `.env` needs to be recreated to test the graph half): stated a durable fact ("my name is Debanjan, I prefer short answers") in a Document Q&A conversation, confirmed it extracted and stored in Chroma with `agent_id=docqa` (inspected directly); a **second, different** Document Q&A conversation recalled the name correctly; a **Story Developer** conversation asked the same "what do you know about me?" and correctly recalled nothing
- Caught and fixed a real extraction bug in this same test run: on a near-fact-free exchange, the extractor fabricated a fact by echoing the literal text of a few-shot example in `backend/memory/pipeline.py`'s extraction prompt — "the user previously misspelled their name and corrected it to Sarah Connor," which never happened in that conversation. Root cause: the examples were passed as separate `{"role": "user"/"assistant"}` chat messages, the same shape as real conversation turns, so a cheap/nano-tier model could mistake them for prior context. Fixed by folding the examples into plain text inside the system prompt instead (structurally can't be confused with real turns anymore), adding an explicit "these are reference only" instruction, and adding a verbatim-match filter that drops any extracted fact matching an example's output text as a safety net. Reverified: the identical reproduction case (Story Developer, "what do you know about me?") now extracts 0 facts instead of the fabricated one; real fact extraction on a genuine multi-fact message still works correctly (3/3 facts, no regression)
- No memory browser/admin UI yet (Remembrane's Streamlit frontend has Memories/Knowledge-Graph/Settings pages; not ported) — memory operates silently. See "Next" below

**Core chat**
- FastAPI backend + Streamlit frontend, `deepagents` (LangGraph) agent, streaming responses ([backend/online_pipeline/main.py](backend/online_pipeline/main.py), [frontend/app.py](frontend/app.py))
- RAG via FAISS (local, file-based) + local sentence-transformers embeddings, no external key needed for retrieval ([backend/retrieval.py](backend/retrieval.py))
- Generation via OpenRouter, OpenAI-compatible ([backend/agents/docqa/agent.py](backend/agents/docqa/agent.py))

**Knowledge bases**
- Config-driven via `kb.yaml` per folder, live-discovered (no restart) ([backend/kb.py](backend/kb.py))
- Offline ingestion: incremental (hash-tracked, skips unchanged files), PDF/DOCX/MD/TXT, per-file chunk-id manifest for precise delete-on-change/removal ([backend/offline_pipeline/ingest.py](backend/offline_pipeline/ingest.py), [backend/offline_pipeline/loaders.py](backend/offline_pipeline/loaders.py))
- 3 seeded KBs: `product-docs`, `hr-policies`, `sec_10q` (23 real SEC 10-Q filings via [backend/offline_pipeline/download_sec_10q.py](backend/offline_pipeline/download_sec_10q.py))
- arXiv paper downloader ([backend/offline_pipeline/download_arxiv.py](backend/offline_pipeline/download_arxiv.py)), ported from `AIR-FastAgent` (otherwise a duplicate of the already-reviewed `AIB-DocQaAgent` scratch repo — same `pyproject.toml`, same fast-agent scripts, nothing else new). Simplified from the source: HTML is stripped to plain text with stdlib `html.parser` instead of kept raw (no new `bs4` dependency), and the PDF fallback saves the `.pdf` directly instead of shelling out to `pdftotext` — `loaders.py` already parses PDFs natively, so no external binary dependency at all. Verified live: real topic search via the `arxiv` package, 3/3 papers saved through the HTML→text path, output already one of `loaders.py`'s supported types (`.txt`/`.pdf`), so nothing further needed to make it KB-ingestible
- All data under `data/`: `kbs/` (ingested), `raw/` (scratch downloads), `uploads/` (CSV uploads), `charts/` (generated), `faiss_indexes/` (generated) — no more repo-root scatter

**Agent behavior & answer quality**
- Asks a clarifying follow-up only when *retrieved results* are genuinely ambiguous (multi-company/multi-period), not before searching ([backend/agents/docqa/agent.py](backend/agents/docqa/agent.py) system prompt)
- Multi-turn follow-ups get rewritten into self-contained queries before searching (resolves pronouns/implied subjects like "what about Q2?") — verified live: a bare follow-up correctly resolved to the prior turn's company+metric and cited the right filing
- Inline citations: retrieval keeps chunk metadata (`source`, `doc_title`, `chunk_index`) through to the tool output, model cites `(Source: filename)`, prompt hardened against citing sources not actually returned that turn (caught a real nano hallucination during testing) ([backend/retrieval.py](backend/retrieval.py), [backend/agents/docqa/agent.py](backend/agents/docqa/agent.py))
- Fixed a real retrieval recall gap: chunks now get their document title prefixed into the *embedded* text at ingest time (e.g. `[2026 Q1 AAPL]`), not just stored as metadata — exact-document queries ("Apple Q1 2026 revenue") were losing to semantically-louder chunks from the wrong filing. Verified: a query that previously returned "I don't have that figure" now returns the correct number, correctly cited ([backend/offline_pipeline/ingest.py](backend/offline_pipeline/ingest.py)); retrieval `k` bumped 4→6 for more recall headroom ([backend/retrieval.py](backend/retrieval.py))
- Smart model router: heuristic (keyword/length, no LLM call) picks nano vs mini model per question via `wrap_model_call` middleware, now with a labeled eval harness (`python -m backend.agents.docqa.router`, 100% on 12 examples spanning all 3 KBs) instead of untested vibes ([backend/agents/docqa/router.py](backend/agents/docqa/router.py), [backend/config.py](backend/config.py))
- Hybrid retrieval: BM25 (sparse/keyword) alongside FAISS (dense/semantic), merged via Reciprocal Rank Fusion, per-KB index at `data/faiss_indexes/<slug>/bm25.pkl`, rebuilt in full on every ingest run (no incremental API for BM25, cheap enough not to need one) ([backend/retrieval.py](backend/retrieval.py), [backend/offline_pipeline/ingest.py](backend/offline_pipeline/ingest.py)). Caught and fixed a real regression during testing: without stopword removal, BM25's raw term frequency let an unrelated company's filing outrank the correct one (a short chunk repeating "what"/"was"/"in" plus one real term outscored the actual answer) — added minimal stopword filtering, verified live before/after: wrong-company result eliminated, correct filing consistently retrieved
- Retrieval eval harness, finally filling the "no eval harness for retrieval quality" gap: `backend/offline_pipeline/eval_retrieval.py`, built on Vectara's `open_ragbench` benchmark (real queries + ground-truth relevant papers) — sourced from reviewing `AIR-OpenRAG` (its own scoring scripts were just similarity-metric scratch code with no actual qrels-based scoring; the eval loop — Recall@k, MRR — is new). Deliberately reuses the real pipeline instead of a parallel one: fetched papers become a normal KB (`ragbench-eval`, ingested via the unmodified `backend.offline_pipeline.ingest`), scored with the unmodified `backend.retrieval.search()`. Verified live end-to-end: fetched a 50-paper slice, ingested (8,698 chunks), ran the eval — 100% Recall@1/3/5/10, MRR 1.0 on 190 queries. Spot-checked one query's actual ranked results by hand to rule out a scoring bug (confirmed real: correct paper at rank 1, two *different* papers genuinely present at ranks 7 and 10) — the perfect score is a real artifact of the small 50-paper sample (benchmark questions are paper-specific, far less distractor competition than the full corpus), not a broken scorer; a larger `MAX_PAPERS` would give a more realistic number

**Ops**
- Structured logging throughout (stdlib `logging`, `LOG_LEVEL` env var) — request lifecycle, retrieval timing, router decisions ([backend/logging_config.py](backend/logging_config.py))

**Persistence**
- Conversations + messages persisted to a local SQLite file (`data/shonku.db`, stdlib `sqlite3`, no new dependency) ([backend/db.py](backend/db.py))
- Server-owned history: `/chat` takes `{conversation_id, message}` only — `agent_type`/`source_type`/`source_ref` come from the conversation record, the client can no longer override or smuggle a different source into an existing thread
- Endpoints: `POST /conversations`, `GET /conversations`, `GET /conversations/{id}`, `GET /conversations/{id}/messages`
- Frontend built around `st.query_params` + server-fetched history instead of `session_state` — survives a real browser refresh and a backend restart, plus a sidebar to browse/reopen past conversations per agent+source ([frontend/app.py](frontend/app.py))
- Verified live: raw DB inspection, a real `uvicorn` process restart, cross-source isolation, and a genuine browser navigation recovering full history from just the URL

## Next

**1. Knowledge base / dataset admin**
- Still folder + `kb.yaml` + manual `ingest` run for docqa — no admin UI (explicitly deferred from day one, still true)
- No access control on which KBs/datasets a user can see/query

**2. Frontend**
- Streamlit is functional, not the Linear/VS Code/Notion-grade UI the product vision describes — would need a real frontend (Next.js/React) if that bar matters for this phase

**3. Data Analyst follow-ups**
- Model is a fixed default (router's `hard_model`) for the whole agent — no per-message easy/hard routing like docqa has
- `run_python_code` executes arbitrary LLM-generated code (inherited from the source repo as-is) — fine for local single-user use, would need sandboxing before any shared/hosted deployment
- No eval harness for retrieval/answer quality on this agent yet (docqa now has one for its router; Data Analyst has none)

**4. Residual retrieval limitation**
- Adjacent-quarter ambiguity: dense embeddings still struggle to distinguish two filings from the *same* company whose MD&A boilerplate is nearly identical quarter to quarter (e.g. Q1 vs Q2 2026 AAPL) — hybrid retrieval fixed wrong-company results but this narrower case remains. A metadata-aware exact-period filter (parse quarter/year from the query, restrict candidates before ranking) would be the next lever, not attempted here

**5. Recommendation agent follow-ups**
- Image-upload search deferred (text-only v1) — the source repo's CLIP text+image blending is straightforward to re-add once there's a reason to
- Catalog is a 425-item category-sampled slice (not the full 44k HuggingFace dataset), so coverage is thin in places — a text-only query outside the sample (e.g. "blue winter jacket") can legitimately return zero results
- ChromaDB is now a second vector-store dependency alongside FAISS — acceptable since the modality (image+text CLIP vectors) is genuinely different from KB text chunks, but worth knowing if dependency surface ever needs trimming
- No eval harness for this agent either (same gap noted for Data Analyst)

**6. Story Developer / Mystery Generator follow-ups**
- No eval harness for either (same gap noted for Data Analyst and Recommendation)
- Mystery Generator's per-conversation `data/mysteries/<id>/` directories are never cleaned up — same "nothing gets pruned yet" limitation as `shonku.db`
- The path-discipline fix (deterministic filename + mandatory `ls`/`read_file` before judging) is prompt-level, not a hard guarantee — worth an occasional spot-check if the underlying model changes

**7. Ingestion quality — docling + chonkie**
- Sourced from reviewing `AIR-DataIngestion` (mostly a scratch arXiv chunking/embedding benchmark, not directly portable — but two pieces of it are worth pulling in)
- **docling for PDF→Markdown**: swap `PyPDFLoader` in [backend/offline_pipeline/loaders.py](backend/offline_pipeline/loaders.py) for `docling`'s structure-aware conversion (`DocumentConverter().convert(...).export_to_markdown()`) — preserves headers/tables/sections instead of a flat per-page text dump. Real candidate for retrieval-quality improvement on dense KBs like `sec_10q`, but needs testing against actual chunk quality on a real KB before committing — PDF parsers vary a lot on messy real-world documents
- **chonkie's `RecursiveChunker`**: alternative to the `RecursiveCharacterTextSplitter` already used in [backend/offline_pipeline/ingest.py](backend/offline_pipeline/ingest.py) — token-count-aware chunking. Lower priority than docling: current splitter already works, this is a new dependency for uncertain upside
- Not taken from that repo: arXiv-specific downloader, LM Studio local embedding, CSV/Parquet output — none of it touches FAISS/BM25/serving, and its BM25 usage (plain `BM25Retriever.from_documents()`, no dense fusion) is less capable than Shonku's existing RRF hybrid merge

**8. Raw LangGraph workflow builder — if a looping agent is ever needed**
- Sourced from reviewing `AIR-DeepAgent` (mostly empty skeleton — its two `deep-agent-v*.py` files are 100% `# TODO` stubs, nothing functional to port)
- One real, working piece: `langgraph_builder.py`'s `LangGraphBuilder` — declarative YAML-driven `StateGraph` construction (nodes/edges/conditional-routing/state-schema) instead of hand-wiring one in Python
- Not needed today — every Shonku agent goes through `deepagents.create_deep_agent()`, which already covers subagent delegation + tools + prompts, no agent hand-builds a raw graph
- Worth reaching for specifically if a future agent needs genuine iterative/looping control flow that deepagents' subagent-delegation model doesn't express well — e.g. a "deep research" style agent: decompose question → search → retrieve → analyze → synthesize → loop-if-insufficient → generate report (the exact shape stubbed, unimplemented, in that repo)

**9. Alternative embedding/vector-store implementations — if ever needed**
- Sourced from reviewing `AIR-OpenRAG` (alongside the eval harness already built into Done, above) — mostly scratch code otherwise, these two pieces are just worth knowing about, not adopted
- **`OpenRouterEmbeddings`** (cloud embedding via OpenRouter, tested there with `qwen/qwen3-embedding-8b`) — confirms OpenRouter genuinely serves an `/embeddings` endpoint, which earlier assumptions in this project doubted. A real alternative to the current free local MiniLM (`backend/retrieval.py`) if embedding quality ever becomes the bottleneck — but trades away the current zero-cost, zero-network-dependency retrieval path for per-call API cost and a new failure mode (retrieval breaks if the embedding API is down). Not adopted now — no evidence local embeddings are the limiting factor
- **`FaissCosineStore`** (hand-rolled bare-FAISS cosine store, no langchain wrapper) — a lower-level reference implementation. No functional gain over what's already built (langchain FAISS + the hybrid BM25/RRF layer on top) — would be a lateral rewrite. Worth revisiting only if the langchain FAISS wrapper itself ever becomes a real constraint

**10. Flashcards/Quiz agent — candidate, if a 6th agent is wanted**
- Sourced from reviewing `AIF-Quiz`, a standalone Streamlit app (real personal tool, not a prototype — actual study data for Python/CS topics against an Obsidian vault at `~/Documents/Vault13`). Different in kind from every repo reviewed so far: **zero LLM/agent framework**, pure algorithmic
- **`sr_engine.py`**: solid FSRS-4.5 spaced-repetition scheduler (stability/difficulty/retrievability, leech detection) — stdlib-only (`json`/`math`/`datetime`), no new dependency, directly reusable as-is
- **`vault_parser.py`**: parses `.md` files tagged `#flashcards` for Q&A blocks + YAML frontmatter (category/topic/tags), plus a manual-cards JSON store — tied to Obsidian vault conventions specifically, would need a source-picker concept Shonku doesn't have yet (a filesystem vault path, not a KB folder or dataset)
- **`quiz_engine.py`**: MCQ generation using *other cards' answers* as distractors (no LLM) — the crude part; an LLM-backed version (real distractors, explanations) would fit Shonku's existing OpenRouter generation path better
- Doesn't map onto any existing Shonku agent shape — no retrieval, no `deepagents.create_deep_agent()` graph, no generation in the source at all. Would be a genuinely new agent: **Flashcards/Quiz**, source = a vault path instead of a KB/dataset/catalog
- Not started — needs a scoping decision (keep FSRS+quiz scoring as-is and just add LLM-generated distractors/explanations? or go further and let the agent field free-form questions about a card's topic using the vault as context, closer to docqa) before implementation

**11. Multilingual / Bangla support — only if the product ever needs it**
- Sourced from reviewing `AIF-GolpoForge`/`bangla_sentiment_miner` — not a chatbot, not LLM-based, no direct port. Classical ML: TF-IDF (word/char n-grams) + LinearSVC sweep, plus a Word2Vec/FastText embedding variant, trained on the SentNoB Bengali sentiment dataset (real Train/Val/Test CSVs present)
- Nothing here is reusable as-is — no agent framework, needs an external `cc.bn.300.vec` FastText vector file (not included, large) for the embedding path, and solves sentiment classification, not chat/retrieval/generation
- Only relevant *if* Shonku ever needs multilingual (Bangla) input handling — e.g. docqa answering in Bangla, or a source KB with Bangla documents. Current retrieval embeddings (`sentence-transformers` MiniLM, `backend/retrieval.py`) and OpenRouter generation already handle non-English text passably out of the box; this repo's value would only be as a reference for Bangla-specific text normalization (`preprocess.py`'s numeral normalization) or a Bangla-specific embedding model swap, not a wholesale port
- Not started — no current product requirement for non-English support

**12. Memory layer follow-ups**
- No browser/admin UI (Remembrane's Streamlit frontend has Memories/Knowledge-Graph/Settings pages — deliberately not ported this pass, memory operates silently). Real fast-follow if the user ever wants to see/edit/delete what's been remembered
- The `.env`-configured Neo4j AuraDB instance (`ff65f0a5...`) no longer resolves (DNS NXDOMAIN, not just paused) — needs a fresh free instance created at [console.neo4j.io](https://console.neo4j.io) and `.env` updated before entity-graph-expanded recall can be tested/used; vector-only memory works today regardless
- No memory deletion/expiry — same "nothing gets pruned yet" limitation as `shonku.db`/`data/mysteries/`
- No eval harness for extraction/recall quality (same gap noted for Data Analyst/Recommendation/Story Developer/Mystery Generator)

**13. Production readiness**
- No auth (explicitly deferred)
- No tests — zero real test suite; the router eval is a start, not coverage
- No rate limiting / abuse protection on `/chat`
- No deployment setup (Docker, CI, hosting target)
- Secrets are plain `.env` — fine for local dev, not for shipping
- No conversation deletion/archival — `shonku.db` only grows

## Suggested order

Everything left is scope decisions, not obvious next steps — say which direction matters most.
