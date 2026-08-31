# memory

Cross-cutting long-term memory, consumed by every agent (not an agent
itself). Ported from `AIF-Remembrane`'s Mem0 architecture (Extract → Search
→ Update → Store), adapted to Shonku's sync backend and local embeddings.
Scoped **per-agent**: a fact learned in one Document Q&A conversation is
recalled in any other Document Q&A conversation, never in Story Developer or
elsewhere. `conversation_id` is stored as provenance only, not filtered.

- `models.py` — `MemoryScope` (user_id + agent_id filter, run_id = tag only),
  `Memory`, `ExtractedFact`, `MemoryOperation`, `MemoryUpdateResult`,
  `ScoredMemory`, `GraphNode`/`GraphRelationship`.
- `chroma_store.py` — vector store. Embeddings from
  `backend.retrieval.get_embeddings()` (local MiniLM, free) — not
  OpenRouter. Persisted at `CHROMA_PERSIST_PATH` (`data/chroma/`, gitignored).
- `neo4j_store.py` — entity graph (Neo4j AuraDB), sync driver. Optional: if
  connection fails at startup, `graph_enabled=False` and memory runs
  vector-only — never breaks chat.
- `pipeline.py` — extraction + ADD/UPDATE/DELETE/SKIP decision LLM calls
  (`LLM_MODEL` env var), context synthesis for prompt injection. Few-shot
  examples live as plain text inside the system prompt (not chat-turn
  messages) plus a verbatim-match filter — fixed a real bug where a weak
  model echoed example text as a fabricated fact.
- `manager.py` — `MemoryManager` orchestrator: `add()`, `search()`,
  `get_context()`. `get_context()` never raises — a Neo4j/Chroma failure
  degrades to no recall, not a broken chat request.
- `__init__.py` — `get_memory_manager()`, lazy singleton, same pattern as
  `backend.retrieval.get_embeddings()`.

Wired into `backend/online_pipeline/main.py`'s `/chat`: `get_context()`
called before agent construction (result passed to every `make_*_agent()`
as `memory_context`), `add()` fired in a background thread after the
response streams — zero added latency either way.
