# docqa

Document Q&A agent. Source = a KB (`data/kbs/<slug>/`). Answers using only
`search_knowledge_base` tool results, cites `[Source: ...]` passages.

- `agent.py` — `make_kb_agent(kb_slug, memory_context="")`: builds the search
  tool (wraps `backend.retrieval.search`), constructs the deepagent with
  `route_model` middleware (easy/hard model per turn) and the memory-context
  block appended to the system prompt if present.
- `router.py` — heuristic (keyword/length, no LLM call) picks easy vs hard
  model per question. Eval: `python -m backend.agents.docqa.router`.

Called from `backend/online_pipeline/main.py`'s `/chat` when
`agent_type == "docqa"`.
