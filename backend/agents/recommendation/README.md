# recommendation

Product Recommendations agent. Source = fixed fashion catalog, no picker.
Ported from `AIB-RecommendationAgent`'s procedural pipeline, wrapped as a
single deepagents tool so it fits the same `.stream()` shape as every other
agent.

- `agent.py` — `make_recommendation_agent(memory_context="")`: one tool
  (`search_products`) running `_run_pipeline`: LLM query expansion → CLIP
  embed → ChromaDB vector search → LLM rerank with match explanations.
  Text-only for v1 (image-upload search deferred). Results tagged
  `[Image: filename]` for the frontend to render.
- `embedding_service.py` — CLIP (`open_clip`), text/image → 512-d vectors.
- `vector_store.py` — ChromaDB wrapper, `data/products/chroma_db` (seeded,
  committed — different Chroma instance than `backend/memory/`'s).
- `llm_service.py` — query expansion + reranking prompts.
- `models.py` — `Product` / `SearchResponse` / `RankedProduct` /
  `QueryExpansion`.

Called from `backend/online_pipeline/main.py`'s `/chat` when
`agent_type == "recommendation"`.
