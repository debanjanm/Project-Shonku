# recommendation

Product Recommendations agent. Source = fixed fashion catalog, no picker.
Ported from `AIB-RecommendationAgent`'s procedural pipeline, wrapped as a
single deepagents tool so it fits the same `.stream()` shape as every other
agent.

- `agent.py` — `make_recommendation_agent(memory_context="")`: two tools —
  `search_products` running `run_pipeline()` (LLM query expansion → CLIP
  embed → ChromaDB vector search → LLM rerank with match explanations,
  deduped by product id since the LLM's rerank JSON occasionally repeats
  one) and `get_product_details` (direct ID lookup via
  `VectorStore.get_product`, for a follow-up about a specific
  previously-shown item instead of a fresh fuzzy search). Text-only for v1
  (image-upload search deferred). Results tagged `[Image: filename]
  [ID: product_id]` — the frontend renders the image and strips both tags
  from displayed text.
- `skills/outfit-composition/` — deepagents Agent Skill: decompose a
  compound outfit ask (e.g. "an outfit for a summer wedding") into one
  `search_products` call per garment category instead of one query that
  can't represent multiple distinct items.
- `embedding_service.py` — CLIP (`open_clip`), text/image → 512-d vectors.
- `vector_store.py` — ChromaDB wrapper, `data/products/chroma_db` (seeded,
  committed — different Chroma instance than `backend/memory/`'s).
- `llm_service.py` — query expansion + reranking prompts.
- `models.py` — `Product` / `SearchResponse` / `RankedProduct` /
  `QueryExpansion`.
- `eval_recommendation.py` — leave-one-out eval (no labeled ground truth
  exists for this catalog): builds a query from a sampled product's own
  tags, checks whether that product comes back in its own results.
  Reports Recall@k/MRR for the raw candidate stage and the real
  `run_pipeline()` output separately. Run:
  `python -m backend.agents.recommendation.eval_recommendation`

Called from `backend/online_pipeline/main.py`'s `/chat` when
`agent_type == "recommendation"`.
