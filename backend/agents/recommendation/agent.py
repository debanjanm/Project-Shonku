"""Deep agent wrapping the product search pipeline (LLM query expansion ->
CLIP embedding -> ChromaDB vector search -> LLM reranking) as a single tool.

Ported from AIB-RecommendationAgent's services/search_pipeline.py, collapsed
into one tool function (text-only for v1 — image-upload search deferred) so
this agent stays a normal deepagents-compiled graph like docqa/data_analyst,
instead of adding a non-agentic branch to /chat's streaming loop.
"""

import logging
import os

from deepagents import create_deep_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from backend.agents.recommendation.embedding_service import get_embedding_service
from backend.agents.recommendation.llm_service import LLMService
from backend.agents.recommendation.models import Product
from backend.agents.recommendation.vector_store import VectorStore
from backend.config import get_router_config

logger = logging.getLogger(__name__)

TOP_K_CANDIDATES = 20
TOP_K_RESULTS = 10

_BASE_SYSTEM_PROMPT = """You are the Project Shonku product recommendation assistant. Answer using ONLY the
`search_products` tool's results as your source of truth for the product catalog.

Always call `search_products` with the user's question first. Present the results as a short list:
product name, brand, price, and the match reason already provided by the tool — don't invent details
not present in the tool output. Each result is tagged `[Image: filename]`; always keep that tag verbatim
in your answer right after the product it belongs to, so the UI can render the image.

If no products match well, say so plainly instead of forcing a recommendation. Be concise and direct.
Do not use file or shell tools."""


def get_model() -> ChatOpenAI:
    model_name = get_router_config().hard_model
    logger.info("creating recommendation chat model via openrouter model=%s", model_name)
    return ChatOpenAI(
        model=model_name,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        streaming=True,
    )


def _run_pipeline(query: str) -> list[dict]:
    """Text-only port of SearchPipeline.run(): expand -> embed -> retrieve -> rerank."""
    llm = LLMService()
    embedder = get_embedding_service()
    store = VectorStore()

    expansion = llm.expand_query(query, image_provided=False)

    query_embedding = embedder.embed_query(text=expansion.expanded_query)

    category_filter = expansion.search_filters.get("category")
    if category_filter and category_filter.lower() == "unknown":
        category_filter = None

    n_candidates = min(TOP_K_CANDIDATES, store.count())
    if n_candidates == 0:
        return []

    candidates = store.query(embedding=query_embedding, n_results=n_candidates, category_filter=category_filter)
    reranked = llm.rerank(original_query=query, intent=expansion.extracted_intent, candidates=candidates)

    meta_lookup = {c["id"]: c["metadata"] for c in candidates}
    cosine_scores = {c["id"]: c["similarity_score"] for c in candidates}

    results = []
    for item in reranked[:TOP_K_RESULTS]:
        pid = item.get("id")
        meta = meta_lookup.get(pid)
        if meta is None:
            continue
        product = Product.from_chroma_metadata(meta)
        results.append(
            {
                "product": product,
                "similarity_score": cosine_scores.get(pid, 0.0),
                "rerank_score": float(item.get("score", 0.0)),
                "match_reason": item.get("match_reason", ""),
            }
        )
    return results


def make_recommendation_agent(memory_context: str = ""):
    logger.info("building recommendation deep agent")
    model = get_model()
    system_prompt = _BASE_SYSTEM_PROMPT + (f"\n\n{memory_context}" if memory_context else "")

    @tool
    def search_products(query: str) -> str:
        """Search the product catalog for items matching the user's query."""
        logger.info("tool search_products called query=%r", query)
        results = _run_pipeline(query)
        if not results:
            logger.warning("tool search_products found nothing query=%r", query)
            return "No matching products found in the catalog."

        lines = []
        for r in results:
            p: Product = r["product"]
            lines.append(
                f"- **{p.name}** ({p.brand}) — ${p.price:.2f}\n"
                f"  Match: {r['match_reason']} (similarity {r['similarity_score']:.2f})\n"
                f"  [Image: {p.image_filename}]"
            )
        return "\n\n".join(lines)

    return create_deep_agent(
        model=model,
        tools=[search_products],
        system_prompt=system_prompt,
    )
