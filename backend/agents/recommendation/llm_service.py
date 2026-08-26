"""LLM query expansion + reranking. Ported from AIB-RecommendationAgent's
services/llm_service.py, adapted to reuse Shonku's router hard_model and
OPENROUTER_API_KEY convention instead of a separate OPENROUTER_MODEL/BASE_URL.
"""

import json
import logging
import os

from openai import OpenAI

from backend.agents.recommendation.models import QueryExpansion
from backend.config import get_router_config

logger = logging.getLogger(__name__)

EXPANSION_SYSTEM_PROMPT = """You are a product search assistant. Analyze user queries and extract structured search intent to improve product retrieval.
Always respond with valid JSON only. No markdown, no explanation."""

EXPANSION_USER_TEMPLATE = """Analyze this product search query and return a JSON object with exactly these keys:
- "expanded_query": A richer, more descriptive search phrase including synonyms, attributes, and style descriptors. Keep under 50 words.
- "extracted_intent": {{
    "category": One of [footwear, outerwear, bags, electronics, home, apparel, accessories, personal_care, sporting_goods, unknown],
    "key_attributes": List of product attributes mentioned or implied (max 5),
    "style_keywords": List of aesthetic/style descriptors (max 3),
    "price_sensitivity": "budget" | "mid-range" | "premium" | "unknown"
  }}
- "search_filters": {{ "category": same as extracted_intent.category }} (omit key if "unknown")

User query: "{query}"
Image provided: {image_provided}"""

RERANK_SYSTEM_PROMPT = """You are a product recommendation expert. Given a user's search intent and candidate products, rank them by relevance.
Always respond with valid JSON only. No markdown, no explanation."""

RERANK_USER_TEMPLATE = """The user searched for: "{original_query}"
Their intent: {intent_json}

Here are {n} candidate products retrieved by vector search:
{candidates_json}

Return a JSON array of ALL {n} candidates sorted by relevance (most relevant first).
Each item must have:
- "id": the product id (string)
- "score": relevance score float 0.0-1.0 (1.0 = perfect match)
- "match_reason": 1-2 sentence explanation of why this product matches (or doesn't match) the user's intent

Focus on semantic relevance to the user's implicit needs, not just keyword overlap."""


class LLMService:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        self.model = model or get_router_config().hard_model
        self.client = OpenAI(api_key=self.api_key, base_url="https://openrouter.ai/api/v1")

    def expand_query(self, query: str, image_provided: bool = False) -> QueryExpansion:
        """
        Use the LLM to expand the user query into a richer search phrase
        and extract structured intent. Falls back to the original query if
        the LLM call fails.
        """
        try:
            user_msg = EXPANSION_USER_TEMPLATE.format(
                query=query,
                image_provided=str(image_provided).lower(),
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXPANSION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=400,
                timeout=20,
            )
            raw = response.choices[0].message.content.strip()
            data = json.loads(raw)
            return QueryExpansion(
                expanded_query=data.get("expanded_query", query),
                extracted_intent=data.get("extracted_intent", {}),
                search_filters=data.get("search_filters", {}),
            )
        except Exception:
            logger.warning("expand_query failed, using original query", exc_info=True)
            return QueryExpansion(expanded_query=query, extracted_intent={}, search_filters={})

    def rerank(self, original_query: str, intent: dict, candidates: list[dict]) -> list[dict]:
        """
        Use the LLM to rerank candidates by semantic relevance.
        Returns a list of {id, score, match_reason} dicts sorted by score desc.
        Falls back to cosine order if the LLM call fails.
        """
        if not candidates:
            return []

        candidate_summaries = [
            {
                "id": c["id"],
                "name": c["metadata"]["name"],
                "brand": c["metadata"]["brand"],
                "category": c["metadata"]["category"],
                "price": c["metadata"]["price"],
                "description": c["metadata"]["description"][:200],
                "tags": c["metadata"]["tags"],
            }
            for c in candidates
        ]

        try:
            user_msg = RERANK_USER_TEMPLATE.format(
                original_query=original_query,
                intent_json=json.dumps(intent, indent=2),
                n=len(candidates),
                candidates_json=json.dumps(candidate_summaries, indent=2),
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": RERANK_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=1500,
                timeout=30,
            )
            raw = response.choices[0].message.content.strip()
            ranked = json.loads(raw)

            valid = [r for r in ranked if isinstance(r, dict) and "id" in r and "score" in r]
            valid.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
            return valid

        except Exception:
            logger.warning("rerank failed, falling back to cosine order", exc_info=True)
            return [
                {
                    "id": c["id"],
                    "score": c["similarity_score"],
                    "match_reason": "Retrieved by vector similarity search.",
                }
                for c in candidates
            ]
