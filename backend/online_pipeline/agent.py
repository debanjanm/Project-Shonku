"""Deep agent bound to a single KB's retrieval tool."""

import logging
import os

from deepagents import create_deep_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from backend.config import get_router_config
from backend.online_pipeline.router import choose_model
from backend.retrieval import search

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Project Shonku chat assistant. Answer the user's question using ONLY the
`search_knowledge_base` tool's results as your source of truth for the currently selected knowledge base.

Always call `search_knowledge_base` with the user's question first — do not ask a clarifying question before
searching, and do not assume the knowledge base is ambiguous just because the question alone could be. After
reviewing the results, ask ONE concise follow-up question only if the results themselves are genuinely
ambiguous — e.g. they span multiple distinct companies, documents, or time periods that could each answer
differently, and the user didn't say which. If the results clearly point to one answer, answer directly.

If the knowledge base doesn't contain the answer, say so plainly instead of guessing. Each passage you get
back is tagged `[Source: ...]` — when you state a fact from it, cite that exact source name in parentheses
right after, e.g. "18 days (Source: pto.md)". Only ever cite a source name that appears verbatim in a
`[Source: ...]` tag from THIS turn's tool results — never invent, guess, or reuse a source name from memory
or a different question. If a fact has no matching source tag, don't cite one for it. Be concise and direct.
Do not use file or shell tools."""


def get_model(model_name: str) -> ChatOpenAI:
    logger.info("creating chat model via openrouter model=%s", model_name)
    return ChatOpenAI(
        model=model_name,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        streaming=True,
    )


def _latest_human_text(messages) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


@wrap_model_call
def route_model(request: ModelRequest, handler) -> ModelResponse:
    """Pick easy/hard model per call based on the latest human message."""
    message = _latest_human_text(request.state["messages"])
    model = get_model(choose_model(message))
    return handler(request.override(model=model))


def make_kb_agent(kb_slug: str):
    logger.info("building deep agent kb_slug=%s", kb_slug)

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the currently selected knowledge base for relevant passages."""
        logger.info("tool search_knowledge_base called kb_slug=%s query=%r", kb_slug, query)
        docs = search(kb_slug, query)
        if not docs:
            logger.warning("tool search_knowledge_base found nothing kb_slug=%s query=%r", kb_slug, query)
            return "No relevant passages found in this knowledge base."
        passages = []
        for doc in docs:
            source = doc.metadata.get("source") or doc.metadata.get("doc_title") or "unknown"
            passages.append(f"[Source: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(passages)

    return create_deep_agent(
        model=get_model(get_router_config().easy_model),  # default; route_model overrides per call
        tools=[search_knowledge_base],
        system_prompt=SYSTEM_PROMPT,
        middleware=[route_model],
    )
