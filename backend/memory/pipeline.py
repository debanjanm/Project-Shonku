"""Memory pipeline: extract facts, decide ADD/UPDATE/DELETE/SKIP, format for
injection. Ported from AIF-Remembrane's pipeline/{extractor,updater,
synthesizer}.py, collapsed into one file — same convention as
backend/agents/data_analyst/sql_pipeline.py (multi-stage LLM pipeline, one
file, not one-file-per-stage). Uses the same sync langchain_openai.ChatOpenAI
client backend/agents/docqa/agent.py already uses for OpenRouter, called
directly instead of through create_deep_agent.
"""

import json
import logging
import os
from datetime import datetime

from langchain_openai import ChatOpenAI

from backend.memory.models import ExtractedFact, MemoryOperation, MemoryScope, MemoryType, OperationType, ScoredMemory

logger = logging.getLogger(__name__)


def _get_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "openai/gpt-5.4-nano"),
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )


def _complete_json(messages: list[dict], temperature: float) -> dict:
    model = _get_model()
    response = model.invoke(messages, temperature=temperature)
    text = response.content if isinstance(response.content, str) else str(response.content)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


_EXTRACT_SYSTEM_PROMPT = """You are a memory extraction assistant for an AI agent system.

Your task is to analyze a conversation and extract key facts worth remembering long-term.
These are facts about the USER — their preferences, biographical details, goals, skills,
relationships, and any other information that would help an AI assistant serve them better
in future conversations.

Rules:
- Only extract facts with lasting relevance (skip pleasantries, filler, or one-time requests)
- Write each fact as a concise, standalone third-person statement about the user
  (e.g. "The user prefers Python over Java" not "I like Python")
- IMPORTANT: If the user corrects or updates previously stated information (e.g. "actually my name is X",
  "I meant Y", "I misspelled — it's Z"), extract the CORRECTED fact. Corrections are high-value facts.
- For each fact, identify named entities (people, places, topics, tools)
- Assign confidence: 1.0 = stated explicitly, 0.8 = strongly implied, 0.6 = inferred
- Classify memory_type: "user" (preferences/bio), "agent" (agent context), "run" (session-only)
- If the conversation has no memorable facts (e.g. it's just a greeting, or the user asked what you
  know about them and nothing new was actually said), return an EMPTY facts list. Do not invent a fact
  just to have something to return.

Return ONLY valid JSON in this exact schema:
{
  "facts": [
    {
      "content": "<fact as a statement>",
      "entities": ["<entity1>", "<entity2>"],
      "confidence": <0.0 to 1.0>,
      "memory_type": "user" | "agent" | "run"
    }
  ]
}

If there are no memorable facts, return {"facts": []}.

Below are two EXAMPLES showing the expected input/output shape. They are reference material only —
NOT part of any real conversation. Never extract a fact from the example text itself; only extract
facts from the actual conversation given in the user's message, which follows after "Conversation:".

Example 1 — input:
Conversation:
User: I'm a backend engineer at a startup in Berlin. I mostly work with Python and FastAPI.
Assistant: That's great! Do you use any databases?
User: Yes, PostgreSQL mostly. I hate dealing with ORMs though.

Example 1 — output:
{"facts": [
  {"content": "The user is a backend engineer at a startup in Berlin.", "entities": ["Berlin", "backend engineer", "startup"], "confidence": 1.0, "memory_type": "user"},
  {"content": "The user primarily works with Python and FastAPI.", "entities": ["Python", "FastAPI"], "confidence": 1.0, "memory_type": "user"},
  {"content": "The user uses PostgreSQL as their primary database.", "entities": ["PostgreSQL"], "confidence": 1.0, "memory_type": "user"},
  {"content": "The user dislikes working with ORMs.", "entities": ["ORMs"], "confidence": 1.0, "memory_type": "user"}
]}

Example 2 — input:
Conversation:
User: Sorry, I misspelled my name earlier — it's Sarah Connor, not Sara Connor.
Assistant: No problem, Sarah Connor!

Example 2 — output:
{"facts": [
  {"content": "The user's name is Sarah Connor.", "entities": ["Sarah Connor"], "confidence": 1.0, "memory_type": "user"}
]}
"""

# Guards against the model echoing example content as if it were a real
# extracted fact (observed live on a nano-tier model) — reject any fact that
# matches an example's output verbatim, even though the prompt above now also
# tells it not to.
_EXAMPLE_FACT_CONTENTS = frozenset({
    "the user is a backend engineer at a startup in berlin.",
    "the user primarily works with python and fastapi.",
    "the user uses postgresql as their primary database.",
    "the user dislikes working with orms.",
    "the user's name is sarah connor.",
})


def extract_facts(user_message: str, assistant_message: str, scope: MemoryScope) -> list[ExtractedFact]:
    """Extract memorable facts from one user/assistant exchange."""
    conversation_text = f"User: {user_message}\nAssistant: {assistant_message}"
    llm_messages = [
        {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Conversation:\n{conversation_text}\n\nExtract memorable facts."},
    ]
    try:
        result = _complete_json(llm_messages, temperature=0.1)
        raw_facts = result.get("facts", [])
    except Exception as e:
        logger.warning("memory extraction failed, skipping user_id=%s error=%s", scope.user_id, e)
        return []

    facts: list[ExtractedFact] = []
    for item in raw_facts:
        try:
            content = item["content"]
            if content.strip().lower() in _EXAMPLE_FACT_CONTENTS:
                logger.warning("dropped extracted fact matching a few-shot example verbatim user_id=%s", scope.user_id)
                continue
            facts.append(ExtractedFact(
                content=content,
                entities=item.get("entities", []),
                confidence=float(item.get("confidence", 1.0)),
                memory_type=MemoryType(item.get("memory_type", "user")),
            ))
        except (KeyError, ValueError):
            continue
    logger.info("extracted facts count=%d user_id=%s", len(facts), scope.user_id)
    return facts


_UPDATE_SYSTEM_PROMPT = """You are a memory management system for an AI agent.

Given a list of NEW FACTS extracted from a conversation and a list of EXISTING MEMORIES
that are related to those facts, determine what to do with each new fact.

For each new fact, choose exactly one action:
- ADD: This is genuinely new information — no existing memory covers it
- UPDATE: An existing memory covers the same topic but needs to be corrected or expanded
  (provide the merged/improved content)
- DELETE: This fact directly contradicts an existing memory which should now be removed
  (the new fact itself will be added as a replacement)
- SKIP: The existing memories already capture this fact — no action needed

Rules:
- Avoid redundant memories; prefer UPDATE/SKIP over ADD when possible
- When updating, write the merged content as a complete, standalone sentence
- Only mark DELETE when the new fact clearly contradicts a specific existing memory

Return ONLY valid JSON:
{
  "operations": [
    {
      "fact_index": <int>,
      "action": "ADD" | "UPDATE" | "DELETE" | "SKIP",
      "target_memory_id": "<id of existing memory to update/delete, or null>",
      "merged_content": "<improved content for UPDATE, or null>"
    }
  ]
}
"""


def compute_operations(new_facts: list[ExtractedFact], existing_memories: list[ScoredMemory]) -> list[MemoryOperation]:
    if not new_facts:
        return []
    if not existing_memories:
        return [MemoryOperation(action=OperationType.ADD, fact=fact) for fact in new_facts]

    facts_text = "\n".join(f"[{i}] {fact.content}" for i, fact in enumerate(new_facts))
    existing_text = "\n".join(f"[id={sm.memory.id}] {sm.memory.content}" for sm in existing_memories)
    llm_messages = [
        {"role": "system", "content": _UPDATE_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"NEW FACTS:\n{facts_text}\n\nEXISTING MEMORIES:\n{existing_text}\n\n"
            "Determine the operation for each new fact."
        )},
    ]
    try:
        result = _complete_json(llm_messages, temperature=0.0)
        raw_ops = result.get("operations", [])
    except Exception as e:
        logger.warning("memory update decision failed, defaulting to ADD error=%s", e)
        return [MemoryOperation(action=OperationType.ADD, fact=fact) for fact in new_facts]

    operations: list[MemoryOperation] = []
    covered_indices: set[int] = set()
    for op in raw_ops:
        try:
            idx = int(op["fact_index"])
            if idx >= len(new_facts):
                continue
            operations.append(MemoryOperation(
                action=OperationType(op["action"]), fact=new_facts[idx],
                target_id=op.get("target_memory_id"), merged_content=op.get("merged_content"),
            ))
            covered_indices.add(idx)
        except (KeyError, ValueError):
            continue

    for i, fact in enumerate(new_facts):
        if i not in covered_indices:
            operations.append(MemoryOperation(action=OperationType.ADD, fact=fact))
    return operations


_CONTEXT_HEADER = "## Relevant memories about this user\n"


def synthesize(scored_memories: list[ScoredMemory], max_memories: int = 10, min_score: float = 0.4) -> str:
    """Format scored memories into an LLM-injectable context block."""
    filtered = [sm for sm in scored_memories if sm.score >= min_score]
    if not filtered:
        return ""
    ordered = sorted(
        filtered,
        key=lambda sm: (round(sm.score, 1), sm.memory.metadata.updated_at or datetime.min),
        reverse=True,
    )
    lines = [_CONTEXT_HEADER]
    for i, sm in enumerate(ordered[:max_memories], 1):
        lines.append(f"{i}. {sm.memory.content}")
    return "\n".join(lines)
