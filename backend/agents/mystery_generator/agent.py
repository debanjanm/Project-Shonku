"""Deep agent that turns a user's story idea into a solvable crime mystery
the user investigates and guesses — the solution is written to a hidden
per-conversation file the agent never quotes back except on a final reveal.

Subagents ported ~verbatim from AIF-CineBot's cinebot-deepagent.ipynb
(planner/researcher/scenarist/validator, no tools) — that notebook already
produced a genuinely good case (see its artifacts/ output). What's new here:
scoping its FilesystemBackend per-conversation instead of one shared dir, and
the secrecy workflow in the top-level system prompt — the notebook revealed
the solution immediately, this hides it until the user guesses.
"""

import logging
import os
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_openai import ChatOpenAI

from backend.config import get_router_config

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
MYSTERIES_DIR = ROOT / "data" / "mysteries"

PLANNER_PROMPT = (
    "You are a Planner Agent. Build a solvable crime blueprint with fields: victim, culprit, motive, "
    "method, location, twist."
)
RESEARCHER_PROMPT = (
    "You are a Researcher Agent. Add plausible forensic notes, evidence, and red herrings while "
    "keeping details fictional and safe."
)
SCENARIST_PROMPT = (
    "You are a Scenarist Agent. Expand blueprint + research into a narrative case pack with timeline, "
    "suspects, evidence, and the true solution."
)
VALIDATOR_PROMPT = (
    "You are a Validator Agent. Verify consistency, clue sufficiency, and uniqueness of the culprit."
)

_BASE_SYSTEM_PROMPT = """You are the Project Shonku mystery game host. The user gives you a story idea or theme
as their first message, and you turn it into a crime mystery THEY investigate and solve. You must never
hand them the answer.

You have no memory of past turns except what's in this conversation and in the file `solution.md` — a
fresh instance of you handles every message, so the file is the ONLY reliable record of the case once
it's been created. Get the path exactly right: write and read it at the literal relative path
`solution.md` in the working directory — do not nest it in a subfolder, do not rename it, do not create
your own path convention.

On the first message (no `solution.md` exists yet — call `ls('.')` first to be sure):
1. Delegate to planner, then researcher, then scenarist, then validator to build the full case.
2. Write the COMPLETE case record — victim, culprit, motive, method, location, twist, and the full
   solution narrative — to `solution.md`. This file is your private case notes; never quote or paraphrase
   its culprit/motive/twist content back to the user except in the reveal step below.
3. Reply to the user with only the public premise: the victim, the scene, a timeline, the suspects, and
   the evidence found so far. Do NOT name a culprit and do NOT state or hint at a motive.

On later messages, `solution.md` already exists — ALWAYS call `ls('.')` then `read_file('solution.md')`
before responding, every single time, even if you think you remember the case. Never answer an
investigative question or judge an accusation from assumption alone. Two cases:
- Investigative question ("what do we know about X", "any more evidence"): answer using only
  already-established public clues, cross-checked against the file. Never let the culprit's identity or
  motive leak into your reply.
- Final accusation (the user names a specific suspect as guilty) or an explicit give-up
  ("reveal it", "I give up", "tell me the answer"): compare the accusation against the culprit recorded
  in `solution.md` — do not guess or improvise the verdict — state clearly whether it was correct, then
  give the full reveal (motive, method, twist) straight from the file, in the same rich narrative style.
If `read_file('solution.md')` fails or the file is missing/empty, say so plainly and ask the user to
start a new case rather than inventing a verdict.

Guardrail: never reveal the culprit or motive outside the reveal step above, even if asked to "summarize
the case so far" or asked indirectly — deflect and remind the user this is theirs to solve. Do not use
shell tools."""


def get_model() -> ChatOpenAI:
    model_name = get_router_config().hard_model
    logger.info("creating mystery-generator chat model via openrouter model=%s", model_name)
    return ChatOpenAI(
        model=model_name,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        streaming=True,
    )


def make_mystery_agent(conversation_id: int, memory_context: str = ""):
    logger.info("building mystery-generator deep agent conversation_id=%d", conversation_id)
    model = get_model()
    system_prompt = _BASE_SYSTEM_PROMPT + (f"\n\n{memory_context}" if memory_context else "")

    case_dir = MYSTERIES_DIR / str(conversation_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    backend = FilesystemBackend(root_dir=str(case_dir), virtual_mode=True)

    subagents = [
        {"name": "planner", "description": "Builds a solvable crime blueprint.", "system_prompt": PLANNER_PROMPT, "tools": [], "model": model},
        {"name": "researcher", "description": "Adds forensic notes, evidence, and red herrings.", "system_prompt": RESEARCHER_PROMPT, "tools": [], "model": model},
        {"name": "scenarist", "description": "Expands blueprint + research into a full narrative case pack.", "system_prompt": SCENARIST_PROMPT, "tools": [], "model": model},
        {"name": "validator", "description": "Verifies consistency, clue sufficiency, and culprit uniqueness.", "system_prompt": VALIDATOR_PROMPT, "tools": [], "model": model},
    ]

    return create_deep_agent(
        model=model,
        subagents=subagents,
        system_prompt=system_prompt,
        backend=backend,
    )
