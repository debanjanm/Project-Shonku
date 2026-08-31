"""Deep agent that develops a user's initial story draft into a structured
concept + key scene. Adapted from AIF-CineBot's Google ADK film-pitch
pipeline, trimmed to writer + scene-director + editor (storyline/title/
scenes only — no casting/cinematography/marketing) and reimplemented in
deepagents to match this codebase's stack. Stateless, like docqa
— the "source" is just the user's own message, nothing to persist.
"""

import logging
import os

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

from backend.config import get_router_config

logger = logging.getLogger(__name__)

WRITER_PROMPT = (
    "You are a world-class screenwriter. Given the user's initial idea, develop a unique and "
    "captivating film concept. "
    "1. Title & Genre: a catchy title and a specific genre mix (e.g. 'Neo-Noir Cyberpunk'). "
    "2. Logline: a one-sentence hook. "
    "3. Synopsis: a structured three-act synopsis (Setup, Confrontation, Resolution) with clear stakes. "
    "4. Character Profiles: 3-4 main characters with distinct personalities, motivations, and flaws. "
    "Ensure thematic depth and emotional resonance."
)

SCENE_DIRECTOR_PROMPT = (
    "You are an award-winning film director. Given the concept and characters developed so far, write "
    "one pivotal scene. "
    "1. Scene Heading: standard slugline (e.g. INT. ABANDONED WAREHOUSE - NIGHT). "
    "2. Atmosphere: lighting, sound design, mood, in rich detail. "
    "3. Action & Blocking: precise stage directions showing character movement and interaction. "
    "4. Dialogue: natural, subtext-rich dialogue that reveals character conflict. "
    "Show, don't tell."
)

EDITOR_PROMPT = (
    "You are a senior film editor and script doctor. Review the concept and scene, and any user "
    "feedback on them. "
    "1. Pacing & Flow: fix dragging sections or abrupt transitions. "
    "2. Dialogue Polish: sharpen lines to sound authentic and impactful. "
    "3. Clarity & Impact: sharpen descriptions for visual and emotional punch. "
    "4. Consistency: keep character voices and plot points consistent. "
    "When the user asks for a revision on a later turn, apply it directly and return the updated material "
    "— don't just describe the change."
)

_BASE_SYSTEM_PROMPT = """You are the Project Shonku story development assistant. The user gives you a rough
story idea (any genre) as their first message.

Workflow for a first message:
1. Delegate to the writer subagent to develop title, genre, logline, synopsis, and characters.
2. Delegate to the scene-director subagent to write one pivotal scene from that concept.
3. Delegate to the editor subagent to polish both.
4. Present the final result as one structured answer: Title & Genre, Logline, Synopsis, Characters, Scene.

For later messages, treat them as feedback/revision requests and delegate to the editor subagent to apply
the requested change, then present the updated material. Be concise and direct. Do not use file or shell
tools."""


def get_model() -> ChatOpenAI:
    model_name = get_router_config().hard_model
    logger.info("creating story-developer chat model via openrouter model=%s", model_name)
    return ChatOpenAI(
        model=model_name,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        streaming=True,
    )


def make_story_agent(memory_context: str = ""):
    logger.info("building story-developer deep agent")
    model = get_model()
    system_prompt = _BASE_SYSTEM_PROMPT + (f"\n\n{memory_context}" if memory_context else "")

    subagents = [
        {
            "name": "writer",
            "description": "Develops title, genre, logline, synopsis, and character profiles from a story idea.",
            "system_prompt": WRITER_PROMPT,
            "tools": [],
            "model": model,
        },
        {
            "name": "scene-director",
            "description": "Writes one pivotal scene (slugline, atmosphere, blocking, dialogue) from a concept.",
            "system_prompt": SCENE_DIRECTOR_PROMPT,
            "tools": [],
            "model": model,
        },
        {
            "name": "editor",
            "description": "Polishes pacing, dialogue, and consistency; applies revision feedback on later turns.",
            "system_prompt": EDITOR_PROMPT,
            "tools": [],
            "model": model,
        },
    ]

    return create_deep_agent(
        model=model,
        subagents=subagents,
        system_prompt=system_prompt,
    )
