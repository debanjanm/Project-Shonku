"""FastAPI backend: agents, KBs/datasets, conversations (SQLite-persisted),
streamed chat, chart serving.
"""

import json
import logging
import re
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import AIMessageChunk
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from backend import db  # noqa: E402
from backend.agents.data_analyst.agent import make_data_analyst_agent  # noqa: E402
from backend.agents.data_analyst.tools import BUILT_IN_DATASETS, CHARTS_DIR  # noqa: E402
from backend.agents.docqa.agent import make_kb_agent  # noqa: E402
from backend.agents.mystery_generator.agent import make_mystery_agent  # noqa: E402
from backend.agents.recommendation.agent import make_recommendation_agent  # noqa: E402
from backend.agents.story_developer.agent import make_story_agent  # noqa: E402
from backend.kb import get_kb, list_knowledge_bases  # noqa: E402
from backend.logging_config import configure_logging  # noqa: E402
from backend.memory import get_memory_manager  # noqa: E402
from backend.memory.models import MemoryScope  # noqa: E402

configure_logging()
logger = logging.getLogger(__name__)
db.init_db()

DEFAULT_USER_ID = "shonku_user"  # single-user app, no auth — one fixed memory owner

try:
    memory_manager = get_memory_manager()
except Exception:
    # A paused Neo4j AuraDB free instance (or bad credentials) shouldn't take
    # the whole backend down — chat just runs without memory recall/storage.
    logger.exception("memory manager unavailable at startup, continuing without it")
    memory_manager = None

ROOT = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PRODUCT_IMAGES_DIR = ROOT / "data" / "products" / "images"

RECOMMENDATION_SOURCE_TYPE = "catalog"
RECOMMENDATION_SOURCE_REF = "fashion-500"
FREEFORM_SOURCE_TYPE = "brief"
FREEFORM_SOURCE_REF = "freeform"

AGENTS = [
    {
        "id": "docqa",
        "name": "Document Q&A",
        "description": "Chat with a curated knowledge base.",
    },
    {
        "id": "data_analyst",
        "name": "Data Analyst",
        "description": "Analyze datasets: statistics, SQL, charts, and reports.",
    },
    {
        "id": "recommendation",
        "name": "Product Recommendations",
        "description": "Search a product catalog by natural language, powered by CLIP + LLM reranking.",
    },
    {
        "id": "story_developer",
        "name": "Story Developer",
        "description": "Develop your story idea into a title, synopsis, characters, and a key scene.",
    },
    {
        "id": "mystery_generator",
        "name": "Mystery Generator",
        "description": "Turns your idea into a crime mystery you investigate and solve.",
    },
]

app = FastAPI(title="Project Shonku")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    conversation_id: int
    message: str


class CreateConversationRequest(BaseModel):
    agent_type: str
    source_type: str
    source_ref: str


def _validate_source(agent_type: str, source_type: str, source_ref: str) -> None:
    if agent_type == "docqa":
        if source_type != "kb":
            raise HTTPException(status_code=400, detail="docqa requires source_type='kb'")
        try:
            get_kb(source_ref)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    elif agent_type == "data_analyst":
        if source_type == "dataset":
            if source_ref not in BUILT_IN_DATASETS:
                raise HTTPException(status_code=404, detail=f"Unknown dataset: {source_ref}")
        elif source_type == "csv":
            if not Path(source_ref).exists():
                raise HTTPException(status_code=404, detail=f"Uploaded file not found: {source_ref}")
        else:
            raise HTTPException(status_code=400, detail="data_analyst requires source_type='dataset' or 'csv'")
    elif agent_type == "recommendation":
        if source_type != RECOMMENDATION_SOURCE_TYPE or source_ref != RECOMMENDATION_SOURCE_REF:
            raise HTTPException(
                status_code=400,
                detail=f"recommendation requires source_type='{RECOMMENDATION_SOURCE_TYPE}', source_ref='{RECOMMENDATION_SOURCE_REF}'",
            )
    elif agent_type in ("story_developer", "mystery_generator"):
        if source_type != FREEFORM_SOURCE_TYPE or source_ref != FREEFORM_SOURCE_REF:
            raise HTTPException(
                status_code=400,
                detail=f"{agent_type} requires source_type='{FREEFORM_SOURCE_TYPE}', source_ref='{FREEFORM_SOURCE_REF}'",
            )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown agent_type: {agent_type}")


def _extract_chart_filenames(text: str) -> list[str]:
    found = []
    for match in re.finditer(r"chart_[\w\-]+\.png", text):
        name = match.group(0)
        if (CHARTS_DIR / name).exists() and name not in found:
            found.append(name)
    return found


def _extract_product_images(text: str) -> list[str]:
    found = []
    for match in re.finditer(r"\[Image:\s*([^\]]+)\]", text):
        name = match.group(1).strip()
        if (PRODUCT_IMAGES_DIR / name).exists() and name not in found:
            found.append(name)
    return found


@app.get("/agents")
def list_agents():
    return AGENTS


@app.get("/kbs")
def list_kbs():
    return [{"slug": kb.slug, "name": kb.name, "description": kb.description} for kb in list_knowledge_bases()]


@app.get("/datasets")
def list_datasets():
    return [{"name": name, "description": desc} for name, desc in BUILT_IN_DATASETS.items()]


@app.post("/uploads")
async def upload_csv(file: UploadFile = File(...)):
    safe_name = Path(file.filename).name
    dest = UPLOAD_DIR / safe_name
    dest.write_bytes(await file.read())
    logger.info("uploaded csv saved to %s", dest)
    return {"path": str(dest)}


@app.post("/conversations")
def create_conversation(req: CreateConversationRequest):
    _validate_source(req.agent_type, req.source_type, req.source_ref)
    conversation_id = db.create_conversation(req.agent_type, req.source_type, req.source_ref)
    return db.get_conversation(conversation_id)


@app.get("/conversations")
def list_conversations(agent_type: str | None = None, source_type: str | None = None, source_ref: str | None = None):
    return db.list_conversations(agent_type, source_type, source_ref)


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: int):
    conversation = db.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail=f"Unknown conversation: {conversation_id}")
    return conversation


@app.get("/conversations/{conversation_id}/messages")
def get_conversation_messages(conversation_id: int):
    if db.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown conversation: {conversation_id}")
    return db.get_messages(conversation_id)


@app.get("/charts/{filename}")
def get_chart(filename: str):
    safe_name = Path(filename).name
    path = CHARTS_DIR / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown chart: {filename}")
    return FileResponse(path, media_type="image/png")


@app.get("/products/images/{filename}")
def get_product_image(filename: str):
    safe_name = Path(filename).name
    path = PRODUCT_IMAGES_DIR / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown product image: {filename}")
    return FileResponse(path)


def _store_memory_background(user_message: str, assistant_message: str, scope: MemoryScope) -> None:
    """Runs the extract->update->store memory pipeline off the request thread
    so it never adds latency to the streamed chat response."""
    try:
        result = memory_manager.add(user_message, assistant_message, scope)
        logger.info("memory stored user_id=%s %s", scope.user_id, result.summary())
    except Exception:
        logger.exception("memory storage failed user_id=%s", scope.user_id)


@app.post("/chat")
def chat(req: ChatRequest):
    conversation = db.get_conversation(req.conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail=f"Unknown conversation: {req.conversation_id}")
    agent_type = conversation["agent_type"]
    source_type = conversation["source_type"]
    source_ref = conversation["source_ref"]

    logger.info(
        "chat request conversation_id=%d agent_type=%s source_type=%s source_ref=%s message_len=%d",
        req.conversation_id, agent_type, source_type, source_ref, len(req.message),
    )

    db.append_message(req.conversation_id, "user", req.message)
    db.set_title_if_unset(req.conversation_id, req.message)
    messages = [{"role": m["role"], "content": m["content"]} for m in db.get_messages(req.conversation_id)]

    memory_scope = MemoryScope(user_id=DEFAULT_USER_ID, agent_id=agent_type, run_id=str(req.conversation_id))
    memory_context = memory_manager.get_context(req.message, memory_scope) if memory_manager else ""

    if agent_type == "docqa":
        agent = make_kb_agent(source_ref, memory_context=memory_context)
    elif agent_type == "recommendation":
        agent = make_recommendation_agent(memory_context=memory_context)
    elif agent_type == "story_developer":
        agent = make_story_agent(memory_context=memory_context)
    elif agent_type == "mystery_generator":
        agent = make_mystery_agent(req.conversation_id, memory_context=memory_context)
    else:
        agent = make_data_analyst_agent(source_type, source_ref, memory_context=memory_context)

    def event_stream():
        start = time.monotonic()
        chunk_count = 0
        parts: list[str] = []
        try:
            for chunk, _metadata in agent.stream({"messages": messages}, stream_mode="messages"):
                if not isinstance(chunk, AIMessageChunk):
                    continue
                text = chunk.content
                if isinstance(text, list):
                    text = "".join(part.get("text", "") for part in text if isinstance(part, dict) and part.get("type") == "text")
                if text:
                    chunk_count += 1
                    parts.append(text)
                    yield f"data: {json.dumps({'delta': text})}\n\n"
            full_text = "".join(parts)
            if full_text:
                db.append_message(req.conversation_id, "assistant", full_text)
                if memory_manager:
                    threading.Thread(
                        target=_store_memory_background,
                        args=(req.message, full_text, memory_scope),
                        daemon=True,
                    ).start()
            if agent_type == "data_analyst" and full_text:
                chart_files = _extract_chart_filenames(full_text)
                if chart_files:
                    yield f"data: {json.dumps({'chart_paths': chart_files})}\n\n"
            if agent_type == "recommendation" and full_text:
                product_images = _extract_product_images(full_text)
                if product_images:
                    yield f"data: {json.dumps({'product_images': product_images})}\n\n"
            logger.info(
                "chat response conversation_id=%d agent_type=%s chunks=%d chars=%d duration=%.2fs",
                req.conversation_id, agent_type, chunk_count, len(full_text), time.monotonic() - start,
            )
        except Exception:
            logger.exception("chat stream failed conversation_id=%d agent_type=%s", req.conversation_id, agent_type)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
