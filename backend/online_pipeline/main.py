"""FastAPI backend: list KBs, stream a KB-grounded chat response."""

import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from backend.kb import get_kb, list_knowledge_bases  # noqa: E402
from backend.logging_config import configure_logging  # noqa: E402
from backend.online_pipeline.agent import make_kb_agent  # noqa: E402

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Project Shonku")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    kb_slug: str
    message: str
    history: list[dict] = []


@app.get("/kbs")
def list_kbs():
    return [{"slug": kb.slug, "name": kb.name, "description": kb.description} for kb in list_knowledge_bases()]


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        get_kb(req.kb_slug)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    logger.info("chat request kb_slug=%s message_len=%d", req.kb_slug, len(req.message))

    agent = make_kb_agent(req.kb_slug)
    messages = [*req.history, {"role": "user", "content": req.message}]

    def event_stream():
        start = time.monotonic()
        chunk_count = 0
        char_count = 0
        try:
            for chunk, _metadata in agent.stream({"messages": messages}, stream_mode="messages"):
                if not isinstance(chunk, AIMessageChunk):
                    continue
                text = chunk.content
                if isinstance(text, list):
                    text = "".join(part.get("text", "") for part in text if isinstance(part, dict) and part.get("type") == "text")
                if text:
                    chunk_count += 1
                    char_count += len(text)
                    yield f"data: {json.dumps({'delta': text})}\n\n"
            logger.info(
                "chat response kb_slug=%s chunks=%d chars=%d duration=%.2fs",
                req.kb_slug, chunk_count, char_count, time.monotonic() - start,
            )
        except Exception:
            logger.exception("chat stream failed kb_slug=%s", req.kb_slug)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
