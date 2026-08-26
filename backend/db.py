"""SQLite conversation persistence. Open-a-connection-per-call: simplest
thread-safe pattern for FastAPI's threadpool-executed sync handlers at this
scale, no pooling needed.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "shonku.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_type TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)
    logger.info("db initialized at %s", DB_PATH)


def create_conversation(agent_type: str, source_type: str, source_ref: str) -> int:
    now = _now()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (agent_type, source_type, source_ref, title, created_at, updated_at) "
            "VALUES (?, ?, ?, '', ?, ?)",
            (agent_type, source_type, source_ref, now, now),
        )
        conversation_id = cur.lastrowid
    logger.info(
        "created conversation id=%d agent_type=%s source_type=%s source_ref=%s",
        conversation_id, agent_type, source_type, source_ref,
    )
    return conversation_id


def get_conversation(conversation_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
    return dict(row) if row else None


def list_conversations(
    agent_type: str | None = None,
    source_type: str | None = None,
    source_ref: str | None = None,
) -> list[dict]:
    query = "SELECT * FROM conversations"
    clauses = []
    params: list = []
    for field, value in (("agent_type", agent_type), ("source_type", source_type), ("source_ref", source_ref)):
        if value is not None:
            clauses.append(f"{field} = ?")
            params.append(value)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY updated_at DESC"
    with _connect() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def get_messages(conversation_id: int) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def append_message(conversation_id: int, role: str, content: str) -> None:
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, now),
        )
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))


def set_title_if_unset(conversation_id: int, text: str) -> None:
    title = text.strip()[:60]
    with _connect() as conn:
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ? AND title = ''",
            (title, conversation_id),
        )
