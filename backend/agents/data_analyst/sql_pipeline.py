"""
NL-to-SQL pipeline inspired by Uber QueryGPT and DIN-SQL.

Stages:
  1. Schema context  — rich column info (types, nulls, sample values)
  2. Few-shot RAG    — TF-IDF retrieval of similar past queries from SQLite memory
  3. CoT generation  — chain-of-thought decomposition before writing SQL
  4. Execute + retry — run the SQL; on error, feed message back to LLM (up to max_retries)
  5. Memory write    — persist successful (question → SQL) pair for future few-shots
"""

import json
import logging
import sqlite3
import textwrap
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ─────────────────────── query memory DDL ─────────────────────────────

_MEMORY_DDL = """
CREATE TABLE IF NOT EXISTS query_examples (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset       TEXT    NOT NULL,
    question      TEXT    NOT NULL,
    sql           TEXT    NOT NULL,
    result_preview TEXT,
    created_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_qex_dataset ON query_examples(dataset);
"""


def _memory_conn(memory_db: str) -> sqlite3.Connection:
    conn = sqlite3.connect(memory_db, check_same_thread=False)
    conn.executescript(_MEMORY_DDL)
    conn.commit()
    return conn


# ─────────────────────── schema context ───────────────────────────────


def build_schema_context(df: pd.DataFrame, dataset_label: str, sample_n: int = 3) -> str:
    """
    Build a rich schema string for the LLM prompt.

    Example output:
        TABLE: df  (source: titanic)
        ROWS: 891   COLUMNS: 12

        COLUMN DETAILS:
          • "pclass"      dtype=object   nulls=0.0%   samples=[1, 2, 3]
          • "survived"    dtype=int64    nulls=0.0%   samples=[0, 1, 0]
          ...
    """
    lines = [
        f"TABLE: df  (source: {dataset_label})",
        f"ROWS: {len(df):,}   COLUMNS: {len(df.columns)}",
        "",
        "COLUMN DETAILS:",
    ]
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_pct = round(df[col].isna().mean() * 100, 1)
        try:
            raw_samples = df[col].dropna().head(sample_n).tolist()
            samples_str = ", ".join(repr(s) for s in raw_samples)
        except Exception:
            samples_str = "N/A"
        col_quoted = f'"{col}"'
        lines.append(
            f"  • {col_quoted:<38} dtype={dtype:<12} nulls={null_pct}%   "
            f"samples=[{samples_str}]"
        )

    # Add a note about SQLite quoting requirement
    lines += [
        "",
        "NOTE: Column names with spaces or special characters MUST be double-quoted in SQL.",
        '      e.g.  SELECT "petal length (cm)" FROM df',
    ]
    return "\n".join(lines)


# ─────────────────────── few-shot memory ──────────────────────────────


def save_query(
    question: str,
    sql: str,
    dataset_label: str,
    result_preview: str = "",
    memory_db: str = "agent_memory.db",
) -> None:
    """Persist a verified (question → SQL) pair so future queries can use it as a few-shot example."""
    try:
        conn = _memory_conn(memory_db)
        conn.execute(
            "INSERT INTO query_examples (dataset, question, sql, result_preview, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                dataset_label,
                question,
                sql,
                result_preview[:600],
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        conn.close()
        logger.info("Saved successful query to memory: %s…", question[:60])
    except Exception:
        logger.exception("Failed to save query to memory")


def retrieve_few_shots(
    question: str,
    dataset_label: str,
    memory_db: str = "agent_memory.db",
    top_k: int = 4,
    min_score: float = 0.05,
) -> list[dict[str, str]]:
    """
    Retrieve the top-k most semantically similar past (question, SQL) pairs
    using TF-IDF cosine similarity. Falls back gracefully if the table is empty.
    """
    try:
        conn = _memory_conn(memory_db)
        rows = conn.execute(
            "SELECT question, sql FROM query_examples "
            "WHERE dataset = ? ORDER BY id DESC LIMIT 100",
            (dataset_label,),
        ).fetchall()
        conn.close()

        if not rows:
            return []

        past_questions = [r[0] for r in rows]
        past_sqls = [r[1] for r in rows]

        # TF-IDF over past questions + current question (appended last)
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        corpus = past_questions + [question]
        tfidf_matrix = vectorizer.fit_transform(corpus)
        scores = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]

        top_indices = scores.argsort()[::-1][:top_k]
        return [
            {
                "question": past_questions[i],
                "sql": past_sqls[i],
                "score": round(float(scores[i]), 3),
            }
            for i in top_indices
            if scores[i] >= min_score
        ]
    except Exception:
        logger.exception("Few-shot retrieval failed")
        return []


# ─────────────────────── LLM call wrapper ─────────────────────────────


def _llm(model: Any, system_prompt: str, user_prompt: str) -> str:
    """Invoke a LangChain chat model with a system + user message pair."""
    from langchain_core.messages import HumanMessage, SystemMessage

    response = model.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    return response.content.strip()


# ─────────────────────── CoT SQL generation ───────────────────────────

_GENERATION_SYSTEM = textwrap.dedent("""
    You are an expert SQL analyst converting natural language to SQLite SQL.

    STRICT RULES:
    - The table is ALWAYS named "df". Never use any other table name.
    - Only reference columns that appear in the SCHEMA block.
    - Double-quote ALL column names that contain spaces, parentheses, or special chars.
    - Use SQLite-compatible syntax only (no ILIKE, no ARRAY, use LIKE for patterns).
    - Do NOT end the SQL with a semicolon.
    - Return exactly ONE SQL statement (SELECT only — no INSERT/UPDATE/DELETE).

    YOUR RESPONSE FORMAT (follow exactly, including the labels):

    THINKING:
    <Step-by-step reasoning: identify the key columns, required filters, aggregations,
     groupings, orderings, and any edge cases like NULL handling.>

    QUERY_PLAN:
    <Concise bullet list of what the final SQL must achieve.>

    SQL:
    <The single final SQL query — plain text, no markdown fences.>
""").strip()


def _build_generation_prompt(
    question: str,
    schema_ctx: str,
    few_shots: list[dict],
) -> str:
    parts: list[str] = [f"SCHEMA:\n{schema_ctx}", ""]

    if few_shots:
        parts.append("FEW-SHOT REFERENCE QUERIES (similar past questions — use for style guidance):")
        for i, ex in enumerate(few_shots, 1):
            parts.append(f"\n  [{i}] Question : {ex['question']}")
            parts.append(f"       SQL      : {ex['sql']}")
        parts.append("")

    parts.append(f"QUESTION: {question}")
    return "\n".join(parts)


def _parse_cot_response(response: str) -> tuple[str, str, str]:
    """Extract (thinking, plan, sql) from the CoT response."""
    thinking = plan = sql = ""

    if "THINKING:" in response:
        after_thinking = response.split("THINKING:", 1)[1]
        if "QUERY_PLAN:" in after_thinking:
            thinking = after_thinking.split("QUERY_PLAN:", 1)[0].strip()
            after_plan = after_thinking.split("QUERY_PLAN:", 1)[1]
            if "SQL:" in after_plan:
                plan = after_plan.split("SQL:", 1)[0].strip()
                sql = after_plan.split("SQL:", 1)[1].strip()
            else:
                plan = after_plan.strip()
        elif "SQL:" in after_thinking:
            thinking = after_thinking.split("SQL:", 1)[0].strip()
            sql = after_thinking.split("SQL:", 1)[1].strip()
        else:
            thinking = after_thinking.strip()
    elif "SQL:" in response:
        sql = response.split("SQL:", 1)[1].strip()
    else:
        sql = response.strip()

    # Strip any accidental markdown code fences from the SQL
    sql = _clean_sql(sql)
    return thinking, plan, sql


def _clean_sql(raw: str) -> str:
    """Remove markdown fences and leading sql keyword if present."""
    raw = raw.strip()
    # Remove triple-backtick fences
    if raw.startswith("```"):
        lines = raw.splitlines()
        # Drop opening fence (possibly ```sql) and closing fence
        inner = [l for l in lines[1:] if not l.strip().startswith("```")]
        raw = "\n".join(inner).strip()
    # Remove a lone "sql" keyword at the very start
    if raw.lower().startswith("sql\n"):
        raw = raw[4:].strip()
    # Strip trailing semicolons
    raw = raw.rstrip(";").strip()
    return raw


# ─────────────────────── self-correction loop ─────────────────────────

_CORRECTION_SYSTEM = textwrap.dedent("""
    You are an expert SQLite debugger.
    A SQL query was executed and returned an error. Your job is to fix it.

    RULES:
    - Only reference columns from the provided schema.
    - Table name is always "df".
    - Double-quote column names with spaces or special characters.
    - Return ONLY the corrected SQL — no explanation, no markdown, no semicolons.
""").strip()


def _build_correction_prompt(
    question: str,
    bad_sql: str,
    error_msg: str,
    schema_ctx: str,
    attempt: int,
) -> str:
    return (
        f"SCHEMA:\n{schema_ctx}\n\n"
        f"ORIGINAL QUESTION: {question}\n\n"
        f"FAILED SQL (attempt {attempt}):\n{bad_sql}\n\n"
        f"ERROR MESSAGE:\n{error_msg}\n\n"
        "Write the corrected SQL query:"
    )


# ─────────────────────── SQL executor ─────────────────────────────────


def _run_sql(sql: str, df: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    """Execute sql against an in-memory SQLite copy of df. Returns (df, None) or (None, error_str)."""
    conn = sqlite3.connect(":memory:")
    try:
        df.to_sql("df", conn, index=False, if_exists="replace")
        result = pd.read_sql_query(sql, conn)
        return result, None
    except Exception as exc:
        return None, str(exc)
    finally:
        conn.close()


# ─────────────────────── main pipeline ────────────────────────────────


def nl_to_sql(
    question: str,
    df: pd.DataFrame,
    dataset_label: str,
    model: Any,
    memory_db: str = "agent_memory.db",
    max_retries: int = 3,
    max_result_rows: int = 200,
) -> dict[str, Any]:
    """
    Full NL→SQL pipeline.

    Returns a dict:
      status        : "success" | "failed"
      question      : original NL question
      sql           : final SQL (corrected if needed)
      rows_returned : int (only on success)
      columns       : list[str]
      data          : list[dict]  (up to max_result_rows)
      thinking      : CoT reasoning text
      query_plan    : bullet-list plan
      few_shots_used: int
      attempts      : int (1 = first try worked)
      error         : str (only on failure)
    """
    # ── 1. Schema context
    schema_ctx = build_schema_context(df, dataset_label)

    # ── 2. Few-shot retrieval
    few_shots = retrieve_few_shots(question, dataset_label, memory_db=memory_db)
    logger.info(
        "NL-to-SQL start: dataset=%s question=%.60s few_shots=%d",
        dataset_label, question, len(few_shots),
    )

    # ── 3. CoT generation (attempt 1)
    gen_prompt = _build_generation_prompt(question, schema_ctx, few_shots)
    cot_response = _llm(model, _GENERATION_SYSTEM, gen_prompt)
    thinking, query_plan, sql = _parse_cot_response(cot_response)
    logger.info("Generated SQL attempt 1: %.200s", sql)

    # ── 4. Execute + self-correction loop
    result_df: pd.DataFrame | None = None
    last_error: str | None = None

    for attempt in range(1, max_retries + 1):
        result_df, error = _run_sql(sql, df)

        if result_df is not None:
            last_error = None
            logger.info("SQL succeeded on attempt %d", attempt)
            break

        last_error = error
        logger.warning("SQL attempt %d failed: %s", attempt, error)

        if attempt < max_retries:
            correction_prompt = _build_correction_prompt(
                question, sql, error, schema_ctx, attempt
            )
            corrected = _llm(model, _CORRECTION_SYSTEM, correction_prompt)
            sql = _clean_sql(corrected)
            logger.info("Corrected SQL attempt %d: %.200s", attempt + 1, sql)

    # ── 5. Persist successful query for future few-shots
    if result_df is not None:
        preview = result_df.head(3).to_json(orient="records", default_handler=str)
        save_query(question, sql, dataset_label, preview, memory_db=memory_db)

        result_df = result_df.head(max_result_rows)
        safe = result_df.astype(object).where(pd.notna(result_df), None)
        return {
            "status": "success",
            "question": question,
            "sql": sql,
            "rows_returned": len(result_df),
            "columns": list(result_df.columns),
            "data": safe.to_dict(orient="records"),
            "thinking": thinking,
            "query_plan": query_plan,
            "few_shots_used": len(few_shots),
            "attempts": attempt,
        }

    return {
        "status": "failed",
        "question": question,
        "sql": sql,
        "error": last_error,
        "thinking": thinking,
        "query_plan": query_plan,
        "few_shots_used": len(few_shots),
        "attempts": attempt,
    }


# ─────────────────── convenience: list memory stats ───────────────────


def memory_stats(memory_db: str = "agent_memory.db") -> dict[str, Any]:
    """Return a summary of the query memory store."""
    try:
        conn = _memory_conn(memory_db)
        total = conn.execute("SELECT COUNT(*) FROM query_examples").fetchone()[0]
        by_dataset = conn.execute(
            "SELECT dataset, COUNT(*) FROM query_examples GROUP BY dataset ORDER BY 2 DESC"
        ).fetchall()
        recent = conn.execute(
            "SELECT question, sql, created_at FROM query_examples ORDER BY id DESC LIMIT 5"
        ).fetchall()
        conn.close()
        return {
            "total_examples": total,
            "by_dataset": {r[0]: r[1] for r in by_dataset},
            "recent": [{"question": r[0], "sql": r[1], "at": r[2]} for r in recent],
        }
    except Exception as exc:
        return {"error": str(exc)}
