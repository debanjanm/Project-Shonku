"""Deep agent bound to a single data source (built-in dataset or uploaded CSV).

Ported from AIB-DataAnalystAgent/app.py's build_agent(), adapted:
  - No SqliteSaver checkpointer — Shonku's backend/db.py owns conversation
    history for every agent type, fed in fresh each call (same as docqa).
  - Source binding (which exact dataset_name/csv_path to use) moves from a
    per-turn build_user_prompt wrapper into the system prompt, built once
    per conversation — matches how docqa's SYSTEM_PROMPT works.
  - Model defaults to Shonku's router hard_model instead of a separate
    OPENROUTER_MODEL env var — one model config surface for the platform.
"""

import logging
import os

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

from backend.agents.data_analyst import tools as t
from backend.config import get_router_config

logger = logging.getLogger(__name__)


def get_model() -> ChatOpenAI:
    model_name = get_router_config().hard_model
    logger.info("creating data-analyst chat model via openrouter model=%s", model_name)
    return ChatOpenAI(
        model=model_name,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        streaming=True,
    )


def _source_block(source_type: str, source_ref: str) -> str:
    if source_type == "csv":
        return (
            f"Active source type: csv\n"
            f"CSV path: {source_ref}\n"
            "Dataset name: N/A\n"
            "Tool rule: Always pass csv_path=\"" + source_ref + "\" on every tool call and "
            "every subagent delegation. Never invent a dataset_name or a different csv_path."
        )
    return (
        f"Active source type: built_in_dataset\n"
        "CSV path: N/A\n"
        f"Dataset name: {source_ref}\n"
        "Tool rule: Always pass dataset_name=\"" + source_ref + "\" on every tool call and "
        "every subagent delegation. Never invent a csv_path or a different dataset_name."
    )


def make_data_analyst_agent(source_type: str, source_ref: str, memory_context: str = ""):
    logger.info("building data-analyst deep agent source_type=%s source_ref=%s", source_type, source_ref)

    model = get_model()
    t.set_model(model)

    subagents = [
        {
            "name": "data-analyst",
            "description": (
                "Performs deep data analysis: inspects structure, computes statistics, "
                "detects outliers, analyses correlations, and runs Python code for custom analysis."
            ),
            "system_prompt": (
                "You are a rigorous data analyst. Use the available tools to inspect the "
                "dataset schema, compute summary statistics, find correlations, detect outliers, "
                "and run Python to answer specific questions. Always cite column names and "
                "numbers in your findings. Be concise but precise."
            ),
            "tools": t.ALL_TOOLS,
            "model": model,
        },
        {
            "name": "viz-analyst",
            "description": (
                "Specialises in creating charts and visual patterns from data: histograms, "
                "scatter plots, bar charts, box plots, heatmaps, line charts, pair plots, and pie charts."
            ),
            "system_prompt": (
                "You are a data visualisation expert. When asked to show patterns, distributions, "
                "trends, or relationships, choose the most appropriate chart type and call "
                "generate_chart. Always explain what the chart reveals. "
                "Suggest follow-up visualisations when relevant."
            ),
            "tools": [t.generate_chart, t.inspect_data],
            "model": model,
        },
        {
            "name": "sql-analyst",
            "description": (
                "Answers data questions using SQL. Handles natural language questions, "
                "complex aggregations, filtering, grouping, ranking, window functions, "
                "and multi-step queries. Uses an NL-to-SQL pipeline with chain-of-thought "
                "reasoning and automatic self-correction on errors."
            ),
            "system_prompt": (
                "You are an expert SQL analyst with a powerful NL-to-SQL pipeline.\n\n"
                "TOOL SELECTION RULES:\n"
                "  • Use smart_sql_query for ANY natural language question about the data.\n"
                "    It handles schema linking, chain-of-thought decomposition, and auto-corrects\n"
                "    SQL errors. It also learns from past successful queries.\n"
                "  • Use run_sql_query ONLY when you already have a precise SQL string\n"
                "    that you want to execute directly (e.g., a follow-up variation of a query).\n"
                "  • Use query_memory_stats to see how much few-shot context is available.\n\n"
                "OUTPUT FORMAT:\n"
                "  Always show:\n"
                "    1. The generated SQL (from the 'sql' field of the result)\n"
                "    2. The chain-of-thought reasoning (from 'thinking') — summarised\n"
                "    3. The result with interpretation\n"
                "    4. Any interesting follow-up queries worth running\n\n"
                "For multi-step questions, break them into sub-questions and call smart_sql_query\n"
                "for each step. Combine the results in your final answer."
            ),
            "tools": [t.smart_sql_query, t.run_sql_query, t.query_memory_stats, t.inspect_data],
            "model": model,
        },
        {
            "name": "report-writer",
            "description": "Converts technical analysis findings into a clear business-friendly executive summary.",
            "system_prompt": (
                "You are an executive report writer. Convert technical findings into a concise, "
                "well-structured summary with key insights, risks, and recommended next steps. "
                "Avoid jargon. Bullet the most important points."
            ),
            "model": model,
        },
    ]

    system_prompt = (
        "You are a multi-agent data analyst supervisor. You have access to a rich toolset:\n"
        "  • inspect_data        — schema, preview, missing values\n"
        "  • summarize_numeric   — descriptive statistics\n"
        "  • value_counts        — frequency of categorical values\n"
        "  • correlation_matrix  — Pearson correlations between numeric columns\n"
        "  • detect_outliers     — IQR-based outlier detection per column\n"
        "  • smart_sql_query     — NL-to-SQL pipeline (CoT + schema linking + auto-correct)\n"
        "  • run_sql_query       — execute a raw SQL SELECT directly\n"
        "  • query_memory_stats  — inspect the few-shot query memory store\n"
        "  • run_python_code     — run custom Python with 'df' pre-loaded\n"
        "  • generate_chart      — create and save charts (histogram, scatter, bar, box,\n"
        "                          heatmap, line, pairplot, pie)\n\n"
        "Delegation rules:\n"
        "  • Deep statistical analysis            → data-analyst subagent\n"
        "  • Any SQL or data question             → sql-analyst subagent\n"
        "    (prefer smart_sql_query over run_sql_query for natural language questions)\n"
        "  • Charts / visualisations              → viz-analyst subagent\n"
        "  • Executive summary / business report  → report-writer subagent\n\n"
        f"{_source_block(source_type, source_ref)}\n\n"
        "Always produce a clear, actionable final answer. When charts are created, "
        "mention the chart_path so the UI can display them."
        + (f"\n\n{memory_context}" if memory_context else "")
    )

    return create_deep_agent(
        model=model,
        tools=t.ALL_TOOLS,
        subagents=subagents,
        system_prompt=system_prompt,
    )
