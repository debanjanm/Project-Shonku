"""Data-analyst tools, ported from AIB-DataAnalystAgent/app.py.

Ported near-verbatim; adapted paths to live under data/ (CHARTS_DIR, default
memory_db) and dropped the standalone script's own logging.basicConfig call —
backend.logging_config already configures logging centrally.
"""

import io
import json
import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
)

from backend.agents.data_analyst.sql_pipeline import memory_stats, nl_to_sql

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHARTS_DIR = ROOT / "data" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MEMORY_DB = str(ROOT / "data" / "shonku.db")

BUILT_IN_DATASETS = {
    "titanic":       "🚢 Titanic — passenger survival (891 rows, 12 cols)",
    "iris":          "🌸 Iris — flower species classification (150 rows, 5 cols)",
    "wine":          "🍷 Wine — wine quality recognition (178 rows, 14 cols)",
    "breast_cancer": "🔬 Breast Cancer — tumour classification (569 rows, 31 cols)",
    "diabetes":      "💉 Diabetes — disease progression (442 rows, 11 cols)",
    "tips":          "🍽️ Tips — restaurant tipping data (244 rows, 7 cols)",
    "penguins":      "🐧 Penguins — Palmer penguins species (344 rows, 7 cols)",
    "diamonds":      "💎 Diamonds — price & quality (53940 rows, 10 cols)",
    "flights":       "✈️ Flights — monthly airline passengers (144 rows, 3 cols)",
    "mpg":           "🚗 MPG — car fuel efficiency (398 rows, 9 cols)",
    "planets":       "🪐 Planets — discovered exoplanets (1035 rows, 6 cols)",
}

# Module-level model singleton — set by make_data_analyst_agent() so tool
# functions can call the LLM directly (smart_sql_query).
_model = None
_memory_db: str = DEFAULT_MEMORY_DB


def set_model(model, memory_db: str = DEFAULT_MEMORY_DB) -> None:
    global _model, _memory_db
    _model = model
    _memory_db = memory_db


# ──────────────────────────── data loaders ────────────────────────────


def _load_csv(csv_path: str) -> pd.DataFrame:
    logger.info("Loading CSV from %s", csv_path)
    path = Path(csv_path)
    if not path.exists():
        logger.error("CSV file not found: %s", csv_path)
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(path)


def _load_named_dataset(dataset_name: str) -> pd.DataFrame:
    name = dataset_name.lower()
    logger.info("Loading built-in dataset: %s", name)

    if name == "iris":
        bunch = load_iris(as_frame=True)
        return bunch.frame.copy()
    elif name == "wine":
        bunch = load_wine(as_frame=True)
        return bunch.frame.copy()
    elif name == "breast_cancer":
        bunch = load_breast_cancer(as_frame=True)
        return bunch.frame.copy()
    elif name == "diabetes":
        bunch = load_diabetes(as_frame=True)
        return bunch.frame.copy()
    elif name == "titanic":
        bunch = fetch_openml(name="titanic", version=1, as_frame=True)
        return bunch.frame.copy()
    elif name in ("tips", "penguins", "diamonds", "flights", "mpg", "planets"):
        try:
            import seaborn as sns
            return sns.load_dataset(name)
        except Exception as e:
            raise ValueError(f"Could not load seaborn dataset '{name}': {e}")
    else:
        raise ValueError(
            "Unsupported dataset. Choose one of: " + ", ".join(BUILT_IN_DATASETS)
        )


def _load_table(csv_path: str | None = None, dataset_name: str | None = None) -> pd.DataFrame:
    logger.debug("Resolving table source csv=%s dataset=%s", csv_path, dataset_name)
    if csv_path:
        return _load_csv(csv_path)
    if dataset_name:
        return _load_named_dataset(dataset_name)
    raise ValueError("Provide either a CSV path or a dataset name.")


def _json_safe_records(df: pd.DataFrame) -> list[dict]:
    safe_df = df.astype(object).where(pd.notna(df), None)
    return safe_df.to_dict(orient="records")


# ─────────────────────────────── tools ───────────────────────────────


def inspect_data(
    csv_path: str | None = None,
    dataset_name: str | None = None,
    rows: int = 5,
) -> str:
    """Return a quick schema and preview for a CSV file or sklearn/seaborn dataset."""
    logger.info("Tool inspect_data: csv=%s dataset=%s rows=%s", csv_path, dataset_name, rows)
    df = _load_table(csv_path=csv_path, dataset_name=dataset_name)
    payload = {
        "source": csv_path or dataset_name,
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns],
        "preview": _json_safe_records(df.head(rows)),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
    }
    return json.dumps(payload, indent=2, default=str)


def summarize_numeric(
    csv_path: str | None = None,
    dataset_name: str | None = None,
) -> str:
    """Return summary statistics (count, mean, std, min, max, percentiles) for all numeric columns."""
    logger.info("Tool summarize_numeric: csv=%s dataset=%s", csv_path, dataset_name)
    df = _load_table(csv_path=csv_path, dataset_name=dataset_name)
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return "No numeric columns found."
    return numeric_df.describe().to_json(indent=2, default_handler=str)


def value_counts(
    column: str,
    csv_path: str | None = None,
    dataset_name: str | None = None,
    limit: int = 10,
) -> str:
    """Return top N value counts for a specific column (great for categorical columns)."""
    logger.info("Tool value_counts: column=%s csv=%s dataset=%s limit=%s", column, csv_path, dataset_name, limit)
    df = _load_table(csv_path=csv_path, dataset_name=dataset_name)
    if column not in df.columns:
        return f"Column '{column}' not found. Available columns: {list(df.columns)}"
    counts = df[column].astype(str).value_counts(dropna=False).head(limit)
    return counts.to_json(indent=2)


def correlation_matrix(
    csv_path: str | None = None,
    dataset_name: str | None = None,
) -> str:
    """Return the Pearson correlation matrix for all numeric columns. Useful for identifying relationships."""
    logger.info("Tool correlation_matrix: csv=%s dataset=%s", csv_path, dataset_name)
    df = _load_table(csv_path=csv_path, dataset_name=dataset_name)
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return "No numeric columns found."
    corr = numeric_df.corr().round(3)
    result = {
        "columns": list(corr.columns),
        "matrix": corr.to_dict(),
        "top_correlations": [],
    }
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append({
                "col_a": cols[i],
                "col_b": cols[j],
                "correlation": round(float(corr.iloc[i, j]), 3),
            })
    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    result["top_correlations"] = pairs[:10]
    return json.dumps(result, indent=2, default=str)


def detect_outliers(
    column: str,
    csv_path: str | None = None,
    dataset_name: str | None = None,
) -> str:
    """Detect outliers in a numeric column using the IQR (interquartile range) method."""
    logger.info("Tool detect_outliers: column=%s csv=%s dataset=%s", column, csv_path, dataset_name)
    df = _load_table(csv_path=csv_path, dataset_name=dataset_name)
    if column not in df.columns:
        return f"Column '{column}' not found. Available columns: {list(df.columns)}"
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return f"Column '{column}' has no numeric values."
    q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return json.dumps({
        "column": column,
        "total_rows": int(len(series)),
        "outlier_count": int(len(outliers)),
        "outlier_pct": round(len(outliers) / len(series) * 100, 2),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "lower_fence": round(lower, 4),
        "upper_fence": round(upper, 4),
        "sample_outlier_values": sorted(outliers.head(20).tolist()),
    }, indent=2)


def run_sql_query(
    query: str,
    csv_path: str | None = None,
    dataset_name: str | None = None,
) -> str:
    """Run a SQL SELECT query on the dataset. The table is always named 'df'. Returns up to 200 rows."""
    logger.info("Tool run_sql_query: csv=%s dataset=%s query=%s", csv_path, dataset_name, query[:100])
    df = _load_table(csv_path=csv_path, dataset_name=dataset_name)
    conn = sqlite3.connect(":memory:")
    try:
        df.to_sql("df", conn, index=False, if_exists="replace")
        result_df = pd.read_sql_query(query, conn)
        result_df = result_df.head(200)
        return json.dumps({
            "query": query,
            "rows_returned": len(result_df),
            "columns": list(result_df.columns),
            "data": _json_safe_records(result_df),
        }, indent=2, default=str)
    except Exception as e:
        logger.error("SQL error: %s", e)
        return json.dumps({"error": str(e), "query": query})
    finally:
        conn.close()


def smart_sql_query(
    question: str,
    csv_path: str | None = None,
    dataset_name: str | None = None,
) -> str:
    """
    Convert a natural language question into SQL and execute it using the full NL-to-SQL pipeline.

    Pipeline stages:
      1. Schema context  — injects column names, dtypes, null %, and sample values into the prompt
      2. Few-shot RAG    — retrieves the most similar past (question → SQL) pairs from memory
      3. CoT generation  — chain-of-thought: THINKING → QUERY_PLAN → SQL before writing the query
      4. Self-correction — executes the SQL; if it fails, feeds the error back to the LLM and retries
                           (up to 3 attempts automatically)
      5. Memory write    — saves every successful query so future similar questions get better answers

    Use this instead of run_sql_query for ANY natural language question about the data.
    Use run_sql_query only when you already have a precise SQL string to execute directly.
    """
    if _model is None:
        return json.dumps({"error": "LLM model not initialised. make_data_analyst_agent() must be called first."})

    logger.info("Tool smart_sql_query: question=%.80s csv=%s dataset=%s", question, csv_path, dataset_name)
    df = _load_table(csv_path=csv_path, dataset_name=dataset_name)
    dataset_label = csv_path or dataset_name or "unknown"

    result = nl_to_sql(
        question=question,
        df=df,
        dataset_label=dataset_label,
        model=_model,
        memory_db=_memory_db,
    )
    return json.dumps(result, indent=2, default=str)


def query_memory_stats(
    csv_path: str | None = None,
    dataset_name: str | None = None,
) -> str:
    """
    Show statistics about the NL-to-SQL query memory store:
    total saved examples, breakdown by dataset, and the 5 most recent queries.
    Useful for understanding how much few-shot context is available.
    """
    logger.info("Tool query_memory_stats")
    stats = memory_stats(memory_db=_memory_db)
    return json.dumps(stats, indent=2, default=str)


def run_python_code(
    code: str,
    csv_path: str | None = None,
    dataset_name: str | None = None,
) -> str:
    """
    Execute Python code for custom analysis. 'df' is pre-loaded as a pandas DataFrame.
    'pd' (pandas) is also available. Print results to see them. Returns stdout output.
    """
    logger.info("Tool run_python_code: csv=%s dataset=%s code_length=%s", csv_path, dataset_name, len(code))
    df = _load_table(csv_path=csv_path, dataset_name=dataset_name)
    namespace: dict = {"df": df.copy(), "pd": pd, "json": json}
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        exec(code, namespace)  # noqa: S102
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        return output.strip() if output.strip() else "Code executed successfully (no printed output)."
    except Exception as e:
        sys.stdout = old_stdout
        logger.error("Python execution error: %s", e)
        return f"Python execution error: {e}"


def generate_chart(
    chart_type: str,
    csv_path: str | None = None,
    dataset_name: str | None = None,
    x_col: str | None = None,
    y_col: str | None = None,
    hue_col: str | None = None,
    title: str | None = None,
) -> str:
    """
    Generate and save a chart as a PNG file. Returns the saved file path.

    chart_type options:
      - histogram  : distribution of one numeric column (x_col)
      - scatter    : x_col vs y_col, optionally colored by hue_col
      - bar        : value counts of a categorical column (x_col)
      - box        : box plot of numeric column (y_col), grouped by x_col
      - heatmap    : correlation heatmap of all numeric columns
      - line       : line plot of y_col (optionally over x_col)
      - pairplot   : pairwise scatter matrix (first 5 numeric cols, hue_col optional)
      - pie        : pie chart of top value counts in x_col
    """
    logger.info("Tool generate_chart: type=%s csv=%s dataset=%s x=%s y=%s", chart_type, csv_path, dataset_name, x_col, y_col)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted")

    df = _load_table(csv_path=csv_path, dataset_name=dataset_name)
    chart_type = chart_type.lower().strip()
    title = title or f"{chart_type.replace('_', ' ').title()} — {csv_path or dataset_name}"
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")
    chart_path = CHARTS_DIR / f"chart_{chart_type}_{timestamp}.png"

    try:
        if chart_type == "pairplot":
            plt.close("all")
            num_cols = df.select_dtypes(include="number").columns[:5].tolist()
            plot_df = df[num_cols + ([hue_col] if hue_col and hue_col in df.columns else [])]
            pair_fig = sns.pairplot(
                plot_df,
                hue=hue_col if hue_col and hue_col in df.columns else None,
                diag_kind="kde",
            )
            pair_fig.fig.suptitle(title, y=1.02, fontsize=13)
            pair_fig.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close("all")

        elif chart_type == "pie":
            col = x_col or df.select_dtypes(include=["object", "category"]).columns[0]
            counts = df[col].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140)
            ax.set_title(title, fontsize=13)
            plt.tight_layout()
            fig.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close("all")

        elif chart_type == "heatmap":
            numeric_df = df.select_dtypes(include="number")
            corr = numeric_df.corr()
            size = max(8, len(corr.columns))
            fig, ax = plt.subplots(figsize=(size, max(6, size - 2)))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
            ax.set_title(title, fontsize=13)
            plt.tight_layout()
            fig.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close("all")

        elif chart_type == "histogram":
            col = x_col or df.select_dtypes(include="number").columns[0]
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
            ax.set_xlabel(col, fontsize=11)
            ax.set_ylabel("Frequency", fontsize=11)
            ax.set_title(title, fontsize=13)
            plt.tight_layout()
            fig.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close("all")

        elif chart_type == "scatter":
            num_cols = df.select_dtypes(include="number").columns
            _x = x_col if x_col and x_col in df.columns else (num_cols[0] if len(num_cols) > 0 else None)
            _y = y_col if y_col and y_col in df.columns else (num_cols[1] if len(num_cols) > 1 else num_cols[0])
            if _x is None or _y is None:
                return "Not enough numeric columns for scatter plot."
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                data=df, x=_x, y=_y,
                hue=hue_col if hue_col and hue_col in df.columns else None,
                ax=ax, alpha=0.65, edgecolor="white", linewidth=0.3,
            )
            ax.set_xlabel(_x, fontsize=11)
            ax.set_ylabel(_y, fontsize=11)
            ax.set_title(title, fontsize=13)
            plt.tight_layout()
            fig.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close("all")

        elif chart_type == "bar":
            col = x_col or df.select_dtypes(include=["object", "category"]).columns[0]
            counts = df[col].value_counts().head(15)
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette="muted")
            ax.set_xlabel(col, fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.set_title(title, fontsize=13)
            plt.xticks(rotation=40, ha="right")
            plt.tight_layout()
            fig.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close("all")

        elif chart_type == "box":
            num_cols = df.select_dtypes(include="number").columns
            _y = y_col if y_col and y_col in df.columns else (num_cols[0] if len(num_cols) > 0 else None)
            if _y is None:
                return "No numeric column available for box plot."
            fig, ax = plt.subplots(figsize=(12, 6))
            if x_col and x_col in df.columns:
                sns.boxplot(data=df, x=x_col, y=_y, ax=ax, palette="muted")
                plt.xticks(rotation=40, ha="right")
            else:
                plot_cols = list(num_cols[:8])
                sns.boxplot(data=df[plot_cols], ax=ax, palette="muted")
                plt.xticks(rotation=40, ha="right")
            ax.set_title(title, fontsize=13)
            plt.tight_layout()
            fig.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close("all")

        elif chart_type == "line":
            num_cols = df.select_dtypes(include="number").columns
            _y = y_col if y_col and y_col in df.columns else (num_cols[0] if len(num_cols) > 0 else None)
            if _y is None:
                return "No numeric column available for line plot."
            fig, ax = plt.subplots(figsize=(12, 6))
            if x_col and x_col in df.columns:
                ax.plot(df[x_col], df[_y], linewidth=1.5, color="steelblue")
                ax.set_xlabel(x_col, fontsize=11)
            else:
                ax.plot(df[_y].values, linewidth=1.5, color="steelblue")
                ax.set_xlabel("Index", fontsize=11)
            ax.set_ylabel(_y, fontsize=11)
            ax.set_title(title, fontsize=13)
            plt.tight_layout()
            fig.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close("all")

        else:
            return (
                f"Unsupported chart_type '{chart_type}'. "
                "Choose from: histogram, scatter, bar, box, heatmap, line, pairplot, pie"
            )

        logger.info("Chart saved to %s", chart_path)
        return json.dumps({
            "chart_path": str(chart_path),
            "chart_type": chart_type,
            "title": title,
            "status": "success",
        })

    except Exception as e:
        plt.close("all")
        logger.error("Chart generation error: %s", e)
        return json.dumps({"error": str(e), "chart_type": chart_type, "status": "failed"})


ALL_TOOLS = [
    inspect_data,
    summarize_numeric,
    value_counts,
    correlation_matrix,
    detect_outliers,
    run_sql_query,
    smart_sql_query,
    query_memory_stats,
    run_python_code,
    generate_chart,
]
