"""Retrieval quality eval harness, built on Vectara's open_ragbench benchmark
(real queries + ground-truth relevant papers). Sourced from reviewing
AIR-OpenRAG: its fetch/download logic is ported here; its scoring scripts
were just similarity-metric scratch code with no actual qrels-based scoring,
so the eval loop itself (recall@k, MRR) is new.

Deliberately reuses Shonku's existing pipeline instead of building a parallel
one: the fetched corpus becomes a normal KB under data/kbs/ragbench-eval/,
ingested via the unchanged `backend.offline_pipeline.ingest` (same FAISS+BM25
hybrid index everything else uses), then scored with the unchanged
`backend.retrieval.search()` — this evaluates the real retrieval path, not a
simulation of it.

Relevance is judged at paper level (did the correct paper's chunk appear in
the top-k), not exact section — ingestion re-chunks freely by character
count, so section-level ground truth doesn't survive that process anyway.

Run:
    python -m backend.offline_pipeline.eval_retrieval --fetch
        # downloads a slice of open_ragbench, builds data/kbs/ragbench-eval/
    python -m backend.offline_pipeline.ingest --kb ragbench-eval
        # normal ingest — builds the FAISS+BM25 index like any other KB
    python -m backend.offline_pipeline.eval_retrieval --run
        # scores backend.retrieval.search() against the ground truth
"""

import argparse
import json
import logging
from pathlib import Path

from backend.kb import KB_DATA_DIR
from backend.logging_config import configure_logging
from backend.retrieval import search

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DOWNLOAD_DIR = ROOT / "data" / "raw" / "ragbench_download"
EVAL_QUERIES_PATH = ROOT / "data" / "raw" / "ragbench_eval_queries.json"
EVAL_REPORT_PATH = ROOT / "data" / "raw" / "ragbench_eval_report.json"
KB_SLUG = "ragbench-eval"

REPO_ID = "vectara/open_ragbench"
REPO_SUBFOLDER = "pdf/arxiv"
MAX_PAPERS = 50  # keep the default eval run fast/cheap — not the full benchmark
K_VALUES = (1, 3, 5, 10)


def _normalize_qrels(raw: dict) -> dict[str, set[str]]:
    """qrels.json values may be a single {doc_id: ...} dict or a list of them
    per query_id — normalize both shapes to {query_id: {doc_id, ...}}."""
    qrels: dict[str, set[str]] = {}
    for query_id, value in raw.items():
        entries = value if isinstance(value, list) else [value]
        qrels[query_id] = {e["doc_id"] for e in entries if isinstance(e, dict) and "doc_id" in e}
    return qrels


def _paper_to_markdown(paper: dict) -> str:
    lines = [f"# {paper.get('title', '')}", ""]
    authors = paper.get("authors")
    if authors:
        lines += [f"Authors: {', '.join(authors)}", ""]
    abstract = paper.get("abstract")
    if abstract:
        lines += ["## Abstract", abstract, ""]
    for section in paper.get("sections", []):
        lines += [section.get("text", ""), ""]
    return "\n".join(lines)


def fetch() -> None:
    from huggingface_hub import snapshot_download

    configure_logging(plain=True)
    logger.info("downloading %s (%s)...", REPO_ID, REPO_SUBFOLDER)
    download_path = Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=f"{REPO_SUBFOLDER}/*",
            local_dir=str(DOWNLOAD_DIR),
        )
    )
    source_base = download_path / REPO_SUBFOLDER

    queries_by_id = json.loads((source_base / "queries.json").read_text())
    qrels_by_id = _normalize_qrels(json.loads((source_base / "qrels.json").read_text()))

    corpus_dir = source_base / "corpus"
    corpus_files = sorted(corpus_dir.glob("*.json"))[:MAX_PAPERS]
    kb_dir = KB_DATA_DIR / KB_SLUG
    kb_dir.mkdir(parents=True, exist_ok=True)

    selected_paper_ids = set()
    for corpus_file in corpus_files:
        paper = json.loads(corpus_file.read_text())
        paper_id = paper.get("id") or corpus_file.stem
        selected_paper_ids.add(paper_id)
        (kb_dir / f"{paper_id}.md").write_text(_paper_to_markdown(paper))

    (kb_dir / "kb.yaml").write_text(
        "name: RAGBench Eval\n"
        f"description: {len(selected_paper_ids)}-paper slice of Vectara's open_ragbench, for retrieval eval only.\n"
    )
    logger.info("wrote %d papers to %s", len(selected_paper_ids), kb_dir)

    eval_queries = []
    for query_id, query_meta in queries_by_id.items():
        relevant = qrels_by_id.get(query_id, set()) & selected_paper_ids
        if relevant:
            eval_queries.append(
                {"query_id": query_id, "query": query_meta.get("query", ""), "relevant_doc_ids": sorted(relevant)}
            )

    EVAL_QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_QUERIES_PATH.write_text(json.dumps(eval_queries, indent=2))
    logger.info(
        "wrote %d eval queries (with a relevant paper in the selected slice) to %s",
        len(eval_queries), EVAL_QUERIES_PATH,
    )
    logger.info("next: python -m backend.offline_pipeline.ingest --kb %s", KB_SLUG)


def run_eval() -> None:
    configure_logging(plain=True)
    if not EVAL_QUERIES_PATH.exists():
        raise SystemExit(f"{EVAL_QUERIES_PATH} not found — run --fetch first.")
    eval_queries = json.loads(EVAL_QUERIES_PATH.read_text())
    if not eval_queries:
        raise SystemExit("No eval queries found.")

    max_k = max(K_VALUES)
    hits_at_k = {k: 0 for k in K_VALUES}
    reciprocal_ranks = []

    for entry in eval_queries:
        results = search(KB_SLUG, entry["query"], k=max_k)
        retrieved_paper_ids = [doc.metadata.get("doc_title") for doc in results]
        relevant = set(entry["relevant_doc_ids"])

        rank = next((i + 1 for i, pid in enumerate(retrieved_paper_ids) if pid in relevant), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        for k in K_VALUES:
            if rank is not None and rank <= k:
                hits_at_k[k] += 1

    n = len(eval_queries)
    report = {
        "num_queries": n,
        "recall_at_k": {k: round(hits_at_k[k] / n, 4) for k in K_VALUES},
        "mrr": round(sum(reciprocal_ranks) / n, 4),
    }
    EVAL_REPORT_PATH.write_text(json.dumps(report, indent=2))

    logger.info("=== Retrieval eval: %s (%d queries) ===", KB_SLUG, n)
    for k in K_VALUES:
        logger.info("Recall@%-2d: %.1f%%", k, report["recall_at_k"][k] * 100)
    logger.info("MRR:       %.4f", report["mrr"])
    logger.info("report written to %s", EVAL_REPORT_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGBench-based retrieval eval harness.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fetch", action="store_true", help="Download the benchmark slice and build the KB.")
    group.add_argument("--run", action="store_true", help="Score retrieval against the ground truth.")
    args = parser.parse_args()

    if args.fetch:
        fetch()
    else:
        run_eval()


if __name__ == "__main__":
    main()
