"""Recommendation pipeline eval harness. No external labeled ground truth
exists for this catalog (unlike docqa's ragbench benchmark) — no dataset of
"query -> correct product" pairs. Leave-one-out design instead: build a
synthetic query from a sampled product's own tags, then check whether that
exact product comes back in its own results. Not a perfect signal — several
near-identical catalog siblings (many similar "black women's bags") can
legitimately outrank the exact source item without that being a pipeline
bug — but real, reproducible, and scored the same Recall@k/MRR way
eval_retrieval.py already established, for the same honest reasons.

Two recall numbers, not one, isolating which stage a regression would come
from:
  - Candidate stage: raw CLIP+ChromaDB recall on the query text as typed,
    before LLM query expansion or reranking. An approximation of stage 1
    alone — production actually embeds the LLM-expanded query, not the raw
    one, before this step — but a useful "is the raw vector search even in
    the neighborhood" signal, isolated from the two LLM-dependent stages.
  - Final stage: the real run_pipeline() unmodified — zero drift from what
    a user actually gets, same "reuse the real pipeline, not a parallel
    simulation of it" principle eval_retrieval.py already established.

Run:
    python -m backend.agents.recommendation.eval_recommendation
"""

import json
import logging
import random
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(ROOT / ".env")  # run_pipeline() calls LLMService(), which needs OPENROUTER_API_KEY

from backend.agents.recommendation.agent import TOP_K_CANDIDATES, run_pipeline  # noqa: E402
from backend.agents.recommendation.embedding_service import get_embedding_service  # noqa: E402
from backend.agents.recommendation.models import Product  # noqa: E402
from backend.agents.recommendation.vector_store import VectorStore  # noqa: E402
from backend.logging_config import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

REPORT_PATH = ROOT / "data" / "raw" / "recommendation_eval_report.json"

SAMPLE_SIZE = 50  # fast/cheap default — matches eval_retrieval.py's MAX_PAPERS
SEED = 42
K_VALUES = (1, 3, 5, 10)


def _build_query(product: Product) -> str:
    """Realistic keyword-style e-commerce query from the item's own tags —
    not the full product name, which would make this a trivial string-match
    test instead of a real semantic-search one."""
    return " ".join(product.tags)


def run_eval(sample_size: int = SAMPLE_SIZE) -> None:
    configure_logging(plain=True)
    store = VectorStore()
    total = store.count()
    if total == 0:
        raise SystemExit("Product catalog is empty — nothing to eval.")

    all_products = store.list_products(limit=total)
    random.Random(SEED).shuffle(all_products)
    sample = all_products[:sample_size]

    embedder = get_embedding_service()

    candidate_hits = {k: 0 for k in K_VALUES}
    final_hits = {k: 0 for k in K_VALUES}
    candidate_rr: list[float] = []
    final_rr: list[float] = []

    for i, product in enumerate(sample, 1):
        query = _build_query(product)
        logger.info("[%d/%d] %s -> %r", i, len(sample), product.id, query)

        query_embedding = embedder.embed_query(text=query)
        candidates = store.query(embedding=query_embedding, n_results=min(TOP_K_CANDIDATES, total))
        candidate_ids = [c["id"] for c in candidates]
        c_rank = next((r + 1 for r, pid in enumerate(candidate_ids) if pid == product.id), None)
        candidate_rr.append(1.0 / c_rank if c_rank else 0.0)
        for k in K_VALUES:
            if c_rank is not None and c_rank <= k:
                candidate_hits[k] += 1

        results = run_pipeline(query)
        final_ids = [r["product"].id for r in results]
        f_rank = next((r + 1 for r, pid in enumerate(final_ids) if pid == product.id), None)
        final_rr.append(1.0 / f_rank if f_rank else 0.0)
        for k in K_VALUES:
            if f_rank is not None and f_rank <= k:
                final_hits[k] += 1

    n = len(sample)
    report = {
        "sample_size": n,
        "candidate_recall_at_k": {k: round(candidate_hits[k] / n, 4) for k in K_VALUES},
        "candidate_mrr": round(sum(candidate_rr) / n, 4),
        "final_recall_at_k": {k: round(final_hits[k] / n, 4) for k in K_VALUES},
        "final_mrr": round(sum(final_rr) / n, 4),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    logger.info("=== Recommendation eval (n=%d) ===", n)
    logger.info("-- Candidate stage: raw CLIP+ChromaDB, before LLM expansion/rerank --")
    for k in K_VALUES:
        logger.info("Recall@%-2d: %.1f%%", k, report["candidate_recall_at_k"][k] * 100)
    logger.info("MRR:       %.4f", report["candidate_mrr"])
    logger.info("-- Final stage: real run_pipeline(), top-10 as shown to the user --")
    for k in K_VALUES:
        logger.info("Recall@%-2d: %.1f%%", k, report["final_recall_at_k"][k] * 100)
    logger.info("MRR:       %.4f", report["final_mrr"])
    logger.info("report written to %s", REPORT_PATH)


def main() -> None:
    run_eval()


if __name__ == "__main__":
    main()
