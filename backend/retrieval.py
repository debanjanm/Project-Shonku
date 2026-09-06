"""Build and query per-KB hybrid (BM25 + FAISS) indexes.

Embeddings run locally (sentence-transformers) so retrieval needs no API key.
"""

import logging
import pickle
import re
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "faiss_indexes"
BM25_FILENAME = "bm25.pkl"
RRF_K = 60  # standard Reciprocal Rank Fusion constant

_embeddings: HuggingFaceEmbeddings | None = None
_loaded_indexes: dict[str, FAISS] = {}
_loaded_bm25: dict[str, dict | None] = {}


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info("loading embedding model sentence-transformers/all-MiniLM-L6-v2")
        start = time.monotonic()
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info("embedding model loaded in %.2fs", time.monotonic() - start)
    return _embeddings


# Without stopword removal, BM25's raw term-frequency scoring lets a short
# chunk that happens to repeat common words ("what", "was", "in") plus one
# or two real query terms outscore a longer, genuinely relevant chunk — this
# is standard IR practice, not optional polish (confirmed empirically: an
# unrelated company's filing was outranking the correct one before this).
_STOPWORDS = frozenset("""
a an the of in on at to for with and or is was were be been being
this that these those it its as by from what which who whom
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in _STOPWORDS]


# Adjacent-quarter fix: dense embeddings can't reliably tell apart two
# filings from the same company whose boilerplate barely changes quarter to
# quarter. When the query names both a year and a quarter explicitly, narrow
# candidates to that exact period before ranking — conservative on purpose:
# a year or quarter named alone isn't a strong enough signal to safely
# exclude candidates, so it falls through to today's unfiltered behavior.
_YEAR_RE = re.compile(r"\b(\d{4})\b")
_QUARTER_RE = re.compile(r"\bQ([1-4])\b", re.IGNORECASE)


def _parse_query_period(query: str) -> tuple[int, int] | None:
    year_match = _YEAR_RE.search(query)
    quarter_match = _QUARTER_RE.search(query)
    if not year_match or not quarter_match:
        return None
    return int(year_match.group(1)), int(quarter_match.group(1))


def _filter_by_period(docs: list[Document], year: int, quarter: int) -> list[Document]:
    """Keeps a doc if it has no fiscal_year metadata at all (never excludes
    non-quarterly KBs) or its fiscal_year/fiscal_quarter match exactly."""
    return [
        d for d in docs
        if d.metadata.get("fiscal_year") is None
        or (d.metadata.get("fiscal_year") == year and d.metadata.get("fiscal_quarter") == quarter)
    ]


def save_index(kb_slug: str, index: FAISS) -> None:
    out_dir = INDEX_DIR / kb_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    index.save_local(str(out_dir))


def save_bm25(kb_slug: str, docs: list[Document], ids: list[str]) -> None:
    out_dir = INDEX_DIR / kb_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    bm25 = BM25Okapi([tokenize(doc.page_content) for doc in docs])
    with open(out_dir / BM25_FILENAME, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": docs, "ids": ids}, f)
    logger.info("saved bm25 index kb_slug=%s docs=%d", kb_slug, len(docs))


def load_index_or_none(kb_slug: str) -> FAISS | None:
    idx_dir = INDEX_DIR / kb_slug
    if not idx_dir.exists():
        return None
    index = FAISS.load_local(str(idx_dir), get_embeddings(), allow_dangerous_deserialization=True)
    logger.debug("loaded index kb_slug=%s vectors=%d", kb_slug, index.index.ntotal)
    return index


def load_bm25_or_none(kb_slug: str) -> dict | None:
    if kb_slug not in _loaded_bm25:
        path = INDEX_DIR / kb_slug / BM25_FILENAME
        if path.exists():
            with open(path, "rb") as f:
                _loaded_bm25[kb_slug] = pickle.load(f)
            logger.debug("loaded bm25 index kb_slug=%s", kb_slug)
        else:
            _loaded_bm25[kb_slug] = None
    return _loaded_bm25[kb_slug]


def load_index(kb_slug: str) -> FAISS:
    if kb_slug not in _loaded_indexes:
        index = load_index_or_none(kb_slug)
        if index is None:
            logger.error("no FAISS index for kb_slug=%s", kb_slug)
            raise FileNotFoundError(f"No FAISS index for KB '{kb_slug}'. Run `python -m backend.offline_pipeline.ingest` first.")
        _loaded_indexes[kb_slug] = index
    return _loaded_indexes[kb_slug]


def _rrf_merge(ranked_lists: list[list[Document]], k: int) -> list[Document]:
    """Reciprocal Rank Fusion: combine several ranked result lists by rank
    position only (1/(rank+RRF_K) per list), not raw score — avoids having to
    normalize incomparable scales (FAISS distance vs. BM25 score)."""
    scores: dict[str, float] = {}
    doc_by_key: dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = f"{doc.metadata.get('source')}:{doc.metadata.get('chunk_index')}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + RRF_K)
            doc_by_key.setdefault(key, doc)
    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [doc_by_key[key] for key in ranked_keys[:k]]


def search(kb_slug: str, query: str, k: int = 6) -> list[Document]:
    logger.info("search kb_slug=%s k=%d query=%r", kb_slug, k, query)
    start = time.monotonic()
    index = load_index(kb_slug)
    bundle = load_bm25_or_none(kb_slug)
    period = _parse_query_period(query)

    if bundle is None:
        # Not yet re-ingested since hybrid search was added — fall back to
        # plain dense search rather than erroring.
        fetch_k = k * 3 if period else k
        results = index.similarity_search(query, k=fetch_k)
        if period:
            results = _filter_by_period(results, *period)[:k]
        logger.info(
            "search kb_slug=%s (dense-only, no bm25 index) returned %d results in %.3fs",
            kb_slug, len(results), time.monotonic() - start,
        )
        return results

    # Over-fetch a wider pool when a period was named so narrowing to that
    # exact quarter doesn't starve the final top-k.
    fetch_k = max(k * 2, 12) * (3 if period else 1)
    dense_results = index.similarity_search(query, k=fetch_k)

    query_tokens = tokenize(query)
    bm25_scores = bundle["bm25"].get_scores(query_tokens)
    sparse_order = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:fetch_k]
    sparse_results = [bundle["docs"][i] for i in sparse_order]

    if period:
        dense_results = _filter_by_period(dense_results, *period)
        sparse_results = _filter_by_period(sparse_results, *period)

    results = _rrf_merge([dense_results, sparse_results], k=k)
    logger.info(
        "search kb_slug=%s (hybrid) returned %d results in %.3fs",
        kb_slug, len(results), time.monotonic() - start,
    )
    return results
