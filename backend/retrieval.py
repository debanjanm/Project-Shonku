"""Build and query per-KB FAISS indexes.

Embeddings run locally (sentence-transformers) so retrieval needs no API key.
"""

import logging
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "faiss_indexes"

_embeddings: HuggingFaceEmbeddings | None = None
_loaded_indexes: dict[str, FAISS] = {}


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info("loading embedding model sentence-transformers/all-MiniLM-L6-v2")
        start = time.monotonic()
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info("embedding model loaded in %.2fs", time.monotonic() - start)
    return _embeddings


def save_index(kb_slug: str, index: FAISS) -> None:
    out_dir = INDEX_DIR / kb_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    index.save_local(str(out_dir))


def load_index_or_none(kb_slug: str) -> FAISS | None:
    idx_dir = INDEX_DIR / kb_slug
    if not idx_dir.exists():
        return None
    index = FAISS.load_local(str(idx_dir), get_embeddings(), allow_dangerous_deserialization=True)
    logger.debug("loaded index kb_slug=%s vectors=%d", kb_slug, index.index.ntotal)
    return index


def load_index(kb_slug: str) -> FAISS:
    if kb_slug not in _loaded_indexes:
        index = load_index_or_none(kb_slug)
        if index is None:
            logger.error("no FAISS index for kb_slug=%s", kb_slug)
            raise FileNotFoundError(f"No FAISS index for KB '{kb_slug}'. Run `python -m backend.offline_pipeline.ingest` first.")
        _loaded_indexes[kb_slug] = index
    return _loaded_indexes[kb_slug]


def search(kb_slug: str, query: str, k: int = 4) -> list[Document]:
    logger.info("search kb_slug=%s k=%d query=%r", kb_slug, k, query)
    start = time.monotonic()
    index = load_index(kb_slug)
    results = index.similarity_search(query, k=k)
    logger.info(
        "search kb_slug=%s returned %d results in %.3fs",
        kb_slug, len(results), time.monotonic() - start,
    )
    return results
