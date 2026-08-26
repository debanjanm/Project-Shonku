"""Incremental offline ingestion: data/kbs/<slug>/* -> FAISS index.

Only re-embeds files whose content hash changed since the last run; removed
files have their chunks deleted from the index. State lives in
faiss_indexes/<slug>/manifest.json alongside the index itself.

Run:
    python -m backend.offline_pipeline.ingest                 # all KBs, incremental
    python -m backend.offline_pipeline.ingest --kb hr-policies # just this KB
    python -m backend.offline_pipeline.ingest --rebuild        # wipe + full re-embed
"""

import argparse
import hashlib
import json
import logging
import shutil
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.kb import KnowledgeBase, KB_DATA_DIR, list_knowledge_bases
from backend.logging_config import configure_logging
from backend.offline_pipeline.loaders import SUPPORTED_EXTENSIONS, load_source_file
from backend.retrieval import INDEX_DIR, get_embeddings, load_index_or_none, save_bm25, save_index
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _discover_source_files(kb_dir: Path) -> dict[str, Path]:
    files = {}
    for path in sorted(kb_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files[str(path.relative_to(kb_dir))] = path
    return files


def _load_manifest(index_dir: Path) -> dict:
    manifest_path = index_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return {"files": {}}
    return json.loads(manifest_path.read_text())


def _save_manifest(index_dir: Path, manifest: dict) -> None:
    (index_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))


def _split_file(kb: KnowledgeBase, relpath: str, path: Path) -> tuple[list[Document], list[str]]:
    docs = load_source_file(path)
    chunks = SPLITTER.split_documents(docs)
    ids = [uuid.uuid4().hex for _ in chunks]
    for i, (chunk, chunk_id) in enumerate(zip(chunks, ids)):
        chunk.metadata.update(
            {
                "source": relpath,
                "kb_slug": kb.slug,
                "chunk_index": i,
                "doc_title": path.stem,
            }
        )
        # Prefix the embedded text with the document title so a chunk's own
        # embedding carries its document identity (e.g. "2026 Q1 AAPL") even
        # when the chunk's prose never restates it — otherwise exact-document
        # queries ("Apple Q1 2026 revenue") can lose to chunks that happen to
        # discuss the topic more verbosely but belong to the wrong filing.
        chunk.page_content = f"[{path.stem}]\n{chunk.page_content}"
    return chunks, ids


def ingest_kb(kb: KnowledgeBase, *, rebuild: bool = False) -> None:
    kb_dir = KB_DATA_DIR / kb.slug
    index_dir = INDEX_DIR / kb.slug

    if rebuild and index_dir.exists():
        shutil.rmtree(index_dir)

    manifest = {"files": {}} if rebuild else _load_manifest(index_dir)
    # An index with no manifest entries is orphaned (e.g. built by a prior
    # non-incremental ingest run) — its vectors aren't accounted for, so
    # reusing it would silently accumulate untracked duplicates. Treat it
    # as absent and rebuild fresh instead.
    index = None if (rebuild or not manifest["files"]) else load_index_or_none(kb.slug)

    current_files = _discover_source_files(kb_dir)
    current_hashes = {relpath: _hash_file(path) for relpath, path in current_files.items()}

    new_count = changed_count = removed_count = unchanged_count = 0

    # Removed files: delete their chunks, drop from manifest.
    for relpath in list(manifest["files"]):
        if relpath not in current_hashes:
            chunk_ids = manifest["files"][relpath]["chunk_ids"]
            if index is not None and chunk_ids:
                index.delete(chunk_ids)
            del manifest["files"][relpath]
            removed_count += 1

    # New / changed files: re-embed.
    for relpath, path in current_files.items():
        prior = manifest["files"].get(relpath)
        if prior is not None and prior["hash"] == current_hashes[relpath]:
            unchanged_count += 1
            continue

        if prior is not None:
            if index is not None and prior["chunk_ids"]:
                index.delete(prior["chunk_ids"])
            changed_count += 1
        else:
            new_count += 1

        chunks, ids = _split_file(kb, relpath, path)
        if not chunks:
            manifest["files"][relpath] = {"hash": current_hashes[relpath], "chunk_ids": []}
            continue

        if index is None:
            index = FAISS.from_documents(chunks, get_embeddings(), ids=ids)
        else:
            index.add_documents(chunks, ids=ids)

        manifest["files"][relpath] = {"hash": current_hashes[relpath], "chunk_ids": ids}

    total_chunks = sum(len(f["chunk_ids"]) for f in manifest["files"].values())
    if total_chunks == 0:
        if index_dir.exists():
            shutil.rmtree(index_dir)
        logger.warning("[%s] no ingestible content, skipped (removed stale index if any)", kb.slug)
        return

    save_index(kb.slug, index)
    # BM25 has no incremental add/delete API, so it's always rebuilt in full
    # from the FAISS docstore (cheap: pure term-frequency stats, no embedding
    # calls) — this also means a plain (non --rebuild) ingest run backfills
    # bm25.pkl for any KB that predates hybrid search.
    all_docs = list(index.docstore._dict.values())
    all_ids = list(index.docstore._dict.keys())
    save_bm25(kb.slug, all_docs, all_ids)
    _save_manifest(index_dir, manifest)
    logger.info(
        "[%s] new=%d changed=%d removed=%d unchanged=%d -> %d chunks",
        kb.slug, new_count, changed_count, removed_count, unchanged_count, total_chunks,
    )


def main() -> None:
    configure_logging(plain=True)
    parser = argparse.ArgumentParser(description="Ingest KB source files into FAISS indexes.")
    parser.add_argument("--kb", action="append", dest="slugs", help="Limit to this KB slug (repeatable).")
    parser.add_argument("--rebuild", action="store_true", help="Wipe and fully re-embed the selected KB(s).")
    args = parser.parse_args()

    kbs = list_knowledge_bases()
    if args.slugs:
        kbs = [kb for kb in kbs if kb.slug in args.slugs]
        missing = set(args.slugs) - {kb.slug for kb in kbs}
        if missing:
            raise SystemExit(f"Unknown KB slug(s): {', '.join(sorted(missing))}")

    for kb in kbs:
        ingest_kb(kb, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
