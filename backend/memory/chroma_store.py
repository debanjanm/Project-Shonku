"""ChromaDB vector store for memories.

Ported from AIF-Remembrane's clients/chroma_client.py, with one real change:
embeddings come from backend.retrieval.get_embeddings() (local
sentence-transformers, 384-dim, free) instead of the OpenRouter embedding
API — same model every other KB/eval in Shonku already uses.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb import Collection
from chromadb.api.types import Documents, Embeddings, Where
from chromadb.utils.embedding_functions import EmbeddingFunction

from backend.memory.models import Memory, MemoryMetadata, MemoryScope, MemoryType, ScoredMemory

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent


class _NoOpEmbeddingFunction(EmbeddingFunction):
    """We always pass pre-computed embeddings directly; this only stops
    ChromaDB from registering its own default (different dim) model."""

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        raise RuntimeError("_NoOpEmbeddingFunction should never be invoked directly.")


def _build_where(scope: MemoryScope) -> Optional[Where]:
    return scope.to_filter()  # type: ignore[return-value]


def _doc_to_memory(id: str, document: str, metadata: dict, embedding: Optional[list[float]] = None) -> Memory:
    scope = MemoryScope(
        user_id=metadata.get("user_id", ""),
        agent_id=metadata.get("agent_id") or None,
        run_id=metadata.get("run_id") or None,
    )
    meta = MemoryMetadata(
        scope=scope,
        memory_type=MemoryType(metadata.get("memory_type", MemoryType.USER.value)),
        confidence=float(metadata.get("confidence", 1.0)),
    )
    return Memory(id=id, content=document, metadata=meta, embedding=embedding)


class ChromaMemoryStore:
    def __init__(self) -> None:
        persist_path = os.environ.get("CHROMA_PERSIST_PATH", "./data/chroma")
        path = Path(persist_path)
        if not path.is_absolute():
            path = ROOT / persist_path
        path.mkdir(parents=True, exist_ok=True)
        self._collection_name = os.environ.get("CHROMA_COLLECTION", "memories")
        self._chroma = chromadb.PersistentClient(path=str(path))
        self._collection: Optional[Collection] = self._chroma.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=_NoOpEmbeddingFunction(),
        )
        logger.info("chroma memory collection ready collection=%s path=%s", self._collection_name, path)

    def upsert(self, memories: list[Memory]) -> list[str]:
        if not memories:
            return []
        ids = [m.id for m in memories]
        documents = [m.content for m in memories]
        metadatas = [m.metadata.to_dict() for m in memories]
        embeddings = [m.embedding for m in memories if m.embedding is not None]
        kwargs: dict = {"ids": ids, "documents": documents, "metadatas": metadatas}
        if len(embeddings) == len(memories):
            kwargs["embeddings"] = embeddings
        self._collection.upsert(**kwargs)
        return ids

    def search(self, query_embedding: list[float], scope: MemoryScope, top_k: int = 10) -> list[ScoredMemory]:
        where = _build_where(scope)
        if self._collection.count() == 0:
            return []
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding], n_results=top_k, where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            err = str(e)
            if "greater than number of elements" in err or "Number of requested results" in err:
                try:
                    results = self._collection.query(
                        query_embeddings=[query_embedding], n_results=1, where=where,
                        include=["documents", "metadatas", "distances"],
                    )
                except Exception:
                    return []
            else:
                raise

        scored: list[ScoredMemory] = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        for mid, doc, meta, dist in zip(ids, docs, metas, distances):
            similarity = 1.0 - (dist / 2.0)  # cosine distance 0-2 -> similarity 0-1
            scored.append(ScoredMemory(memory=_doc_to_memory(mid, doc, meta), score=similarity))
        return sorted(scored, key=lambda x: x.score, reverse=True)

    def get_by_id(self, memory_id: str) -> Optional[Memory]:
        results = self._collection.get(ids=[memory_id], include=["documents", "metadatas"])
        if not results["ids"]:
            return None
        return _doc_to_memory(results["ids"][0], results["documents"][0], results["metadatas"][0])

    def get_all(self, scope: MemoryScope) -> list[Memory]:
        results = self._collection.get(where=_build_where(scope), include=["documents", "metadatas"])
        return [_doc_to_memory(mid, doc, meta) for mid, doc, meta in
                zip(results["ids"], results["documents"], results["metadatas"])]

    def delete(self, memory_ids: list[str]) -> None:
        if memory_ids:
            self._collection.delete(ids=memory_ids)

    def delete_all(self, scope: MemoryScope) -> int:
        results = self._collection.get(where=_build_where(scope), include=[])
        ids = results["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)
