"""MemoryManager — orchestrates Extract -> Search -> Update -> Store across
ChromaDB (vectors) and Neo4j (entity graph). Sync port of AIF-Remembrane's
memory/manager.py.
"""

import logging

from backend.memory import pipeline
from backend.memory.chroma_store import ChromaMemoryStore
from backend.memory.models import (
    GraphNode, GraphRelationship, Memory, MemoryScope,
    MemoryUpdateResult, OperationType, ScoredMemory,
)
from backend.memory.neo4j_store import Neo4jGraphStore
from backend.retrieval import get_embeddings

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self, vector_store: ChromaMemoryStore, graph_store: Neo4jGraphStore | None) -> None:
        self._vector = vector_store
        self._graph = graph_store

    def add(self, user_message: str, assistant_message: str, scope: MemoryScope) -> MemoryUpdateResult:
        """Full pipeline: Extract -> Search -> Update/Delete -> Store."""
        facts = pipeline.extract_facts(user_message, assistant_message, scope)
        if not facts:
            return MemoryUpdateResult()

        all_entity_names = list({e for fact in facts for e in fact.entities})
        combined_query = " ".join(fact.content for fact in facts)
        existing_memories = self._search(combined_query, scope, top_k=20, entity_names=all_entity_names)

        operations = pipeline.compute_operations(facts, existing_memories)
        self._embed_operations(operations)
        result = self._apply_operations(operations, scope)
        self._update_graph(facts, result, scope)
        return result

    def search(self, query: str, scope: MemoryScope, top_k: int = 10) -> list[ScoredMemory]:
        return self._search(query, scope, top_k=top_k)

    def get_context(self, query: str, scope: MemoryScope, top_k: int = 8) -> str:
        """Formatted memory context for system-prompt injection. Never raises —
        a Neo4j/Chroma hiccup (e.g. a paused AuraDB free instance) degrades to
        no memory recall instead of breaking the chat request."""
        try:
            memories = self._search(query, scope, top_k=top_k)
            return pipeline.synthesize(memories)
        except Exception as e:
            logger.warning("memory context fetch failed, continuing without it error=%s", e)
            return ""

    def get_all(self, scope: MemoryScope) -> list[Memory]:
        return self._vector.get_all(scope)

    def delete_all(self, scope: MemoryScope) -> int:
        return self._vector.delete_all(scope)

    # ── internal ──────────────────────────────────────────────────────────

    def _search(self, query: str, scope: MemoryScope, top_k: int, entity_names: list[str] | None = None) -> list[ScoredMemory]:
        query_embedding = get_embeddings().embed_query(query)
        vector_results = self._vector.search(query_embedding, scope, top_k=top_k)

        graph_memory_ids: set[str] = set()
        if entity_names and self._graph is not None:
            try:
                related_entities = self._graph.find_related_entities(entity_names, depth=2)
                for entity in related_entities:
                    graph_memory_ids.update(self._graph.get_entity_memory_ids(entity.id))
            except Exception as e:
                logger.warning("graph search failed, using vector-only results error=%s", e)

        seen_ids = {sm.memory.id for sm in vector_results}
        extra: list[ScoredMemory] = []
        for mid in graph_memory_ids:
            if mid in seen_ids:
                continue
            memory = self._vector.get_by_id(mid)
            if memory:
                extra.append(ScoredMemory(memory=memory, score=0.5))
                seen_ids.add(mid)

        results = vector_results + extra
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _embed_operations(self, operations: list) -> None:
        contents, indices = [], []
        for i, op in enumerate(operations):
            if op.action in (OperationType.ADD, OperationType.UPDATE):
                contents.append(op.merged_content or op.fact.content)
                indices.append(i)
        if not contents:
            return
        try:
            embeddings = get_embeddings().embed_documents(contents)
            for idx, embedding in zip(indices, embeddings):
                operations[idx].fact.embedding = embedding
        except Exception as e:
            logger.warning("embedding failed, memories stored without vectors error=%s", e)

    def _apply_operations(self, operations: list, scope: MemoryScope) -> MemoryUpdateResult:
        result = MemoryUpdateResult()
        to_upsert: list[Memory] = []

        for op in operations:
            if op.action == OperationType.SKIP:
                result.skipped += 1
                continue

            if op.action == OperationType.DELETE and op.target_id:
                self._vector.delete([op.target_id])
                if self._graph is not None:
                    self._graph.delete_memory_node(op.target_id)
                result.deleted.append(op.target_id)
                new_mem = Memory.new(op.fact.content, scope, op.fact.memory_type, op.fact.confidence)
                new_mem.embedding = op.fact.embedding
                to_upsert.append(new_mem)
                result.added.append(new_mem.id)

            elif op.action == OperationType.UPDATE and op.target_id:
                content = op.merged_content or op.fact.content
                existing = self._vector.get_by_id(op.target_id)
                if existing:
                    existing.content = content
                    existing.metadata.confidence = op.fact.confidence
                    existing.embedding = op.fact.embedding
                    to_upsert.append(existing)
                    result.updated.append(op.target_id)
                else:
                    new_mem = Memory.new(content, scope, op.fact.memory_type, op.fact.confidence)
                    new_mem.embedding = op.fact.embedding
                    to_upsert.append(new_mem)
                    result.added.append(new_mem.id)

            elif op.action == OperationType.ADD:
                new_mem = Memory.new(op.fact.content, scope, op.fact.memory_type, op.fact.confidence)
                new_mem.embedding = op.fact.embedding
                to_upsert.append(new_mem)
                result.added.append(new_mem.id)

        if to_upsert:
            self._vector.upsert(to_upsert)
        logger.info("memory update applied user_id=%s %s", scope.user_id, result.summary())
        return result

    def _update_graph(self, facts: list, result: MemoryUpdateResult, scope: MemoryScope) -> None:
        if self._graph is None:
            return
        try:
            all_entities: dict[str, GraphNode] = {}
            for fact in facts:
                for name in fact.entities:
                    if name not in all_entities:
                        entity = GraphNode(name=name)
                        entity.id = self._graph.upsert_entity(entity)
                        all_entities[name] = entity

            for fact in facts:
                entity_ids = [all_entities[e].id for e in fact.entities if e in all_entities]
                for memory_id in result.added + result.updated:
                    if entity_ids:
                        self._graph.link_memory_to_entities(memory_id, entity_ids, scope.user_id, scope.agent_id)
                for i, eid1 in enumerate(entity_ids):
                    for eid2 in entity_ids[i + 1:]:
                        self._graph.upsert_relationship(
                            GraphRelationship(source_id=eid1, target_id=eid2, rel_type="RELATED_TO", strength=1.0)
                        )
        except Exception as e:
            logger.warning("graph update failed (non-fatal) error=%s", e)
