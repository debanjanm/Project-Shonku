"""Cross-agent memory layer: long-term facts about the user, extracted from
every conversation and injected into every agent's system prompt.

Backed by ChromaDB (vectors, local embeddings) + Neo4j AuraDB (entity graph).
Ported from AIF-Remembrane's Mem0-architecture pipeline. See manager.py for
the orchestrator, models.py for the data shapes.

Usage:
    from backend.memory import get_memory_manager
    manager = get_memory_manager()
    context = manager.get_context(query, scope)
"""

import logging

from backend.memory.chroma_store import ChromaMemoryStore
from backend.memory.manager import MemoryManager
from backend.memory.neo4j_store import Neo4jGraphStore

logger = logging.getLogger(__name__)

_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    global _manager
    if _manager is None:
        vector_store = ChromaMemoryStore()
        try:
            graph_store: Neo4jGraphStore | None = Neo4jGraphStore()
            graph_store.setup_schema()
        except Exception:
            # Entity-graph expansion is an enhancement, not a requirement —
            # vector memory (extract/search/store) works fine on its own.
            logger.exception("neo4j unavailable, memory running vector-only (no entity graph)")
            graph_store = None
        _manager = MemoryManager(vector_store, graph_store)
        logger.info("memory manager ready graph_enabled=%s", graph_store is not None)
    return _manager
