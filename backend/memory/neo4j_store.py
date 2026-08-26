"""Neo4j AuraDB graph store for the memory entity graph.

Ported from AIF-Remembrane's clients/neo4j_client.py using the sync
neo4j.GraphDatabase driver instead of AsyncGraphDatabase — Shonku's backend
has no async anywhere else (main.py handlers, db.py, retrieval.py are all
sync), so keeping this sync too avoids introducing asyncio for one feature.

Graph schema:
    Nodes:  Entity {id, name, normalized_name, entity_type}
            User   {user_id}
    Rels:   (:MemoryRef)-[:MENTIONS]->(:Entity)
            (:Entity)-[:RELATED_TO {strength}]->(:Entity)
            (:User)-[:HAS_MEMORY]->(:MemoryRef)
"""

import logging
import os
from typing import Optional

from neo4j import GraphDatabase

from backend.memory.models import EntityType, GraphNode, GraphRelationship

logger = logging.getLogger(__name__)


class Neo4jGraphStore:
    def __init__(self) -> None:
        uri = os.environ["NEO4J_URI"]
        user = os.environ["NEO4J_USERNAME"]
        password = os.environ["NEO4J_PASSWORD"]
        self._database = os.environ.get("NEO4J_DATABASE") or None
        self._driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_pool_size=50)
        self._driver.verify_connectivity()
        logger.info("connected to neo4j uri=%s", uri)

    def close(self) -> None:
        self._driver.close()

    def _session(self):
        return self._driver.session(database=self._database)

    def setup_schema(self) -> None:
        """Create indexes/constraints. Idempotent, safe to call every startup."""
        queries = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.normalized_name)",
        ]
        with self._session() as session:
            for q in queries:
                session.run(q)
        logger.info("neo4j schema ready")

    def upsert_entity(self, entity: GraphNode) -> str:
        query = """
        MERGE (e:Entity {normalized_name: $normalized_name})
        ON CREATE SET e.id = $id, e.name = $name, e.entity_type = $entity_type
        ON MATCH  SET e.entity_type = $entity_type
        RETURN e.id AS id
        """
        with self._session() as session:
            record = session.run(
                query, id=entity.id, name=entity.name,
                normalized_name=entity.normalized_name, entity_type=entity.entity_type.value,
            ).single()
            return record["id"]

    def upsert_relationship(self, rel: GraphRelationship) -> None:
        query = f"""
        MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
        MERGE (a)-[r:{rel.rel_type}]->(b)
        SET r += $props
        """
        with self._session() as session:
            session.run(query, source_id=rel.source_id, target_id=rel.target_id,
                        props={**rel.properties, "strength": rel.strength})

    def link_memory_to_entities(self, memory_id: str, entity_ids: list[str], user_id: str,
                                 agent_id: Optional[str] = None) -> None:
        with self._session() as session:
            session.run("MERGE (u:User {user_id: $user_id})", user_id=user_id)
            session.run(
                """
                MERGE (m:MemoryRef {memory_id: $memory_id})
                ON CREATE SET m.user_id = $user_id, m.agent_id = $agent_id
                WITH m
                MATCH (u:User {user_id: $user_id})
                MERGE (u)-[:HAS_MEMORY]->(m)
                """,
                memory_id=memory_id, user_id=user_id, agent_id=agent_id or "",
            )
            for eid in entity_ids:
                session.run(
                    """
                    MATCH (m:MemoryRef {memory_id: $memory_id})
                    MATCH (e:Entity {id: $entity_id})
                    MERGE (m)-[:MENTIONS]->(e)
                    """,
                    memory_id=memory_id, entity_id=eid,
                )

    def find_related_entities(self, entity_names: list[str], depth: int = 2) -> list[GraphNode]:
        if not entity_names:
            return []
        normalized = [n.lower().strip() for n in entity_names]
        query_apoc = """
        MATCH (e:Entity)
        WHERE e.normalized_name IN $names
        CALL apoc.path.subgraphNodes(e, {relationshipFilter: 'RELATED_TO>|<RELATED_TO', maxLevel: $depth})
        YIELD node
        RETURN DISTINCT node.id AS id, node.name AS name,
               node.normalized_name AS normalized_name, node.entity_type AS entity_type
        """
        # AuraDB free tier may not have APOC — fall back to plain variable-length match.
        query_no_apoc = """
        MATCH (e:Entity)-[:RELATED_TO*1..2]-(related:Entity)
        WHERE e.normalized_name IN $names
        RETURN DISTINCT related.id AS id, related.name AS name,
               related.normalized_name AS normalized_name, related.entity_type AS entity_type
        UNION
        MATCH (e:Entity)
        WHERE e.normalized_name IN $names
        RETURN e.id AS id, e.name AS name, e.normalized_name AS normalized_name, e.entity_type AS entity_type
        """
        with self._session() as session:
            try:
                records = list(session.run(query_apoc, names=normalized, depth=depth))
            except Exception:
                records = list(session.run(query_no_apoc, names=normalized))

        nodes: list[GraphNode] = []
        for rec in records:
            try:
                etype = EntityType(rec["entity_type"])
            except ValueError:
                etype = EntityType.UNKNOWN
            nodes.append(GraphNode(id=rec["id"], name=rec["name"], entity_type=etype))
        return nodes

    def get_entity_memory_ids(self, entity_id: str) -> list[str]:
        query = "MATCH (m:MemoryRef)-[:MENTIONS]->(e:Entity {id: $entity_id}) RETURN m.memory_id AS memory_id"
        with self._session() as session:
            return [r["memory_id"] for r in session.run(query, entity_id=entity_id)]

    def delete_memory_node(self, memory_id: str) -> None:
        with self._session() as session:
            session.run("MATCH (m:MemoryRef {memory_id: $memory_id}) DETACH DELETE m", memory_id=memory_id)

    def delete_user_data(self, user_id: str) -> int:
        query = """
        MATCH (u:User {user_id: $user_id})
        OPTIONAL MATCH (u)-[:HAS_MEMORY]->(m:MemoryRef)
        WITH u, collect(m) AS memories
        FOREACH (m IN memories | DETACH DELETE m)
        DETACH DELETE u
        RETURN size(memories) AS deleted_count
        """
        with self._session() as session:
            record = session.run(query, user_id=user_id).single()
            return int(record["deleted_count"]) if record else 0
