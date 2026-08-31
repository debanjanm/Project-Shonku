"""Data model for the cross-agent memory layer.

Ported from AIF-Remembrane's models/memory.py + models/graph.py. MemoryScope
deliberately drops agent_id as a filter dimension — see manager.py — since
Shonku's memory is shared across all agents, not per-agent like the source.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class MemoryType(str, Enum):
    USER = "user"      # preferences, biographical facts
    AGENT = "agent"     # agent-specific context
    RUN = "run"         # session-only


class OperationType(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    SKIP = "SKIP"


@dataclass
class MemoryScope:
    """Ownership/visibility scope of a memory. agent_id silos memory per
    agent (a Document Q&A fact never surfaces in Story Developer) and IS
    used to filter retrieval. run_id tags a memory with the conversation it
    came from for provenance but is never used to filter retrieval — recall
    spans every conversation with the same agent, not just the one a fact
    was learned in."""
    user_id: str
    agent_id: Optional[str] = None
    run_id: Optional[str] = None

    def to_filter(self) -> dict:
        conditions: list[dict] = [{"user_id": {"$eq": self.user_id}}]
        if self.agent_id:
            conditions.append({"agent_id": {"$eq": self.agent_id}})
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def to_metadata(self) -> dict:
        return {"user_id": self.user_id, "agent_id": self.agent_id or "", "run_id": self.run_id or ""}


@dataclass
class MemoryMetadata:
    scope: MemoryScope
    memory_type: MemoryType = MemoryType.USER
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            **self.scope.to_metadata(),
            "memory_type": self.memory_type.value,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class Memory:
    id: str
    content: str
    metadata: MemoryMetadata
    embedding: Optional[list[float]] = None

    @classmethod
    def new(cls, content: str, scope: MemoryScope, memory_type: MemoryType = MemoryType.USER,
            confidence: float = 1.0) -> "Memory":
        return cls(
            id=str(uuid.uuid4()),
            content=content,
            metadata=MemoryMetadata(scope=scope, memory_type=memory_type, confidence=confidence),
        )


@dataclass
class ScoredMemory:
    memory: Memory
    score: float  # 0.0-1.0 similarity


@dataclass
class ExtractedFact:
    content: str
    entities: list[str] = field(default_factory=list)
    confidence: float = 1.0
    memory_type: MemoryType = MemoryType.USER
    embedding: Optional[list[float]] = None  # filled in by manager before store


@dataclass
class MemoryOperation:
    action: OperationType
    fact: ExtractedFact
    target_id: Optional[str] = None
    merged_content: Optional[str] = None


@dataclass
class MemoryUpdateResult:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    skipped: int = 0

    def summary(self) -> dict:
        return {
            "added": len(self.added), "updated": len(self.updated),
            "deleted": len(self.deleted), "skipped": self.skipped,
        }


class EntityType(str, Enum):
    PERSON = "PERSON"
    TOPIC = "TOPIC"
    EVENT = "EVENT"
    PREFERENCE = "PREFERENCE"
    PLACE = "PLACE"
    TOOL = "TOOL"
    SKILL = "SKILL"
    UNKNOWN = "UNKNOWN"


@dataclass
class GraphNode:
    name: str
    entity_type: EntityType = EntityType.UNKNOWN
    id: Optional[str] = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())

    @property
    def normalized_name(self) -> str:
        return self.name.lower().strip()


@dataclass
class GraphRelationship:
    source_id: str
    target_id: str
    rel_type: str
    strength: float = 1.0
    properties: dict = field(default_factory=dict)
