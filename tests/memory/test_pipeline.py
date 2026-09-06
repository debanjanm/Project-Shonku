import json
from datetime import datetime, timedelta

import pytest

from backend.memory import pipeline
from backend.memory.models import (
    ExtractedFact, Memory, MemoryMetadata, MemoryScope, MemoryType, OperationType, ScoredMemory,
)

SCOPE = MemoryScope(user_id="u1", agent_id="docqa")


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeModel:
    """Stand-in for ChatOpenAI: .invoke(messages, temperature=...) -> _FakeResponse."""

    def __init__(self, content: str | None = None, side_effect: Exception | None = None):
        self._content = content
        self._side_effect = side_effect

    def invoke(self, messages, temperature=0.0):
        if self._side_effect:
            raise self._side_effect
        return _FakeResponse(self._content)


def _patch_model(monkeypatch, content: str | None = None, side_effect: Exception | None = None):
    monkeypatch.setattr(pipeline, "_get_model", lambda: _FakeModel(content, side_effect))


# --- extract_facts ---------------------------------------------------------

def test_extract_facts_normal(monkeypatch):
    payload = json.dumps({
        "facts": [
            {"content": "The user likes tea.", "entities": ["tea"], "confidence": 0.9, "memory_type": "user"},
        ]
    })
    _patch_model(monkeypatch, content=payload)
    facts = pipeline.extract_facts("I like tea.", "Noted!", SCOPE)
    assert len(facts) == 1
    assert facts[0].content == "The user likes tea."
    assert facts[0].confidence == 0.9
    assert facts[0].memory_type == MemoryType.USER


def test_extract_facts_strips_markdown_fence(monkeypatch):
    payload = "```json\n" + json.dumps({"facts": [{"content": "The user likes tea."}]}) + "\n```"
    _patch_model(monkeypatch, content=payload)
    facts = pipeline.extract_facts("I like tea.", "Noted!", SCOPE)
    assert len(facts) == 1
    assert facts[0].content == "The user likes tea."


def test_extract_facts_drops_example_leak(monkeypatch):
    # Verbatim match (case-insensitive) against a few-shot example's output —
    # the regression this guard exists for (see docs/PLAN.md's Done section).
    payload = json.dumps({"facts": [{"content": "The user's name is Sarah Connor."}]})
    _patch_model(monkeypatch, content=payload)
    facts = pipeline.extract_facts("hi", "hello", SCOPE)
    assert facts == []


def test_extract_facts_skips_malformed_item(monkeypatch):
    payload = json.dumps({"facts": [{"entities": ["x"]}]})  # missing required "content"
    _patch_model(monkeypatch, content=payload)
    facts = pipeline.extract_facts("hi", "hello", SCOPE)
    assert facts == []


def test_extract_facts_returns_empty_on_llm_error(monkeypatch):
    _patch_model(monkeypatch, side_effect=RuntimeError("boom"))
    facts = pipeline.extract_facts("hi", "hello", SCOPE)
    assert facts == []


# --- compute_operations -----------------------------------------------------

def _fact(content="fact") -> ExtractedFact:
    return ExtractedFact(content=content)


def test_compute_operations_empty_facts_returns_empty():
    assert pipeline.compute_operations([], []) == []


def test_compute_operations_no_existing_memories_defaults_to_add():
    facts = [_fact("a"), _fact("b")]
    ops = pipeline.compute_operations(facts, [])
    assert [op.action for op in ops] == [OperationType.ADD, OperationType.ADD]


def _scored_memory(content="existing") -> ScoredMemory:
    memory = Memory.new(content, SCOPE)
    return ScoredMemory(memory=memory, score=0.8)


def test_compute_operations_uncovered_fact_defaults_to_add(monkeypatch):
    payload = json.dumps({"operations": []})  # LLM covers none of the facts
    _patch_model(monkeypatch, content=payload)
    facts = [_fact("a")]
    ops = pipeline.compute_operations(facts, [_scored_memory()])
    assert [op.action for op in ops] == [OperationType.ADD]


def test_compute_operations_parses_update(monkeypatch):
    existing = _scored_memory()
    payload = json.dumps({
        "operations": [
            {"fact_index": 0, "action": "UPDATE", "target_memory_id": existing.memory.id,
             "merged_content": "merged"},
        ]
    })
    _patch_model(monkeypatch, content=payload)
    facts = [_fact("a")]
    ops = pipeline.compute_operations(facts, [existing])
    assert len(ops) == 1
    assert ops[0].action == OperationType.UPDATE
    assert ops[0].target_id == existing.memory.id
    assert ops[0].merged_content == "merged"


def test_compute_operations_falls_back_to_add_on_llm_error(monkeypatch):
    _patch_model(monkeypatch, side_effect=RuntimeError("boom"))
    facts = [_fact("a"), _fact("b")]
    ops = pipeline.compute_operations(facts, [_scored_memory()])
    assert [op.action for op in ops] == [OperationType.ADD, OperationType.ADD]


# --- synthesize (pure, no LLM) ----------------------------------------------

def _scored(content: str, score: float, updated_at: datetime) -> ScoredMemory:
    metadata = MemoryMetadata(scope=SCOPE, updated_at=updated_at)
    return ScoredMemory(memory=Memory(id=content, content=content, metadata=metadata), score=score)


def test_synthesize_empty_below_threshold_returns_empty_string():
    now = datetime.utcnow()
    assert pipeline.synthesize([_scored("low", 0.1, now)], min_score=0.4) == ""


def test_synthesize_filters_and_formats():
    now = datetime.utcnow()
    memories = [_scored("high", 0.9, now), _scored("low", 0.1, now)]
    result = pipeline.synthesize(memories, min_score=0.4)
    assert "high" in result
    assert "low" not in result
    assert result.startswith(pipeline._CONTEXT_HEADER)


def test_synthesize_ties_break_on_updated_at():
    now = datetime.utcnow()
    older = _scored("older", 0.85, now - timedelta(days=1))
    newer = _scored("newer", 0.85, now)
    result = pipeline.synthesize([older, newer], min_score=0.4)
    assert result.index("newer") < result.index("older")


def test_synthesize_truncates_to_max_memories():
    now = datetime.utcnow()
    memories = [_scored(f"m{i}", 0.9, now) for i in range(15)]
    result = pipeline.synthesize(memories, max_memories=3, min_score=0.4)
    item_lines = [line for line in result.splitlines() if line and line[0].isdigit()]
    assert len(item_lines) == 3
