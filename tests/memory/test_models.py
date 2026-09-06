from backend.memory.models import MemoryScope


def test_to_filter_user_only():
    scope = MemoryScope(user_id="u1")
    assert scope.to_filter() == {"user_id": {"$eq": "u1"}}


def test_to_filter_user_and_agent():
    scope = MemoryScope(user_id="u1", agent_id="docqa")
    assert scope.to_filter() == {
        "$and": [{"user_id": {"$eq": "u1"}}, {"agent_id": {"$eq": "docqa"}}],
    }


def test_to_filter_never_includes_run_id():
    scope = MemoryScope(user_id="u1", agent_id="docqa", run_id="42")
    filter_ = scope.to_filter()
    assert "run_id" not in str(filter_)


def test_to_metadata_defaults_none_to_empty_string():
    scope = MemoryScope(user_id="u1")
    assert scope.to_metadata() == {"user_id": "u1", "agent_id": "", "run_id": ""}


def test_to_metadata_all_fields_set():
    scope = MemoryScope(user_id="u1", agent_id="docqa", run_id="42")
    assert scope.to_metadata() == {"user_id": "u1", "agent_id": "docqa", "run_id": "42"}
