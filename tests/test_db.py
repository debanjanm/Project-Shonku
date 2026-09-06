import pytest

from backend import db


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_shonku.db")
    db.init_db()


def test_create_and_get_conversation():
    conv_id = db.create_conversation("docqa", "kb", "hr-policies")
    conv = db.get_conversation(conv_id)
    assert conv["agent_type"] == "docqa"
    assert conv["source_type"] == "kb"
    assert conv["source_ref"] == "hr-policies"
    assert conv["title"] == ""


def test_get_conversation_missing_returns_none():
    assert db.get_conversation(999) is None


def test_append_message_and_get_messages_ordered():
    conv_id = db.create_conversation("docqa", "kb", "hr-policies")
    db.append_message(conv_id, "user", "hello")
    db.append_message(conv_id, "assistant", "hi there")
    messages = db.get_messages(conv_id)
    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "hello"),
        ("assistant", "hi there"),
    ]


def test_append_message_bumps_updated_at():
    conv_id = db.create_conversation("docqa", "kb", "hr-policies")
    before = db.get_conversation(conv_id)["updated_at"]
    db.append_message(conv_id, "user", "hello")
    after = db.get_conversation(conv_id)["updated_at"]
    assert after >= before


def test_list_conversations_filters():
    db.create_conversation("docqa", "kb", "hr-policies")
    db.create_conversation("docqa", "kb", "sec_10q")
    db.create_conversation("recommendation", "catalog", "fixed")

    assert len(db.list_conversations()) == 3
    assert len(db.list_conversations(agent_type="docqa")) == 2
    assert len(db.list_conversations(source_ref="sec_10q")) == 1
    assert len(db.list_conversations(agent_type="recommendation", source_type="catalog")) == 1
    assert len(db.list_conversations(agent_type="story_developer")) == 0


def test_set_title_if_unset_only_sets_once():
    conv_id = db.create_conversation("docqa", "kb", "hr-policies")
    text = "  What is the PTO policy?  " + "x" * 100
    db.set_title_if_unset(conv_id, text)
    title = db.get_conversation(conv_id)["title"]
    assert title == text.strip()[:60]

    db.set_title_if_unset(conv_id, "a different message")
    assert db.get_conversation(conv_id)["title"] == title  # unchanged
