"""Regression check for Mystery Generator's file-path discipline.

Real bug this guards against (see docs/PLAN.md's Done section): /chat builds
a fresh deepagents graph every turn, no persistent process state. The first
fix attempt let the agent write solution.md to a self-chosen path — the next
turn's fresh instance couldn't find it and fabricated a confident-but-wrong
verdict instead of reporting the failure. The current prompt fixes this with
a deterministic path + mandatory ls/read_file before judging, but that's
prompt-level, not a hard guarantee — worth an occasional spot-check if the
underlying model changes, hence this script.

Runs against the real app in-process (fastapi.testclient.TestClient) — real
/chat and /conversations endpoints, real backend.db writes, real subagent
LLM calls. Not a simulation, same principle as eval_retrieval.py and the
recommendation eval.

Written as a plain-assert pytest-discoverable function (test_ prefix, no
pytest import) so it costs nothing today and gets picked up for free once
the production-readiness roadmap's pytest suite happens.

Run:
    python -m backend.agents.mystery_generator.test_path_discipline
"""

import json
import logging
import re
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from backend.agents.mystery_generator.agent import MYSTERIES_DIR
from backend.online_pipeline.main import app

logger = logging.getLogger(__name__)

OPENING_PROMPT = "A theft at a small art gallery during a gala."
GIVE_UP_PROMPT = "I give up, reveal the answer."
WRONG_ACCUSATION = "I accuse Reginald Ashworth-Pemberton the Third."

_CULPRIT_RE = re.compile(r"culprit\**:?\**\s*\**([A-Z][A-Za-z'\-]+(?:\s[A-Z][A-Za-z'\-]+){0,3})", re.IGNORECASE)


def _chat(client: TestClient, conversation_id: int, message: str) -> str:
    """Sends one /chat turn, returns the fully assembled response text."""
    resp = client.post("/chat", json={"conversation_id": conversation_id, "message": message})
    resp.raise_for_status()
    text = ""
    for line in resp.text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            break
        text += json.loads(payload).get("delta", "")
    return text


def _new_mystery_conversation(client: TestClient) -> int:
    resp = client.post(
        "/conversations",
        json={"agent_type": "mystery_generator", "source_type": "brief", "source_ref": "freeform"},
    )
    resp.raise_for_status()
    return resp.json()["id"]


def _extract_culprit(solution_text: str) -> str | None:
    """Best-effort — the system prompt tells the agent to record culprit as
    a field, but nothing enforces exact formatting on free-form prose."""
    match = _CULPRIT_RE.search(solution_text)
    return match.group(1).strip() if match else None


def _shares_distinctive_words(a: str, b: str, min_len: int = 6, min_shared: int = 3) -> bool:
    """Loose "did this response actually draw from this file" check — shared
    long/distinctive words, not exact phrasing (the agent paraphrases)."""
    words_a = {w.lower() for w in re.findall(r"[A-Za-z']+", a) if len(w) >= min_len}
    words_b = {w.lower() for w in re.findall(r"[A-Za-z']+", b) if len(w) >= min_len}
    return len(words_a & words_b) >= min_shared


def test_mystery_path_discipline() -> None:
    client = TestClient(app)

    # 1-2: create a case, assert the deterministic path.
    conv_id = _new_mystery_conversation(client)
    opening = _chat(client, conv_id, OPENING_PROMPT)
    assert opening, "first turn returned an empty response"

    solution_path = MYSTERIES_DIR / str(conv_id) / "solution.md"
    assert solution_path.exists(), (
        f"solution.md not found at the deterministic path {solution_path} — "
        "the agent wrote it somewhere else (the exact original bug)"
    )
    solution_text = solution_path.read_text()
    assert solution_text.strip(), "solution.md exists but is empty"

    culprit = _extract_culprit(solution_text)
    if culprit is None:
        logger.warning("could not extract a culprit name from solution.md — skipping the accusation checks")
    elif culprit.lower() in opening.lower():
        # Regex-extracted name, best-effort — could be a false positive (a
        # red-herring suspect legitimately sharing a name), so this is a
        # warning, not a hard assert like the checks below.
        logger.warning("possible leak: extracted culprit %r appears in the first-turn public premise", culprit)

    # 3: give-up path — the response must actually reflect the real file,
    # not be fabricated from nothing (the exact original failure mode).
    reveal = _chat(client, conv_id, GIVE_UP_PROMPT)
    assert reveal, "give-up turn returned an empty response"
    assert _shares_distinctive_words(reveal, solution_text), (
        "reveal response shares no distinctive words with the actual solution.md — "
        "looks fabricated rather than read from the file"
    )

    if culprit is not None:
        # 4: correct accusation, same conversation — must confirm.
        correct_verdict = _chat(client, conv_id, f"I accuse {culprit}.")
        assert culprit.lower() in correct_verdict.lower(), (
            f"correct accusation of {culprit!r} didn't name them in the verdict"
        )

        # 5: wrong accusation, a fresh separate conversation — must deny,
        # never confirm a name that was never in *that* conversation's file.
        other_conv_id = _new_mystery_conversation(client)
        _chat(client, other_conv_id, OPENING_PROMPT)
        wrong_verdict = _chat(client, other_conv_id, WRONG_ACCUSATION)
        denial_markers = ("incorrect", "wrong", "not the culprit", "not correct", "isn't", "is not", "denied", "mistaken")
        assert any(marker in wrong_verdict.lower() for marker in denial_markers), (
            f"wrong accusation wasn't denied: {wrong_verdict!r}"
        )

    logger.info("mystery generator path-discipline check passed (culprit extracted: %s)", culprit is not None)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        test_mystery_path_discipline()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
    else:
        print("PASS")
