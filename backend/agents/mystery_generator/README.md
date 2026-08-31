# mystery_generator

Mystery Generator agent. Source = none — the user's first message is the
theme. Turns it into a crime mystery the user investigates and solves; the
agent never reveals the solution except on a final accusation or give-up.
Subagents ported ~verbatim from `AIF-CineBot`'s deepagents notebook
(planner/researcher/scenarist/validator, no tools) — the secrecy workflow on
top is new.

- `agent.py` — `make_mystery_agent(conversation_id, memory_context="")`:
  per-conversation `FilesystemBackend(root_dir=data/mysteries/<id>/,
  virtual_mode=True)` — the ONLY state carried between turns, since `/chat`
  builds a fresh agent every message. First turn: planner → researcher →
  scenarist → validator build the full case, written to `solution.md`
  (culprit/motive/method/twist — never quoted back). Later turns: `ls` +
  `read_file('solution.md')` before every response, mandatory — this exact
  discipline was added after a real bug (see `docs/PLAN.md`) where a fresh agent
  instance couldn't find a wrongly-pathed solution file and fabricated a
  wrong verdict instead of reporting the failure.

Called from `backend/online_pipeline/main.py`'s `/chat` when
`agent_type == "mystery_generator"`.
