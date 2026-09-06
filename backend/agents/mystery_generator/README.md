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
  wrong verdict instead of reporting the failure. The planner/validator
  subagents also get a `fair-play-mystery-design` skill — since a second,
  stable backend can't just be bolted on alongside the per-conversation
  case-file one, both are combined behind one
  `deepagents.backends.composite.CompositeBackend(default=case_backend,
  routes={"/skills/": skills_backend})` so `solution.md` and `/skills/...`
  route correctly through the same shared backend object.
- `skills/fair-play-mystery-design/` — classic detective-fiction fairness
  checklist (every clue shown before the reveal, red herrings plausible but
  explicable, exactly one logically-supportable culprit) for the
  planner/validator subagents to consult during case creation.
- `test_path_discipline.py` — regression check for that exact bug: creates a
  real mystery via the actual `/chat` endpoint (`TestClient`, in-process),
  asserts `solution.md` lands at the deterministic path, and that a later
  turn's reveal/verdict genuinely reflects the file instead of being
  fabricated. Plain `assert`s, pytest-discoverable but doesn't need pytest
  installed to run. Run: `python -m backend.agents.mystery_generator.test_path_discipline`
- `eval_mystery_generator.py` — scored quality eval (LLM-as-judge, no ground
  truth for generative writing), complementing `test_path_discipline.py`'s
  single-bug regression guard: runs 10 varied seed themes through the real
  agent, judges each case's public premise against its private `solution.md`
  on 5 boolean criteria mirrored verbatim from the `fair-play-mystery-design`
  skill (exactly one culprit, no culprit/motive leak in the premise, every
  clue shown before the reveal, red herrings plausible, internal
  consistency). Synthetic high-offset `conversation_id`s, self-cleans its
  `data/mysteries/<id>/` dirs after each run. Run:
  `python -m backend.agents.mystery_generator.eval_mystery_generator`

Called from `backend/online_pipeline/main.py`'s `/chat` when
`agent_type == "mystery_generator"`.
