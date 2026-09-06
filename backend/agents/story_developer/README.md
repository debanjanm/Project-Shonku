# story_developer

Story Developer agent. Source = none — the user's own first message is the
brief. Stateless, like docqa. Adapted from `AIF-CineBot`'s
Google ADK film-pitch pipeline, trimmed to 3 roles (from the source's 6:
dropped casting/cinematography/marketing).

- `agent.py` — `make_story_agent(memory_context="")`: 3 subagents run
  sequentially on the first message —
  **writer** (title/genre/logline/synopsis/characters) →
  **scene-director** (one pivotal scene: slugline, atmosphere, blocking,
  dialogue; has its own `screenplay-formatting` skill) → **editor**
  (pacing/dialogue/consistency polish, also handles later-turn revision
  requests).
- `skills/screenplay-formatting/` — deepagents Agent Skill scoped to the
  scene-director subagent only: industry-standard slugline/action-line/
  dialogue conventions, detailed enough not to belong permanently inlined
  in the always-loaded system prompt.
- `eval_story_developer.py` — LLM-as-judge eval (no ground truth exists for
  generative writing, so Recall@k/MRR doesn't apply): runs a fixed set of
  10 varied seed ideas through the real agent, a second LLM scores each
  output against 7 boolean criteria drawn from the system prompt + the
  screenplay-formatting skill (title/genre, logline, three-act synopsis,
  3-4 characters, scene slugline, scene dialogue, coherence). Reports
  per-criterion + overall pass rate. Run:
  `python -m backend.agents.story_developer.eval_story_developer`

Called from `backend/online_pipeline/main.py`'s `/chat` when
`agent_type == "story_developer"`.
