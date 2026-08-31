# story_developer

Story Developer agent. Source = none — the user's own first message is the
brief. Stateless, like docqa. Adapted from `AIF-CineBot`'s
Google ADK film-pitch pipeline, trimmed to 3 roles (from the source's 6:
dropped casting/cinematography/marketing).

- `agent.py` — `make_story_agent(memory_context="")`: 3 subagents run
  sequentially on the first message —
  **writer** (title/genre/logline/synopsis/characters) →
  **scene-director** (one pivotal scene: slugline, atmosphere, blocking,
  dialogue) → **editor** (pacing/dialogue/consistency polish, also handles
  later-turn revision requests).

Called from `backend/online_pipeline/main.py`'s `/chat` when
`agent_type == "story_developer"`.
