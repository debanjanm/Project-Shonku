# Changelog

Format loosely follows [Keep a Changelog](https://keepachangelog.com/). No
version tags yet — this project doesn't cut releases, so entries are grouped
by commit instead. Full engineering narrative (why, what broke, what got
verified live) stays in [PLAN.md](PLAN.md) — this file is the terse index.

## [Unreleased]

### Removed
- Data Analyst agent (`backend/agents/data_analyst/`) — tabular-data work now
  lives in a separate dedicated repo. Also removed its only touchpoints:
  `/datasets`, `/uploads`, `/charts/{filename}` endpoints, the frontend's
  dataset/CSV picker and chart rendering, and the
  `pandas`/`scikit-learn`/`matplotlib`/`seaborn` dependencies

### Added
- `docs/` folder — `PLAN.md` moved here, this changelog added alongside it
- README for every backend component (`docqa`, `recommendation`,
  `story_developer`, `mystery_generator`, `memory`) plus `backend/`,
  `frontend/`, `data/`
- 8-phase production-readiness roadmap in `docs/PLAN.md`, ranked by
  interview signal vs. effort (tests, Docker, CI, security, auth, docs
  polish, deployment)

### Fixed
- `.gitignore` gap: `.env copy`/`.env copy.example` (space, not a dot)
  matched neither the old `.env` nor `.env.*` pattern and held live
  credentials — replaced with a single `.env*` pattern. No secret was ever
  actually committed (confirmed via `git log --diff-filter=A`)

## [da19de3] — 2026-08-26 — "stable version pushed"
Cross-agent memory layer (Extract → Search → Update → Store on ChromaDB +
Neo4j, ported from `AIF-Remembrane`, scoped per-agent), the five (now four)
agents restructured under `backend/agents/`, hybrid FAISS+BM25 retrieval,
SQLite conversation persistence, arXiv/SEC downloaders. Full detail in
`PLAN.md`'s Done section.

## [365833f] — 2026-08-20 — "Structure Updated"
Initial MVP scaffold: FastAPI backend, Streamlit frontend, Document Q&A
agent.

## [711f29f] — 2026-08-07 — "Initial commit"
