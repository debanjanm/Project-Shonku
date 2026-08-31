# frontend

Streamlit UI. One file: `app.py` — agent picker → source picker (skipped for
Recommendation/Story Developer/Mystery Generator, which have no real source
choice) → chat.

Built around `st.query_params` + server-fetched history instead of
`session_state` — survives a real browser refresh and a backend restart.
Sidebar lists past conversations per agent+source. Never touches backend
disk directly: charts and product images are fetched over HTTP
(`/charts/{filename}`, `/products/images/{filename}`), not read from
`data/` locally.

Run: `streamlit run frontend/app.py` (needs the backend up on `:8000`).
