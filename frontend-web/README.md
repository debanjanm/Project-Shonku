# frontend-web

Next.js (App Router, TypeScript) chat UI — Tailwind + shadcn/ui. Replaces
`frontend/app.py` (Streamlit), full feature parity, verified live against
the real backend. Talks to the same FastAPI backend on `:8000`, no backend
changes needed.

Run: `npm install && npm run dev` — serves on `:3000`. Needs
`NEXT_PUBLIC_BACKEND_URL` in `.env.local` (defaults to `http://localhost:8000`
if unset).

- `app/page.tsx` — thin `<Suspense>` wrapper (required for `useSearchParams`)
- `app/chat-app.tsx` — everything for the Chat tab: all state, all
  data-fetching effects, the agent/source-picker derivation, the
  conversation-lock rule, `handleSend`
- `app/kbs/page.tsx` — Knowledge Base admin tab: list + create + select on
  the left, selected KB's documents/upload/ingest/delete on the right
- `components/` — presentational pieces: `agent-picker`, `source-picker`,
  `conversation-history`, `chat-messages`, `chat-input`, `product-image-grid`,
  `top-nav` (shared header, both tabs), `kb-list`, `kb-detail`,
  `kb-create-dialog`, plus `ui/` (shadcn primitives — copy-in, not a runtime
  dependency)
- `lib/api.ts` — typed fetch wrappers for every backend endpoint, plus
  `streamChat()`: a hand-written `fetch()` + `ReadableStream` SSE parser,
  since `/chat` is POST and the browser's native `EventSource` only does GET
- `lib/types.ts` — `Agent`/`Kb`/`Conversation`/`Message`/`KbDocument`/`IngestResult`,
  matching the backend's actual JSON shapes exactly (`Message` has no `id`
  field — `backend/db.py`'s `get_messages` only selects
  `role`/`content`/`created_at`, a real bug caught during testing when the
  type assumed one existed)

State model: `conversation_id` in the URL query string is the source of
truth for the active conversation (mirrors Streamlit's `st.query_params`) —
survives a page refresh. A `loadedConversationIdRef` guards against the
URL-driven "load conversation" effect racing `handleSend`'s optimistic local
state when a new conversation gets created mid-send (another real bug caught
live: creating a conversation changes the URL, which would otherwise
re-trigger a redundant fetch of the same conversation while it's still
streaming).
