# Project Shonku MVP Architecture

The MVP chat app is built in Python:

- **Agent**: `deepagents` (built on LangGraph) drives the conversation and
  decides when to call the knowledge-base search tool.
- **Retrieval**: each Knowledge Base is embedded and indexed into its own
  local FAISS vector index. Embeddings are generated locally with a
  `sentence-transformers` model, so no external API key is needed for
  retrieval.
- **Generation**: chat responses are generated through OpenRouter, which
  exposes an OpenAI-compatible API in front of many model providers.
- **Backend**: FastAPI exposes `/kbs` to list available Knowledge Bases and
  `/chat` to stream a grounded answer for a message.
- **Frontend**: a Streamlit app provides the Knowledge Base picker and chat
  interface, calling the FastAPI backend over HTTP.

There is no database in the MVP — FAISS indexes are files on disk, and
conversation history lives only in the browser session while you're chatting.
