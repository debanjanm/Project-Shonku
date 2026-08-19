"""Streamlit chat UI: pick a KB, chat with it. Talks to the FastAPI backend."""

import json
import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("SHONKU_BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Project Shonku", page_icon="💬", layout="wide")
st.title("Project Shonku — Chat")


@st.cache_data(ttl=60)
def get_kbs():
    resp = requests.get(f"{BACKEND_URL}/kbs", timeout=10)
    resp.raise_for_status()
    return resp.json()


kbs = get_kbs()
kb_slug_by_name = {kb["name"]: kb["slug"] for kb in kbs}

with st.sidebar:
    st.header("Knowledge Base")
    selected_name = st.selectbox("Choose a knowledge base", list(kb_slug_by_name.keys()))
    selected_slug = kb_slug_by_name[selected_name]
    kb_info = next(kb for kb in kbs if kb["slug"] == selected_slug)
    st.caption(kb_info["description"])
    if st.button("New chat"):
        st.session_state.messages = []

if "messages" not in st.session_state or st.session_state.get("kb_slug") != selected_slug:
    st.session_state.messages = []
    st.session_state.kb_slug = selected_slug

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask something about this knowledge base...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        history = st.session_state.messages[:-1]
        with requests.post(
            f"{BACKEND_URL}/chat",
            json={"kb_slug": selected_slug, "message": prompt, "history": history},
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8")
                if not decoded.startswith("data: "):
                    continue
                data = decoded[len("data: "):]
                if data == "[DONE]":
                    break
                delta = json.loads(data).get("delta", "")
                full_text += delta
                placeholder.markdown(full_text)
        st.session_state.messages.append({"role": "assistant", "content": full_text})
