"""Streamlit chat UI: pick an agent type, then its source (KB or dataset/CSV),
then chat. DB (via backend) is the source of truth for conversation history,
not session_state — survives a real browser refresh via conversation_id in
the URL query string.
"""

import json
import os
import re

import requests
import streamlit as st

BACKEND_URL = os.environ.get("SHONKU_BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Project Shonku", page_icon="💬", layout="wide")
st.title("Project Shonku — Chat")


@st.cache_data(ttl=60)
def get_agents():
    resp = requests.get(f"{BACKEND_URL}/agents", timeout=10)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=60)
def get_kbs():
    resp = requests.get(f"{BACKEND_URL}/kbs", timeout=10)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=60)
def get_datasets():
    resp = requests.get(f"{BACKEND_URL}/datasets", timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_conversation(conversation_id: str):
    resp = requests.get(f"{BACKEND_URL}/conversations/{conversation_id}", timeout=10)
    return resp.json() if resp.status_code == 200 else None


def list_conversations(agent_type: str, source_type: str, source_ref: str):
    resp = requests.get(
        f"{BACKEND_URL}/conversations",
        params={"agent_type": agent_type, "source_type": source_type, "source_ref": source_ref},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def get_messages(conversation_id: int):
    resp = requests.get(f"{BACKEND_URL}/conversations/{conversation_id}/messages", timeout=10)
    resp.raise_for_status()
    return resp.json()


def create_conversation(agent_type: str, source_type: str, source_ref: str):
    resp = requests.post(
        f"{BACKEND_URL}/conversations",
        json={"agent_type": agent_type, "source_type": source_type, "source_ref": source_ref},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def upload_csv(uploaded_file) -> str:
    resp = requests.post(
        f"{BACKEND_URL}/uploads",
        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["path"]


def render_charts(content: str) -> None:
    names = list(dict.fromkeys(re.findall(r"chart_[\w\-]+\.png", content)))
    if not names:
        return
    cols = st.columns(min(len(names), 2))
    for i, name in enumerate(names):
        with cols[i % 2]:
            st.image(f"{BACKEND_URL}/charts/{name}", use_container_width=True)


def render_product_images(content: str) -> None:
    names = list(dict.fromkeys(re.findall(r"\[Image:\s*([^\]]+)\]", content)))
    if not names:
        return
    cols = st.columns(min(len(names), 3))
    for i, name in enumerate(names):
        with cols[i % 3]:
            st.image(f"{BACKEND_URL}/products/images/{name.strip()}", use_container_width=True)


agents = get_agents()
agent_name_by_id = {a["name"]: a["id"] for a in agents}
agent_names = list(agent_name_by_id.keys())

conversation_id = st.query_params.get("conversation_id")
active_conversation = get_conversation(conversation_id) if conversation_id else None

default_agent_type = active_conversation["agent_type"] if active_conversation else st.session_state.get("last_agent_type", agents[0]["id"])
default_agent_index = [a["id"] for a in agents].index(default_agent_type) if default_agent_type in [a["id"] for a in agents] else 0

with st.sidebar:
    st.header("Agent")
    selected_agent_name = st.selectbox("Choose an agent", agent_names, index=default_agent_index)
    agent_type = agent_name_by_id[selected_agent_name]
    st.session_state.last_agent_type = agent_type
    st.caption(next(a["description"] for a in agents if a["id"] == agent_type))

    st.divider()

    if agent_type == "docqa":
        st.header("Knowledge Base")
        kbs = get_kbs()
        kb_slug_by_name = {kb["name"]: kb["slug"] for kb in kbs}
        default_slug = active_conversation["source_ref"] if active_conversation and active_conversation["agent_type"] == "docqa" else st.session_state.get("last_kb_slug", kbs[0]["slug"])
        default_index = list(kb_slug_by_name.values()).index(default_slug) if default_slug in kb_slug_by_name.values() else 0
        selected_name = st.selectbox("Choose a knowledge base", list(kb_slug_by_name.keys()), index=default_index)
        selected_slug = kb_slug_by_name[selected_name]
        st.session_state.last_kb_slug = selected_slug
        kb_info = next(kb for kb in kbs if kb["slug"] == selected_slug)
        st.caption(kb_info["description"])
        source_type, source_ref = "kb", selected_slug
    elif agent_type == "recommendation":
        source_type, source_ref = "catalog", "fashion-500"
        st.caption("Searching a 425-item fashion product catalog.")
    elif agent_type in ("story_developer", "mystery_generator"):
        source_type, source_ref = "brief", "freeform"
        st.caption("No source to pick — just describe your idea in the chat.")
    else:
        st.header("Data Source")
        source_choice = st.radio("Choose data source", ["Built-in dataset", "Upload CSV"])
        if source_choice == "Built-in dataset":
            datasets = get_datasets()
            dataset_by_name = {d["name"]: d["description"] for d in datasets}
            dataset_name = st.selectbox(
                "Select dataset", list(dataset_by_name.keys()),
                index=list(dataset_by_name.keys()).index(st.session_state.get("last_dataset", "titanic"))
                if st.session_state.get("last_dataset", "titanic") in dataset_by_name else 0,
                format_func=lambda x: dataset_by_name.get(x, x),
            )
            st.session_state.last_dataset = dataset_name
            source_type, source_ref = "dataset", dataset_name
        else:
            uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
            if uploaded_file is not None:
                st.session_state.uploaded_csv_path = upload_csv(uploaded_file)
                st.success(f"Loaded: {uploaded_file.name}")
            source_type = "csv"
            source_ref = st.session_state.get("uploaded_csv_path")
            if not source_ref:
                st.info("Upload a CSV file to begin.")

    # Switching agent/source away from the loaded conversation leaves it
    # (a conversation stays locked to the agent+source it started with).
    if active_conversation and (active_conversation["agent_type"] != agent_type or active_conversation["source_ref"] != source_ref):
        del st.query_params["conversation_id"]
        st.rerun()

    st.divider()
    if st.button("New chat", disabled=not source_ref):
        new_conversation = create_conversation(agent_type, source_type, source_ref)
        st.query_params["conversation_id"] = str(new_conversation["id"])
        st.rerun()

    if source_ref:
        st.divider()
        st.caption("History")
        for conv in list_conversations(agent_type, source_type, source_ref):
            label = conv["title"] or "New chat"
            is_active = active_conversation is not None and conv["id"] == active_conversation["id"]
            if st.button(("→ " if is_active else "") + label, key=f"conv-{conv['id']}", use_container_width=True):
                st.query_params["conversation_id"] = str(conv["id"])
                st.rerun()

if active_conversation:
    for msg in get_messages(active_conversation["id"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_charts(msg["content"])
                render_product_images(msg["content"])
else:
    st.info("Start typing to begin a new chat, or pick one from the sidebar.")

CHAT_PLACEHOLDERS = {
    "story_developer": "Describe your story idea...",
    "mystery_generator": "Describe your story idea — I'll turn it into a mystery you solve...",
}
prompt = st.chat_input(CHAT_PLACEHOLDERS.get(agent_type, "Ask something..."), disabled=not source_ref)
if prompt:
    if active_conversation is None:
        active_conversation = create_conversation(agent_type, source_type, source_ref)
        st.query_params["conversation_id"] = str(active_conversation["id"])

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        chart_names: list[str] = []
        product_image_names: list[str] = []
        with requests.post(
            f"{BACKEND_URL}/chat",
            json={"conversation_id": active_conversation["id"], "message": prompt},
            stream=True,
            timeout=180,
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
                payload = json.loads(data)
                if "delta" in payload:
                    full_text += payload["delta"]
                    placeholder.markdown(full_text)
                elif "chart_paths" in payload:
                    chart_names = payload["chart_paths"]
                elif "product_images" in payload:
                    product_image_names = payload["product_images"]
        if chart_names:
            cols = st.columns(min(len(chart_names), 2))
            for i, name in enumerate(chart_names):
                with cols[i % 2]:
                    st.image(f"{BACKEND_URL}/charts/{name}", use_container_width=True)
        if product_image_names:
            cols = st.columns(min(len(product_image_names), 3))
            for i, name in enumerate(product_image_names):
                with cols[i % 3]:
                    st.image(f"{BACKEND_URL}/products/images/{name}", use_container_width=True)

    st.rerun()
