import os
import time
import httpx
import streamlit as st

st.markdown("### Fraud Analyst Chatbot")
st.markdown("Answers are grounded in RBI guidelines, PCI DSS v4.0, and FinShield documentation.")

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "should_stream" not in st.session_state:
    st.session_state["should_stream"] = False

# Typing simulator for st.write_stream
def stream_text(text: str):
    for char in text:
        yield char
        time.sleep(0.005)

# Handle message submission
def handle_user_query(query: str):
    st.session_state["messages"].append({"role": "user", "content": query})
    st.session_state["should_stream"] = True
    
    with st.spinner("Retrieving relevant policy documents and generating answer..."):
        try:
            backend_url = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000").rstrip("/")
            with httpx.Client(timeout=30.0) as client:
                r = client.post(f"{backend_url}/chat", json={"query": query})
                if r.status_code == 200:
                    res = r.json()
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": res["answer"],
                        "sources": res.get("sources", [])
                    })
                else:
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": f"API Error (status code {r.status_code}): {r.text}",
                        "sources": []
                    })
        except Exception as exc:
            st.session_state["messages"].append({
                "role": "assistant",
                "content": f"Error: Could not retrieve answer. Details: {str(exc)}",
                "sources": []
            })

# Sidebar clear conversation button
with st.sidebar:
    if st.button("Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["should_stream"] = False
        st.rerun()

# 4 pre-built quick question buttons
st.markdown("#### Suggested Questions")
col1, col2, col3, col4 = st.columns(4)
if col1.button("What is a card-testing attack?", use_container_width=True):
    handle_user_query("What is a card-testing attack?")
    st.rerun()
if col2.button("RBI fraud reporting requirements?", use_container_width=True):
    handle_user_query("RBI fraud reporting requirements?")
    st.rerun()
if col3.button("PCI DSS velocity thresholds?", use_container_width=True):
    handle_user_query("PCI DSS velocity thresholds?")
    st.rerun()
if col4.button("How to handle a false positive?", use_container_width=True):
    handle_user_query("How to handle a false positive?")
    st.rerun()

st.divider()

# Render chat history
for i, msg in enumerate(st.session_state["messages"]):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and i == len(st.session_state["messages"]) - 1 and st.session_state["should_stream"]:
            st.session_state["should_stream"] = False
            st.write_stream(stream_text(msg["content"]))
        else:
            st.write(msg["content"])
            
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Source Documents Cited"):
                for src in msg["sources"]:
                    st.write(f"- {src}")

# Chat input
user_input = st.chat_input("Ask a regulatory compliance or fraud policy question...")
if user_input:
    handle_user_query(user_input)
    st.rerun()
