"""Streamlit Dashboard Entry Point for FinShield AI."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import httpx
import streamlit as st

# Set page config without emojis
st.set_page_config(
    page_title="FinShield",
    page_icon="shield",
    layout="wide"
)

# Common styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117, #161b22);
        border-right: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar & System Status Check
# ---------------------------------------------------------------------------
api_status_html = '<span style="color: #f78166; font-weight: bold;">Offline</span>'
model_version = "unknown"
uptime_str = "N/A"
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000").rstrip("/")

try:
    with httpx.Client(timeout=1.0) as client:
        r = client.get(f"{BACKEND_API_URL}/health")
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "healthy" or data.get("status") == "ok":
                api_status_html = '<span style="color: #3fb950; font-weight: bold;">Online</span>'
                uptime_sec = data.get("uptime_seconds", 0.0)
                if uptime_sec > 3600:
                    uptime_str = f"{uptime_sec / 3600:.1f} hours"
                elif uptime_sec > 60:
                    uptime_str = f"{uptime_sec / 60:.1f} mins"
                else:
                    uptime_str = f"{uptime_sec:.0f} secs"
                
                # Fetch model version
                r_info = client.get(f"{BACKEND_API_URL}/model/info")
                if r_info.status_code == 200:
                    info_data = r_info.json()
                    model_version = info_data.get("model_name", "xgboost_tuned")
except Exception:
    pass

with st.sidebar:
    st.markdown("### FinShield")
    st.markdown("Production Fraud Intelligence")
    st.divider()
    
    st.markdown("#### System Status")
    st.markdown(f"API Status: {api_status_html}", unsafe_allow_html=True)
    st.markdown(f"Model Version: **{model_version}**")
    st.markdown(f"API Uptime: **{uptime_str}**")
    st.divider()

# ---------------------------------------------------------------------------
# Routing and Navigation
# ---------------------------------------------------------------------------
monitor_page = st.Page("pages/monitor.py", title="Live Monitor")
performance_page = st.Page("pages/monitor_performance.py", title="Model Performance")
explainer_page = st.Page("pages/explainer.py", title="Transaction Explainer")
chatbot_page = st.Page("pages/chatbot.py", title="Fraud Analyst Chatbot")

pg = st.navigation({
    "Navigation": [
        monitor_page,
        performance_page,
        explainer_page,
        chatbot_page
    ]
})

# Consistent Header on every page
st.markdown("<h1 style='margin-bottom: 0px;'>FinShield</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #8b949e; font-size: 1.1rem; margin-top: 5px; margin-bottom: 10px;'>AI-Powered Fraud Detection Platform</p>", unsafe_allow_html=True)
st.divider()

# Run the selected page
pg.run()
