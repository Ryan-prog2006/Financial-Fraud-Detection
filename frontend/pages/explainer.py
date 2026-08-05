"""Transaction explainer page for FinShield Dashboard."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import httpx
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.markdown("### Transaction Explainer")
st.markdown("Score a transaction and view a detailed SHAP feature explanation and regulatory reference.")

# Load presets (cached)
@st.cache_data(show_spinner=False)
def load_data_presets() -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    csv_path = "sample_creditcard.csv"
    if not os.path.exists(csv_path):
        csv_path = "../backend/data/processed/creditcard_processed.csv"
    try:
        df = pd.read_csv(csv_path)
        fraud_df = df[df["Class"] == 1]
        legit_df = df[df["Class"] == 0]
        return fraud_df, legit_df
    except Exception:
        return None, None

fraud_df, legit_df = load_data_presets()

# Fetch scaler parameters for unscaling
scaler_mean = 88.3496193
scaler_scale = 250.11967014
try:
    import pickle
    scaler_path = "scaler.pkl"
    if not os.path.exists(scaler_path):
        scaler_path = "../backend/data/processed/scaler.pkl"
    with open(scaler_path, "rb") as fh:
        scaler = pickle.load(fh)
        scaler_mean = float(scaler.mean_[0])
        scaler_scale = float(scaler.scale_[0])
except Exception:
    pass

# Initialize session state for input fields
if "exp_amount" not in st.session_state:
    st.session_state["exp_amount"] = 85.0
if "exp_time_hour" not in st.session_state:
    st.session_state["exp_time_hour"] = 12.0
if "exp_is_night" not in st.session_state:
    st.session_state["exp_is_night"] = 0
if "exp_is_weekend" not in st.session_state:
    st.session_state["exp_is_weekend"] = 0
for v in range(1, 29):
    key = f"exp_v{v}"
    if key not in st.session_state:
        st.session_state[key] = 0.0

# Input method selection
input_method = st.radio(
    "Select Input Method",
    ["Enter transaction manually", "Load random sample from dataset"],
    horizontal=True
)

if input_method == "Load random sample from dataset":
    if fraud_df is not None and len(fraud_df) > 0:
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Load random FRAUD", use_container_width=True):
                row = fraud_df.sample(n=1).iloc[0]
                st.session_state["exp_amount"] = round(float(row["Amount"] * scaler_scale + scaler_mean), 2)
                st.session_state["exp_time_hour"] = round(float(row["Hour"]), 1)
                st.session_state["exp_is_night"] = int(row["Is_night"])
                st.session_state["exp_is_weekend"] = int(row["Is_weekend"])
                for v in range(1, 29):
                    st.session_state[f"exp_v{v}"] = float(row[f"V{v}"])
                st.toast("Random FRAUD transaction loaded.")
        with col_btn2:
            if st.button("Load random LEGIT", use_container_width=True):
                row = legit_df.sample(n=1).iloc[0]
                st.session_state["exp_amount"] = round(float(row["Amount"] * scaler_scale + scaler_mean), 2)
                st.session_state["exp_time_hour"] = round(float(row["Hour"]), 1)
                st.session_state["exp_is_night"] = int(row["Is_night"])
                st.session_state["exp_is_weekend"] = int(row["Is_weekend"])
                for v in range(1, 29):
                    st.session_state[f"exp_v{v}"] = float(row[f"V{v}"])
                st.toast("Random LEGIT transaction loaded.")
    else:
        st.warning("Processed dataset not found or empty. Using manual inputs only.")

# Transaction Input Form
st.divider()
with st.form("explainer_form"):
    st.markdown("#### Transaction Input parameters")
    
    col_amt, col_hour, col_night, col_wkd = st.columns(4)
    with col_amt:
        amount_in = st.number_input("Amount (euros)", min_value=0.01, value=st.session_state["exp_amount"], step=0.01)
    with col_hour:
        hour_in = st.number_input("Time Hour", min_value=0.0, max_value=23.9, value=st.session_state["exp_time_hour"], step=0.1)
    with col_night:
        night_in = st.selectbox("Is Night", [0, 1], index=int(st.session_state["exp_is_night"]))
    with col_wkd:
        wkd_in = st.selectbox("Is Weekend", [0, 1], index=int(st.session_state["exp_is_weekend"]))
        
    st.markdown("#### PCA Components (V1 - V28)")
    v_cols = st.columns(7)
    v_inputs = {}
    for v in range(1, 29):
        col_idx = (v - 1) % 7
        with v_cols[col_idx]:
            v_inputs[f"V{v}"] = st.number_input(f"V{v}", value=st.session_state[f"exp_v{v}"], format="%.4f")
            
    submit_btn = st.form_submit_button("Analyse Transaction", use_container_width=True, type="primary")

if submit_btn:
    # Prepare payload
    payload = {
        **v_inputs,
        "time_hour": hour_in,
        "amount": amount_in,
        "is_night": int(night_in),
        "is_weekend": int(wkd_in)
    }
    
    # Query API /predict
    with st.spinner("Analyzing transaction..."):
        try:
            with httpx.Client(timeout=30.0) as client:
                backend_url = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000").rstrip("/")
                r = client.post(f"{backend_url}/predict", json=payload)
                if r.status_code == 200:
                    result = r.json()
                    
                    st.divider()
                    st.markdown("### Analysis Results")
                    
                    col_gauge, col_box = st.columns(2)
                    
                    with col_box:
                        risk_level = result["risk_level"]
                        risk_colors = {
                            "LOW": {"bg": "#1a3a1e", "text": "#3fb950"},
                            "MEDIUM": {"bg": "#2a2a10", "text": "#e3b341"},
                            "HIGH": {"bg": "#3d2a10", "text": "#ffa657"},
                            "CRITICAL": {"bg": "#3d1f1f", "text": "#f78166"}
                        }
                        color_info = risk_colors.get(risk_level, {"bg": "#21262d", "text": "#c9d1d9"})
                        
                        st.markdown(
                            f'<div style="background-color: {color_info["bg"]}; border: 1px solid {color_info["text"]}; '
                            f'border-radius: 10px; padding: 25px; text-align: center; margin-top: 15px;">'
                            f'<h2 style="color: {color_info["text"]}; margin: 0;">{risk_level} RISK</h2>'
                            f'<p style="color: #8b949e; font-size: 1.1rem; margin: 10px 0 0 0;">'
                            f'Fraud Probability: <b>{result["fraud_probability"]:.4f}</b></p>'
                            f'<p style="color: #8b949e; font-size: 0.9rem; margin: 5px 0 0 0;">'
                            f'Processing Time: {result["processing_time_ms"]} ms</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        
                    with col_gauge:
                        prob_pct = result["fraud_probability"] * 100
                        fig_g = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prob_pct,
                            title={"text": "Fraud Probability Score"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": color_info["text"]},
                                "bgcolor": "#21262d",
                                "steps": [
                                    {"range": [0, 30], "color": "#1a3a1e"},
                                    {"range": [30, 50], "color": "#2a2a10"},
                                    {"range": [50, 75], "color": "#3d2a10"},
                                    {"range": [75, 100], "color": "#3d1f1f"}
                                ]
                            },
                            number={"suffix": "%", "font": {"size": 40}}
                        ))
                        fig_g.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#c9d1d9"),
                            height=250,
                            margin=dict(t=40, b=10, l=10, r=10)
                        )
                        st.plotly_chart(fig_g, use_container_width=True)
                        
                    st.divider()
                    
                    # SHAP Chart
                    st.markdown("#### Top 5 SHAP Feature Contributions")
                    features = [f["feature"] for f in result["top_features"]]
                    shap_values = [f["shap_value"] for f in result["top_features"]]
                    bar_colors = ["#f78166" if v > 0 else "#3fb950" for v in shap_values]
                    
                    fig_shap = go.Figure(go.Bar(
                        x=shap_values[::-1],
                        y=features[::-1],
                        orientation='h',
                        marker_color=bar_colors[::-1],
                        text=[f"{v:+.4f}" for v in shap_values[::-1]],
                        textposition='outside'
                    ))
                    fig_shap.update_layout(
                        xaxis=dict(title="SHAP Value"),
                        yaxis=dict(title="Feature"),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(28,35,51,0.6)",
                        font=dict(color="#c9d1d9"),
                        height=280,
                        margin=dict(t=20, b=20, l=20, r=60)
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                    
                    # LLM narrative and policy reference
                    st.divider()
                    st.markdown("#### LLM Fraud Narrative & Regulatory Policy Reference")
                    st.markdown(
                        f'<div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; '
                        f'padding: 20px; color: #c9d1d9; font-size: 1.05rem; line-height: 1.6;">'
                        f'<b>Explanation:</b><br>{result["explanation"]}<br><br>'
                        f'<b>Policy Reference:</b> {result["policy_reference"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.error(f"API Error (status code {r.status_code}): {r.text}")
        except Exception as exc:
            st.error(f"An error occurred during API call: {exc}")
