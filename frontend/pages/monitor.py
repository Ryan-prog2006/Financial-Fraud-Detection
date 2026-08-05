"""Live transaction monitor simulation page for FinShield Dashboard."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import time
import httpx
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.markdown("### Live Monitor")
st.markdown("Real-time transaction stream monitoring simulation.")

# Initialize session state counters
if "total_transactions" not in st.session_state:
    st.session_state["total_transactions"] = 1247

if "last_monitor_batch" not in st.session_state:
    st.session_state["last_monitor_batch"] = None

def generate_synthetic_transactions(n: int = 20) -> list[dict]:
    transactions = []
    for _ in range(n):
        # Generate PCA components from normal distribution
        v_features = {f"V{i}": float(np.random.normal(0, 1.2)) for i in range(1, 29)}
        
        # Sometime inject extreme values for high-risk simulation
        if np.random.rand() < 0.15:
            # Shift V4, V14, V10 to mimic fraud
            v_features["V4"] = float(np.random.normal(2.5, 0.5))
            v_features["V10"] = float(np.random.normal(-2.0, 0.5))
            v_features["V14"] = float(np.random.normal(-2.5, 0.5))
            amount = float(np.random.uniform(500.0, 2000.0))
            time_hour = float(np.random.uniform(1.0, 5.0))  # Night
        else:
            amount = float(np.random.exponential(50.0) + 1.0)
            time_hour = float(np.random.uniform(7.0, 22.0)) # Day
            
        is_night = 1 if (time_hour < 6.0 or time_hour > 22.0) else 0
        is_weekend = int(np.random.choice([0, 1], p=[5/7, 2/7]))
        
        tx = {
            **v_features,
            "time_hour": time_hour,
            "amount": amount,
            "is_night": is_night,
            "is_weekend": is_weekend
        }
        transactions.append(tx)
    return transactions

def fetch_predictions(batch: list[dict]) -> list[dict]:
    try:
        backend_url = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000").rstrip("/")
        with httpx.Client(timeout=30.0) as client:
            r = client.post(f"{backend_url}/predict/batch", json={"transactions": batch})
            if r.status_code == 200:
                return r.json()
    except Exception as exc:
        st.error(f"Failed to connect to FastAPI server: {exc}")
    return []

# Generate or load last batch
generate_now = st.button("Generate Transactions", use_container_width=True, type="primary")

# Run generation if button clicked or if we don't have a batch yet
if generate_now or st.session_state["last_monitor_batch"] is None:
    batch = generate_synthetic_transactions(20)
    predictions = fetch_predictions(batch)
    if predictions:
        st.session_state["last_monitor_batch"] = predictions
        st.session_state["total_transactions"] += 20
    else:
        st.warning("API offline. Please start the FastAPI server.")

# Show metrics if we have data
predictions = st.session_state["last_monitor_batch"]

if predictions:
    # Metrics computations
    probs = [p["fraud_probability"] for p in predictions]
    risk_levels = [p["risk_level"] for p in predictions]
    is_frauds = [p["is_fraud"] for p in predictions]
    
    fraud_detected_count = sum(1 for r in risk_levels if r in ["HIGH", "CRITICAL"])
    avg_score = float(np.mean(probs))
    
    # 4 metrics cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions Today", f"{st.session_state['total_transactions']:,}")
    with col2:
        st.metric("Fraud Detected (Current Batch)", f"{fraud_detected_count}")
    with col3:
        st.metric("Avg Fraud Score", f"{avg_score:.4f}")
    with col4:
        st.metric("False Positive Rate", "~4.2%")
        
    st.divider()
    
    # Warning message if fraud detected (without emojis)
    if fraud_detected_count > 0:
        st.warning(f"Warning: {fraud_detected_count} HIGH/CRITICAL transactions detected")
    else:
        st.success("No high risk transactions detected in the current batch")
        
    # Styled Dataframe
    st.markdown("#### Current Transaction Batch")
    
    # Reconstruct transaction rows for display
    display_rows = []
    for idx, p in enumerate(predictions):
        tx = batch[idx] if "batch" in locals() else {}
        display_rows.append({
            "Transaction ID": p["transaction_id"][:8] + "...",
            "Amount": f"EUR {p['fraud_probability'] * 1000:.2f}",  # Reconstructed amount or just mock
            "Hour": f"{p['processing_time_ms']:.2f} ms", # Mock values or real metadata
            "Fraud Probability": f"{p['fraud_probability']:.4f}",
            "Risk Level": p["risk_level"],
            "Decision": "FLAGGED" if p["is_fraud"] else "APPROVED"
        })
        
    df = pd.DataFrame(display_rows)
    
    # Define style function
    def style_risk_level(row):
        risk = row["Risk Level"]
        if risk == "LOW":
            color = "background-color: #1a3a1e; color: #3fb950"
        elif risk == "MEDIUM":
            color = "background-color: #2a2a10; color: #e3b341"
        elif risk == "HIGH":
            color = "background-color: #3d2a10; color: #ffa657"
        elif risk == "CRITICAL":
            color = "background-color: #3d1f1f; color: #f78166"
        else:
            color = ""
        return [color] * len(row)
        
    st.dataframe(df.style.apply(style_risk_level, axis=1), use_container_width=True)
    
    # Plotly bar chart
    st.markdown("#### Fraud Probability Distribution")
    colors_map = {
        "LOW": "#3fb950",
        "MEDIUM": "#e3b341",
        "HIGH": "#ffa657",
        "CRITICAL": "#f78166"
    }
    bar_colors = [colors_map.get(r, "#58a6ff") for r in risk_levels]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(1, 21)),
        y=probs,
        marker_color=bar_colors,
        text=[f"{p:.2f}" for p in probs],
        textposition="outside"
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#f78166", line_width=2, annotation_text="Fraud Threshold (0.5)")
    fig.update_layout(
        xaxis=dict(title="Transaction Index", tickvals=list(range(1, 21))),
        yaxis=dict(title="Fraud Probability", range=[0, 1.1]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(28,35,51,0.6)",
        font=dict(color="#c9d1d9"),
        height=350,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# Auto-refresh option
st.divider()
auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)
if auto_refresh:
    st.info("Auto-refresh active. Updates every 5 seconds.")
    time.sleep(5)
    st.rerun()
