"""Self-contained Hugging Face Spaces Streamlit Application for FinShield."""

from __future__ import annotations

import os
import time
import uuid
import pickle
import numpy as np
import pandas as pd
import shap
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

# Setup page configuration
st.set_page_config(
    page_title="FinShield - Fraud Detection",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
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

# Detect artifacts paths
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
VECTORSTORE_PATH = ARTIFACTS_DIR / "vectorstore"

# ---------------------------------------------------------------------------
# Section 1 - Model & RAG Loading (Cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            return None, None
        with open(MODEL_PATH, "rb") as fh:
            model = pickle.load(fh)
        with open(SCALER_PATH, "rb") as fh:
            scaler = pickle.load(fh)
        return model, scaler
    except Exception:
        return None, None

PROMPT_TEMPLATE = """You are a fraud detection expert and compliance officer at an Indian bank.
You have access to RBI guidelines, PCI DSS requirements, and fraud pattern 
documentation. Answer the analyst's question accurately and cite the specific 
document and section your answer comes from.

Context from policy documents:
{context}

Analyst question: {question}

Provide a clear, structured answer. If the context contains relevant policy 
references, cite them explicitly (e.g., "Per RBI Master Directions Section X...").
If the question cannot be answered from the context, say so clearly.

Answer:"""

@st.cache_resource(show_spinner=False)
def load_rag_chain():
    try:
        if not VECTORSTORE_PATH.exists():
            return None, None
            
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        from langchain_core.prompts import PromptTemplate
        from langchain.chains import RetrievalQA
        from langchain_groq import ChatGroq
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma(
            persist_directory=str(VECTORSTORE_PATH),
            embedding_function=embeddings
        )
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key or groq_api_key.strip() == "":
            return None, vector_store
            
        llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant", temperature=0.0)
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        return chain, vector_store
    except Exception:
        return None, None

# Load resources
model, scaler = load_model()
rag_chain, vector_store = load_rag_chain()

# Sidebar header and page selection
with st.sidebar:
    st.markdown("### FinShield")
    st.markdown("Production Fraud Intelligence")
    st.divider()
    
    page = st.radio(
        "Navigation",
        ["Live Transaction Analyser", "RAG Chatbot", "About Platform"]
    )
    
    st.divider()
    st.markdown("#### System Configuration")
    if model is not None:
        st.markdown("Model: **XGBoost Tuned**")
    else:
        st.markdown("Model: **Missing**")
        
    if os.getenv("GROQ_API_KEY"):
        st.markdown("Groq LLM: **Connected**")
    else:
        st.markdown("Groq LLM: **Missing Key**")

# Header section
st.markdown("<h1 style='margin-bottom: 0px;'>FinShield</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #8b949e; font-size: 1.1rem; margin-top: 5px; margin-bottom: 10px;'>AI-Powered Fraud Detection Platform</p>", unsafe_allow_html=True)
st.divider()

# Verify resources are loaded
if model is None or scaler is None:
    st.warning("Model and scaler files not found in artifacts/. Please run setup_artifacts.py to copy them.")

# ---------------------------------------------------------------------------
# Section 2 - Live Transaction Analyser
# ---------------------------------------------------------------------------
if page == "Live Transaction Analyser":
    st.markdown("### Live Transaction Analyser")
    st.markdown("Enter transaction details manually to get a real-time risk classification and SHAP driver values.")
    
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        st.markdown("#### Input Parameters")
        amount = st.number_input("Amount (INR)", min_value=1.00, max_value=100000.00, value=5000.00, step=100.00)
        hour = st.slider("Hour of Day", min_value=0, max_value=23, value=14)
        v14 = st.slider("V14", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
        v10 = st.slider("V10", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
        v4 = st.slider("V4", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
        is_night = st.checkbox("Night-time Transaction")
        is_weekend = st.checkbox("Weekend Transaction")
        
        analyse_btn = st.button("Analyse Transaction", use_container_width=True, type="primary")
        
    with col_output:
        if analyse_btn:
            if model is None or scaler is None:
                st.error("Model not available for inference.")
            else:
                with st.spinner("Processing features and running model inference..."):
                    # Preprocess locally
                    # Conversion rate logic or raw mapping
                    amount_val = float(amount)
                    amount_log = np.log1p(amount_val)
                    mean_amt = float(scaler.mean_[0])
                    std_amt = float(scaler.scale_[0])
                    amount_zscore = (amount_val - mean_amt) / (std_amt + 1e-9)
                    
                    scaled_vals = scaler.transform([[amount_val, amount_log, amount_zscore]])
                    scaled_amount = scaled_vals[0][0]
                    scaled_amount_log = scaled_vals[0][1]
                    scaled_amount_zscore = scaled_vals[0][2]
                    
                    features = {f"V{i}": 0.0 for i in range(1, 29)}
                    features["V4"] = float(v4)
                    features["V10"] = float(v10)
                    features["V14"] = float(v14)
                    
                    features["Amount"] = scaled_amount
                    features["Hour"] = float(hour)
                    features["Day"] = 0.0
                    features["Is_night"] = int(is_night)
                    features["Is_weekend"] = int(is_weekend)
                    features["Amount_log"] = scaled_amount_log
                    features["Amount_zscore"] = scaled_amount_zscore
                    features["Is_round_amount"] = int(amount_val % 1.0 == 0)
                    features["Is_small_amount"] = int(amount_val < 5.0)
                    features["V14_V4_interaction"] = float(v14 * v4)
                    features["V14_V10_interaction"] = float(v14 * v10)
                    features["V14_squared"] = float(v14 ** 2)
                    features["V10_squared"] = float(v10 ** 2)
                    
                    # Columns in exact order
                    feature_names = [
                        'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
                        'Hour', 'Day', 'Is_night', 'Is_weekend', 'Amount_log', 'Amount_zscore',
                        'Is_round_amount', 'Is_small_amount', 'V14_V4_interaction',
                        'V14_V10_interaction', 'V14_squared', 'V10_squared'
                    ]
                    
                    df_preprocessed = pd.DataFrame([features])[feature_names]
                    
                    # Prediction
                    prob = float(model.predict_proba(df_preprocessed)[0, 1])
                    
                    # Risk Mapping
                    if prob < 0.3:
                        risk_level = "LOW"
                        color_info = {"bg": "#1a3a1e", "text": "#3fb950"}
                    elif prob < 0.5:
                        risk_level = "MEDIUM"
                        color_info = {"bg": "#2a2a10", "text": "#e3b341"}
                    elif prob < 0.75:
                        risk_level = "HIGH"
                        color_info = {"bg": "#3d2a10", "text": "#ffa657"}
                    else:
                        risk_level = "CRITICAL"
                        color_info = {"bg": "#3d1f1f", "text": "#f78166"}
                        
                    # UI Box
                    st.markdown(
                        f'<div style="background-color: {color_info["bg"]}; border: 1px solid {color_info["text"]}; '
                        f'border-radius: 10px; padding: 20px; text-align: center;">'
                        f'<h3 style="color: {color_info["text"]}; margin: 0;">Risk Level: {risk_level}</h3>'
                        f'<p style="color: #8b949e; margin: 5px 0 0 0;">Fraud Probability: {prob:.4f}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    st.progress(prob)
                    
                    # SHAP Explanation
                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer(df_preprocessed)
                    shap_arr = shap_vals.values[0]
                    
                    top_indices = np.argsort(np.abs(shap_arr))[::-1][:5]
                    top_features = []
                    for idx in top_indices:
                        feat_name = df_preprocessed.columns[idx]
                        val = float(shap_arr[idx])
                        top_features.append({
                            "feature": feat_name,
                            "shap_value": val,
                            "direction": "positive" if val > 0 else "negative"
                        })
                        
                    # Plotly chart
                    st.markdown("#### Top SHAP Driver Contributions")
                    feats_list = [f["feature"] for f in top_features]
                    vals_list = [f["shap_value"] for f in top_features]
                    bar_colors = ["#f78166" if v > 0 else "#3fb950" for v in vals_list]
                    
                    fig = go.Figure(go.Bar(
                        x=vals_list[::-1],
                        y=feats_list[::-1],
                        orientation='h',
                        marker_color=bar_colors[::-1],
                        text=[f"{v:+.4f}" for v in vals_list[::-1]],
                        textposition='outside'
                    ))
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(28,35,51,0.6)",
                        font=dict(color="#c9d1d9"),
                        height=250,
                        margin=dict(t=10, b=10, l=10, r=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # LLM Explanation call
                    top_feat_1 = top_features[0]["feature"]
                    top_feat_2 = top_features[1]["feature"]
                    
                    explanation = f"Transaction flagged due to {top_feat_1} and {top_feat_2}. Fraud probability: {prob:.1%}."
                    
                    groq_key = os.getenv("GROQ_API_KEY")
                    if groq_key and prob > 0.3:
                        try:
                            from langchain_core.messages import HumanMessage
                            from langchain_groq import ChatGroq
                            llm = ChatGroq(groq_api_key=groq_key, model="llama-3.1-8b-instant", temperature=0.0)
                            prompt = (
                                f"Explain in exactly two simple sentences why a financial transaction was flagged as suspicious "
                                f"by our machine learning model. The transaction amount is INR {amount_val:.2f} "
                                f"and was made at hour {hour:.1f}. The primary risk factors are "
                                f"{top_feat_1} (SHAP value: {top_features[0]['shap_value']:+.4f}) and {top_feat_2} (SHAP value: {top_features[1]['shap_value']:+.4f}). "
                                f"Fraud probability is {prob:.1%}. Keep it plain English, direct, and concise."
                            )
                            response = llm.invoke([HumanMessage(content=prompt)])
                            explanation = response.content.strip()
                        except Exception:
                            pass
                            
                    st.markdown("#### LLM Fraud Explanation")
                    st.info(explanation)
        else:
            st.info("Set parameters on the left and click Analyse Transaction.")

# ---------------------------------------------------------------------------
# Section 3 - RAG Chatbot
# ---------------------------------------------------------------------------
elif page == "RAG Chatbot":
    st.markdown("### Fraud Analyst Chatbot")
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        
    if "should_stream" not in st.session_state:
        st.session_state["should_stream"] = False
        
    def stream_text(text: str):
        for char in text:
            yield char
            time.sleep(0.005)
            
    def handle_chatbot_query(query: str):
        st.session_state["messages"].append({"role": "user", "content": query})
        st.session_state["should_stream"] = True
        
        with st.spinner("Searching compliance documents..."):
            try:
                groq_key = os.getenv("GROQ_API_KEY")
                if not groq_key or not rag_chain:
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": "Set GROQ_API_KEY in Space secrets to enable the chatbot.",
                        "sources": []
                    })
                else:
                    response = rag_chain.invoke({"query": query})
                    answer = response.get("result", "").strip()
                    
                    sources = []
                    source_docs = response.get("source_documents", [])
                    for doc in source_docs:
                        source_path = doc.metadata.get("source", "")
                        source_name = os.path.basename(source_path)
                        if source_name and source_name not in sources:
                            sources.append(source_name)
                            
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
            except Exception as exc:
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": f"Error querying chatbot: {exc}",
                    "sources": []
                })
                
    st.markdown("#### Suggested Questions")
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("What is a card-testing attack?", use_container_width=True):
        handle_chatbot_query("What is a card-testing attack?")
        st.rerun()
    if col2.button("RBI fraud reporting requirements?", use_container_width=True):
        handle_chatbot_query("RBI fraud reporting requirements?")
        st.rerun()
    if col3.button("PCI DSS velocity thresholds?", use_container_width=True):
        handle_chatbot_query("PCI DSS velocity thresholds?")
        st.rerun()
    if col4.button("How to handle a false positive?", use_container_width=True):
        handle_chatbot_query("How to handle a false positive?")
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
                with st.expander("Sources Cited"):
                    for src in msg["sources"]:
                        st.write(f"- {src}")
                        
    chat_input = st.chat_input("Ask a fraud policy or compliance question...")
    if chat_input:
        handle_chatbot_query(chat_input)
        st.rerun()
        
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["should_stream"] = False
        st.rerun()

# ---------------------------------------------------------------------------
# Section 4 - About Platform
# ---------------------------------------------------------------------------
elif page == "About Platform":
    st.markdown("### About FinShield")
    st.markdown(
        "FinShield is a production-grade end-to-end AI fraud detection and intelligence platform. "
        "It combines classical machine learning, sequential deep learning models, and LLM-powered "
        "decision reasoning to secure digital banking and payment systems."
    )
    
    st.markdown("#### System Architecture")
    st.markdown(
        "1. **Classical ML Layer**: XGBoost and LightGBM ensemble optimized via Optuna (PR-AUC 0.8756).\n"
        "2. **Deep Learning Layer**: PyTorch BiLSTM sequential model for temporal fraud signature detection.\n"
        "3. **LLM + RAG Layer**: LangChain framework querying vectorized compliance rules (RBI Master Circulars, PCI DSS v4.0) with Groq Llama 3 models.\n"
        "4. **MLOps and Governance**: MLflow tracking, Evidently AI data drift validation, and automated retraining pipelines."
    )
    
    st.markdown("#### Baseline vs Production Performance")
    performance_data = {
        "Model": ["Baseline Logistic Regression", "LightGBM Tuned", "XGBoost Tuned", "FinShield Ensemble (Production)"],
        "PR-AUC": [0.7021, 0.8723, 0.8725, 0.8756],
        "F1 Score": [0.6512, 0.8723, 0.8300, 0.8705]
    }
    st.dataframe(pd.DataFrame(performance_data), use_container_width=True, hide_index=True)
    
    st.markdown("#### Links and Resources")
    st.markdown(
        "- **GitHub Repository**: [FinShield Repository](https://github.com/YOUR_USERNAME/finshield)\n"
        "- **Hugging Face Space**: Deployed active space"
    )
