# FinShield — AI Fraud Detection Platform

## Project Identity
You are building FinShield, a production-grade end-to-end AI fraud detection and 
intelligence platform. This is a serious portfolio project for a data science student 
targeting high-paying ML/AI jobs in India (Razorpay, PhonePe, HDFC Bank, Navi, 
Quantiphi). Every file you write must be production quality — not tutorial quality.

## Project Goal
Detect financial transaction fraud using a 5-layer system:
1. Classical ML — XGBoost + LightGBM + Isolation Forest ensemble
2. Deep Learning — PyTorch BiLSTM for sequential fraud patterns
3. LLM + RAG — LangChain + ChromaDB for fraud explanation and policy Q&A
4. MLOps — MLflow + Evidently AI + Prefect for model monitoring and retraining
5. Deployment — FastAPI + Streamlit + Docker + GitHub Actions CI/CD on Hugging Face Spaces

## Tech Stack
- Python 3.11
- pandas, numpy, scikit-learn, imbalanced-learn (SMOTE)
- XGBoost, LightGBM for classical ML
- PyTorch for deep learning (never Keras or TensorFlow)
- LangChain, ChromaDB, sentence-transformers, RAGAS for RAG
- MLflow for experiment tracking and model registry
- Evidently AI for drift detection
- Prefect for pipeline orchestration
- FastAPI + Pydantic for API
- Streamlit for dashboard
- Docker + docker-compose for containerisation
- GitHub Actions for CI/CD
- DVC for data version control
- Pandera for data validation
- SHAP for model explainability
- Optuna for hyperparameter tuning (never GridSearchCV)

## Project Structure
finshield/
├── data/
│   ├── raw/           # original CSVs — never modify these
│   └── processed/     # pipeline output only
├── notebooks/
│   └── figures/       # all EDA plots saved here as PNG
├── src/
│   ├── data/          # pipeline, preprocessing, validation
│   ├── models/        # training, evaluation, ensemble
│   ├── llm/           # RAG, LangChain, vector store
│   ├── mlops/         # drift detection, retraining, MLflow
│   └── api/           # FastAPI app
├── tests/             # pytest unit tests for every module
├── dashboard/         # Streamlit app
├── docker/            # Dockerfiles
├── .dvc/
├── requirements.txt
├── docker-compose.yml
├── AGENTS.md          # this file
└── README.md
