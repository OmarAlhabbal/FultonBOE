import json, os, io
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report

st.set_page_config(page_title="FCS Dropout Risk Dashboard", page_icon="🎓", layout="wide")

MODEL_DIR = "models"
DATA_DIR = "data"
MODEL_PATH = os.path.join(MODEL_DIR, "logit_model.pkl")
COLS_PATH = os.path.join(MODEL_DIR, "columns.json")
CSV_PATH = os.path.join(DATA_DIR, "combined_data.csv")

# ---------- Utility helpers ----------
@st.cache_data
def read_csv(path):
    return pd.read_csv(path)

def guess_target(df):
    for t in ["dropout", "Dropout", "is_dropout", "label", "target"]:
        if t in df.columns:
            return t
    return None

def split_cols(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    num = X.select_dtypes(include=["number"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return X, y, num, cat

def build_model(num, cat):
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
    ])
    clf = LogisticRegression(max_iter=200)
    return Pipeline([("pre", pre), ("clf", clf)])

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(COLS_PATH):
        model = joblib.load(MODEL_PATH)
        with open(COLS_PATH) as f:
            cols = json.load(f)
        return model, cols
    # fallback: train on CSV
    if os.path.exists(CSV_PATH):
        df = read_csv(CSV_PATH)
        target = guess_target(df)
        X, y, num, cat = split_cols(df, target)
        pipe = build_model(num, cat)
        pipe.fit(X, y)
        return pipe, list(X.columns)
    else:
        st.error("No model or data found.")
        st.stop()

def score(model, df):
    p = model.predict_proba(df)[:, 1]
    pred = (p >= 0.5).astype(int)
    out = df.copy()
    out["dropout_prob"] = p
    out["dropout_pred"] = pred
    return out

# ---------- Sidebar ----------
page = st.sidebar.radio("Select view", ["Single Student", "Batch Scoring", "Model Report"])
model, columns = load_model()

# ---------- Pages ----------
if page == "Single Student":
    st.header("Single Student Prediction")
    inputs = {}
    col1, col2 = st.columns(2)
    for i, c in enumerate(columns):
        with (col1 if i % 2 == 0 else col2):
            inputs[c] = st.text_input(c, value="0")
    if st.button("Predict"):
        df = pd.DataFrame([{c: float(v) if v.replace('.', '', 1).isdigit() else v for c, v in inputs.items()}])
        df = df[columns]
        result = score(model, df)
        prob = float(result.loc[0, "dropout_prob"])
        st.success(f"Predicted dropout probability: {prob:.3f}")
        st.progress(prob)

elif page == "Batch Scoring":
    st.header("Batch Scoring")
    upload = st.file_uploader("Upload a CSV file", type=["csv"])
    if upload:
        df = pd.read_csv(upload)
        try:
            df = df[columns]
            results = score(model, df)
            st.dataframe(results.head())
            st.download_button("Download results", results.to_csv(index=False), "scored.csv")
        except Exception as e:
            st.error(f"Column mismatch: {e}")

else:
    st.header("Model Report")
    if os.path.exists(CSV_PATH):
        df = read_csv(CSV_PATH)
        t = guess_target(df)
        if t:
            X, y, *_ = split_cols(df, t)
            X = X[columns]
            p = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, p)
            st.metric("AUC", f"{auc:.3f}")
            preds = (p >= 0.5).astype(int)
            rep = pd.DataFrame(classification_report(y, preds, output_dict=True)).T
            st.dataframe(rep)
    st.caption("Dashboard built for FCS Dropout Analysis team.")
