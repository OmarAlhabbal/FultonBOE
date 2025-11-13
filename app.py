import os, io, json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
import joblib

st.set_page_config(page_title="FCS Dropout Risk", page_icon="🎓", layout="wide")

MODEL_DIR = "models"
DATA_DIR = "data"
MODEL_PATH = os.path.join(MODEL_DIR, "logit_model.pkl")
COLS_PATH = os.path.join(MODEL_DIR, "columns.json")
CSV_PATH = os.path.join(DATA_DIR, "combined_data.csv")
LIKELY_TARGETS = ["dropout", "DropOut", "Dropout", "is_dropout", "label", "target"]
LIKELY_IDS = ["student_id", "StudentID", "id"]

def guess_target(df):
    for t in LIKELY_TARGETS:
        if t in df.columns:
            return t
    return None

def split_cols(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    num = X.select_dtypes(include=["number"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return X, y, num, cat

def build_pipe(num, cat):
    pre = ColumnTransformer([
        ("num", StandardScaler(with_mean=False), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])
    clf = LogisticRegression(max_iter=300)
    return Pipeline([("pre", pre), ("clf", clf)])

@st.cache_resource
def load_or_train():
    # 1) Load pre-trained model if present
    if os.path.exists(MODEL_PATH) and os.path.exists(COLS_PATH):
        model = joblib.load(MODEL_PATH)
        with open(COLS_PATH) as f:
            cols = json.load(f)
        return model, cols, None, None

    # 2) Train quick model from CSV if present
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)

    tgt = guess_target(df)
    if tgt is None:
        raise RuntimeError("Target column not found, rename your label to 'dropout'")

    # Normalize label if strings
    if df[tgt].dtype == object:
        df[tgt] = (
            df[tgt]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0})
        )

    # Drop rows where target is missing
    df = df.dropna(subset=[tgt])

    # Replace infinities with NaN, then drop any remaining NaNs in features
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    X, y, num, cat = split_cols(df, tgt)
    pipe = build_pipe(num, cat).fit(X, y)
    return pipe, list(X.columns), {"trained_on_csv": True}, tgt



    # 3) Nothing found
    return None, None, None, None

def ensure_feature_order(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[cols]

def score_df(model, X):
    p = model.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)
    out = X.copy()
    out["dropout_prob"] = p
    out["dropout_pred"] = pred
    return out

st.sidebar.title("FCS Dropout Risk")
page = st.sidebar.radio("Select view", ["Single Student", "Batch Scoring", "Model Report"])

model, model_cols, meta, tgt = load_or_train()

if model is None:
    st.error("No model or data found. Upload either models/logit_model.pkl and models/columns.json, or data/combined_data.csv with a 'dropout' column.")
    st.stop()

if page == "Single Student":
    st.header("Single Student")
    st.caption("Enter features, then score.")
    col1, col2 = st.columns(2)
    vals = {}
    for i, c in enumerate(model_cols):
        with (col1 if i % 2 == 0 else col2):
            vals[c] = st.text_input(c, "0")
    if st.button("Score"):
        row = {k: (float(v) if v.replace(".","",1).lstrip("-").isdigit() else v) for k, v in vals.items()}
        X = pd.DataFrame([row])[model_cols]
        res = score_df(model, X)
        prob = float(res.loc[0, "dropout_prob"])
        st.success(f"Predicted dropout probability: {prob:.3f}")
        st.progress(min(max(prob, 0), 1))

elif page == "Batch Scoring":
    st.header("Batch Scoring")
    up = st.file_uploader("Upload CSV to score", type=["csv"])
    if up:
        df = pd.read_csv(up)
        X = ensure_feature_order(df, model_cols)
        scored = score_df(model, X)
        st.dataframe(scored.head(20), use_container_width=True)
        st.download_button("Download scored CSV", scored.to_csv(index=False), "scored_dropout.csv")

else:
    st.header("Model Report")
    if os.path.exists(CSV_PATH) and tgt is not None:
        df = pd.read_csv(CSV_PATH)
        X, y, *_ = split_cols(df, tgt)
        X = ensure_feature_order(X, model_cols)
        p = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, p)
        st.metric("AUC on dataset", f"{auc:.3f}")
        preds = (p >= 0.5).astype(int)
        rep = pd.DataFrame(classification_report(y, preds, output_dict=True)).T
        st.dataframe(rep, use_container_width=True)
    else:
        st.write("No evaluation dataset available.")


