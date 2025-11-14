import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
import os

st.set_page_config(page_title="Dropout Dashboard", layout="wide")

CSV_PATH = "data/combined_data.csv"

TARGET_NAMES = ["dropout", "DropOut", "Dropout", "label", "target"]

@st.cache_resource
def load_and_train():
    if not os.path.exists(CSV_PATH):
        return None, None, None

    df = pd.read_csv(CSV_PATH)

    target = None
    for t in TARGET_NAMES:
        if t in df.columns:
            target = t
            break

    if target is None:
        return None, None, None

    # Clean dataset
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    X = df.drop(columns=[target])
    y = df[target]

    # Convert target to 0/1
    if y.dtype == object:
        y = y.astype(str).str.lower().map({"yes":1, "true":1, "1":1, "no":0, "false":0, "0":0}).fillna(0).astype(int)

    # Identify cols
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    model = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=300))
    ])

    model.fit(X, y)

    return model, list(X.columns), target


model, cols, target = load_and_train()

if model is None:
    st.error("Upload a CSV with a dropout column in data/combined_data.csv")
    st.stop()

st.sidebar.title("Dropout Dashboard")
choice = st.sidebar.radio("Choose view", ["Single Student", "Batch Scoring", "Model Report"])

# Single student
if choice == "Single Student":
    st.header("Single Student Prediction")
    inputs = {}
    for c in cols:
        inputs[c] = st.text_input(c, "0")
    if st.button("Predict"):
        row = {}
        for k,v in inputs.items():
            try: row[k] = float(v)
            except: row[k] = v
        df = pd.DataFrame([row])
        df = df.fillna(0)
        pred = model.predict_proba(df)[0,1]
        st.success(f"Dropout probability: {pred:.3f}")
        st.progress(pred)

# Batch scoring
elif choice == "Batch Scoring":
    st.header("Batch Scoring")
    upload = st.file_uploader("Upload CSV", type=["csv"])
    if upload:
        df = pd.read_csv(upload).fillna(0)
        df = df.reindex(columns=cols)
        probs = model.predict_proba(df)[:,1]
        df["dropout_prob"] = probs
        df["dropout_pred"] = (probs>=0.5).astype(int)
        st.dataframe(df.head())
        st.download_button("Download Scored CSV", df.to_csv(index=False), "scored.csv")

# Model report
else:
    st.header("Model Report")
    df = pd.read_csv(CSV_PATH).fillna(0)
    X = df[cols]
    y = df[target]
    probs = model.predict_proba(X)[:,1]
    try:
        auc = roc_auc_score(y, probs)
        st.metric("AUC", f"{auc:.3f}")
    except:
        st.write("Could not compute AUC")
    preds = (probs>=0.5).astype(int)
    rep = pd.DataFrame(classification_report(y, preds, output_dict=True)).T
    st.dataframe(rep)
