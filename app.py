import os
import json
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

# include both DropOut and dropout just in case
LIKELY_TARGETS = ["dropout", "DropOut", "Dropout", "is_dropout", "label", "target"]
LIKELY_IDS = ["student_id", "StudentID", "id"]


def guess_target(df: pd.DataFrame) -> str | None:
    for t in LIKELY_TARGETS:
        if t in df.columns:
            return t
    return None


def prepare_data(df: pd.DataFrame, target: str):
    """
    Clean the dataframe for modeling:
    - map string labels to 0 or 1
    - drop rows with missing target
    - replace inf with NaN
    - fill numeric NaN with column medians
    - fill categorical NaN with 'Unknown'
    """
    # handle target mapping if needed
    if df[target].dtype == object:
        mapped = (
            df[target]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "yes": 1,
                    "y": 1,
                    "true": 1,
                    "1": 1,
                    "no": 0,
                    "n": 0,
                    "false": 0,
                    "0": 0,
                }
            )
        )
        # if mapping produced NaNs everywhere, fall back to original
        if mapped.notna().sum() > 0:
            df[target] = mapped

    # drop rows where target is missing
    df = df.dropna(subset=[target])

    # replace infinities
    df = df.replace([np.inf, -np.inf], np.nan)

    # split X and y
    X = df.drop(columns=[target])
    y = df[target]

    # identify numeric and categorical columns
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # fill numeric NaN with median
    if num_cols:
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    # fill categorical NaN with placeholder
    if cat_cols:
        X[cat_cols] = X[cat_cols].fillna("Unknown")

    return X, y, num_cols, cat_cols


def build_pipe(num_cols, cat_cols) -> Pipeline:
    transformers = []
    if num_cols:
        transformers.append(
            ("num", StandardScaler(with_mean=False), num_cols)
        )
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        )

    if not transformers:
        raise RuntimeError("No usable feature columns found for the model.")

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    clf = LogisticRegression(max_iter=300)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe


@st.cache_resource
def load_or_train():
    # 1. Load pre trained model if it exists
    if os.path.exists(MODEL_PATH) and os.path.exists(COLS_PATH):
        model = joblib.load(MODEL_PATH)
        with open(COLS_PATH, "r") as f:
            cols = json.load(f)
        return model, cols, None, None

    # 2. Train from CSV if available
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)

        target = guess_target(df)
        if target is None:
            raise RuntimeError("Target column not found. Rename your label to dropout.")

        X, y, num_cols, cat_cols = prepare_data(df, target)
        pipe = build_pipe(num_cols, cat_cols)
        pipe.fit(X, y)

        model_cols = list(X.columns)
        return pipe, model_cols, {"trained_on_csv": True}, target

    # 3. Nothing found
    return None, None, None, None


def ensure_feature_order(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[cols]


def score_df(model, X: pd.DataFrame) -> pd.DataFrame:
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    out = X.copy()
    out["dropout_prob"] = probs
    out["dropout_pred"] = preds
    return out


# sidebar
st.sidebar.title("FCS Dropout Risk Dashboard")
page = st.sidebar.radio("Select view", ["Single Student", "Batch Scoring", "Model Report"])

# load model
model, model_cols, meta, target_name = load_or_train()

if model is None:
    st.error("No model or data found. Add a trained model or upload data/combined_data.csv.")
    st.stop()

# page: single student
if page == "Single Student":
    st.header("Single Student Prediction")
    col1, col2 = st.columns(2)
    inputs = {}

    for i, col in enumerate(model_cols):
        with (col1 if i % 2 == 0 else col2):
            inputs[col] = st.text_input(col, "0")

    if st.button("Predict"):
        row = {}
        for k, v in inputs.items():
            # try numeric, fall back to string
            try:
                row[k] = float(v)
            except ValueError:
                row[k] = v

        X_input = pd.DataFrame([row])
        X_input = ensure_feature_order(X_input, model_cols)

        result = score_df(model, X_input)
        prob = float(result.loc[0, "dropout_prob"])

        st.success(f"Predicted dropout probability: {prob:.3f}")
        st.progress(min(max(prob, 0), 1))

# page: batch scoring
elif page == "Batch Scoring":
    st.header("Batch Scoring")
    uploaded = st.file_uploader("Upload CSV with the same feature columns", type=["csv"])

    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        X_up = ensure_feature_order(df_up, model_cols)
        scored = score_df(model, X_up)

        st.subheader("Preview")
        st.dataframe(scored.head(20), use_container_width=True)

        csv_bytes = scored.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download scored CSV",
            data=csv_bytes,
            file_name="scored_dropout.csv",
            mime="text/csv",
        )

# page: model report
else:
    st.header("Model Report")

    if os.path.exists(CSV_PATH) and target_name is not None:
        df_all = pd.read_csv(CSV_PATH)
        X_all, y_all, _, _ = prepare_data(df_all, target_name)
        X_all = ensure_feature_order(X_all, model_cols)

        probs = model.predict_proba(X_all)[:, 1]
        preds = (probs >= 0.5).astype(int)

        try:
            auc = roc_auc_score(y_all, probs)
            st.metric("AUC on dataset", f"{auc:.3f}")
        except Exception:
            st.write("Could not compute AUC for this dataset.")

        report = classification_report(y_all, preds, output_dict=True)
        rep_df = pd.DataFrame(report).T
        st.subheader("Classification report")
        st.dataframe(rep_df, use_container_width=True)
    else:
        st.write("No evaluation dataset available.")
