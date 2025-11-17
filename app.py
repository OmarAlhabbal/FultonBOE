import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.compose import ColumnTransformer
import os

st.set_page_config(page_title="Dropout Dashboard", layout="wide")

CSV_PATH = "data/combined_data.csv"

TARGET_NAMES = ["dropout", "DropOut", "Dropout", "label", "target"]

# Variables from Appendix A (adjust here if your list changes)
APPX_FEATURES = [
    "present_pct",
    "HS_PctEarned",
    "HS_IsRepeater",
    "MAP_TestPercentile",
    "num_discipline",         
    "HS_SenseofBelonging",
]


def risk_level(prob: float) -> str:
    """
    Map probability to risk level.
    Adjust thresholds here to match your paper if needed.
    """
    if prob < 0.2:
        return "Low"
    elif prob < 0.5:
        return "Medium"
    else:
        return "High"


@st.cache_resource
def load_and_train():
    if not os.path.exists(CSV_PATH):
        return None, None, None, None

    df = pd.read_csv(CSV_PATH)

    # find target column
    target = None
    for t in TARGET_NAMES:
        if t in df.columns:
            target = t
            break
    if target is None:
        return None, None, None, None

    # build num_discipline from yearly columns if not already there
    if "num_discipline" not in df.columns:
        disc_cols = [c for c in df.columns if c.startswith("num_discipline_")]
        if disc_cols:
            df["num_discipline"] = df[disc_cols].sum(axis=1)
        else:
            df["num_discipline"] = 0

    # keep only Appendix A features that actually exist
    feature_cols = [c for c in APPX_FEATURES if c in df.columns]

    # clean data
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    X = df[feature_cols]
    y = df[target]

    # convert target to 0/1 if it is strings
    if y.dtype == object:
        y = (
            y.astype(str)
            .str.lower()
            .map({"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0})
            .fillna(0)
            .astype(int)
        )

    # numeric only, simple pipeline
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", feature_cols),
        ],
        remainder="drop",
    )

    model = Pipeline(
        [
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=300)),
        ]
    )

    model.fit(X, y)

    return model, feature_cols, target, df


model, cols, target, full_df = load_and_train()

if model is None:
    st.error("Upload a CSV with a dropout column in data/combined_data.csv")
    st.stop()

st.sidebar.title("Dropout Dashboard")
choice = st.sidebar.radio(
    "Choose view",
    ["Single Student", "Batch Scoring", "Model Report", "Credits and Repeats"],
)

# Single student
if choice == "Single Student":
    st.header("Single Student Prediction (Appendix A variables only)")

    inputs = {}
    col1, col2 = st.columns(2)

    for i, c in enumerate(cols):
        with (col1 if i % 2 == 0 else col2):
            inputs[c] = st.text_input(c, "0")

    if st.button("Predict"):
        row = {}
        for k, v in inputs.items():
            try:
                row[k] = float(v)
            except ValueError:
                row[k] = v

        df_input = pd.DataFrame([row])
        df_input = df_input.fillna(0)
        df_input = df_input.reindex(columns=cols)

        prob = float(model.predict_proba(df_input)[0, 1])
        level = risk_level(prob)

        st.subheader(f"Risk level: **{level}**")
        st.caption(f"(Predicted dropout probability: {prob:.3f})")
        st.progress(prob)

# Batch scoring
elif choice == "Batch Scoring":
    st.header("Batch Scoring")

    st.write(
        "Upload a CSV that has at least these columns: "
        + ", ".join(cols)
        + ". It should not include the dropout label."
    )

    upload = st.file_uploader("Upload CSV", type=["csv"])
    if upload:
        df_up = pd.read_csv(upload)

        # create num_discipline if needed
        if "num_discipline" not in df_up.columns:
            disc_cols = [c for c in df_up.columns if c.startswith("num_discipline_")]
            if disc_cols:
                df_up["num_discipline"] = df_up[disc_cols].sum(axis=1)
            else:
                df_up["num_discipline"] = 0

        df_up = df_up.fillna(0)
        df_up = df_up.reindex(columns=cols)

        probs = model.predict_proba(df_up)[:, 1]
        df_up["dropout_prob"] = probs
        df_up["risk_level"] = [risk_level(p) for p in probs]
        df_up["dropout_pred"] = (probs >= 0.5).astype(int)

        st.subheader("Preview")
        st.dataframe(df_up.head())

        st.download_button(
            "Download Scored CSV", df_up.to_csv(index=False), "scored_with_risk.csv"
        )

# Model report
elif choice == "Model Report":
    st.header("Model Report (Appendix A variables)")

    df = full_df.fillna(0)
    X = df[cols]
    y = df[target]

    probs = model.predict_proba(X)[:, 1]

    try:
        auc = roc_auc_score(y, probs)
        st.metric("AUC", f"{auc:.3f}")
    except Exception:
        st.write("Could not compute AUC")

    preds = (probs >= 0.5).astype(int)
    rep = pd.DataFrame(classification_report(y, preds, output_dict=True)).T
    st.dataframe(rep)

# Credits and Repeats dashboard
else:
    st.header("Credits and Repeats Dashboard")

    df = full_df.copy()

    # make sure columns exist
    if "HS_PctEarned" not in df.columns or "HS_IsRepeater" not in df.columns:
        st.error("HS_PctEarned and HS_IsRepeater must be in the data.")
    else:
        df["HS_IsRepeater"] = df["HS_IsRepeater"].fillna(0)

        col1, col2 = st.columns(2)
        with col1:
            avg_credits = df["HS_PctEarned"].mean()
            st.metric("Average percent of credits earned", f"{avg_credits:.1f}")

        with col2:
            repeat_rate = df["HS_IsRepeater"].mean()
            st.metric("Percent of students who repeated", f"{100 * repeat_rate:.1f}%")

        st.subheader("Distribution of credits earned")
        st.caption("Histogram of HS_PctEarned")
        st.bar_chart(df["HS_PctEarned"].value_counts().sort_index())

        st.subheader("Repeater vs credits")
        st.caption("Average HS_PctEarned by repeater status")
        grp = df.groupby("HS_IsRepeater")["HS_PctEarned"].mean().rename(
            {0: "Non repeaters", 1: "Repeaters"}
        )
        st.bar_chart(grp)
