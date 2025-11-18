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

# Label names we will accept for dropout
TARGET_NAMES = ["DropOut", "dropout", "Dropout", "label", "target"]

# Core predictors for the dropout logistic model from the paper
# (RIT, discipline, repeater, percent credits, sense of belonging)
APPX_FEATURES = [
    "MAP_TestRITScore",
    "num_discipline",       # we will build this if needed
    "HS_IsRepeater",
    "HS_PctEarned",
    "HS_SenseofBelonging",
]


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def risk_level(prob: float) -> str:
    """
    Map probability to risk category based on Table 12 in the paper:
    [0.00, 0.20]  -> Low Risk
    (0.20, 0.40]  -> Moderately Low Risk
    (0.40, 0.60]  -> Moderate Risk
    (0.60, 0.80]  -> Moderately High Risk
    (0.80, 1.00]  -> High Risk
    """
    if prob <= 0.20:
        return "Low Risk"
    elif prob <= 0.40:
        return "Moderately Low Risk"
    elif prob <= 0.60:
        return "Moderate Risk"
    elif prob <= 0.80:
        return "Moderately High Risk"
    else:
        return "High Risk"


@st.cache_resource
def load_and_train():
    if not os.path.exists(CSV_PATH):
        return None, None, None, None

    df = pd.read_csv(CSV_PATH)

    # find dropout target column
    target = None
    for t in TARGET_NAMES:
        if t in df.columns:
            target = t
            break
    if target is None:
        return None, None, None, None

    # build num_discipline from yearly columns if needed
    if "num_discipline" not in df.columns:
        disc_cols = [c for c in df.columns if c.startswith("num_discipline_")]
        if disc_cols:
            df["num_discipline"] = df[disc_cols].sum(axis=1)
        else:
            df["num_discipline"] = 0

    # keep only features that actually exist
    feature_cols = [c for c in APPX_FEATURES if c in df.columns]

    # clean
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    X = df[feature_cols]
    y = df[target]

    # convert target to 0 or 1 if it is strings
    if y.dtype == object:
        y = (
            y.astype(str)
            .str.lower()
            .map({"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0})
            .fillna(0)
            .astype(int)
        )

    # numeric only
    pre = ColumnTransformer(
        transformers=[("num", "passthrough", feature_cols)],
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
    st.error("Upload a CSV with a dropout label in data/combined_data.csv")
    st.stop()

# --------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------
st.sidebar.title("Dropout Dashboard")
choice = st.sidebar.radio(
    "Choose view",
    ["Single Student", "Batch Scoring", "Model Report", "Credits and Repeats"],
)

# --------------------------------------------------------------------
# Single Student view
# --------------------------------------------------------------------
if choice == "Single Student":
    st.header("Single Student Prediction (Appendix A variables)")

    inputs = {}
    col1, col2 = st.columns(2)

    for i, c in enumerate(cols):
        label = c.replace("_", " ")
        with (col1 if i % 2 == 0 else col2):
            inputs[c] = st.text_input(label, "0")

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

        st.subheader(f"Risk level: {level}")
        st.caption(f"Predicted dropout probability: {prob:.3f}")
        st.progress(prob)

        st.markdown(
            """
            **How to read this**  
            - These risk bands follow the paper:  
              - 0.00 to 0.20  Low Risk  
              - 0.20 to 0.40  Moderately Low Risk  
              - 0.40 to 0.60  Moderate Risk  
              - 0.60 to 0.80  Moderately High Risk  
              - 0.80 to 1.00  High Risk
            """
        )

# --------------------------------------------------------------------
# Batch Scoring
# --------------------------------------------------------------------
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

        # rebuild num_discipline if needed
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

        st.subheader("Preview of scored data")
        st.dataframe(df_up.head())

        st.download_button(
            "Download Scored CSV",
            df_up.to_csv(index=False),
            "scored_with_risk.csv",
        )

# --------------------------------------------------------------------
# Model Report
# --------------------------------------------------------------------
elif choice == "Model Report":
    st.header("Model Report (Appendix A variables)")

    df = full_df.fillna(0)
    X = df[cols]
    y = df[target]

    probs = model.predict_proba(X)[:, 1]

    # AUC
    try:
        auc = roc_auc_score(y, probs)
        st.metric("AUC", f"{auc:.3f}")
    except Exception:
        st.write("Could not compute AUC")

    # confusion metrics
    preds = (probs >= 0.5).astype(int)
    rep = pd.DataFrame(classification_report(y, preds, output_dict=True)).T
    st.subheader("Classification report")
    st.dataframe(rep)

    # risk profile distribution like Table 13
    st.subheader("Risk profile distribution (all labeled students)")
    risk_series = pd.Series([risk_level(p) for p in probs], name="risk_level")
    order = [
        "Low Risk",
        "Moderately Low Risk",
        "Moderate Risk",
        "Moderately High Risk",
        "High Risk",
    ]
    counts = risk_series.value_counts().reindex(order).fillna(0).astype(int)
    dist_df = counts.rename("count").to_frame()
    st.table(dist_df)
    st.bar_chart(dist_df)

# --------------------------------------------------------------------
# Credits and Repeats Dashboard
# --------------------------------------------------------------------
else:
    st.header("Credits and Repeats Dashboard")

    df = full_df.copy()

    missing = []
    for col in ["HS_PctEarned", "HS_IsRepeater"]:
        if col not in df.columns:
            missing.append(col)

    if missing:
        st.error(
            "Missing required columns in data: "
            + ", ".join(missing)
            + ". Cannot build credits and repeats dashboard."
        )
    else:
        df["HS_PctEarned"] = df["HS_PctEarned"].astype(float)
        df["HS_IsRepeater"] = df["HS_IsRepeater"].fillna(0).astype(int)

        # top level metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_credits = df["HS_PctEarned"].mean()
            st.metric("Average percent of credits earned", f"{avg_credits:.1f}")
        with col2:
            pct_below_80 = (df["HS_PctEarned"] < 0.8).mean() * 100
            st.metric("Percent of students below 80% credits", f"{pct_below_80:.1f}%")
        with col3:
            repeat_rate = df["HS_IsRepeater"].mean() * 100
            st.metric("Percent of students who repeated a grade", f"{repeat_rate:.1f}%")

        # band credits and compute dropout rates by band if target exists
        if target in df.columns:
            st.subheader("Dropout rate by percent credits earned")

            bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
            labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
            df["credit_band"] = pd.cut(
                df["HS_PctEarned"], bins=bins, labels=labels, include_lowest=True
            )

            band_rates = (
                df.groupby("credit_band")[target].mean().reindex(labels) * 100
            )  # percent dropout
            band_df = band_rates.rename("Dropout rate (%)").to_frame()
            st.table(band_df)
            st.bar_chart(band_df)

        # repeater vs credits (like your previous chart but cleaner)
        st.subheader("Credits earned by repeater status")
        grp = (
            df.groupby("HS_IsRepeater")["HS_PctEarned"]
            .mean()
            .rename({0: "Non repeaters", 1: "Repeaters"})
        )
        grp_df = grp.rename("Average HS_PctEarned").to_frame()
        st.bar_chart(grp_df)
