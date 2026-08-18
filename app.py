"""
app.py — Streamlit demo app for the fetal health classifiers.

Features required by the assignment:
  a. Dataset upload option (CSV)
  b. Model selection dropdown
  c. Display of evaluation metrics
  d. Confusion matrix / classification report

Run locally with:  streamlit run app.py
"""

import pickle

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

st.set_page_config(page_title="Fetal Health Classifier", page_icon="🩺", layout="wide")

TARGET_COL = "fetal_health"
CLASS_LABELS = {1: "Normal", 2: "Suspect", 3: "Pathological"}

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "K-Nearest Neighbors": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
}


@st.cache_resource
def load_artifacts():
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    models = {}
    for name, path in MODEL_FILES.items():
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
    return scaler, models


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    st.markdown(
        "<h1 style='text-align:center;'>🩺 Fetal Health Classifier</h1>"
        "<p style='text-align:center;color:#A855F7;'>Compare 5 classification models on CTG data</p>",
        unsafe_allow_html=True,
    )

    scaler, models = load_artifacts()

    # --- Sidebar: model selection ---
    st.sidebar.header("Controls")
    model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))
    model = models[model_name]

    # --- Upload zone ---
    st.subheader("1. Upload test data (CSV)")
    uploaded = st.file_uploader(
        "Upload a CSV with the same feature columns as training data, plus the "
        f"'{TARGET_COL}' label column (use the provided test_data.csv).",
        type="csv",
    )

    if uploaded is None:
        st.info("Upload test_data.csv to see predictions and metrics.")
        st.stop()

    df = pd.read_csv(uploaded)

    if TARGET_COL not in df.columns:
        st.error(f"Uploaded file must contain a '{TARGET_COL}' column.")
        st.stop()

    X = df.drop(TARGET_COL, axis=1)
    y_true = df[TARGET_COL].astype(int)

    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        st.error(f"Feature mismatch with training data: {e}")
        st.stop()

    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)

    # --- Metrics ---
    st.subheader(f"2. Evaluation metrics — {model_name}")
    metrics = compute_metrics(y_true, y_pred, y_proba)
    cols = st.columns(len(metrics))
    for col, (metric_name, value) in zip(cols, metrics.items()):
        col.metric(metric_name, f"{value:.3f}")

    # --- Confusion matrix + classification report ---
    st.subheader("3. Confusion matrix & classification report")
    c1, c2 = st.columns(2)

    with c1:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with c2:
        report = classification_report(y_true, y_pred, zero_division=0)
        st.text(report)

    # --- Compare all models on the same uploaded data ---
    st.subheader("4. Compare all models on this data")
    rows = []
    for name, m in models.items():
        pred = m.predict(X_scaled)
        proba = m.predict_proba(X_scaled)
        row = compute_metrics(y_true, pred, proba)
        row["Model"] = name
        rows.append(row)
    comparison_df = pd.DataFrame(rows).set_index("Model").round(3)
    st.dataframe(comparison_df, use_container_width=True)

    best_model = comparison_df["MCC"].idxmax()
    st.success(f"Best model on this data (by MCC): **{best_model}**")


if __name__ == "__main__":
    main()
