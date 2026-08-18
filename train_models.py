"""
train_models.py
================
Trains 5 classification models on the Fetal Health dataset, evaluates each
on 6 metrics, and saves the trained models + scaler + test set to disk.

Run this once locally (or in BITS Virtual Lab) before building/deploying
the Streamlit app. You can also copy these cells into a Jupyter notebook
(train_models.ipynb) if your assignment requires a notebook submission.

Dataset: https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification
Place fetal_health.csv in the same folder as this script before running.
"""

import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
DATA_PATH = "/Users/harikesh/Downloads/fetal_health.csv"
TARGET_COL = "fetal_health"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("Class distribution:\n", df[TARGET_COL].value_counts())
    print("Missing values:", df.isnull().sum().sum())
    return df


def split_and_scale(df: pd.DataFrame):
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # IMPORTANT: transform only on test data, never fit_transform (avoids leakage)
    X_test_scaled = scaler.transform(X_test)

    return X, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }


def train_all(models, X_train_scaled, y_train):
    trained = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained[name] = model
        print(f"Trained: {name}")
    return trained


def evaluate_all(trained_models, X_test_scaled, y_test):
    rows = []
    all_preds = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        all_preds[name] = y_pred

        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "AUC": roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"),
                "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "MCC": matthews_corrcoef(y_test, y_pred),
            }
        )
    results_df = pd.DataFrame(rows).set_index("Model")
    return results_df, all_preds


def plot_confusion_matrices(trained_models, all_preds, y_test, out_path="confusion_matrices.png"):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for ax, (name, y_pred) in zip(axes, all_preds.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax, cbar=False)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    # hide unused subplot (5 models in a 2x3 grid leaves one empty)
    for ax in axes[len(all_preds):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved confusion matrix grid to {out_path}")


def save_artifacts(trained_models, scaler, X_test, y_test):
    os.makedirs("model", exist_ok=True)

    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    filename_map = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "K-Nearest Neighbors": "knn",
        "Naive Bayes": "naive_bayes",
        "Random Forest": "random_forest",
    }
    for name, model in trained_models.items():
        with open(f"model/{filename_map[name]}.pkl", "wb") as f:
            pickle.dump(model, f)

    # Save UNSCALED test data (with labels) — the app applies scaling itself
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values
    test_df.to_csv("test_data.csv", index=False)
    print("Saved model/*.pkl and test_data.csv")


def main():
    df = load_data()
    X, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(df)

    models = build_models()
    trained_models = train_all(models, X_train_scaled, y_train)

    results_df, all_preds = evaluate_all(trained_models, X_test_scaled, y_test)
    print("\n=== Evaluation Results ===")
    print(results_df.round(4))
    results_df.round(4).to_csv("metrics_summary.csv")

    plot_confusion_matrices(trained_models, all_preds, y_test)
    save_artifacts(trained_models, scaler, X_test, y_test)

    print("\nDone. Use metrics_summary.csv values to fill the README comparison table.")


if __name__ == "__main__":
    main()
