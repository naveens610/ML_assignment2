# Fetal Health Classifier

## a. Problem Statement
Cardiotocogram (CTG) exams give obstetricians a set of numeric signals about
fetal heart rate and uterine contractions. This project builds and compares
five classification models that predict fetal health status — **Normal**,
**Suspect**, or **Pathological** — from these CTG features, and exposes the
models through an interactive Streamlit app so predictions and metrics can
be explored on new test data.

## b. Dataset Description
- **Source:** [Fetal Health Classification, Kaggle](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification)
- **Instances:** 2,126
- **Features:** 21 numeric CTG-derived features (baseline heart rate,
  accelerations, decelerations, histogram statistics, etc.)
- **Target:** `fetal_health` — 1 = Normal (~78%), 2 = Suspect (~14%),
  3 = Pathological (~8%)
- **Class balance:** Imbalanced — handled via `class_weight='balanced'`
  where supported, and macro-averaged metrics throughout.

## c. GitHub Repository Link
`<PASTE YOUR PUBLIC REPO URL HERE>`

## d. Models Used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ |
| Decision Tree | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ |
| kNN | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ |
| Naive Bayes | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ |
| Random Forest (Ensemble) | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ |

> Values come straight out of `metrics_summary.csv`, produced by
> `train_models.py`. Copy them in after you run training on your own data.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | _Write 2–3 sentences based on your own results._ |
| Decision Tree | _Write 2–3 sentences based on your own results._ |
| kNN | _Write 2–3 sentences based on your own results._ |
| Naive Bayes | _Write 2–3 sentences based on your own results._ |
| Random Forest (Ensemble) | _Write 2–3 sentences based on your own results._ |
| **Overall Winner for your dataset?** | _State which model and why._ |

## Live App
`<PASTE YOUR STREAMLIT COMMUNITY CLOUD LINK HERE>`

## How to Run Locally
```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train_models.py         # trains models, saves pickles + test_data.csv
streamlit run app.py
```

## Repository Structure
```
fetal-health-classifier/
├── app.py                  # Streamlit app
├── train_models.py         # training + evaluation script
├── requirements.txt
├── README.md
├── test_data.csv           # held-out test set (unscaled) for the app
├── confusion_matrices.png  # 2x3 grid of confusion matrices
├── metrics_summary.csv     # raw metrics table used above
├── .streamlit/config.toml  # dark theme
└── model/
    ├── scaler.pkl
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl
```
