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
| Logistic Regression | 0.8568 | 0.9602 | 0.7406 | 0.8497 | 0.7798 | 0.6868 |
| Decision Tree | 0.8873 | 0.8786 | 0.8150 | 0.8496 | 0.8268 | 0.7188 |
| kNN | 0.8709 | 0.9563 | 0.8090 | 0.6900 | 0.7374 | 0.6168 |
| Naive Bayes | 0.8099 | 0.8759 | 0.6500 | 0.7011 | 0.6569 | 0.5737 |
| Random Forest (Ensemble) | 0.9319 | 0.9812 | 0.8920 | 0.8625 | 0.8764 | 0.8097 |

> Values come straight out of `metrics_summary.csv`, produced by
> `train_models.py`. Copy them in after you run training on your own data.

### Observations

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression achieved an accuracy of 0.8568 and the highest AUC among the non-ensemble models at 0.9602. It performed reasonably well in identifying the three fetal health classes, although its Precision (0.7406) and F1 score (0.7798) were lower than those of the Decision Tree and Random Forest. |
| Decision Tree | Decision Tree achieved an accuracy of 0.8873, with Precision of 0.8150, Recall of 0.8496, and F1 score of 0.8268. It performed better than Logistic Regression, KNN, and Naive Bayes on most classification metrics, although its AUC of 0.8786 was lower than that of Logistic Regression and Random Forest. |
| kNN | kNN achieved an accuracy of 0.8709 and a strong AUC of 0.9563. However, its Recall (0.6900) and F1 score (0.7374) were relatively lower, indicating weaker overall class-wise performance compared with the Decision Tree and Random Forest. |
| Naive Bayes | Naive Bayes produced the lowest overall performance, with an accuracy of 0.8099, Precision of 0.6500, Recall of 0.7011, F1 score of 0.6569, and MCC of 0.5737. Although its AUC was 0.8759, it was less effective than the other models for overall classification performance. |
| Random Forest (Ensemble) | Random Forest achieved the best overall performance, with the highest Accuracy (0.9319), AUC (0.9812), Precision (0.8920), F1 score (0.8764), and MCC (0.8097). Its Recall of 0.8625 was also the highest among the five models, making Random Forest the strongest model for this dataset. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** is the overall winner because it achieved the best Accuracy, AUC, Precision, Recall, F1, and MCC among all five models. Its ensemble approach provides the strongest and most consistent classification performance on the Fetal Health dataset. |

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
