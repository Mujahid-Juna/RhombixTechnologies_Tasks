# Credit Scoring Model
**Rhombix Technologies — Machine Learning Internship — Task 2 (Month 2)**

## Overview
This project builds a machine learning model that predicts whether a loan
applicant is **creditworthy**, based on financial features like income,
existing debt, credit history, and payment behavior — the same core idea
banks and lenders use to automate loan approval decisions.

## Dataset
- **Source:** a realistic **synthetic** dataset (`generate_data.py`), built
  from genuine credit-risk rules (higher debt-to-income = riskier, more late
  payments = riskier, longer credit history = safer, etc.) plus randomness,
  so it isn't a trivially "easy" pattern — just like real credit risk.
- **Why synthetic?** Real credit datasets require an internet download to
  obtain. Generating the data locally keeps this project fully
  self-contained — anyone can run it with zero downloads.
- **Size:** 3,000 applicant records.
- **Features (10):** age, annual income, employment years, existing debt,
  credit history length, number of late payments, credit utilization ratio,
  loan amount requested, bankruptcy history, debt-to-income ratio.
- **Target:** `creditworthy` (1 = yes, 0 = no) — roughly balanced (55%/45%).

## Approach
1. **Generate/load the data**, check the class balance.
2. **Split** into 80% train / 20% test (stratified).
3. **Scale** features with `StandardScaler`.
4. **Train 4 models** and compare them:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting
5. **Evaluate** with accuracy, precision/recall/F1, ROC-AUC, and confusion
   matrices — ROC-AUC is used to pick the best model since it better reflects
   performance on imbalanced/risk-based classification than accuracy alone.
6. **Visualize**: class balance, confusion matrix, model comparison chart,
   ROC curves, and feature importance (which factors matter most).
7. **Save** the best model + scaler with `joblib` for reuse without retraining.

## Results
| Model                | Accuracy | ROC-AUC |
|-----------------------|----------|---------|
| **Logistic Regression** ⭐ | 78.17%   | **0.8686** |
| Gradient Boosting      | 78.17%   | 0.8545  |
| Random Forest          | 78.00%   | 0.8523  |
| Decision Tree          | 73.00%   | 0.8146  |

Logistic Regression was selected as the best model (highest ROC-AUC) and
saved to `models/best_model.joblib`.

**Top predictive factors** (from the model's coefficients): debt-to-income
ratio, annual income, and bankruptcy history — which matches how real-world
credit scoring works.

## Project Structure
```
credit_scoring_task/
├── generate_data.py          # creates the synthetic applicant dataset
├── credit_scoring.py         # main training + evaluation pipeline
├── predict.py                 # loads the saved model, predicts on example applicants
├── requirements.txt
├── README.md
├── credit_data.csv            # generated dataset (created on first run)
├── models/
│   ├── best_model.joblib      # trained Logistic Regression model
│   └── scaler.joblib          # feature scaler used at training time
└── outputs/
    ├── class_balance.png
    ├── confusion_matrix.png
    ├── model_comparison.png
    ├── roc_curves.png
    └── feature_importance.png
```

## How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data, train all models, evaluate, and save the best one
python3 credit_scoring.py

# 3. Load the saved model and predict on example applicants
python3 predict.py
```
All charts are saved automatically to the `outputs/` folder.

## Key Learnings
- How to build a realistic synthetic dataset when real data isn't accessible.
- How to handle a binary classification problem (creditworthy vs not).
- Why ROC-AUC is a better model-selection metric than plain accuracy for
  risk-based decisions.
- How to interpret model coefficients / feature importance to explain
  *why* a model made a decision — critical in finance, where decisions must
  be explainable.
- How to persist and reuse a trained model for new predictions.

## Possible Extensions
- Train on a real dataset (e.g., the UCI German Credit dataset or
  LendingClub data) once internet access is available.
- Add SHAP values for per-applicant explainability.
- Deploy behind a simple Flask/FastAPI endpoint as a loan-decision API.
- Add fairness checks across demographic groups (a real concern in credit
  scoring systems).

---
**Intern:** Mujahid Juna
**Internship:** Machine Learning — Rhombix Technologies
**Task:** Task 2 – Credit Scoring Model (Month 2)
