# Task 2: Predicting Credit Card Approvals

## Objective
Build an automatic credit card approval predictor using Logistic Regression with Hyperparameter Optimization (GridSearchCV), handling missing values, categorical encoding, feature scaling, and imbalanced data (SMOTE).

## Dataset
UCI Credit Screening Dataset (crx.data)

## How to Run
1. Create virtual environment: `python -m venv venv`
2. Activate it: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
3. Install dependencies: `pip install -r requirements.txt`
4. Run preprocessing: `python src/data_preprocessing.py`
5. Train & Tune Model: `python src/model_training.py`
6. Make a Prediction: `python src/predict.py`

## Results
Best parameters and F1-scores are saved in the `outputs/` folder.

**Intern:** Mujahid Juna Internship: Machine Learning — Rhombix Technologies **Task:** Task 2 – Credit Card Approval Prediction
