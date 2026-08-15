"""
predict.py
Load the trained credit scoring model and predict creditworthiness for
a few example applicants. Demonstrates how the saved model would be used
in a real application (e.g., a loan approval system).

Usage:
    python3 predict.py
"""

import joblib
import pandas as pd

FEATURES = [
    "age", "annual_income", "employment_years", "existing_debt",
    "credit_history_years", "num_late_payments", "credit_utilization",
    "loan_amount_requested", "has_bankruptcy", "debt_to_income",
]


def main():
    model = joblib.load("models/best_model.joblib")
    scaler = joblib.load("models/scaler.joblib")

    # A few example applicants to test the model on.
    # Feel free to edit these values and re-run to try your own scenarios!
    applicants = pd.DataFrame([
        {  # Strong applicant: high income, low debt, long history, no late payments
            "age": 42, "annual_income": 95000, "employment_years": 15,
            "existing_debt": 8000, "credit_history_years": 20,
            "num_late_payments": 0, "credit_utilization": 0.15,
            "loan_amount_requested": 15000, "has_bankruptcy": 0,
            "debt_to_income": 8000 / 95000,
        },
        {  # Risky applicant: low income, high debt, short history, late payments
            "age": 24, "annual_income": 28000, "employment_years": 1,
            "existing_debt": 22000, "credit_history_years": 2,
            "num_late_payments": 5, "credit_utilization": 0.9,
            "loan_amount_requested": 20000, "has_bankruptcy": 1,
            "debt_to_income": 22000 / 28000,
        },
        {  # Borderline applicant
            "age": 33, "annual_income": 52000, "employment_years": 5,
            "existing_debt": 15000, "credit_history_years": 8,
            "num_late_payments": 2, "credit_utilization": 0.5,
            "loan_amount_requested": 12000, "has_bankruptcy": 0,
            "debt_to_income": 15000 / 52000,
        },
    ])

    X_scaled = scaler.transform(applicants[FEATURES])
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    labels = ["Strong applicant", "Risky applicant", "Borderline applicant"]
    print("\nCredit Scoring Predictions")
    print("=" * 50)
    for label, pred, proba in zip(labels, predictions, probabilities):
        verdict = "CREDITWORTHY" if pred == 1 else "NOT CREDITWORTHY"
        print(f"{label}: {verdict}  (confidence: {proba*100:.1f}%)")


if __name__ == "__main__":
    main()
