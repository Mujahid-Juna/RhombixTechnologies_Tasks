"""
generate_data.py
Generates a realistic synthetic credit-applicant dataset.

Why synthetic data?
Real-world credit datasets (like German Credit or LendingClub) require an
internet download. To keep this project fully self-contained and runnable
anywhere with no downloads, this script generates a realistic dataset using
sensible financial rules + randomness - the same approach banks use
internally to stress-test scoring models before using real data.

The rules encode genuine credit-risk logic (higher debt-to-income = riskier,
more late payments = riskier, longer credit history = safer, etc.), then add
random "noise" so it isn't a trivial, perfectly separable pattern - just like
real credit risk isn't 100% predictable either.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)
N_SAMPLES = 3000


def generate_dataset(n=N_SAMPLES):
    age = RNG.integers(21, 65, n)
    annual_income = RNG.normal(55000, 22000, n).clip(12000, 220000)
    employment_years = RNG.integers(0, 35, n)
    existing_debt = RNG.normal(15000, 12000, n).clip(0, 120000)
    credit_history_years = RNG.integers(0, 30, n)
    num_late_payments = RNG.poisson(1.2, n)
    credit_utilization = RNG.uniform(0, 1, n)  # % of available credit used
    loan_amount_requested = RNG.normal(18000, 9000, n).clip(1000, 80000)
    has_bankruptcy = RNG.choice([0, 1], size=n, p=[0.93, 0.07])

    debt_to_income = existing_debt / annual_income

    # Rule-based "credit risk score" - higher = more creditworthy.
    # Weights reflect realistic factors used in real scoring models.
    score = (
        0.35 * (annual_income / annual_income.max())
        - 0.30 * debt_to_income
        + 0.15 * (credit_history_years / 30)
        - 0.20 * (num_late_payments / (num_late_payments.max() + 1))
        - 0.15 * credit_utilization
        - 0.25 * has_bankruptcy
        + 0.10 * (employment_years / 35)
        - 0.05 * (loan_amount_requested / loan_amount_requested.max())
    )

    # Add random noise so it's not a perfectly clean separation (real life isn't).
    score_noisy = score + RNG.normal(0, 0.12, n)

    # Threshold at the median-ish point, then label creditworthy (1) / not (0).
    threshold = np.percentile(score_noisy, 45)
    creditworthy = (score_noisy > threshold).astype(int)

    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income.round(2),
        "employment_years": employment_years,
        "existing_debt": existing_debt.round(2),
        "credit_history_years": credit_history_years,
        "num_late_payments": num_late_payments,
        "credit_utilization": credit_utilization.round(3),
        "loan_amount_requested": loan_amount_requested.round(2),
        "has_bankruptcy": has_bankruptcy,
        "debt_to_income": debt_to_income.round(3),
        "creditworthy": creditworthy,
    })
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("credit_data.csv", index=False)
    print(f"Generated {len(df)} synthetic applicant records -> credit_data.csv")
    print(f"Creditworthy: {df['creditworthy'].sum()} "
          f"({df['creditworthy'].mean()*100:.1f}%)")
    print(f"Not creditworthy: {(df['creditworthy']==0).sum()} "
          f"({(df['creditworthy']==0).mean()*100:.1f}%)")
