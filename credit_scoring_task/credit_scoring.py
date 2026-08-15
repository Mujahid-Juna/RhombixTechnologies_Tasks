"""
Credit Scoring Model
Rhombix Technologies - Machine Learning Internship - Task 2 (Month 2)

Predicts whether a loan applicant is creditworthy based on financial
features (income, debt, credit history, etc.). Covers the full ML pipeline:
data generation -> preprocessing -> training -> evaluation -> visualization.

Dataset: a realistic SYNTHETIC dataset (see generate_data.py) built from
sensible credit-risk rules, since real credit datasets require an internet
download. This keeps the whole project runnable offline with no downloads.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
)
import joblib

from generate_data import generate_dataset

OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "age", "annual_income", "employment_years", "existing_debt",
    "credit_history_years", "num_late_payments", "credit_utilization",
    "loan_amount_requested", "has_bankruptcy", "debt_to_income",
]
TARGET = "creditworthy"


def load_data():
    if os.path.exists("credit_data.csv"):
        df = pd.read_csv("credit_data.csv")
    else:
        df = generate_dataset()
        df.to_csv("credit_data.csv", index=False)
    return df


def visualize_target_balance(df, filename="class_balance.png"):
    counts = df[TARGET].value_counts().sort_index()
    labels = ["Not Creditworthy", "Creditworthy"]
    plt.figure(figsize=(6, 5))
    plt.bar(labels, counts.values, color=["#C44E52", "#55A868"])
    plt.ylabel("Number of Applicants")
    plt.title("Class Balance: Creditworthy vs Not")
    for i, v in enumerate(counts.values):
        plt.text(i, v + 15, str(v), ha="center")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved class balance chart -> {path}")


def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    results = {}
    print("\n" + "=" * 55)
    print("TRAINING & EVALUATING MODELS")
    print("=" * 55)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        results[name] = {
            "model": model, "accuracy": acc, "auc": auc,
            "y_pred": y_pred, "y_proba": y_proba,
        }
        print(f"\n{name}: accuracy = {acc:.4f}, ROC-AUC = {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Not Creditworthy", "Creditworthy"]))

    best_name = max(results, key=lambda k: results[k]["auc"])
    print("=" * 55)
    print(f"BEST MODEL: {best_name} (ROC-AUC = {results[best_name]['auc']:.4f})")
    print("=" * 55)
    return results, best_name


def plot_confusion_matrix(y_test, y_pred, model_name, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["Not Creditworthy", "Creditworthy"])
    fig, ax = plt.subplots(figsize=(6.5, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix -> {path}")


def plot_model_comparison(results, filename="model_comparison.png"):
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    aucs = [results[n]["auc"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, accs, width, label="Accuracy", color="#4C72B0")
    ax.bar(x + width/2, aucs, width, label="ROC-AUC", color="#55A868")
    ax.set_ylim(0.5, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Accuracy vs ROC-AUC")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved model comparison chart -> {path}")


def plot_roc_curves(results, y_test, filename="roc_curves.png"):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, res in results.items():
        RocCurveDisplay.from_predictions(y_test, res["y_proba"], name=name, ax=ax)
    ax.set_title("ROC Curves - All Models")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved ROC curves -> {path}")


def plot_feature_importance(model, model_name, filename="feature_importance.png"):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        xlabel = "Importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        xlabel = "abs(Coefficient) - larger = stronger influence"
    else:
        print(f"{model_name} has no usable importance/coefficient attribute, skipping chart.")
        return
    order = np.argsort(importances)
    plt.figure(figsize=(8, 6))
    plt.barh([FEATURES[i] for i in order], importances[order], color="#8172B2")
    plt.xlabel(xlabel)
    plt.title(f"Feature Importance - {model_name}")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved feature importance chart -> {path}")


def main():
    print("Loading / generating dataset...")
    df = load_data()
    print(f"Dataset shape: {df.shape[0]} applicants, {len(FEATURES)} features")
    visualize_target_balance(df)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results, best_name = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test)

    best_model = results[best_name]["model"]
    best_pred = results[best_name]["y_pred"]

    plot_confusion_matrix(y_test, best_pred, best_name)
    plot_model_comparison(results)
    plot_roc_curves(results, y_test)
    plot_feature_importance(best_model, best_name)

    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    print(f"\nSaved trained model -> {MODEL_DIR}/best_model.joblib")
    print(f"Saved scaler        -> {MODEL_DIR}/scaler.joblib")
    print("\nDone! Check the 'outputs' folder for charts and 'models' folder for the saved model.")


if __name__ == "__main__":
    main()
