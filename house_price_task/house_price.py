"""
House Price Prediction Model
Rhombix Technologies - Machine Learning Internship - Task 2 (Month 4)

Predicts the selling price of a house from its size, location, rooms and age.
Full pipeline: data cleaning -> feature engineering -> model comparison ->
hyperparameter tuning -> evaluation -> saved model.

Dataset: realistic SYNTHETIC housing data (see generate_data.py), because
real housing datasets require an internet download.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from generate_data import generate_dataset
from features import add_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES, TARGET

OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    if os.path.exists("house_data.csv"):
        return pd.read_csv("house_data.csv")
    df = generate_dataset()
    df.to_csv("house_data.csv", index=False)
    return df


def build_preprocessor():
    """Fill missing values, scale numbers, one-hot encode the location text."""
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
        "pred": pred,
    }


def plot_actual_vs_predicted(y_test, pred, name, filename="actual_vs_predicted.png"):
    plt.figure(figsize=(6.5, 6))
    plt.scatter(y_test, pred, alpha=0.4, color="#4C72B0")
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.title(f"Actual vs Predicted - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved -> {OUTPUT_DIR}/{filename}")


def plot_residuals(y_test, pred, filename="residuals.png"):
    residuals = y_test - pred
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=40, color="#55A868", edgecolor="white")
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Error (actual - predicted)")
    plt.ylabel("Number of houses")
    plt.title("Prediction Errors (Residuals)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved -> {OUTPUT_DIR}/{filename}")


def plot_model_comparison(results, filename="model_comparison.png"):
    names = list(results.keys())
    r2 = [results[n]["R2"] for n in names]
    mae = [results[n]["MAE"] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(names, r2, color="#4C72B0")
    axes[0].set_title("R2 score (higher is better)")
    axes[0].set_ylim(0.5, 1.0)
    axes[1].bar(names, mae, color="#C44E52")
    axes[1].set_title("Mean Absolute Error (lower is better)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved -> {OUTPUT_DIR}/{filename}")


def plot_feature_importance(pipeline, filename="feature_importance.png"):
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return
    prep = pipeline.named_steps["prep"]
    names = list(NUMERIC_FEATURES) + list(
        prep.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(CATEGORICAL_FEATURES)
    )
    imp = model.feature_importances_
    order = np.argsort(imp)[-12:]
    plt.figure(figsize=(8, 6))
    plt.barh([names[i] for i in order], imp[order], color="#8172B2")
    plt.xlabel("Importance")
    plt.title("What drives house price? (top features)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved -> {OUTPUT_DIR}/{filename}")


def main():
    print("Loading data...")
    raw = load_data()
    print(f"{len(raw)} houses, missing values per column:")
    print(raw.isnull().sum()[raw.isnull().sum() > 0].to_string())

    df = add_features(raw)
    X = df[ALL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---- Step 1: compare several models with default-ish settings ----
    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    results = {}
    print("\n" + "=" * 60 + "\nBASELINE MODEL COMPARISON\n" + "=" * 60)
    for name, model in candidates.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        pipe.fit(X_train, y_train)
        res = evaluate(pipe, X_test, y_test)
        results[name] = res
        print(f"{name:20s} R2={res['R2']:.4f}  MAE={res['MAE']:,.0f}  RMSE={res['RMSE']:,.0f}")
    plot_model_comparison(results)

    # ---- Step 2: hyperparameter tuning on Gradient Boosting ----
    print("\n" + "=" * 60 + "\nHYPERPARAMETER TUNING (GridSearchCV)\n" + "=" * 60)
    base = Pipeline([("prep", build_preprocessor()),
                     ("model", GradientBoostingRegressor(random_state=42))])
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3, 4],
    }
    grid = GridSearchCV(base, param_grid, cv=5, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best cross-validated R2: {grid.best_score_:.4f}")

    best = grid.best_estimator_
    tuned = evaluate(best, X_test, y_test)
    base_gb = results["Gradient Boosting"]
    print(f"\nGradient Boosting BEFORE tuning: R2={base_gb['R2']:.4f}  MAE={base_gb['MAE']:,.0f}")
    print(f"Gradient Boosting AFTER  tuning: R2={tuned['R2']:.4f}  MAE={tuned['MAE']:,.0f}")

    # ---- Step 3: charts + save ----
    plot_actual_vs_predicted(y_test, tuned["pred"], "Tuned Gradient Boosting")
    plot_residuals(y_test, tuned["pred"])
    plot_feature_importance(best)

    joblib.dump(best, os.path.join(MODEL_DIR, "house_price_model.joblib"))
    print(f"\nSaved trained model -> {MODEL_DIR}/house_price_model.joblib")
    print("Done! Try the prediction tool:  python predict.py")


if __name__ == "__main__":
    main()
