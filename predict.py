"""
predict.py
Load the trained model (models/best_model.joblib) and run predictions
on a few sample images from the dataset. This simulates using the model
for real-world inference after training.

Usage:
    python3 predict.py
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

MODEL_PATH = "models/best_model.joblib"
SCALER_PATH = "models/scaler.joblib"


def main():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    digits = load_digits()
    X, y, images = digits.data, digits.target, digits.images

    # Just for demo purposes, grab 6 random samples to "predict" on
    rng = np.random.default_rng(seed=7)
    idx = rng.choice(len(X), size=6, replace=False)

    X_sample = scaler.transform(X[idx])
    predictions = model.predict(X_sample)

    fig, axes = plt.subplots(1, 6, figsize=(12, 3))
    for ax, i, pred in zip(axes, idx, predictions):
        ax.imshow(images[i], cmap="gray")
        correct = "✓" if pred == y[i] else "✗"
        ax.set_title(f"Pred: {pred} {correct}\n(True: {y[i]})", fontsize=10)
        ax.axis("off")
    plt.suptitle("Sample Predictions with Trained Model")
    plt.tight_layout()
    plt.savefig("outputs/prediction_demo.png", dpi=150)
    plt.close()
    print("Saved prediction demo -> outputs/prediction_demo.png")

    for i, pred in zip(idx, predictions):
        print(f"Image #{i}: predicted={pred}, actual={y[i]}, "
              f"{'correct' if pred == y[i] else 'WRONG'}")


if __name__ == "__main__":
    main()
