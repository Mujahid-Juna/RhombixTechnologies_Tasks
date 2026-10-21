"""
Image Recognition System
Rhombix Technologies - Machine Learning Internship - Task 2

This project builds and evaluates machine learning models that recognize
handwritten digit images (0-9). It covers the full ML pipeline:
data loading -> preprocessing -> training -> evaluation -> visualization.

Dataset: scikit-learn's built-in "digits" dataset
(1,797 real 8x8 grayscale images of handwritten digits, each pixel is a
feature with a value from 0-16). This is a smaller, self-contained cousin
of the famous MNIST dataset, so no internet download is required.

Author: <your name here>
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import joblib
import os

OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    """Load the digits dataset and return images, flattened features, and labels."""
    digits = load_digits()
    X = digits.data          # shape (1797, 64) - flattened 8x8 images
    y = digits.target        # shape (1797,)    - labels 0-9
    images = digits.images   # shape (1797, 8, 8) - original 2D images
    return images, X, y


def visualize_samples(images, y, n=10, filename="sample_digits.png"):
    """Save a grid of sample images with their true labels."""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for ax, image, label in zip(axes.ravel(), images[:n], y[:n]):
        ax.imshow(image, cmap="gray")
        ax.set_title(f"Label: {label}")
        ax.axis("off")
    plt.suptitle("Sample Images from the Dataset")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved sample image grid -> {path}")


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train several models, evaluate them, and return the best one."""
    models = {
        "SVM (RBF kernel)": SVC(kernel="rbf", gamma="scale", C=10),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=500, random_state=42
        ),
    }

    results = {}
    print("\n" + "=" * 55)
    print("TRAINING & EVALUATING MODELS")
    print("=" * 55)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {"model": model, "accuracy": acc, "y_pred": y_pred}
        print(f"\n{name}: accuracy = {acc:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    print("=" * 55)
    print(f"BEST MODEL: {best_name} (accuracy = {results[best_name]['accuracy']:.4f})")
    print("=" * 55)
    return results, best_name


def plot_confusion_matrix(y_test, y_pred, model_name, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(f"Confusion Matrix - {model_name}")
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix -> {path}")


def plot_model_comparison(results, filename="model_comparison.png"):
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, accs, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    plt.ylim(0.8, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=15)
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.003, f"{acc:.3f}",
                  ha="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved model comparison chart -> {path}")


def visualize_misclassified(images_test, y_test, y_pred, n=10,
                             filename="misclassified_examples.png"):
    """Show a few images the model got wrong."""
    wrong_idx = np.where(y_test != y_pred)[0]
    if len(wrong_idx) == 0:
        print("No misclassified examples - perfect accuracy on test set!")
        return
    n = min(n, len(wrong_idx))
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for ax, idx in zip(axes.ravel(), wrong_idx[:n]):
        ax.imshow(images_test[idx], cmap="gray")
        ax.set_title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
        ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    plt.suptitle("Misclassified Examples")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved misclassified examples -> {path}")


def main():
    print("Loading dataset...")
    images, X, y = load_data()
    print(f"Dataset shape: {X.shape[0]} images, {X.shape[1]} pixel-features each, "
          f"{len(set(y))} classes (digits 0-9)")

    visualize_samples(images, y)

    # Split into train/test, keeping matching image arrays for visualization later
    X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
        X, y, images, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (helps SVM, KNN, and the neural network converge better)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results, best_name = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test)

    best_model = results[best_name]["model"]
    best_pred = results[best_name]["y_pred"]

    plot_confusion_matrix(y_test, best_pred, best_name)
    plot_model_comparison(results)
    visualize_misclassified(img_test, y_test, best_pred)

    # Save the best model + scaler so it can be reused without retraining
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    print(f"\nSaved trained model -> {MODEL_DIR}/best_model.joblib")
    print(f"Saved scaler        -> {MODEL_DIR}/scaler.joblib")

    print("\nDone! Check the 'outputs' folder for charts and 'models' folder for the saved model.")


if __name__ == "__main__":
    main()
