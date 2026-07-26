# Image Recognition System
**Rhombix Technologies — Machine Learning Internship — Task 2 (Month 1)**

## Overview
This project applies machine learning to **image recognition**: given a small
grayscale image of a handwritten digit, the model predicts which digit (0–9)
it is. This mirrors the task brief — analyzing digital images by their pixel
values to classify them — the same core idea behind face recognition and
other image-based ML systems.

## Dataset
- **Source:** `sklearn.datasets.load_digits` (built into scikit-learn, based
  on real handwritten digit scans — a smaller sibling of the well-known
  MNIST dataset).
- **Size:** 1,797 images, 8×8 pixels each, grayscale (pixel values 0–16).
- **Classes:** 10 (digits 0 through 9).

## Approach
1. **Load & explore** the dataset, visualize sample images.
2. **Preprocess**: flatten each 8×8 image into a 64-value feature vector and
   standardize features with `StandardScaler`.
3. **Train/test split**: 80% train, 20% test (stratified so each digit is
   represented proportionally).
4. **Train 4 models** and compare them:
   - Support Vector Machine (RBF kernel)
   - Random Forest
   - K-Nearest Neighbors
   - Neural Network (MLPClassifier — a multi-layer perceptron)
5. **Evaluate** each with accuracy, precision/recall/F1 (classification
   report), and a confusion matrix.
6. **Visualize** results: sample digits, model comparison bar chart,
   confusion matrix, and misclassified examples.
7. **Save** the best-performing model (and its scaler) to disk with
   `joblib` so it can be reloaded and reused without retraining.

## Results
| Model                  | Accuracy |
|-------------------------|----------|
| **SVM (RBF kernel)** ⭐  | **98.06%** |
| Neural Network (MLP)    | 96.94%   |
| K-Nearest Neighbors     | 96.67%   |
| Random Forest           | 96.39%   |

The SVM was selected as the best model and saved to `models/best_model.joblib`.

## Project Structure
```
image_recognition_task/
├── image_recognition.py     # main training + evaluation pipeline
├── predict.py                # loads the saved model and runs sample predictions
├── requirements.txt
├── README.md
├── models/
│   ├── best_model.joblib     # trained SVM model
│   └── scaler.joblib         # feature scaler used at training time
└── outputs/
    ├── sample_digits.png
    ├── model_comparison.png
    ├── confusion_matrix.png
    ├── misclassified_examples.png
    └── prediction_demo.png
```

## How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all models, evaluate, and save the best one
python3 image_recognition.py

# 3. Load the saved model and predict on new samples
python3 predict.py
```
All charts are saved automatically to the `outputs/` folder.

## Key Learnings
- How to preprocess image data for classical ML models (flattening +
  scaling pixel values).
- How to compare multiple algorithms fairly on the same train/test split.
- How to read and interpret a confusion matrix and classification report.
- How to persist a trained model for reuse (`joblib`) instead of retraining
  every time.

## Possible Extensions
- Swap in a Convolutional Neural Network (CNN) with TensorFlow/Keras on the
  full MNIST dataset (28×28 images, 70,000 samples) for higher accuracy.
- Add real-time webcam digit recognition using OpenCV.
- Deploy the model behind a simple Flask/FastAPI endpoint.

---
**Intern:** _Mujahid Juna_
**Internship:** Machine Learning — Rhombix Technologies
**Task:** Task 2 – Image Recognition
