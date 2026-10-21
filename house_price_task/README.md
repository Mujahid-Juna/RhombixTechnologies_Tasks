# House Price Prediction Model
**Rhombix Technologies — Machine Learning Internship — Task 2 (Month 4)**

## Overview
A regression model that predicts the selling price of a house from its size,
location, number of rooms, age and extras (garage, garden, distance to the
city centre). It includes data cleaning, feature engineering, model
comparison, hyperparameter tuning with GridSearchCV, and a command-line
tool for getting predictions.

## Dataset
- **Synthetic but realistic** (`generate_data.py`): 2,000 houses. Prices follow
  real-estate logic (bigger = pricier, newer = pricier, better neighbourhood =
  pricier, far from centre = cheaper) plus random market noise.
- **Why synthetic?** Real housing datasets need an internet download, and the
  classic Boston Housing dataset has been removed from scikit-learn. Generating
  the data locally means the project runs anywhere with no downloads.
- **Deliberately messy:** missing values are injected into four columns so the
  cleaning step is genuine.
- Because the data is generated from known rules, the results below show that
  the pipeline works correctly. They are not claims about real housing markets.

## Approach
1. **Clean:** fill missing numbers with the median and missing categories with
   the most common value (`SimpleImputer`).
2. **Feature engineering** (`features.py`): `house_age`, `total_rooms`,
   `size_per_room`, `is_new`.
3. **Preprocess:** scale numeric features, one-hot encode `location`, all inside
   a scikit-learn `Pipeline` so new data is treated identically.
4. **Compare 5 models:** Linear Regression, Ridge, Decision Tree, Random Forest,
   Gradient Boosting.
5. **Tune** Gradient Boosting with `GridSearchCV` (5-fold cross-validation)
   over number of trees, learning rate and tree depth.
6. **Evaluate** on a held-out 20% test set with R², MAE and RMSE.
7. **Save** the final pipeline and provide a command-line predictor.

## Results (20% held-out test set)
| Model | R² | MAE |
|---|---|---|
| Gradient Boosting | 0.924 | 22,011 |
| Random Forest | 0.917 | 22,924 |
| Linear Regression | 0.914 | 23,575 |
| Ridge Regression | 0.914 | 23,588 |
| Decision Tree | 0.842 | 31,098 |

**Tuning:** the best settings were 200 trees, learning rate 0.1, max depth 2,
with a cross-validated R² of 0.932. On the test set the tuned model scored
R² 0.922 (MAE 22,348) versus 0.924 (MAE 22,011) untuned. That difference is
within noise, so tuning did not meaningfully help on this dataset. The data
contains about 7% random price noise, which caps how well any model can do.
Choosing settings by cross-validation is still the correct method.

**What drives price** (feature importance chart): house size dominates, then
location (Outskirts and Downtown), then bedrooms and age.

## Project Structure
```
house_price_task/
├── generate_data.py      # creates the synthetic housing dataset
├── features.py           # feature engineering shared by training + prediction
├── house_price.py        # cleaning, comparison, tuning, evaluation, saving
├── predict.py            # command-line price predictor
├── requirements.txt
├── README.md
├── house_data.csv
├── models/house_price_model.joblib
└── outputs/              # charts (comparison, actual vs predicted, residuals, importance)
```

## How to Run
```bash
pip install -r requirements.txt
python house_price.py          # train, tune, evaluate, save
python predict.py --demo       # three example predictions
python predict.py              # interactive: enter your own house
```

## Possible Extensions
- Use XGBoost (the task mentions it; `pip install xgboost`) in place of
  scikit-learn's Gradient Boosting.
- Train on a real dataset such as a Kaggle housing dataset.
- Wrap the model in a small Streamlit dashboard.

---
**Intern:** Mujahid Juna
**Internship:** Machine Learning — Rhombix Technologies
**Task:** Task 2 – House Price Prediction Model (Month 4)
