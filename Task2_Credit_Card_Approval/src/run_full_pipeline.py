"""
COMPLETE CREDIT CARD APPROVAL PIPELINE
This script does everything in one go:
1. Downloads the dataset
2. Cleans the data
3. Handles missing values
4. Scales features
5. Encodes categorical data
6. Applies SMOTE for imbalance
7. Runs GridSearchCV for hyperparameter tuning
8. Shows final results
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import os
import urllib.request

def main():
    print("="*60)
    print("🚀 CREDIT CARD APPROVAL PREDICTION PIPELINE")
    print("="*60)
    
    # 1. LOAD DATA
    print("\n📥 Step 1: Downloading dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
    urllib.request.urlretrieve(url, "crx.data")
    
    column_names = [
        'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 
        'A11', 'A12', 'A13', 'A14', 'A15', 'Target'
    ]
    df = pd.read_csv("crx.data", header=None, names=column_names, na_values='?')
    df['Target'] = df['Target'].map({'+': 1, '-': 0})
    print(f"   ✅ Loaded {len(df)} rows")
    
    # 2. SPLIT DATA
    print("\n📊 Step 2: Splitting data...")
    X = df.drop('Target', axis=1)
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"   ✅ Training: {len(X_train)} rows, Test: {len(X_test)} rows")
    
    # 3. PREPROCESS DATA
    print("\n🔧 Step 3: Preprocessing data...")
    
    # Identify numeric and categorical columns
    num_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
    cat_cols = [col for col in X.columns if X[col].dtype == 'object']
    
    print(f"   🔢 Numeric columns: {len(num_cols)}")
    print(f"   📝 Categorical columns: {len(cat_cols)}")
    
    # Numeric: Impute + Scale
    num_imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_num = num_imputer.fit_transform(X_train[num_cols])
    X_test_num = num_imputer.transform(X_test[num_cols])
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)
    
    # Categorical: Impute + Encode
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_train_cat = cat_imputer.fit_transform(X_train[cat_cols])
    X_test_cat = cat_imputer.transform(X_test[cat_cols])
    
    encoders = {}
    X_train_cat_encoded_list = []
    X_test_cat_encoded_list = []
    for i, col in enumerate(cat_cols):
        le = LabelEncoder()
        combined = np.concatenate([X_train_cat[:, i], X_test_cat[:, i]])
        le.fit(combined)
        X_train_cat_encoded_list.append(le.transform(X_train_cat[:, i]))
        X_test_cat_encoded_list.append(le.transform(X_test_cat[:, i]))
        encoders[col] = le
    
    X_train_cat_encoded = np.array(X_train_cat_encoded_list).T
    X_test_cat_encoded = np.array(X_test_cat_encoded_list).T
    
    # Combine
    X_train_processed = np.hstack([X_train_num_scaled, X_train_cat_encoded])
    X_test_processed = np.hstack([X_test_num_scaled, X_test_cat_encoded])
    
    # 4. APPLY SMOTE
    print("\n🔄 Step 4: Balancing data with SMOTE...")
    print(f"   Original class distribution:")
    print(f"   {y_train.value_counts().to_dict()}")
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    
    print(f"   After SMOTE:")
    print(f"   {pd.Series(y_train_resampled).value_counts().to_dict()}")
    
    # 5. GRIDSEARCHCV
    print("\n🏆 Step 5: Running GridSearchCV (this takes 2-3 minutes)...")
    
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'],
        'class_weight': ['balanced', None]
    }
    
    grid_search = GridSearchCV(
        estimator=log_reg,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    # 6. RESULTS
    print("\n" + "="*60)
    print("🎉 RESULTS")
    print("="*60)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"\n🏆 Best Parameters: {best_params}")
    print(f"📈 Best Cross-Validation F1 Score: {best_score:.4f}")
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_processed)
    
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    
    print(f"\n📊 Test Set Performance:")
    print(f"   ✅ Accuracy: {test_accuracy:.4f}")
    print(f"   ✅ F1-Score: {test_f1:.4f}")
    
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))
    
    # 7. SAVE MODEL
    print("\n💾 Step 6: Saving model...")
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(best_model, 'outputs/best_model.pkl')
    joblib.dump(scaler, 'outputs/scaler.pkl')
    joblib.dump(num_imputer, 'outputs/num_imputer.pkl')
    joblib.dump(cat_imputer, 'outputs/cat_imputer.pkl')
    joblib.dump(encoders, 'outputs/encoders.pkl')
    joblib.dump(num_cols, 'outputs/num_cols.pkl')
    joblib.dump(cat_cols, 'outputs/cat_cols.pkl')
    
    # Save metrics
    with open('outputs/evaluation_metrics.txt', 'w') as f:
        f.write(f"Best Parameters: {best_params}\n")
        f.write(f"Best CV F1 Score: {best_score:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test F1-Score: {test_f1:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))
    
    print("   ✅ Model and metrics saved to 'outputs/' folder!")
    
    # 8. SHOW SUMMARY
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE!")
    print("="*60)
    print("\n📁 Files saved in 'outputs/' folder:")
    print("   - best_model.pkl  (trained model)")
    print("   - evaluation_metrics.txt  (results)")
    print("\n🎯 Next steps:")
    print("   1. Take a screenshot of these results")
    print("   2. Push to GitHub")
    print("   3. Post on LinkedIn tagging @Rhombix Technologies")
    print("="*60)

if __name__ == "__main__":
    main()