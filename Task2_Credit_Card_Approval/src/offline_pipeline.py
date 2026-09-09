"""
FIXED CREDIT CARD APPROVAL PIPELINE
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

def main():
    print("="*50)
    print("CREDIT CARD APPROVAL SYSTEM")
    print("="*50)
    
    # 1. LOAD DATA
    print("\n📥 Loading dataset...")
    
    # Check if crx.data exists
    if not os.path.exists("crx.data"):
        print("❌ Error: crx.data file not found!")
        print("   Make sure the file is in the main project folder.")
        return
    
    # Load with proper data types
    df = pd.read_csv("crx.data", header=None, na_values='?')
    
    # Set column names
    df.columns = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','Target']
    
    # Convert target
    df['Target'] = df['Target'].map({'+': 1, '-': 0})
    
    print(f"   ✅ Loaded {len(df)} rows")
    
    # 2. SPLIT DATA
    print("\n📊 Splitting data...")
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"   Training: {len(X_train)} rows")
    print(f"   Test: {len(X_test)} rows")
    
    # 3. PREPROCESS
    print("\n🔧 Preprocessing...")
    
    # Define columns
    num_cols = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']
    cat_cols = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']
    
    # Convert numeric columns to float
    for col in num_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    
    # Impute numeric
    num_imputer = SimpleImputer(strategy='median')
    X_train_num = num_imputer.fit_transform(X_train[num_cols])
    X_test_num = num_imputer.transform(X_test[num_cols])
    
    # Scale numeric
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)
    
    # Impute categorical
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_train_cat = cat_imputer.fit_transform(X_train[cat_cols].astype(str))
    X_test_cat = cat_imputer.transform(X_test[cat_cols].astype(str))
    
    # Encode categorical
    for i, col in enumerate(cat_cols):
        le = LabelEncoder()
        combined = np.concatenate([X_train_cat[:, i], X_test_cat[:, i]])
        le.fit(combined)
        X_train_cat[:, i] = le.transform(X_train_cat[:, i])
        X_test_cat[:, i] = le.transform(X_test_cat[:, i])
    
    # Convert to float
    X_train_cat = X_train_cat.astype(float)
    X_test_cat = X_test_cat.astype(float)
    
    # Combine
    X_train_processed = np.hstack([X_train_num_scaled, X_train_cat])
    X_test_processed = np.hstack([X_test_num_scaled, X_test_cat])
    
    # 4. SMOTE
    print("\n🔄 Balancing with SMOTE...")
    print(f"   Original: {dict(y_train.value_counts())}")
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    
    print(f"   After SMOTE: {dict(pd.Series(y_train_resampled).value_counts())}")
    
    # 5. GRID SEARCH
    print("\n🏆 Running GridSearchCV (2-3 minutes)...")
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'],
        'class_weight': ['balanced', None]
    }
    
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    # 6. RESULTS
    print("\n" + "="*50)
    print("🎉 RESULTS")
    print("="*50)
    
    print(f"\n🏆 Best Parameters: {grid_search.best_params_}")
    print(f"📈 Best CV F1 Score: {grid_search.best_score_:.4f}")
    
    y_pred = grid_search.best_estimator_.predict(X_test_processed)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n📊 Test Performance:")
    print(f"   ✅ Accuracy: {accuracy:.4f}")
    print(f"   ✅ F1-Score: {f1:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))
    
    # 7. SAVE
    print("\n💾 Saving model...")
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(grid_search.best_estimator_, 'outputs/best_model.pkl')
    
    print("\n✅ DONE!")
    print("📁 Model saved to: outputs/best_model.pkl")
    print("="*50)

if __name__ == "__main__":
    main()