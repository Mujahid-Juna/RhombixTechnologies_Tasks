import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
def train_model():
    # Load preprocessed data (already saved by data_preprocessing.py)
    X_train = joblib.load('outputs/X_train.pkl')
    X_test = joblib.load('outputs/X_test.pkl')
    y_train = joblib.load('outputs/y_train.pkl')
    y_test = joblib.load('outputs/y_test.pkl')
    
    # Define the model
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    
    # Hyperparameter Grid (as required by PDF)
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'],
        'class_weight': ['balanced', None]
    }
    
    print("🚀 Starting GridSearchCV with 5-Fold Cross Validation...")
    grid_search = GridSearchCV(
        estimator=log_reg,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\n🏆 Best Parameters Found:", grid_search.best_params_)
    print("📈 Best Cross-Validation F1 Score:", grid_search.best_score_)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    
    print("\n📊 Test Set Performance:")
    print(f"✅ Accuracy: {test_accuracy:.4f}")
    print(f"✅ F1-Score: {test_f1:.4f}")
    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))
    
    # Save the best model
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(best_model, 'outputs/best_model.pkl')
    
    # Save metrics
    with open('outputs/evaluation_metrics.txt', 'w') as f:
        f.write(f"Best Parameters: {grid_search.best_params_}\n")
        f.write(f"Best CV F1: {grid_search.best_score_:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test F1: {test_f1:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))
    
    print("\n💾 Model and metrics saved to 'outputs/' folder.")

if __name__ == "__main__":
    train_model()