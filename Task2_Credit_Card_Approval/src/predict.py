import joblib
import numpy as np
import pandas as pd

def predict_new_application():
    model = joblib.load('outputs/best_model.pkl')
    scaler = joblib.load('outputs/scaler.pkl')
    encoder = joblib.load('outputs/encoder.pkl')
    num_imputer = joblib.load('outputs/num_imputer.pkl')
    cat_imputer = joblib.load('outputs/cat_imputer.pkl')
    
    print("📝 Enter applicant details (15 features separated by commas):")
    print("(Example: a,30.83,0.0,u,g,w,v,1.25,t,t,1,f,g,00202,0.0)")
    user_input = input("> ").strip().split(',')
    
    if len(user_input) != 15:
        print("❌ Error: Please enter exactly 15 features.")
        return
    
    column_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15']
    new_data = pd.DataFrame([user_input], columns=column_names).replace('?', np.nan)
    
    num_cols = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']
    cat_cols = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']
    
    new_num = num_imputer.transform(new_data[num_cols].astype(float))
    new_cat = cat_imputer.transform(new_data[cat_cols])
    new_num_scaled = scaler.transform(new_num)
    new_cat_encoded = encoder.transform(new_cat)
    final_input = np.hstack([new_num_scaled, new_cat_encoded])
    
    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0]
    
    if prediction == 1:
        print("\n✅ DECISION: APPROVED")
    else:
        print("\n❌ DECISION: REJECTED")
    print(f"🔢 Confidence -> Reject: {probability[0]:.2f}, Approve: {probability[1]:.2f}")

if __name__ == "__main__":
    predict_new_application()
