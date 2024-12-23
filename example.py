import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def test_model(model_path, test_cases):
    """Test the model with various test cases"""
    try:
        # Load the model
        model = joblib.load(model_path)
        print("Model loaded successfully!")
        
        # Print model feature names
        if hasattr(model, 'feature_names_in_'):
            print(f"Model feature names: {model.feature_names_in_}")
        
        # Test each case
        for case in test_cases:
            features = pd.DataFrame(
                [[case['timestamp'], case['amount']]], 
                columns=['Time', 'Amount']
            )
            
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            
            print(f"\nTest case: {case}")
            print(f"Prediction: {'Malicious' if prediction == 0 else 'Legitimate'}")
            print(f"Confidence: {max(probability) * 100:.2f}%")
            
    except Exception as e:
        print(f"Error testing model: {e}")

if __name__ == "__main__":
    MODEL_PATH = r"C:\folder\Crediccard\project\Model\credit_card_fraud_model.pkl"
    
    # Define test cases
    test_cases = [
        {"timestamp": 33541, "amount": 00},
        {"timestamp": 1000, "amount": 50000},
        {"timestamp": 500, "amount": 1000}
    ]
    
    test_model(MODEL_PATH, test_cases)