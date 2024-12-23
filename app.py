from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
MODEL_PATH = r"Model/credit_card_fraud_model.pkl"
FEATURE_NAMES = ['Time', 'Amount']  # Must match model's training features

class ModelManager:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        """Load the ML model with proper error handling"""
        try:
            model = joblib.load(self.model_path)
            logger.info("Model loaded successfully!")
            if hasattr(model, 'feature_names_in_'):
                logger.info(f"Model's feature names: {model.feature_names_in_}")
            return model
        except FileNotFoundError:
            logger.error(f"Error: Model file not found at {self.model_path}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during model load: {e}")
            return None

    def predict(self, features_df):
        """Make prediction using the model"""
        try:
            prediction = self.model.predict(features_df)[0]
            proba = self.model.predict_proba(features_df)[0]
            return {
                'prediction': "Malicious" if prediction == 0 else "Legitimate",
                'confidence': f"{max(proba) * 100:.2f}%"
            }
        except Exception as e:
            raise ValueError(f"Prediction error: {str(e)}")

# Initialize model manager
model_manager = ModelManager(MODEL_PATH)

if model_manager.model is None:
    raise RuntimeError("The model did not load. Exiting application.")

def prepare_features(timestamp, amount):
    """Prepare features in consistent order with proper names"""
    try:
        return pd.DataFrame(
            [[float(timestamp), float(amount)]], 
            columns=FEATURE_NAMES
        )
    except ValueError as e:
        raise ValueError(f"Invalid input values: {str(e)}")

def validate_transaction_data(data):
    """Validate incoming transaction data"""
    if not isinstance(data, dict):
        raise ValueError("Invalid request data format")
    
    required_fields = ['timestamp', 'amount']
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
    
    return True

# Route handlers
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")

@app.route('/verify-transaction')
def verify_transaction():
    return render_template("verify_transaction.html")

@app.route('/blockchain')
def blockchain():
    return render_template("blockchain.html")

@app.route('/classify-transaction', methods=['POST'])
def classify_transaction():
    """Main endpoint for transaction classification"""
    try:
        # Get and validate JSON data
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "No data provided"
            }), 400

        # Validate required fields
        validate_transaction_data(data)

        # Prepare features
        features_df = prepare_features(
            timestamp=data['timestamp'],
            amount=data['amount']
        )

        # Make prediction
        result = model_manager.predict(features_df)

        # Log successful prediction
        logger.info(f"Successfully classified transaction: {result['prediction']}")

        # Return only the prediction and confidence in the response
        return jsonify({
            "status": "success",
            "result": result['prediction'],  # Return prediction string directly
            "confidence": result['confidence'],  # Confidence as percentage
            "details": {
                "timestamp": data['timestamp'],
                "amount": data['amount'],
                "processed_at": datetime.now().isoformat()
            }
        })

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400
    except Exception as e:
        logger.error(f"Unexpected error in classify_transaction: {e}")
        return jsonify({
            "status": "error",
            "message": "An unexpected error occurred processing your request"
        }), 500


@app.route('/health')
def health_check():
    """Endpoint to verify service health"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model_manager.model is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        "status": "error",
        "message": "Requested resource not found"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "status": "error",
        "message": "Internal server error"
    }), 500

if __name__ == "__main__":
    app.run(debug=True)
