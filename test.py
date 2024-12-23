import pickle
import numpy as np

def load_fraud_model(model_path='credit_card_fraud_model.pkl'):
    """Load the saved fraud detection model"""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print("Model loaded successfully!")
        return model
    except FileNotFoundError:
        print("Error: Model file not found!")
        return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def check_transaction(model, time, amount):
    """
    Check if a transaction is fraudulent
    
    Parameters:
    model: The loaded fraud detection model
    time (float): Time of transaction (in seconds from start of day)
    amount (float): Amount of transaction in dollars
    """
    if model is None:
        return "Error: Model not loaded"
    
    try:
        # Reshape the input for prediction
        transaction = np.array([[time, amount]])
        prediction = model.predict(transaction)[0]
        probability = model.predict_proba(transaction)[0]
        
        result = {
            'is_fraud': bool(prediction),
            'confidence': f"{probability[prediction] * 100:.2f}%",
            'prediction': 'Legitimate' if prediction == 1 else 'LEGITIMATE',
            'time': time,
            'amount': amount
        }
        
        return result
    except Exception as e:
        return f"Error making prediction: {str(e)}"

def print_result(result):
    """Print the transaction check result in a formatted way"""
    if isinstance(result, dict):
        print("\n=== Transaction Analysis ===")
        print(f"Time: {result['time']}")
        print(f"Amount: ${result['amount']:.2f}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']}")
        print("========================")
    else:
        print(result)

def main():
    # Load the model
    model = load_fraud_model()
    
    if model is None:
        return
    
    while True:
        print("\nCredit Card Fraud Detection System")
        print("1. Check a transaction")
        print("2. Exit")
        
        choice = input("Enter your choice (1-2): ")
        
        if choice == '2':
            print("Goodbye!")
            break
        
        elif choice == '1':
            try:
                time = float(input("Enter transaction time (in seconds from start of day): "))
                amount = float(input("Enter transaction amount (in dollars): "))
                
                result = check_transaction(model, time, amount)
                print_result(result)
                
            except ValueError:
                print("Error: Please enter valid numbers for time and amount")
            except Exception as e:
                print(f"Error: {str(e)}")
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()