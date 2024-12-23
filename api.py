import requests
import joblib


def validate_transaction():
    """
    Accept CLI input for Time and Amount, validate it with the trained model,
    and determine if the transaction is fraudulent or valid.
    """
    print("\n--- Transaction Validation ---")

    try:
        amount_input= 529.00
        time_input= 472.0
        #time_input = float(input("Enter transaction time (seconds): "))
        #amount_input = float(input("Enter transaction amount: "))
        
        # Validate input using the trained model
        loaded_model = joblib.load('random_forest_model.pkl')
        prediction = loaded_model.predict([[time_input, amount_input]])
        
        # Output prediction result
        if prediction[0] == 1:
            print("The transaction is **FRAUDULENT**.")
        else:
            print("The transaction is **VALID**.")
    except ValueError:
        print("Invalid input. Please enter numerical values for both time and amount.")
    except Exception as e:
        print(f"An error occurred: {e}")
validate_transaction()

'''
url = "http://localhost:5000/predict"

test_case = {
    "amount": 529.00,
    "timestamp": 472.0
}


response = requests.post(url, json=test_case)

print(response.json())
'''