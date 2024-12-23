import scipy.stats as stats
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load data
data = pd.read_csv('creditcard.csv')
data.info()

# Check for missing values
if data.isnull().values.any():
    print("Dataset has missing values. Consider handling them.")
else:
    print("No missing values found.")

# Separate fraud and valid cases
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
print(f"Fraud cases: {fraud.shape[0]}, Valid cases: {valid.shape[0]}")

# Prepare features and labels
X = data[['Time', 'Amount']]
Y = data["Class"]
print(f"Features shape: {X.shape}, Labels shape: {Y.shape}")

xData = X.values
yData = Y.values

# Train-test split
xTrain, xTest, yTrain, yTest = train_test_split(
    xData, yData, test_size=0.2, random_state=40)

# Train Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)
yPred = rfc.predict(xTest)

# Evaluate accuracy
acc = accuracy_score(yTest, yPred)
print("The accuracy of Random Forest is {:.2f}%".format(acc * 100))

# Save the model
joblib.dump(rfc, 'random_forest_model.pkl')
print("Model saved as random_forest_model.pkl")

# Display feature importances
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rfc.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("Features used for training:")
print(feature_importances)

# CLI Input for Transaction Validation
def validate_transaction():
    """
    Accept CLI input for Time and Amount, validate it with the trained model,
    and determine if the transaction is fraudulent or valid.
    """
    print("\n--- Transaction Validation ---")
    try:
        time_input = float(input("Enter transaction time (seconds): "))
        amount_input = float(input("Enter transaction amount: "))
        
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

# Validate a transaction
validate_transaction()
