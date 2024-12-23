import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score, 
                           precision_score, recall_score, f1_score, 
                           matthews_corrcoef, confusion_matrix)

# Load the data
data = pd.read_csv("creditcard.csv")

# Separate fraud and valid transactions
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

# Randomly sample 500 legitimate transactions
valid_sample = valid.sample(n=500, random_state=42)

# Combine fraud and sampled legitimate transactions
balanced_data = pd.concat([fraud, valid_sample])

# Keep only Time and Amount features
features = ['Time', 'Amount']
X = balanced_data[features]
Y = balanced_data['Class']

# Print dataset information
print("Dataset composition:")
print(f"Total transactions: {len(balanced_data)}")
print(f"Fraud Cases: {len(fraud)}")
print(f"Valid Transactions: {len(valid_sample)}")
print(f"Fraud Percentage: {(len(fraud)/len(balanced_data))*100:.2f}%")

# Plot distribution of Amount for both classes
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='Class', y='Amount', data=balanced_data)
plt.title('Amount Distribution by Class')

plt.subplot(1, 2, 2)
sns.boxplot(x='Class', y='Time', data=balanced_data)
plt.title('Time Distribution by Class')
plt.show()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the training data
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# Train Random Forest model
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_sm, y_train_sm)

# Make predictions
y_pred = rfc.predict(X_test)

# Print model evaluation metrics
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"Matthews Correlation Coefficient: {matthews_corrcoef(y_test, y_pred):.4f}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', 
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'])
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()

# Save the model
import pickle
with open('credit_card_fraud_model.pkl', 'wb') as model_file:
    pickle.dump(rfc, model_file)
print("\nModel saved as 'credit_card_fraud_model.pkl'")

# Function to make predictions on new data
def predict_fraud(time, amount):
    """
    Predict if a transaction is fraudulent based on time and amount
    
    Parameters:
    time (float): Time of transaction
    amount (float): Amount of transaction
    
    Returns:
    int: 0 for normal transaction, 1 for fraudulent
    """
    # Reshape the input to match model expectations
    transaction = np.array([[time, amount]])
    return rfc.predict(transaction)[0]

# Example usage
print("\nExample prediction:")
example_time = 50000  # Example time value
example_amount = 100  # Example amount value
prediction = predict_fraud(example_time, example_amount)
print(f"Transaction with Time={example_time} and Amount=${example_amount}")
print(f"Prediction: {'Fraud' if prediction == 1 else 'Normal'}")