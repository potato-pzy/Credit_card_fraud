import pandas as pd

# Load the dataset
data = pd.read_csv(r'C:\folder\Crediccard\project\creditcard.csv')

# Filter fraudulent transactions
fraudulent_data = data[data['Class'] == 1]

# Display first few rows of fraudulent data
print(fraudulent_data.head())