import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import BernoulliNB
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import warnings
import time

# Ignore warnings
warnings.filterwarnings('ignore')


# Load data
def load_data(file_path):
    """Load data from a given file path."""
    return pd.read_csv(file_path)


# Preprocess data
def preprocess_data(df):
    """Handle label encoding and fill missing values."""
    def label_encode(df):
        for col in df.columns:
            if df[col].dtype == 'object':
                label_encoder = LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col])

    # Encode categorical columns
    label_encode(df)
    
    # Handle missing values
    df.fillna(df.mean(), inplace=True)
    return df


# Feature selection
def select_features(X, y, n_features=10):
    """Select the most important features using Recursive Feature Elimination."""
    rfc = RandomForestClassifier()
    rfe = RFE(rfc, n_features_to_select=n_features)
    rfe.fit(X, y)
    selected_features = X.columns[rfe.support_]
    return X[selected_features]


# Scale features
def scale_features(X):
    """Standardize feature values."""
    scaler = StandardScaler()
    return scaler.fit_transform(X)


# Train and evaluate models using cross-validation
def evaluate_models(models, X, y):
    """Evaluate models using cross-validation and track performance."""
    results = []
    for model in models:
        model_name = model.__class__.__name__
        
        # Record start time
        start_time = time.time()
        
        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        # Record training and prediction time
        training_time = time.time() - start_time
        model.fit(X, y)  # Train the model
        prediction_time = time.time() - start_time
        
        # Compute mean score
        train_score = np.mean(scores)
        results.append([model_name, train_score, training_time, prediction_time])
    
    return results


def main():
    """Main execution function."""
    # Load and preprocess data
    data = load_data('creditcard.csv')
    X = data.drop(['Class'], axis=1)
    y = data['Class']
    X = preprocess_data(X)
    y = LabelEncoder().fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    X_train_scaled = scale_features(X_train)
    X_test_scaled = scale_features(X_test)
    
    # Define models to evaluate
    models = [
        KNeighborsClassifier(),
        LogisticRegression(max_iter=10000),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        SVC(),
        LinearSVC(max_iter=10000),
        BernoulliNB(),
        LGBMClassifier(),
        XGBClassifier()
    ]
    
    # Evaluate models using cross-validation
    results = evaluate_models(models, X_train_scaled, y_train)
    
    # Train the best model on the full training set
    best_model = RandomForestClassifier()
    best_model.fit(X_train_scaled, y_train)
    test_score = best_model.score(X_test_scaled, y_test)
    print(f"Test set score of the RandomForestClassifier: {test_score:.4f}")
    
    # Prepare results for display
    data = pd.DataFrame(results, columns=["Model", "Cross-Validation Score", "Training Time (sec)", "Prediction Time (sec)"])
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="Cross-Validation Score", data=data)
    plt.xticks(rotation=45)
    plt.title("Cross-Validation Scores Across Models")
    plt.show()
    
    # Print detailed results
    print(data)


if __name__ == "__main__":
    main()
