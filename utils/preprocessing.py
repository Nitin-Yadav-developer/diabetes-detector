import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    return X, y

def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'model/scaler.pkl')  # Save the scaler for future use
    return X_scaled
