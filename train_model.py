from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from utils.preprocessing import load_data, preprocess_data
from utils.model_utils import save_model

# Load and preprocess data
X, y = load_data('data/diabetes.csv')
X_scaled = preprocess_data(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model
save_model(model, 'model/diabetes_model.pkl')

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
