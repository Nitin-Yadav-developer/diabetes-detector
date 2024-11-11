from flask import Flask, request, render_template
import aiml
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load AIML chatbot
bot = aiml.Kernel()
bot.learn("diabetes_chatbot.aiml")

# Load model and scaler
model = joblib.load('model/diabetes_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Define prediction function
def predict_diabetes(input_data):
    input_data_scaled = scaler.transform([input_data])
    prediction = model.predict(input_data_scaled)
    return "Diabetes Detected" if prediction[0] == 1 else "No Diabetes Detected"

# Define routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["message"]
    response = bot.respond(user_input)

    if user_input == "CHECK DIABETES":
        try:
            # Get user data from the form input fields
            input_data = [
                float(request.form["age"]),
                float(request.form["glucose"]),
                float(request.form["bloodpressure"]),
                float(request.form["skinthickness"]),
                float(request.form["insulin"]),
                float(request.form["bmi"]),
                float(request.form["diabetespedigreefunction"]),
                float(request.form["pregnancies"]),
            ]
            
            # Predict diabetes
            result = predict_diabetes(input_data)
            response = f"Result: {result}"

        except ValueError:
            # Handle non-numeric values or missing data
            response = "Please enter valid numeric values for all fields."
    
    return render_template("index.html", user_input=user_input, bot_response=response)

if __name__ == "__main__":
    app.run(debug=True)
