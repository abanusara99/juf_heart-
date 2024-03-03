from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    # Convert form data to feature vector
    features = [float(data['age']), float(data['sex']), float(data['cp']), float(data['trestbps']), 
                float(data['chol']), float(data['fbs']), float(data['restecg']), float(data['thalach']), 
                float(data['exang']), float(data['oldpeak']), float(data['slope']), float(data['ca']), 
                float(data['thal'])]
    # Make prediction using the models
    prediction = model.predict([features])[0]
    # Determine prediction message
    prediction_message = "This patient has heart disease." if prediction == 1 else "This patient does not have heart disease."
    # Render the HTML template with prediction message
    return render_template('index.html', prediction_message=prediction_message)

if __name__ == '__main__':
    app.run(debug=True)
