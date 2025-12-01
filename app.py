%%writefile app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return "Welcome to the Diabetes Prediction API! Use the /predict endpoint to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        # Ensure the input data is a list of lists (for multiple samples) or a single list (for one sample)
        # Convert dict values to a list in the correct order as per training features
        # Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        # If input is a dictionary for a single prediction
        if isinstance(data, dict):
            input_data = [data[feature] for feature in feature_names]
            input_array = np.array(input_data).reshape(1, -1)
        # If input is a list of dictionaries for multiple predictions
        elif isinstance(data, list):
            input_array = []
            for item in data:
                if isinstance(item, dict):
                    input_array.append([item[feature] for feature in feature_names])
                else:
                    return jsonify({'error': 'Invalid input format. Expected list of dictionaries or a single dictionary.'}), 400
            input_array = np.array(input_array)
        else:
            return jsonify({'error': 'Invalid input format. Expected JSON dictionary or list of dictionaries.'}), 400

        # Scale the input features
        scaled_data = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(scaled_data)
        prediction_proba = model.predict_proba(scaled_data)

        # Prepare response
        results = []
        for i in range(len(prediction)):
            result = {
                'prediction': int(prediction[i]),
                'probability_no_diabetes': prediction_proba[i][0],
                'probability_diabetes': prediction_proba[i][1]
            }
            results.append(result)

        if len(results) == 1:
            return jsonify(results[0])
        else:
            return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

print("Flask app 'app.py' updated with prediction endpoint.")
