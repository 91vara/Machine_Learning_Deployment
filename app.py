from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return "Welcome to the Diabetes Prediction API! Use /predict to POST your data."

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Respond to browser GET requests
    if request.method == 'GET':
        return "Prediction API is running. Use POST with JSON data to get prediction."

    try:
        data = request.get_json(force=True)

        feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]

        # Single prediction
        if isinstance(data, dict):
            input_data = [data[feature] for feature in feature_names]
            input_array = np.array(input_data).reshape(1, -1)

        # Multiple predictions
        elif isinstance(data, list):
            input_array = []
            for item in data:
                input_array.append([item[feature] for feature in feature_names])
            input_array = np.array(input_array)

        else:
            return jsonify({'error': 'Invalid input format'}), 400

        # Scale
        scaled_data = scaler.transform(input_array)

        # Predict
        prediction = model.predict(scaled_data)
        prediction_proba = model.predict_proba(scaled_data)

        results = []
        for i in range(len(prediction)):
            results.append({
                'prediction': int(prediction[i]),
                'probability_no_diabetes': float(prediction_proba[i][0]),
                'probability_diabetes': float(prediction_proba[i][1])
            })

        return jsonify(results[0] if len(results) == 1 else results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

{
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 85,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.35,
    "Age": 35
}
