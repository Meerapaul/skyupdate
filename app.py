from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("weather_prediction_gb_model.pkl")  # Ensure this file exists in the correct directory

# Define custom weather labels based on model output
weather_mapping = {
    0: "Sunny",
    1: "Partly Cloudy",
    2: "Cloudy",
    3: "Light Rain",
    4: "Heavy Rain",
    5: "Thunderstorm",
    6: "Snow",
    7: "Foggy",
    8: "Windy",
    9: "Hailstorm"
}

@app.route('/')
def home():
    return "Welcome to the Weather Prediction API!"

@app.route('/predict', methods=['GET'])
def predict():
    city = request.args.get('city')  # Get city parameter
    
    if not city:
        return jsonify({'error': 'Please provide a city name'}), 400
    
    # Example input (Dummy values, replace with real input data)
    example_input = [
        75, 1012, 0.2, 50, 300, 15, 20, 30, 1010, 0.1, 200
    ]  

    # Correct feature names (must match model.feature_names_in_)
    feature_columns = [
        'BASEL_humidity', 'BASEL_pressure', 'BASEL_precipitation',
        'BASEL_cloud_cover', 'BASEL_global_radiation', 'TOURS_temp_mean',
        'TOURS_humidity', 'TOURS_pressure', 'TOURS_precipitation',
        'TOURS_wind_speed', 'TOURS_global_radiation'
    ]
    
    if len(example_input) != len(feature_columns):
        return jsonify({'error': 'Input feature length does not match the expected features'}), 400

    input_features = pd.DataFrame([example_input], columns=feature_columns)

    # Make prediction
    try:
        numeric_prediction = model.predict(input_features)[0]  # Get the raw output
        weather_condition = weather_mapping.get(int(round(numeric_prediction, 0)), "Unknown")  # Convert number to label

    except Exception as e:
        return jsonify({'error': f"Model prediction failed: {str(e)}"}), 500

    response = {
        "city": city,
        "predicted_weather": weather_condition  # Return text label (e.g., "Partly Cloudy")
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
