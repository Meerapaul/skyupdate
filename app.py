from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import requests
import os

app = Flask(__name__)
CORS(app, resources={"*": {"origins": "*"}})

# Load the trained model
try:
    model = joblib.load("weather_prediction_gb_model.pkl")  # Ensure the model file exists
except Exception as e:
    print("Error loading model:", str(e))
    model = None  # Set model to None if loading fails

# OpenWeatherMap API Key (Replace with your actual key)
OPENWEATHER_API_KEY = "59ef9c4572c7a5646d6f59db4b6890a5"

# Weather mapping for ML predictions
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
    return "Weather Prediction API is running!"

@app.route('/predict', methods=['GET'])
def predict():
    city = request.args.get('city')
    if not city:
        return jsonify({'error': 'Please provide a city name'}), 400

    if model is None:
        return jsonify({'error': 'Model is not loaded properly'}), 500

    # Replace with actual feature extraction logic
    example_input = [75, 1012, 0.2, 50, 300, 15, 20, 30, 1010, 0.1, 200]  
    feature_columns = [
        'BASEL_humidity', 'BASEL_pressure', 'BASEL_precipitation',
        'BASEL_cloud_cover', 'BASEL_global_radiation', 'TOURS_temp_mean',
        'TOURS_humidity', 'TOURS_pressure', 'TOURS_precipitation',
        'TOURS_wind_speed', 'TOURS_global_radiation'
    ]
    
    input_features = pd.DataFrame([example_input], columns=feature_columns)

    try:
        numeric_prediction = model.predict(input_features)[0]
        weather_condition = weather_mapping.get(int(round(numeric_prediction, 0)), "Unknown")

    except Exception as e:
        return jsonify({'error': f"Model prediction failed: {str(e)}"}), 500

    return jsonify({"city": city, "predicted_weather": weather_condition})

@app.route('/weather', methods=['GET'])
def get_weather():
    city = request.args.get('city')
    if not city:
        return jsonify({'error': 'Please provide a city name'}), 400

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"

    try:
        response = requests.get(url)
        weather_data = response.json()

        if response.status_code != 200:
            return jsonify({"error": weather_data.get("message", "Failed to fetch weather")}), response.status_code

        return jsonify(weather_data)

    except Exception as e:
        return jsonify({'error': f"Failed to fetch real-time weather: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
