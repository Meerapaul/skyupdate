from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import requests

app = Flask(__name__)
CORS(app, resources={"*": {"origins": "*"}})

# Load the trained model
try:
    model = joblib.load("weather_prediction_gb_model.pkl")  # Ensure this file exists
except Exception as e:
    print("Error loading model:", str(e))
    model = None  # Set to None if loading fails

# OpenWeatherMap API Key
OPENWEATHER_API_KEY = "59ef9c4572c7a5646d6f59db4b6890a5"

# Weather mapping for ML predictions
weather_mapping = {
    0: "Sunny", 1: "Partly Cloudy", 2: "Cloudy",
    3: "Light Rain", 4: "Heavy Rain", 5: "Thunderstorm",
    6: "Snow", 7: "Foggy", 8: "Windy", 9: "Hailstorm"
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

    predictions = {}
    for day in range(1, 6):  # Predict for the next 5 days
        example_input = [75, 1012, 0.2, 50, 300, 15, 20, 30, 1010, 0.1, 200]  # Replace with real feature extraction
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
            predictions[f"Day {day}"] = weather_condition
        except Exception as e:
            return jsonify({'error': f"Model prediction failed: {str(e)}"}), 500

    return jsonify({"city": city, "predicted_weather": predictions})

@app.route('/weather', methods=['GET'])
def get_weather():
    city = request.args.get('city')
    if not city:
        return jsonify({'error': 'Please provide a city name'}), 400

    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"

    try:
        response = requests.get(url)
        weather_data = response.json()

        if response.status_code != 200:
            return jsonify({"error": weather_data.get("message", "Failed to fetch weather")}), response.status_code

        # Extract next 5 days' forecast
        daily_forecast = {}
        for i in range(0, 40, 8):  # Every 8th entry (each day at noon)
            day_forecast = weather_data['list'][i]
            daily_forecast[f"Day {i//8 + 1}"] = {
                "temperature": day_forecast['main']['temp'],
                "humidity": day_forecast['main']['humidity'],
                "condition": day_forecast['weather'][0]['description']
            }

        return jsonify({"city": city, "forecast": daily_forecast})

    except Exception as e:
        return jsonify({'error': f"Failed to fetch real-time weather: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
