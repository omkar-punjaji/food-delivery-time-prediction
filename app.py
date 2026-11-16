"""Food Delivery Time Prediction - Flask web app.

Run:  python app.py   then open http://127.0.0.1:5000
Needs model/xgb_best.pkl and model/standard_scaler.pkl (both saved with joblib).
"""
import math
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "xgb_best.pkl"
SCALER_PATH = BASE / "model" / "standard_scaler.pkl"

app = Flask(__name__)

scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
if model is None:
    print(f"[!] {MODEL_PATH} not found - copy your xgb_best.pkl there.")

# Column order the scaler (and therefore the model) was trained on
FEATURES = list(scaler.feature_names_in_)

# Same mapping as the notebook
TRAFFIC = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}

# Allowed values. The first value of each list is the category dropped by
# pd.get_dummies(drop_first=True) in the notebook (all-zero dummy columns).
CHOICES = {
    "Type_of_order": ["Buffet", "Drinks", "Meal", "Snack"],
    "Type_of_vehicle": ["bicycle", "electric_scooter", "motorcycle", "scooter"],
    "Festival": ["No", "Yes"],
    "City": ["Metropolitian", "Semi-Urban", "Urban"],
    "Weatherconditions": ["Cloudy", "Fog", "Sandstorms", "Stormy", "Sunny", "Windy"],
}

RMSE_MIN = 3.91  # test RMSE of the tuned XGBoost model (from the notebook)


def haversine_km(lat1, lon1, lat2, lon2):
    """Same Haversine formula used in the notebook (r = 6371 km)."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371 * math.asin(math.sqrt(a))


def num(data, key, lo, hi):
    try:
        v = float(data[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number.")
    if not lo <= v <= hi:
        raise ValueError(f"'{key}' must be between {lo} and {hi}.")
    return v


def cat(data, key):
    v = data.get(key)
    if v not in CHOICES[key]:
        raise ValueError(f"'{key}' must be one of {CHOICES[key]}.")
    return v


def build_row(data):
    """Turn raw form input into the exact feature vector used in training."""
    if data.get("distance_mode") == "coords":
        distance = haversine_km(
            num(data, "rest_lat", -90, 90), num(data, "rest_lon", -180, 180),
            num(data, "drop_lat", -90, 90), num(data, "drop_lon", -180, 180))
    else:
        distance = num(data, "distance_km", 0, 100)
    if distance >= 100:  # the notebook removed trips >= 100 km as bad data
        raise ValueError("Distance must be under 100 km.")

    try:
        hour = int(str(data["order_time"]).split(":")[0])
    except (KeyError, ValueError):
        raise ValueError("Enter a valid order time.")
    if not 0 <= hour <= 23:
        raise ValueError("Enter a valid order time.")

    traffic = data.get("Road_traffic_density")
    if traffic not in TRAFFIC:
        raise ValueError(f"'Road_traffic_density' must be one of {list(TRAFFIC)}.")

    row = dict.fromkeys(FEATURES, 0)
    row.update({
        "Delivery_person_Age": num(data, "Delivery_person_Age", 15, 70),
        "Delivery_person_Ratings": num(data, "Delivery_person_Ratings", 1, 5),
        "distance_km": distance,
        "order_hour": hour,
        "Road_traffic_density": TRAFFIC[traffic],
        "Vehicle_condition": num(data, "Vehicle_condition", 0, 3),
        "multiple_deliveries": num(data, "multiple_deliveries", 0, 3),
    })
    # One-hot columns, e.g. Weatherconditions_Fog = 1 (dropped category = all 0)
    for col in CHOICES:
        col_name = f"{col}_{cat(data, col)}"
        if col_name in row:
            row[col_name] = 1
    return pd.DataFrame([row], columns=FEATURES), distance


@app.route("/")
def index():
    return render_template("index.html", choices=CHOICES, traffic=list(TRAFFIC))


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify(error="Model file missing: put xgb_best.pkl in the model/ folder."), 503
    try:
        X, distance = build_row(request.get_json(force=True, silent=True) or {})
    except ValueError as e:
        return jsonify(error=str(e)), 400
    minutes = float(model.predict(scaler.transform(X))[0])
    minutes = max(minutes, 1.0)
    return jsonify(
        minutes=round(minutes, 1),
        low=max(round(minutes - RMSE_MIN), 1),
        high=round(minutes + RMSE_MIN),
        distance_km=round(distance, 2),
    )


if __name__ == "__main__":
    app.run(debug=True)
