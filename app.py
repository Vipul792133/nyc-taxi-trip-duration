import json
import joblib
import pandas as pd
import xgboost as xgb
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

model = xgb.XGBRegressor()
model.load_model("model.json")

label_encoder = joblib.load("label_encoder.pkl")

with open("features.json") as f:
    FEATURES = json.load(f)

def build_row(form):
    return {
        'passenger_count': float(form.get('passenger_count', 1)),
        'trip_distance': float(form.get('trip_distance', 1)),
        'PULocationID': int(form.get('PULocationID', 1)),
        'DOLocationID': int(form.get('DOLocationID', 1)),
        'RatecodeID': int(form.get('RatecodeID', 1)),
        'store_and_fwd_flag': int(label_encoder.transform([form.get('store_and_fwd_flag', 'N')])[0]),
        'pickup_hour': int(form.get('pickup_hour', 12)),
        'pickup_dayofweek': int(form.get('pickup_dayofweek', 0)),
        'pickup_month': int(form.get('pickup_month', 1)),
        'is_weekend': int(form.get('is_weekend', 0)),
    }

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form if request.form else request.json
    row = build_row(data)
    x = pd.DataFrame([row])[FEATURES]
    pred_seconds = float(model.predict(x)[0])

    result = {
        "predicted_duration_seconds": round(pred_seconds, 1),
        "predicted_duration_minutes": round(pred_seconds / 60, 2),
    }

    if request.form:
        return render_template("index.html", result=result)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)