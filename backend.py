from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load the trained Random Forest model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "fraud_detection_random_forest.pkl"
)

model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return jsonify({
        "status": "success",
        "message": "Credit Card Fraud Detection API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Accept features as a list
        features = data.get("features")

        if features is None:
            return jsonify({
                "error": "No features were provided"
            }), 400

        # Make prediction
        prediction = model.predict([features])[0]

        # Get probability if the model supports it
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba([features])[0][1]
        else:
            probability = None

        return jsonify({
            "prediction": int(prediction),
            "fraud_probability": float(probability)
            if probability is not None else None
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)