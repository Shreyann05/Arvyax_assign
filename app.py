from flask import Flask, request, jsonify
import pandas as pd
from pipeline import EmotionPipeline
import os

app = Flask(__name__)
MODEL_PATH = os.environ.get("MODEL_PATH", "model_artifacts")

try:
    pipeline = EmotionPipeline.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    pipeline = None
    print(f"Model not found - run train.py first. Error: {e}")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": pipeline is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if pipeline is None:
        return jsonify({"error": "Model not loaded. Run train.py first."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    if isinstance(data, dict):
        data = [data]

    df = pd.DataFrame(data)
    if "id" not in df.columns:
        df["id"] = list(range(1, len(df) + 1))

    try:
        preds = pipeline.predict(df)
        return jsonify(preds.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    if pipeline is None:
        return jsonify({"error": "Model not loaded."}), 503

    data = request.get_json()
    if not isinstance(data, list):
        return jsonify({"error": "Expected a JSON array"}), 400

    df = pd.DataFrame(data)
    if "id" not in df.columns:
        df["id"] = list(range(1, len(df) + 1))

    try:
        preds = pipeline.predict(df)
        return jsonify(preds.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
