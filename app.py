# app.py

from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd

app = Flask(__name__)

model_uri = "runs:/<your_run_id>/model"  # Update this after training or use MLflow model registry URI
model = mlflow.sklearn.load_model(model_uri)

@app.route('/best_model_params', methods=['GET'])
def get_best_params():
    return jsonify({"message": "Params available via MLflow UI or run metadata."})

@app.route('/train_model', methods=['POST'])
def trigger_training():
    import subprocess
    subprocess.call(["python", "src/train_tfidf.py"])
    return jsonify({"message": "Model training started and logged to MLflow."})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    reviews = data.get("reviews", [])
    preds = model.predict(reviews)
    labels = ['Negative' if p == 0 else 'Positive' for p in preds]
    return jsonify({"predictions": labels})

if __name__ == '__main__':
    app.run(debug=True)
