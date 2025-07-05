# app.py

from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import mlflow.transformers

app = Flask(__name__)


tfidf_model_uri = "runs:/cf7f1846eb4846348d8a16fd8465b32a/model"
tfidf_model = mlflow.sklearn.load_model(tfidf_model_uri)


# bert_model_uri = "runs:/<your_bert_run_id>/bert_model"
# bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# bert_model = mlflow.transformers.load_model(bert_model_uri)

# === TF-IDF Prediction ===
@app.route('/predict', methods=['POST'])
def predict_tfidf():
    data = request.get_json()
    reviews = data.get("reviews", [])
    preds = tfidf_model.predict(reviews)
    labels = ['Negative' if p == 0 else 'Positive' for p in preds]
    return jsonify({"model": "TF-IDF", "predictions": labels})


# # === BERT Prediction ===
# @app.route('/predict_bert', methods=['POST'])
# def predict_bert():
#     data = request.get_json()
#     reviews = data.get("reviews", [])
#     inputs = bert_tokenizer(reviews, return_tensors="pt", padding=True, truncation=True, max_length=512)

#     with torch.no_grad():
#         outputs = bert_model(**inputs)
#         probs = torch.nn.functional.softmax(outputs.logits, dim=1)
#         preds = torch.argmax(probs, dim=1).tolist()

#     labels = ['Negative' if p == 0 else 'Positive' for p in preds]
#     return jsonify({"model": "BERT", "predictions": labels})


@app.route('/train_model', methods=['POST'])
def trigger_training():
    import subprocess
    subprocess.call(["python", "src/train_tfidf.py"])
    return jsonify({"message": "Model training triggered!"})


@app.route('/best_model_params', methods=['GET'])
def best_params():
    return jsonify({"message": "Check MLflow UI for best hyperparameters."})


if __name__ == '__main__':
    app.run(debug=True, port=5001)
