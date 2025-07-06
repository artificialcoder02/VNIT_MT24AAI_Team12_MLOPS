# app.py
import os
import logging
from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import mlflow.transformers

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
class Config:
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db')
    TFIDF_MODEL_URI = os.getenv('TFIDF_MODEL_URI', 'runs:/cf7f1846eb4846348d8a16fd8465b32a/model')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    PORT = int(os.getenv('PORT', 5001))

# Set MLflow tracking URI
mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)

# Load models
try:
    tfidf_model = mlflow.sklearn.load_model(Config.TFIDF_MODEL_URI)
    logger.info("TF-IDF model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load TF-IDF model: {e}")
    tfidf_model = None

# Uncomment when BERT model is ready
# try:
#     bert_model_uri = "runs:/<your_bert_run_id>/bert_model"
#     bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#     bert_model = mlflow.transformers.load_model(bert_model_uri)
#     logger.info("BERT model loaded successfully")
# except Exception as e:
#     logger.error(f"Failed to load BERT model: {e}")
#     bert_model = None

# Health Check Endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Docker and load balancers"""
    return jsonify({
        "status": "healthy",
        "service": "ML Prediction Service",
        "version": "1.0",
        "models": {
            "tfidf": tfidf_model is not None,
            "bert": False  # Change when BERT is implemented
        }
    })

# === TF-IDF Prediction ===
@app.route('/predict', methods=['POST'])
def predict_tfidf():
    """Predict sentiment using TF-IDF model"""
    try:
        if tfidf_model is None:
            return jsonify({"error": "TF-IDF model not loaded"}), 500
        
        data = request.get_json()
        if not data or 'reviews' not in data:
            return jsonify({"error": "Invalid input. 'reviews' field required"}), 400
        
        reviews = data.get("reviews", [])
        if not reviews:
            return jsonify({"error": "No reviews provided"}), 400
        
        # Make predictions
        preds = tfidf_model.predict(reviews)
        labels = ['Negative' if p == 0 else 'Positive' for p in preds]
        
        logger.info(f"TF-IDF predictions made for {len(reviews)} reviews")
        
        return jsonify({
            "model": "TF-IDF",
            "predictions": labels,
            "count": len(reviews)
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

# === BERT Prediction (commented out) ===
# @app.route('/predict_bert', methods=['POST'])
# def predict_bert():
#     """Predict sentiment using BERT model"""
#     try:
#         if bert_model is None:
#             return jsonify({"error": "BERT model not loaded"}), 500
#         
#         data = request.get_json()
#         if not data or 'reviews' not in data:
#             return jsonify({"error": "Invalid input. 'reviews' field required"}), 400
#         
#         reviews = data.get("reviews", [])
#         if not reviews:
#             return jsonify({"error": "No reviews provided"}), 400
#         
#         inputs = bert_tokenizer(reviews, return_tensors="pt", padding=True, truncation=True, max_length=512)
#         with torch.no_grad():
#             outputs = bert_model(**inputs)
#         probs = torch.nn.functional.softmax(outputs.logits, dim=1)
#         preds = torch.argmax(probs, dim=1).tolist()
#         labels = ['Negative' if p == 0 else 'Positive' for p in preds]
#         
#         logger.info(f"BERT predictions made for {len(reviews)} reviews")
#         
#         return jsonify({
#             "model": "BERT",
#             "predictions": labels,
#             "count": len(reviews)
#         })
#     
#     except Exception as e:
#         logger.error(f"BERT prediction error: {e}")
#         return jsonify({"error": "BERT prediction failed"}), 500

@app.route('/train_model', methods=['POST'])
def trigger_training():
    """Trigger model training"""
    try:
        import subprocess
        result = subprocess.run(["python", "src/train_tfidf.py"], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("Model training completed successfully")
            return jsonify({"message": "Model training completed successfully!"})
        else:
            logger.error(f"Training failed: {result.stderr}")
            return jsonify({"error": "Training failed"}), 500
    
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Training timeout"}), 500
    except Exception as e:
        logger.error(f"Training trigger error: {e}")
        return jsonify({"error": "Failed to trigger training"}), 500

@app.route('/best_model_params', methods=['GET'])
def best_params():
    """Get best model parameters info"""
    return jsonify({
        "message": "Check MLflow UI for best hyperparameters.",
        "mlflow_ui": "http://localhost:5000",
        "tracking_uri": Config.MLFLOW_TRACKING_URI
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        "models": {
            "tfidf": {
                "loaded": tfidf_model is not None,
                "uri": Config.TFIDF_MODEL_URI
            },
            "bert": {
                "loaded": False,
                "uri": "Not configured"
            }
        },
        "mlflow_tracking_uri": Config.MLFLOW_TRACKING_URI
    })

if __name__ == '__main__':
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=Config.PORT)