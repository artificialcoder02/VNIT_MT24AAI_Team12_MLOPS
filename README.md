# VNIT_MT24AAI_Team12_MLOPS
## Members
### MT24AAI018
### MT24AAI059
### MT24AAI068


# MLOps Flask Application with MLflow

A comprehensive MLOps solution demonstrating containerized machine learning model deployment using Flask, MLflow, and Docker.

##  Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Technologies Used](#technologies-used)
- [What is Docker?](#what-is-docker)
- [What is MLflow?](#what-is-mlflow)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Docker Guide](#docker-guide)
- [API Documentation](#api-documentation)
- [MLflow Integration](#mlflow-integration)
- [Best Practices Implemented](#best-practices-implemented)
- [Deployment Options](#deployment-options)
- [Troubleshooting](#troubleshooting)

## 🚀 Project Overview

This project demonstrates a complete MLOps pipeline featuring:
- **Sentiment Analysis Models**: TF-IDF and BERT-based models
- **Model Management**: MLflow for experiment tracking and model versioning
- **Containerization**: Docker for consistent deployment environments
- **RESTful API**: Flask-based web service for model predictions
- **Production Ready**: Following Docker and Flask best practices

##  Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Apps   │───▶│   Flask API     │───▶│   MLflow        │
│                 │    │   (Docker)      │    │   Models        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   ML Models     │
                       │   (TF-IDF/BERT) │
                       └─────────────────┘
```

##  Technologies Used

- **Python 3.10**: Core programming language
- **Flask**: Web framework for API development
- **MLflow**: ML lifecycle management
- **Docker**: Containerization platform
- **scikit-learn**: Machine learning library
- **Transformers**: BERT model implementation
- **PyTorch**: Deep learning framework

## What is Docker?

**Docker** is a containerization platform that packages applications with their dependencies into lightweight, portable containers.

### Key Benefits:
- **Consistency**: "Works on my machine" → "Works everywhere"
- **Isolation**: Each container runs independently
- **Portability**: Run anywhere Docker is installed
- **Scalability**: Easy to scale horizontally
- **Efficiency**: Shares OS kernel, lightweight

### Docker Components:
- **Image**: Blueprint for containers
- **Container**: Running instance of an image
- **Dockerfile**: Instructions to build an image
- **Registry**: Storage for Docker images (Docker Hub, ECR, etc.)

##  What is MLflow?

**MLflow** is an open-source platform for managing the complete machine learning lifecycle.

### Core Components:
1. **Tracking**: Log parameters, metrics, and artifacts
2. **Projects**: Package ML code in reusable format
3. **Models**: Deploy models to various platforms
4. **Registry**: Centralized model store with versioning

### Benefits:
- **Experiment Tracking**: Compare different model runs
- **Model Versioning**: Track model evolution
- **Reproducibility**: Recreate experiments exactly
- **Deployment**: Deploy models to various platforms

## Project Structure

```
mlops-project/
├── app.py                 # Main Flask application
├── Dockerfile            # Docker configuration
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── .dockerignore        # Docker ignore file
├── src/
│   ├── train_tfidf.py   # TF-IDF model training
│   └── train_bert.py    # BERT model training
├── models/              # Saved model files
├── logs/                # Application logs
├── tests/               # Unit tests
└── data/                # Training data
```

## 🔧 Setup Instructions

### Prerequisites
- Python 3.10+
- Docker Desktop
- Git

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mlops-project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   ```bash
   export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
   export FLASK_DEBUG=true
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

## Docker Guide

### Understanding the Dockerfile

Our Dockerfile implements several best practices:

```dockerfile
# Use specific version for reproducibility
FROM python:3.10-slim

# Set metadata labels
LABEL maintainer="your-email@example.com"

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Optimization: Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Health check for monitoring
HEALTHCHECK --interval=30s --timeout=30s CMD curl -f http://localhost:5001/health
```

### Building Docker Image

1. **Build the image**
   ```bash
   docker build -t mlops-app:latest .
   ```

2. **Build with specific tag**
   ```bash
   docker build -t mlops-app:v1.0 .
   ```

3. **Build with no cache**
   ```bash
   docker build --no-cache -t mlops-app:latest .
   ```

### Running Docker Container

1. **Run container**
   ```bash
   docker run -p 5001:5001 mlops-app:latest
   ```

2. **Run with environment variables**
   ```bash
   docker run -p 5001:5001 \
     -e MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
     -e FLASK_DEBUG=false \
     mlops-app:latest
   ```

3. **Run with volume mounting**
   ```bash
   docker run -p 5001:5001 \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/logs:/app/logs \
     mlops-app:latest
   ```

4. **Run in background (detached)**
   ```bash
   docker run -d -p 5001:5001 --name mlops-container mlops-app:latest
   ```

### Docker Management Commands

```bash
# List running containers
docker ps

# Stop container
docker stop mlops-container

# Remove container
docker rm mlops-container

# View logs
docker logs mlops-container

# Execute commands in container
docker exec -it mlops-container /bin/bash

# List images
docker images

# Remove image
docker rmi mlops-app:latest
```

## Where to Store Docker Images

### 1. **Docker Hub** (Public Registry)
```bash
# Tag image
docker tag mlops-app:latest username/mlops-app:latest

# Push to Docker Hub
docker push username/mlops-app:latest

# Pull from Docker Hub
docker pull username/mlops-app:latest
```

### 2. **Amazon ECR** (Private Registry)
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag image
docker tag mlops-app:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/mlops-app:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/mlops-app:latest
```

### 3. **Google Container Registry (GCR)**
```bash
# Configure Docker
gcloud auth configure-docker

# Tag image
docker tag mlops-app:latest gcr.io/project-id/mlops-app:latest

# Push to GCR
docker push gcr.io/project-id/mlops-app:latest
```

### 4. **Azure Container Registry**
```bash
# Login to ACR
az acr login --name myregistry

# Tag image
docker tag mlops-app:latest myregistry.azurecr.io/mlops-app:latest

# Push to ACR
docker push myregistry.azurecr.io/mlops-app:latest
```

## 📡 API Documentation

### Endpoints

#### 1. Health Check
```bash
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "service": "ML Prediction Service",
  "version": "1.0",
  "models": {
    "tfidf": true,
    "bert": false
  }
}
```

#### 2. TF-IDF Prediction
```bash
POST /predict
Content-Type: application/json

{
  "reviews": [
    "This movie was amazing!",
    "Terrible film, waste of time."
  ]
}
```
**Response:**
```json
{
  "model": "TF-IDF",
  "predictions": ["Positive", "Negative"],
  "count": 2
}
```

#### 3. Trigger Training
```bash
POST /train_model
```
**Response:**
```json
{
  "message": "Model training completed successfully!"
}
```

#### 4. Model Information
```bash
GET /model_info
```
**Response:**
```json
{
  "models": {
    "tfidf": {
      "loaded": true,
      "uri": "runs:/cf7f1846eb4846348d8a16fd8465b32a/model"
    },
    "bert": {
      "loaded": false,
      "uri": "Not configured"
    }
  },
  "mlflow_tracking_uri": "sqlite:///mlflow.db"
}
```

### Testing the API

1. **Using curl**
   ```bash
   # Health check
   curl http://localhost:5001/health

   # Make prediction
   curl -X POST http://localhost:5001/predict \
     -H "Content-Type: application/json" \
     -d '{"reviews": ["Great movie!", "Bad film"]}'
   ```

2. **Using Python requests**
   ```python
   import requests
   
   # Health check
   response = requests.get('http://localhost:5001/health')
   print(response.json())
   
   # Make prediction
   data = {"reviews": ["Great movie!", "Bad film"]}
   response = requests.post('http://localhost:5001/predict', json=data)
   print(response.json())
   ```

## 🔬 MLflow Integration

### Starting MLflow UI

```bash
# Start MLflow tracking server
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Access MLflow UI at: `http://localhost:5000`

### Model Training with MLflow

```python
import mlflow
import mlflow.sklearn

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_type", "TF-IDF")
    mlflow.log_param("max_features", 5000)
    
    # Train model
    model = train_model()
    
    # Log metrics
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("f1_score", 0.83)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

### Model Registry

```python
# Register model
mlflow.register_model(
    model_uri="runs:/cf7f1846eb4846348d8a16fd8465b32a/model",
    name="sentiment-analysis-tfidf"
)

# Load model from registry
model = mlflow.sklearn.load_model("models:/sentiment-analysis-tfidf/1")
```

## ✅ Best Practices Implemented

### Docker Best Practices

1. **Multi-stage builds**: Use slim base images
2. **Layer caching**: Copy requirements.txt first
3. **Security**: Non-root user, specific versions
4. **Health checks**: Container monitoring
5. **Environment variables**: Configurable settings
6. **Logging**: Structured logging setup

### Flask Best Practices

1. **Error handling**: Comprehensive exception handling
2. **Logging**: Structured logging with levels
3. **Configuration**: Environment-based config
4. **Validation**: Input validation and sanitization
5. **Health endpoints**: Monitoring and observability

### MLflow Best Practices

1. **Experiment tracking**: Log all parameters and metrics
2. **Model versioning**: Use model registry
3. **Reproducibility**: Log environment and dependencies
4. **Artifact management**: Store models and data

## 🚀 Deployment Options

### 1. **Local Development**
```bash
python app.py
```

### 2. **Docker Container**
```bash
docker run -p 5001:5001 mlops-app:latest
```

### 3. **Docker Compose**
```yaml
version: '3.8'
services:
  mlops-app:
    build: .
    ports:
      - "5001:5001"
    environment:
      - MLFLOW_TRACKING_URI=sqlite:///mlflow.db
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
```


## 🔍 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Find process using port
   lsof -i :5001
   
   # Kill process
   kill -9 <PID>
   ```

2. **Docker build fails**
   ```bash
   # Clear Docker cache
   docker system prune -a
   
   # Build with no cache
   docker build --no-cache -t mlops-app:latest .
   ```

3. **MLflow model not found**
   ```bash
   # Check MLflow runs
   mlflow runs list
   
   # Verify model URI
   mlflow models list
   ```

4. **Memory issues**
   ```bash
   # Increase Docker memory limit
   # Docker Desktop → Settings → Resources → Memory
   ```

### Debugging Commands

```bash
# Check container logs
docker logs mlops-container

# Execute shell in container
docker exec -it mlops-container /bin/bash

# Check container resources
docker stats mlops-container

# Inspect container
docker inspect mlops-container
```

##  Monitoring and Observability

### Application Metrics

- **Health checks**: `/health` endpoint
- **Logging**: Structured JSON logs
- **Performance**: Response time tracking
- **Errors**: Exception tracking

### MLflow Metrics

- **Model performance**: Accuracy, F1-score
- **Training metrics**: Loss, validation scores
- **Model versions**: Track model evolution
- **Experiment comparison**: Compare runs

## Security Considerations

1. **Non-root user**: Container runs as non-root
2. **Minimal base image**: Use slim Python image
3. **Dependency scanning**: Regular security updates
4. **Environment variables**: Secure configuration
5. **Input validation**: Sanitize user inputs

##  Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

##  License

This project is licensed under the MIT License.

