# src/train_tfidf.py

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("data/imdb_reviews.csv")
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

# Define pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.95
    )),
    ('clf', LogisticRegression(class_weight='balanced', solver='liblinear'))
])

# Expanded hyperparameter search
params = {
    'clf__C': [0.01, 0.1, 1, 10, 100],
    'clf__max_iter': [100, 200, 300]
}

grid = GridSearchCV(pipeline, param_grid=params, cv=3, verbose=2, n_jobs=-1)

with mlflow.start_run():
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_param("best_params", grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(grid.best_estimator_, "model")

    print(f"✅ Logged Model with Accuracy: {acc:.4f}")
