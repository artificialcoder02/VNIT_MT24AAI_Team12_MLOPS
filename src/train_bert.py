# src/train_bert.py

import pandas as pd
import numpy as np
import mlflow
import mlflow.transformers
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

# ✅ Load data
df = pd.read_csv("data/imdb_reviews.csv")
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
df = df[['review', 'label']]

# ✅ Split into train/test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ✅ Convert to HuggingFace datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# ✅ Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['review'], padding='max_length', truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# ✅ Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# ✅ Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    save_strategy="epoch",
    logging_steps=100,
    load_best_model_at_end=True,
    report_to="none"
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

# ✅ Train with MLflow tracking
with mlflow.start_run():
    trainer.train()
    eval_result = trainer.evaluate()

    acc = eval_result.get("eval_accuracy")
    loss = eval_result.get("eval_loss")

    mlflow.log_metric("eval_accuracy", acc)
    mlflow.log_metric("eval_loss", loss)
    mlflow.transformers.log_model(
        transformers_model=model,
        artifact_path="bert_model",
        task="text-classification",
        tokenizer=tokenizer
    )

    print(f"✅ BERT Model Logged with Accuracy: {acc:.4f}")
