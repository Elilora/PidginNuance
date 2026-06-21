"""
Sentiment classification using pidgin_text + emotion_category as combined input.
emotion_category is concatenated as a prefix to the text before tokenization.

Usage:
    python train_sentiment_with_emotion_input.py
"""

import json
import numpy as np
from utils import logger
from datasets import Dataset
from data import load_train_data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


model_name = "Davlan/naija-twitter-sentiment-afriberta-large"
output_dir = "./finetuned/sentiment_with_emotion_model"
label_columns_for_deduplication = ["sentiment", "emotion_category", "emotion_secondary","register", "sarcasm_flag", "prosody_match", "intensity",]

# function to tokenise the text data
def tokenize_data(batch):
    return tokenizer(batch["model_input"], truncation=True, padding="max_length", max_length=128)

# function to compute metrics for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return { "accuracy": accuracy_score(labels, predictions),
             "f1": f1_score(labels, predictions, average="weighted"),
             "classification_report": classification_report(labels, predictions, output_dict=True),}


# Load the dataset
data = load_train_data()
data = data.dropna(subset=["pidgin_text", "emotion_category", "sentiment"]).reset_index(drop=True)
print(f"Loaded {len(data)} rows")

# Remove true duplicates
dedup_cols = ["pidgin_text"] + [c for c in label_columns_for_deduplication if c in data.columns]
data = data.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
print(f"After removing true duplicates: {len(data)} rows")

# Concatenate the pidgin_text and emotion_category for better performance
data["model_input"] = "[EMOTION: " + data["emotion_category"].astype(str) + "] " + data["pidgin_text"]
print("\nSample model input:", data["model_input"].iloc[0])

# Encode the labels to integers
encoder = LabelEncoder()
data["label"] = encoder.fit_transform(data["sentiment"])
num_labels = len(encoder.classes_)
print(f"Classes: {encoder.classes_.tolist()}")

id2label = {i: label for i, label in enumerate(encoder.classes_)}
label2id = {label: i for i, label in enumerate(encoder.classes_)}
print("id2label:", id2label)

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data["label"])
print(f"Train: {len(train_data)} | Val: {len(val_data)}")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels,id2label=id2label,label2id=label2id)

# Tokenise the text data
train_encodings = Dataset.from_pandas(train_data[["model_input", "label"]]).map(tokenize_data, batched=True)
val_encodings = Dataset.from_pandas(val_data[["model_input", "label"]]).map(tokenize_data, batched=True)


# Training parameters
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs_sentiment_emotion_input",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    save_total_limit=2,
    report_to="none",
)

# Train the model
trainer = Trainer(model=model, args=training_args, train_dataset=train_encodings,eval_dataset=val_encodings, compute_metrics=compute_metrics)
logger.info("Starting training (text + emotion_category as input)...")
trainer.train()

# Evaluate the model
predictions = trainer.predict(val_encodings)
preds = np.argmax(predictions.predictions, axis=-1)
true = val_data["label"].values
print(classification_report(true, preds, target_names=encoder.classes_))


# Save the model
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
with open(f"{output_dir}/label_map.json", "w") as f:
    json.dump({"classes": encoder.classes_.tolist()}, f, indent=2)

logger.info(f"Saved model to {output_dir}")