import os
import dotenv
from pathlib import Path
from utils import logger
from transformers import pipeline
from huggingface_hub import login
from data import load_test_data
from sklearn.metrics import accuracy_score, classification_report

# Login to Hugging Face Hub
dotenv.load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env.local")
token = os.getenv("HUGGINGFACE_KEY")
if token:
    print("Token found:", token is not None)
    login(token)

# Load data
data_original = load_test_data()
data = data_original.dropna(subset=["pidgin_text", "sentiment","emotion_category"]).reset_index(drop=True)
logger.info("Loaded %d rows", len(data))

# Load model from Hugging Face Hub
model_dir = "Elilora/pidgin-sentiment-afriberta-finetuned_with_emotion"

sentiment_pipeline = pipeline("sentiment-analysis",model=model_dir,tokenizer=model_dir)

# Combined input format to fit the data the model was trained on
data["model_input"] = "[EMOTION: " + data["emotion_category"].astype(str) + "] " + data["pidgin_text"]
texts = data["model_input"].tolist()

logger.info("Running predictions...")

results = sentiment_pipeline(texts, truncation=True, max_length=128)


data["predicted_sentiment"] = [r["label"] for r in results]
data["sentiment_confidence"] = [round(r["score"], 4) for r in results]


#Compute accuracy
true_labels = data["sentiment"].astype(str).str.lower()
pred_labels = data["predicted_sentiment"].astype(str).str.lower()

accuracy = accuracy_score(true_labels, pred_labels)
logger.info(f"\nAccuracy vs ground truth: {accuracy:.4f}\n")
print(classification_report(true_labels, pred_labels))

data.to_csv("sentiment_results_finetuned.csv", index=False)
logger.info("Saved predictions to sentiment_results_finetuned.csv")

print(data[["sentiment", "predicted_sentiment"]].head(20))

