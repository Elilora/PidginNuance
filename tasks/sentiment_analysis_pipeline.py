"""
Reusable sentiment classifier function/class for use as an agent tool.
Wraps the fine-tuned context-aware sentiment model (text + emotion_category input).
"""

import os
import dotenv
from transformers import pipeline
from huggingface_hub import login

dotenv.load_dotenv()
_token = os.getenv("HUGGINGFACE_KEY")
if _token:
    login(_token)

model_dir = "./finetuned/sentiment_with_emotion_model"

# Load once at module level
_sentiment_pipeline = pipeline("sentiment-analysis",model=model_dir,tokenizer=model_dir)


def predict_sentiment(text: str, emotion_category: str) -> dict:
    """
    Predicts sentiment given pidgin text and a known/guessed emotion category.
    """
    
    model_input = f"[EMOTION: {emotion_category}] {text}"
    result = _sentiment_pipeline(model_input, truncation=True, max_length=128)[0]

    return { "sentiment": result["label"].lower(),"confidence": round(result["score"], 4),}


if __name__ == "__main__":
    # Quick manual test
    example = predict_sentiment("You do well", "sarcasm")
    print(example)
