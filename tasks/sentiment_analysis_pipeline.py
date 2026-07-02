import os
import dotenv
from transformers import pipeline
from huggingface_hub import login

dotenv.load_dotenv()

_token = os.getenv("HUGGINGFACE_KEY")
if _token:
    login(_token)


# Load model from huggingface hub
model_dir = "Elilora/pidgin-sentiment-afriberta-finetuned_with_emotion"

# Load at module level
_sentiment_pipeline = pipeline("sentiment-analysis",model=model_dir,tokenizer=model_dir)


def predict_sentiment(text, emotion_category):
    """
    Predicts sentiment given pidgin text and a known/guessed emotion category.
    """
    
    model_input = f"[EMOTION: {emotion_category}] {text}"
    result = _sentiment_pipeline(model_input, truncation=True, max_length=128)[0]

    return { "sentiment": result["label"].lower(),"confidence": round(result["score"], 4),}


if __name__ == "__main__":
    # Quick test
    example = predict_sentiment("You do well", "sarcasm")
    print(example)
