import os
import anthropic
import dotenv
from pathlib import Path

dotenv.load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env.local")

# Emotion class
emotion_classes = ["anger", "betrayal", "celebration", "contempt", "craving","forming", "hustle_energy", "hustle_fatigue", "joy","market_energy", "neutral", "prayer_gratitude", "pride","sarcasm", "shock", "suspicion"]


_client = None

def _get_client():
    global _client 
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found. Check your .env.local file.")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client

def guess_emotion(text):
    """
    Guesses the emotion category of Nigerian Pidgin text using Claude.
    Returns predicted emotion and full list of valid categories.
    """
    client = _get_client()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=50,
        messages=[{"role": "user",
                   "content": (f"You are an expert in Nigerian Pidgin language and culture. "
                            f"Identify the emotion expressed in this pidgin text.\n"
                            f"Choose exactly ONE emotion from this list: {emotion_classes}\n"
                            f"Respond with ONLY the emotion label, nothing else.\n\n"
                            f"Text: {text}\n"
                            f"Emotion:"),}],)

    emotion = response.content[0].text.strip().lower()

    # fall back to "neutral" if LLM returns something outside the list
    if emotion not in emotion_classes:
            for valid in emotion_classes:
                if valid in emotion:
                    emotion = valid
                    break
            else:
                emotion = "neutral"

    return {"emotion_category": emotion, "all_classes": emotion_classes}


if __name__ == "__main__":
    print(guess_emotion("You do well"))
    print(guess_emotion("Wahala dey o"))

