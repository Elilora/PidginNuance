from smolagents import Tool, InferenceClientModel, LiteLLMModel
from tasks.sentiment_analysis_pipeline import predict_sentiment


emotion_classes = ["anger", "betrayal", "celebration", "contempt", "craving","forming", "hustle_energy", "hustle_fatigue", "joy","market_energy", "neutral", "prayer_gratitude", "pride","sarcasm", "shock", "suspicion"]

class EmotionGuesserTool(Tool):
    name = "emotion_guesser_tool"
    description = ("Guesses the emotion category of Nigerian Pidgin text using an LLM. "
                  "Returns predicted emotion and the full list of valid categories.")
    inputs = {"text": {"type": "string", "description": "The raw pidgin text to guess emotion for",}}
    output_type = "object"

    def __init__(self):
        super().__init__()
        #self.model = InferenceClientModel()
        self.model = LiteLLMModel(model_id="anthropic/claude-sonnet-4-6")

    def forward(self, text):
        prompt = (f"You are an expert in Nigerian Pidgin language and culture. "
                f"Identify the emotion expressed in this pidgin text.\n"
                f"Choose exactly ONE emotion from this list: {emotion_classes}\n"
                f"Respond with ONLY the emotion label, nothing else.\n\n"
                f"Text: {text}\n"
                f"Emotion:"
    )
        response = self.model(messages=[{"role": "user", "content": prompt}])
        emotion = response.content.strip().lower()

        # Validate (fall back to "neutral" if LLM returns something outside the list)
        if emotion not in emotion_classes:
            for valid in emotion_classes:
                if valid in emotion:
                    emotion = valid
                    break
            else:
                emotion = "neutral"

        return {"emotion_category": emotion,"all_classes": emotion_classes,"needs_confirmation": True}


class SentimentAnalysisTool(Tool):
    name = "sentiment_analysis_tool"
    description = ("Performs sentiment analysis on Nigerian Pidgin text given the text and its emotion category."
                   "Returns sentiment (positive/negative/neutral) and confidence score.")
    inputs = {"text": {"type": "string","description": "The pidgin text to analyse",},
              "emotion_category": {"type": "string","description": "The emotion category of the pidgin text (e.g. 'sarcasm', 'joy', 'neutral')",},}
    output_type = "object"

    def __init__(self):
        super().__init__()

    def forward(self, text, emotion_category):
        result = predict_sentiment(text, emotion_category)
        return f"Sentiment: {result['sentiment']} (confidence: {result['confidence']})"