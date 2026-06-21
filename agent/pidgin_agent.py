from smolagents import CodeAgent,InferenceClientModel,UserInputTool, LiteLLMModel
from agent.tools import SentimentAnalysisTool, EmotionGuesserTool


def run_agent(text):

    #agent = CodeAgent(tools=[EmotionGuesserTool(), UserInputTool(), SentimentAnalysisTool()], 
                  #model = InferenceClientModel(model_id="Qwen/Qwen2.5-72B-Instruct"))

    agent = CodeAgent(tools=[EmotionGuesserTool(), UserInputTool(), SentimentAnalysisTool()], 
                  model = LiteLLMModel(model_id="anthropic/claude-sonnet-4-6"))

    response = agent.run(f"""You are a Nigerian Pidgin language analysis assistant. Your job is to analyse
                        the sentiment of pidgin text by following these steps:

                        1. On the first attempt, use the emotion_guesser_tool to predict the emotion of the text.
                        2. If needs_confirmation is True (low confidence), use the user_input tool to 
                        show the user your guess and ask them to confirm or provide the correct emotion
                        from the list of valid categories.
                        3. If needs_confirmation is False (high confidence), still briefly tell the user
                        your emotion guess and ask them to confirm before proceeding.
                        4. Once emotion is confirmed, use the sentiment_analysis_tool with the confirmed
                        emotion to predict sentiment.
                        5. Report the final sentiment and confidence to the user clearly.
                        6. Do NOT ask for confirmation more than once.

                        User message: Analyse the sentiment of this Nigerian Pidgin text: '{text}'
                        """)
    return response

if __name__ == "__main__":
    text = input("Enter pidgin text: ")
    result = run_agent(text)
    print("\nFinal result:", result)