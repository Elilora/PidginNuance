import sys
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tasks.emotion_guessing_pipeline import guess_emotion
from tasks.sentiment_analysis_pipeline import predict_sentiment


st.set_page_config(page_title="Agent", layout="wide")
st.title("Pidgin Sentiment Agent")

st.markdown( "Type Nigerian Pidgin text below. The agent will guess the emotion, "
             "let you confirm or correct it, then predict sentiment using that context.")

# State setup 
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stage" not in st.session_state:
    st.session_state.stage = "awaiting_text"
if "pending_text" not in st.session_state:
    st.session_state.pending_text = None
if "guessed_emotion" not in st.session_state:
    st.session_state.guessed_emotion = None
if "emotion_classes" not in st.session_state:
    st.session_state.emotion_classes = []

# Display conversation history 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Get pidgin text from user
if st.session_state.stage == "awaiting_text":
    user_text = st.chat_input("Enter pidgin text to analyse...")
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        st.session_state.pending_text = user_text

        with st.spinner("Guessing emotion..."):
            result = guess_emotion(user_text)

        st.session_state.guessed_emotion = result["emotion_category"]
        st.session_state.emotion_classes = result["all_classes"]

        st.session_state.messages.append({
            "role": "assistant",
            "content": (f"I think the emotion here is **{result['emotion_category']}**. "
                        f"Please confirm or pick the correct one below."),})
        st.session_state.stage = "awaiting_confirmation"
        st.rerun()

# Confirm or correct the guessed emotion
if st.session_state.stage == "awaiting_confirmation":
    with st.chat_message("assistant"):
        classes = st.session_state.emotion_classes
        default_index = (classes.index(st.session_state.guessed_emotion)
                         if st.session_state.guessed_emotion in classes else 0)
        confirmed = st.selectbox("Confirm or select the correct emotion:",options=classes,index=default_index,key="emotion_select",)

        if st.button("Confirm and analyse sentiment"):
            st.session_state.messages.append({"role": "user","content": f"Confirmed emotion: {confirmed}",})

            with st.spinner("Predicting sentiment..."):
                sentiment_result = predict_sentiment(st.session_state.pending_text, confirmed)

            st.session_state.messages.append({"role": "assistant","content": (f"**Sentiment:** {sentiment_result['sentiment']}  \n"f"**Confidence:** {sentiment_result['confidence']}"),})

            # Reset for next analysis
            st.session_state.stage = "awaiting_text"
            st.session_state.pending_text = None
            st.session_state.guessed_emotion = None
            st.session_state.emotion_classes = []
            st.rerun()

# Clear conversation 
if st.session_state.messages:
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.stage = "awaiting_text"
        st.session_state.pending_text = None
        st.session_state.guessed_emotion = None
        st.session_state.emotion_classes = []
        st.rerun()
