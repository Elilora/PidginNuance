import streamlit as st

st.set_page_config(
    page_title="Pidgin Sentiment Dashboard",
    page_icon=None,
    layout="wide",)

st.title("Pidgin Sentiment & Emotion Dashboard")

st.markdown(
    """
    This dashboard analyses sentiment and emotion in Nigerian Pidgin text.

    Use the sidebar to navigate between:

    - **Dataset Overview** — distributions and dataset characteristicss
    - **Agent** — interactively analyse new pidgin text
    """
)
