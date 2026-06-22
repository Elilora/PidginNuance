"""
Entry point for the Pidgin Sentiment & Emotion Dashboard.
Run with: streamlit run streamlit_app/Home.py
(run from the project root, so analysis/data paths resolve correctly)
"""

import streamlit as st

st.set_page_config(
    page_title="Pidgin Sentiment Dashboard",
    page_icon=None,
    layout="wide",
)

st.title("Pidgin Sentiment & Emotion Dashboard")

st.markdown(
    """
    This dashboard analyses sentiment and emotion in Nigerian Pidgin text.

    Use the sidebar to navigate between:

    - **Dataset Overview** — distributions and dataset characteristics
    - **Sentiment & Emotion Analysis** — cross-tabulations and context-dependent findings
    - **Model Performance** — accuracy comparisons across experiments
    - **Agent** — interactively analyse new pidgin text
    """
)
