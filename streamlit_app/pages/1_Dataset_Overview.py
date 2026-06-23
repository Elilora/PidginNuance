
import json
import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards


st.set_page_config(page_title="Dataset Overview", layout="wide")
st.title("Dataset Overview")


data_path = Path("analysis/data/dataset_overview.json")

if not data_path.exists():
    st.error("dataset_overview.json not found.")
    st.stop()

with open(data_path) as f:
    data = json.load(f)

# --- KPI cards ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Rows", data["total_rows"])
col2.metric("Unique Texts", data["unique_texts"])
col3.metric("Emotion Categories", len(data["emotion_distribution"]))
col4.metric("Sarcasm Rate",f"{round(100 * data['sarcasm_flag'].get('yes', 0) / data['total_rows'], 1)}%" if "yes" in data.get("sarcasm_flag", {}) else "N/A",)


style_metric_cards(
    background_color="#B3EBF2",
    border_left_color="#FFFFFF",
    border_color="#FFFFFF",
    box_shadow=True,)

st.divider()

#Sentiment & Emotion distributions 
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment Distribution")
    df = pd.DataFrame(list(data["sentiment_distribution"].items()), columns=["Sentiment", "Count"])
    fig = px.pie(df, names="Sentiment", values="Count", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Emotion Category Distribution")
    df = pd.DataFrame(list(data["emotion_distribution"].items()), columns=["Emotion", "Count"]).sort_values("Count", ascending=True)
    fig = px.bar(df, x="Count", y="Emotion", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Demographics 
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Speaker Gender")
    df = pd.DataFrame(list(data["speaker_gender"].items()), columns=["Gender", "Count"])
    fig = px.pie(df, names="Gender", values="Count")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Register")
    df = pd.DataFrame(list(data["register"].items()), columns=["Register", "Count"])
    fig = px.pie(df, names="Register", values="Count")
    st.plotly_chart(fig, use_container_width=True)

with col3:
    st.subheader("Source Type")
    df = pd.DataFrame(list(data["source_type"].items()), columns=["Source", "Count"])
    fig = px.pie(df, names="Source", values="Count")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Topic Domain")
    df = pd.DataFrame(list(data["topic_domain"].items()), columns=["Topic", "Count"]).sort_values("Count", ascending=True)
    fig = px.bar(df, x="Count", y="Topic", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Intensity Distribution")
    df = pd.DataFrame(list(data["intensity_distribution"].items()), columns=["Intensity", "Count"])
    fig = px.bar(df, x="Intensity", y="Count")
    st.plotly_chart(fig, use_container_width=True)

if "date_trend" in data and data["date_trend"]:
    st.divider()
    st.subheader("Entries Over Time")
    df = pd.DataFrame(list(data["date_trend"].items()), columns=["Month", "Count"]).sort_values("Month")
    fig = px.line(df, x="Month", y="Count", markers=True)
    st.plotly_chart(fig, use_container_width=True)
