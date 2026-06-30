import json
import sys
import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Sentiment & Emotion Analysis", layout="wide")
st.title("Sentiment & Emotion Analysis")

data_path= Path("analysis/data/sentiment_emotion_analysis.json")

if not data_path.exists():
    st.error("sentiment_emotion_analysis.json not found. Run `python -m analysis.compute_analysis` first.")
    st.stop()

with open(data_path) as f:
    data = json.load(f)

# KPI cards 
col1, col2, col3 = st.columns(3)
col1.metric("Context Variants Found", data["context_variant_count"],
            help="Same text, different sentiment due to tone/sarcasm")
col2.metric("True Duplicates Removed", data["true_duplicate_count"],
            help="Identical text AND labels — genuine duplicates")
col3.metric("Sarcasm-Driven Ambiguity",
            f"{data['context_variant_count']} rows",
            help="Rows where same text means different things based on context")

st.divider()

# Emotion vs Sentiment heatmap 
st.subheader("Emotion vs Sentiment Heatmap")
st.caption("Which emotions map to which sentiment classes")

emotion_vs_sentiment = data["emotion_vs_sentiment"]
sentiments = list(next(iter(emotion_vs_sentiment.values())).keys())
emotions = list(emotion_vs_sentiment.keys())
matrix = [[emotion_vs_sentiment[e].get(s, 0) for s in sentiments] for e in emotions]

fig = go.Figure(data=go.Heatmap(
    z=matrix,
    x=sentiments,
    y=emotions,
    colorscale="Blues",
    text=matrix,
    texttemplate="%{text}",
))
fig.update_layout(height=500, xaxis_title="Sentiment", yaxis_title="Emotion")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Sarcasm vs Sentiment + Register vs Sentiment
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sarcasm vs Sentiment")
    sarcasm_data = data["sarcasm_vs_sentiment"]
    rows = []
    for sarcasm_val, sentiment_counts in sarcasm_data.items():
        for sentiment, count in sentiment_counts.items():
            rows.append({"Sarcasm": sarcasm_val, "Sentiment": sentiment, "Count": count})
    df = pd.DataFrame(rows)
    fig = px.bar(df, x="Sarcasm", y="Count", color="Sentiment", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Register vs Sentiment")
    register_data = data["register_vs_sentiment"]
    rows = []
    for register, sentiment_counts in register_data.items():
        for sentiment, count in sentiment_counts.items():
            rows.append({"Register": register, "Sentiment": sentiment, "Count": count})
    df = pd.DataFrame(rows)
    fig = px.bar(df, x="Register", y="Count", color="Sentiment", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Context Variants Table
st.subheader("Context Variants (Same Text, Different Meaning)")
st.caption( "These rows have identical pidgin text but different sentiment/emotion labels, "
    "reflecting tone and sarcasm differences captured by annotators.")

if data["context_variants"]:
    df_variants = pd.DataFrame(data["context_variants"])
    st.dataframe( df_variants,use_container_width=True,
                column_order=["pidgin_text", "train_sentiment", "test_sentiment","train_emotion", "test_emotion","train_sarcasm", "test_sarcasm", ],)
else:
    st.info("No context variants found in the data.")
