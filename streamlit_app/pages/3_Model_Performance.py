import json
import sys
import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Model Performance", layout="wide")
st.title("Model Performance")

data_path = Path("analysis/data/model_performance.json")

if not data_path.exists():
    st.error("model_performance.json not found. Run `python -m analysis.compute_analysis` first.")
    st.stop()

with open(data_path) as f:
    data = json.load(f)

#  Accuracy comparison KPI cards 
st.subheader("Accuracy Comparison")
col1, col2, col3 = st.columns(3)

baseline_acc = data["baseline"]["accuracy"]
finetuned_acc = data["finetuned"]["accuracy"]
improvement = round((finetuned_acc - baseline_acc) * 100, 1)

col1.metric("Pre-trained Baseline (text-only)", f"{baseline_acc * 100:.1f}%")
col2.metric("Fine-tuned (text + emotion context)", f"{finetuned_acc * 100:.1f}%", delta=f"+{improvement}%")
col3.metric("Improvement from Emotion Context", f"+{improvement}%", help="Accuracy gain from providing emotion_category as input alongside text")

st.divider()

# Accuracy comparison bar chart 
st.subheader("Accuracy Across Experiments")
df_acc = pd.DataFrame(data["accuracy_comparison"])
fig = px.bar(
    df_acc, x="model", y="accuracy",
    text=df_acc["accuracy"].apply(lambda x: f"{x*100:.1f}%"),
    color="model",
    labels={"accuracy": "Accuracy", "model": "Model"},)

fig.update_traces(textposition="outside")
fig.update_layout(showlegend=False, yaxis_range=[0, 1.1])
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Per-class F1 comparison
st.subheader("Per-Class F1 Score Comparison")

def extract_per_class_f1(report, model_name):
    rows = []
    for label, metrics in report.items():
        if label in ("accuracy", "macro avg", "weighted avg"):
            continue
        if isinstance(metrics, dict):
            rows.append({ "Class": label, "F1 Score": round(metrics.get("f1-score", 0), 3),"Model": model_name,})
    return rows

rows = []
rows += extract_per_class_f1(data["baseline"]["classification_report"], "Baseline")
rows += extract_per_class_f1(data["finetuned"]["classification_report"], "Fine-tuned")
df_f1 = pd.DataFrame(rows)

fig = px.bar(df_f1, x="Class", y="F1 Score", color="Model", barmode="group",labels={"F1 Score": "F1 Score", "Class": "Sentiment Class"})
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Confusion matrices side by side
st.subheader("Confusion Matrices")
col1, col2 = st.columns(2)

def plot_confusion_matrix(cm, labels, title):
    fig = go.Figure(data=go.Heatmap(z=cm, x=labels, y=labels,colorscale="Blues", text=cm, texttemplate="%{text}",))
    fig.update_layout(title=title,xaxis_title="Predicted",yaxis_title="Actual",height=350,)
    return fig

with col1:
    fig = plot_confusion_matrix(data["baseline"]["confusion_matrix"],data["baseline"]["labels"],"Baseline Model")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = plot_confusion_matrix(data["finetuned"]["confusion_matrix"],data["finetuned"]["labels"],"Fine-tuned Model (text + emotion)")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Misclassified examples 
st.subheader("Misclassified Examples (Fine-tuned Model)")
st.caption("Rows where the model predicted incorrectly ")

if data.get("misclassified_examples"):
    df_misc = pd.DataFrame(data["misclassified_examples"])

    # Filter by true sentiment
    sentiments = ["All"] + sorted(df_misc["true_sentiment"].unique().tolist())
    selected = st.selectbox("Filter by true sentiment:", sentiments)
    if selected != "All":
        df_misc = df_misc[df_misc["true_sentiment"] == selected]

    st.dataframe(df_misc, use_container_width=True)
    st.caption(f"Showing {len(df_misc)} misclassified examples")
else:
    st.success("No misclassified examples found — perfect validation accuracy!")
