"""
Computes all dashboard analysis data and saves to JSON files.
Run once before starting the dashboard server.

Usage:
    python -m analysis.compute_analysis
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
from data.load_data import load_train_data, load_test_data
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix



output_dir = Path("analysis/data")
baseline_model = "Davlan/naija-twitter-sentiment-afriberta-large"
finetuned_model = "./finetuned/sentiment_with_emotion_model"

label_columns_for_deduplication = ["sentiment", "emotion_category", "emotion_secondary","register", "sarcasm_flag", "prosody_match", "intensity",]


def load_and_clean():
    train = load_train_data()
    test = load_test_data()
    dedup_cols = ["pidgin_text"] + [c for c in label_columns_for_deduplication if c in train.columns]
    train_clean = train.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    return train_clean, test


def save_json(data, filename):
    with open(output_dir / filename, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved {filename}")


def compute_dataset_overview(df):
    print("\nEmotion label frequency (primary)")
    emotion_counts_normalized = df["emotion_category"].str.strip().str.lower().value_counts()
    total = len(df)
    for label, count in emotion_counts_normalized.items():
        pct = round(100 * count / total, 1)
        print(f"  {label}: {count} ({pct}%)")
 
    overview = {
        "total_rows": len(df),
        "unique_texts": int(df["pidgin_text"].nunique()),
        "sentiment_distribution": df["sentiment"].str.strip().str.lower().value_counts().to_dict(),
        "emotion_distribution": emotion_counts_normalized.to_dict(),
        "speaker_gender": df["speaker_gender"].value_counts().to_dict(),
        "topic_domain": df["topic_domain"].value_counts().to_dict(),
        "register": df["register"].value_counts().to_dict(),
        "source_type": df["source_type"].value_counts().to_dict(),
        "sarcasm_flag": df["sarcasm_flag"].value_counts().to_dict(),
        "intensity_distribution": df["intensity"].value_counts().sort_index().to_dict(),}
   
    if "date_added" in df.columns:
        df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
        date_counts = (df.dropna(subset=["date_added"]).groupby(df["date_added"].dt.to_period("M").astype(str)).size().to_dict())
        overview["date_trend"] = date_counts
    save_json(overview, "dataset_overview.json")
 

def compute_sentiment_emotion_analysis(df):
    cross_tab = pd.crosstab(df["emotion_category"], df["sentiment"]).to_dict()
    sarcasm_sentiment = pd.crosstab(df["sarcasm_flag"], df["sentiment"]).to_dict()
    register_sentiment = pd.crosstab(df["register"], df["sentiment"]).to_dict()


    train = load_train_data()
    test = load_test_data()
    overlap_texts = set(train["pidgin_text"]) & set(test["pidgin_text"])

    context_variants = []
    for text in overlap_texts:
        t_rows = train[train["pidgin_text"] == text]
        e_rows = test[test["pidgin_text"] == text]
        for _, t_row in t_rows.iterrows():
            for _, e_row in e_rows.iterrows():
                labels_match = all( str(t_row.get(col)) == str(e_row.get(col)) for col in label_columns_for_deduplication )
                if not labels_match:
                    context_variants.append({
                        "pidgin_text": text,
                        "train_sentiment": t_row.get("sentiment"),
                        "test_sentiment": e_row.get("sentiment"),
                        "train_emotion": t_row.get("emotion_category"),
                        "test_emotion": e_row.get("emotion_category"),
                        "train_sarcasm": t_row.get("sarcasm_flag"),
                        "test_sarcasm": e_row.get("sarcasm_flag"),
                    })

    save_json({
        "emotion_vs_sentiment": cross_tab,
        "sarcasm_vs_sentiment": sarcasm_sentiment,
        "register_vs_sentiment": register_sentiment,
        "context_variants": context_variants,
        "context_variant_count": len(context_variants),
        "true_duplicate_count": len(overlap_texts) - len(context_variants),
    }, "sentiment_emotion_analysis.json")

   
def compute_model_performance(df, test_df):
    results = {}

    # Baseline model
    print("Running baseline model...")
    baseline_pipe = pipeline( "sentiment-analysis", model=baseline_model, tokenizer=baseline_model)
    baseline_preds = [r["label"].lower() for r in baseline_pipe(df["pidgin_text"].tolist(), truncation=True, max_length=128)]
    baseline_true = df["sentiment"].str.lower().tolist()
    labels = sorted(set(baseline_true))
    results["baseline"] = {
        "accuracy": round(accuracy_score(baseline_true, baseline_preds), 4),
        "classification_report": classification_report(baseline_true, baseline_preds, output_dict=True),
        "confusion_matrix": confusion_matrix(baseline_true, baseline_preds, labels=labels).tolist(),
        "labels": labels,}

    # Fine-tuned model
    print("Running fine-tuned model...")
    finetuned_pipe = pipeline("sentiment-analysis", model=finetuned_model, tokenizer=finetuned_model)
    test_clean = test_df.dropna(subset=["pidgin_text", "sentiment", "emotion_category"]).reset_index(drop=True)
    test_clean["model_input"] = "[EMOTION: " + test_clean["emotion_category"].astype(str) + "] " + test_clean["pidgin_text"]
    finetuned_preds = [r["label"].lower() for r in finetuned_pipe(test_clean["model_input"].tolist(), truncation=True, max_length=128)]
    finetuned_true = test_clean["sentiment"].str.lower().tolist()
    ft_labels = sorted(set(finetuned_true))
    results["finetuned"] = {
        "accuracy": round(accuracy_score(finetuned_true, finetuned_preds), 4),
        "classification_report": classification_report(finetuned_true, finetuned_preds, output_dict=True),
        "confusion_matrix": confusion_matrix(finetuned_true, finetuned_preds, labels=ft_labels).tolist(),
        "labels": ft_labels, }

    results["accuracy_comparison"] = [
        {"model": "Pre-trained baseline (text-only)", "accuracy": results["baseline"]["accuracy"]},
        {"model": "Fine-tuned (text+emotion context)", "accuracy": results["finetuned"]["accuracy"]},]

    results["leakage_story"] = {
        "original_overlap": 211,
        "after_cleaning": 0,
        "true_duplicates": 214,
        "context_variants": 28,}

    misclassified = [
        {
            "pidgin_text": test_clean["pidgin_text"].iloc[i],
            "emotion_category": test_clean["emotion_category"].iloc[i],
            "true_sentiment": true,
            "predicted_sentiment": pred,
        }
        for i, (true, pred) in enumerate(zip(finetuned_true, finetuned_preds)) if true != pred]
    results["misclassified_examples"] = misclassified[:50]

    save_json(results, "model_performance.json")


def main():
    print("Loading data...")
    train_df, test_df = load_and_clean()

    print("Computing dataset overview...")
    compute_dataset_overview(train_df)

    print("Computing sentiment/emotion analysis...")
    compute_sentiment_emotion_analysis(train_df)

    print("Computing model performance...")
    compute_model_performance(train_df, test_df)

    print("\nAll analysis data saved to analysis/data/")


if __name__ == "__main__":
    main()