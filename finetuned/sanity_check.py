
"""
Sanity check: confirms no leakage between train/val splits used in
train_sentiment_with_emotion_input.py - i.e. no model_input string
appears in both the train and validation sets.

Run this BEFORE or AFTER training to verify the split was clean.

Usage:
    python sanity_check_split.py
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data.load_data import load_train_data

TEXT_COL = "pidgin_text"
EMOTION_COL = "emotion_category"
LABEL_COL = "sentiment"

LABEL_COLS_FOR_DEDUP = [
    "sentiment", "emotion_category", "emotion_secondary",
    "register", "sarcasm_flag", "prosody_match", "intensity",
]


def main():
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("./finetuned/sentiment_with_emotion_model")
    print(config.id2label)
    print(config.label2id)
    # Recreate the exact same data prep steps as the training script,
    # so the split being checked matches what was actually trained on.
    data = load_train_data()
    data = data.dropna(subset=[TEXT_COL, EMOTION_COL, LABEL_COL]).reset_index(drop=True)
    print(f"Loaded {len(data)} rows")

    dedup_cols = [TEXT_COL] + [c for c in LABEL_COLS_FOR_DEDUP if c in data.columns]
    data = data.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    print(f"After removing true duplicates: {len(data)} rows")

    data["model_input"] = "[EMOTION: " + data[EMOTION_COL].astype(str) + "] " + data[TEXT_COL]

    encoder = LabelEncoder()
    data["label"] = encoder.fit_transform(data[LABEL_COL])

    train_data, val_data = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["label"]
    )
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    # Check 1: model_input overlap (the combined text+emotion string)
    model_input_overlap = set(train_data["model_input"]) & set(val_data["model_input"])
    print(f"\nmodel_input overlap between train/val: {len(model_input_overlap)}")

    # Check 2: raw pidgin_text overlap (text alone, ignoring emotion prefix)
    text_overlap = set(train_data[TEXT_COL]) & set(val_data[TEXT_COL])
    print(f"Raw pidgin_text overlap between train/val: {len(text_overlap)}")

    if text_overlap:
        print("\nNote: text overlap without model_input overlap likely means these are")
        print("legitimate context variants (same text, different emotion/sentiment label),")
        print("which is expected and fine - not true leakage.")
        sample = list(text_overlap)[:5]
        for text in sample:
            t_rows = train_data[train_data[TEXT_COL] == text]
            v_rows = val_data[val_data[TEXT_COL] == text]
            print(f"\nText: {text!r}")
            print(f"  Train labels: {t_rows[LABEL_COL].tolist()}")
            print(f"  Val labels:   {v_rows[LABEL_COL].tolist()}")

    if len(model_input_overlap) == 0:
        print("\nNo leakage detected at the model_input level - split is clean.")
    else:
        print("\nWARNING: model_input overlap found - this IS true leakage, investigate.")



if __name__ == "__main__":
    main()

