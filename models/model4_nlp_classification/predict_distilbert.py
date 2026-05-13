#!/usr/bin/env python3
"""
Model 4: NLP Classification — DistilBERT Prediction Script
===========================================================
Drop-in replacement for predict.py once the fine-tuned weights are available.
Weights must be at saved_model/final_model/model.safetensors (see README).

Usage: python predict_distilbert.py
Output: test_data/model4_results.csv
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

MODEL_PATH     = Path("models/model4_nlp_classification/saved_model/final_model")
TEST_DATA_DIR  = Path("test_data/")
OUTPUT_FILE    = TEST_DATA_DIR / "model4_results.csv"
TEST_DATA_FILE = TEST_DATA_DIR / "urbanpulse_311_complaints.csv"

LABELS = {
    0: "Blocked Driveway",
    1: "HEAT/HOT WATER",
    2: "Illegal Parking",
    3: "Noise - Residential",
    4: "Other",
    5: "Snow or Ice",
}

BATCH_SIZE = 64
MAX_LENGTH = 128


def load_model():
    """Load fine-tuned DistilBERT from saved_model/final_model/."""
    weights = MODEL_PATH / "model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(
            f"DistilBERT weights not found at {weights}.\n"
            "Re-run training (python train.py) and commit weights via Git LFS.\n"
            "See README for step-by-step instructions."
        )
    model     = DistilBertForSequenceClassification.from_pretrained(str(MODEL_PATH))
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_PATH))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer


def preprocess_text(texts):
    texts = texts.fillna("").astype(str).str.lower()
    texts = texts.str.replace(r"[^\w\s]", "", regex=True)
    texts = texts.str.replace(r"\s+", " ", regex=True).str.strip()
    return texts.tolist()


def predict(model, tokenizer, texts, batch_size=BATCH_SIZE):
    """Generate predictions on text data.

    Returns a tuple of (predicted_classes, confidence_scores).
    """
    device    = next(model.parameters()).device
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc   = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        if i % 10000 == 0 and i > 0:
            print(f"  {i:,} / {len(texts):,} processed")

    all_probs     = np.vstack(all_probs)
    predicted_ids = np.argmax(all_probs, axis=1)
    confidence    = all_probs.max(axis=1).round(4)
    labels        = [LABELS.get(int(idx), "Other") for idx in predicted_ids]
    return labels, confidence


def main():
    # Load model
    model, tokenizer = load_model()

    # Load test data
    test_df = pd.read_csv(TEST_DATA_FILE)

    # Preprocess text
    texts = preprocess_text(test_df["resolution_description"])

    # Generate predictions
    print(f"Predicting on {len(texts):,} records...")
    predicted_classes, confidence_scores = predict(model, tokenizer, texts)

    # Save results — MUST match output template exactly
    results = pd.DataFrame({
        "id":              test_df["unique_key"],
        "predicted_class": predicted_classes,
        "confidence":      confidence_scores,
    })
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
