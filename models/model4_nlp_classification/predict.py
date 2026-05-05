#!/usr/bin/env python3
"""
Model 4: NLP Classification — Prediction Script
=================================================
Loads your trained model and generates predictions on test data.

Usage: python predict.py
Output: test_data/model4_results.csv
"""
import pandas as pd
from pathlib import Path

import torch
import json

from transformers import( 
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification)

# Paths
MODEL_PATH = Path("models/model4_nlp_classification/saved_model/")
TEST_DATA_DIR = Path("test_data/")
OUTPUT_FILE = TEST_DATA_DIR / "model4_results.csv"


def load_model():
    
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model.eval()

    print("Model Loaded Successfully")
    return model, tokenizer

def load_label_mapping():
    with open(MODEL_PATH / "id2label.json", "r") as f:
        id2label = json.load(f)

    id2label = {int(k): v for k, v in id2label.items()}
    return id2label

def preprocess_text(texts, tokenizer):
    
    return tokenizer(
        list(texts), 
        padding = True,
        truncation = True,
        max_length = 128,
        return_tensors = "pt"
    )


def predict(model, tokenizer, texts, id2label, batch_size=32):
    all_predictions = []
    all_confidences = []

    model.eval()
    texts = list(texts)

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        probabilities = torch.softmax(logits, dim=1)
        confidence_scores, predicted_ids = torch.max(probabilities, dim=1)

        all_predictions.extend([id2label[int(i)] for i in predicted_ids])
        all_confidences.extend(confidence_scores.cpu().numpy())

        if i % 1000 == 0:
            print(f"Processed {i}/{len(texts)} rows")

    return all_predictions, all_confidences

def main():
    # Load model
    # Load test data
    model, tokenizer = load_model()
    id2label = load_label_mapping()

    test_df = pd.read_csv(TEST_DATA_DIR / "test.csv")
    

    predictions, confidence_scores = predict(
        model,
        tokenizer,
        test_df["text"],
        id2label,
        batch_size=32
    )
    
    results = pd.DataFrame({
        "id": test_df.index,
        "predicted_class": predictions,
        "confidence": confidence_scores
    })

    results.to_csv(OUTPUT_FILE, index=False)

    print(f"Predictions saved to {OUTPUT_FILE}")
    # Preprocess text
    # processed = preprocess_text(test_df["text_column"])

    # Generate predictions
    # predictions = predict(model, processed)

    # Save results — MUST match output template exactly
    # results = pd.DataFrame({
    #     "id": test_df["id"],
    #     "predicted_class": predicted_classes,
    #     "confidence": confidence_scores,
    # })
    # results.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    main()
