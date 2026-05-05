#!/usr/bin/env python3
"""
Model 4: NLP Classification — Training Script
===============================================
Train a text classification model on your scenario's text data.

Approaches (pick one):
- TF-IDF + traditional classifier (simplest, often surprisingly good)
- LSTM / GRU neural network
- Fine-tuned transformer (BERT, DistilBERT)

IMPORTANT: Save your vectorizer/tokenizer alongside the model — you'll need
the same text preprocessing at prediction time.
"""
from pathlib import Path
import torch
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
from datasets import Dataset


PROCESSED_DATA = Path("data/raw/smart_city_csvs/urbanpulse_311_complaints.csv")
SAVED_MODEL_DIR = Path("./models/model4_nlp_classification/saved_model")


def load_data():
    """Load text data from data/processed/.

    Use the shared pipeline:
        from pipelines.data_pipeline import load_processed_data
        df = load_processed_data()
    """
    df = load_raw_data()
    return df
    


def preprocess_text(texts):
    """Clean and prepare text for modeling.

    Common steps:
    - Lowercase
    - Handle abbreviations and slang
    - Tokenize
    - Remove stopwords (optional — sometimes they help)

    IMPORTANT: Apply the SAME preprocessing at prediction time.
    """
    # TODO: Clean your text data
    from transformers import DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    texts = list(texts)
    tokenized_texts = tokenizer(texts, truncation = True, max_length=128)
    return tokenized_texts


class ComplaintDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)



def compute_metrics(eval_pred):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    return {
        "accuracy": acc,
        "weighted_f1": f1
    }


def train_model(X_train, y_train, X_val, y_val):
    
    train_dataset = ComplaintDataset(X_train, y_train)
    val_dataset = ComplaintDataset(X_val, y_val)

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)
    data_collator = DataCollatorWithPadding(tokenizer=DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased'))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    for param in model.distilbert.parameters():
        param.requires_grad = False
    for param in model.distilbert.transformer.layer[-2:].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    training_args = TrainingArguments(
        output_dir = SAVED_MODEL_DIR,
        num_train_epochs = 5,
        per_device_train_batch_size = 64,
        per_device_eval_batch_size = 64,
        learning_rate = 1e-5,
        weight_decay = 0.01,
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        eval_strategy = "epoch",
        save_strategy = "epoch",
        logging_strategy = "epoch",
        fp16 = False  
    )

    trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = val_dataset, 
    data_collator = data_collator,
    compute_metrics = compute_metrics,
    )
    
    trainer.train()

    return trainer

    


def evaluate_model(trainer, X_val, y_val):      #model, X_val, y_val):
    
    val_dataset = ComplaintDataset(X_val, y_val)

    predictions = trainer.predict(val_dataset)

    logits = predictions.predictions
    y_true = predictions.label_ids
    y_pred = np.argmax(logits, axis=1)

    print("Weighted F1:")
    print(f1_score(y_true, y_pred, average="weighted"))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

import torch
def quantize_model(model):

    torch.backends.quantized.engine = "qnnpack"

    model.to("cpu")
    model.eval()

    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype = torch.qint8
    )
    return quantized_model



def save_model(trainer):
    
    trainer.save_model(SAVED_MODEL_DIR)
    trainer.tokenizer.save_pretrained(SAVED_MODEL_DIR)
    
def create_complaint_categories(df: pd.DataFrame) -> pd.DataFrame:

  
    top_5 = ['Illegal Parking', 'HEAT/HOT WATER', 'Noise - Residential',
             'Snow or Ice', 'Blocked Driveway']
    df['complaint_category'] = df['complaint_type'].apply(
        lambda x: x if x in top_5 else 'Other'
        )

    print("Complaint category distribution:")
    print(df['complaint_category'].value_counts())

    coverage = df[df['complaint_category'] != 'Other'].shape[0] / len(df) * 100
    print(f"\nTop 5 categories cover {coverage:.1f}% of all complaints")
    print(f"Total classes: {df['complaint_category'].nunique()} (top 5 + Other)")

    return df

print("Starting training pipeline...")
def main():
   

    df = pd.read_csv(Path("data/raw/smart_city_csvs/urbanpulse_311_complaints.csv"))

    df = create_complaint_categories(df)
    urban_df = df.copy()
    urban_df['complaint_type'] = urban_df['complaint_category']
    
    dataset = Dataset.from_pandas(urban_df)

    y = dataset['complaint_category']
    X = dataset['resolution_description']


    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3,
    random_state=42, stratify=y)

    X_train_encoded = preprocess_text(X_train)
    X_val_encoded = preprocess_text(X_val)

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)

    trainer = train_model(X_train_encoded, y_train_encoded, X_val_encoded, y_val_encoded)

    evaluate_model(trainer, X_val_encoded, y_val_encoded)

    quantized_model = quantize_model(trainer.model)

    save_model(trainer)

    torch.save(
        quantized_model.state_dict(),
        SAVED_MODEL_DIR / "distilbert_quantized_state_dict.pth"
    )

  

    print("Training complete, model saved!")


if __name__ == "__main__":
    main()
