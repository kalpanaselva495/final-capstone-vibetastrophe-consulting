from pathlib import Path

import numpy as np
import pandas as pd
import torch
import json

from datasets import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import compute_class_weight, CrossEntropyLoss

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DistilBertConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

DATA_PATH = Path("data/raw/smart_city_csvs/urbanpulse_311_complaints.csv")
SAVED_MODEL_DIR = Path("./models/model4_nlp_classification/saved_model")

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


def load_data():

    df = pd.read_csv(DATA_PATH)
    df = create_complaint_categories(df)
    urban_df = df.copy()
    urban_df['complaint_type'] = urban_df['complaint_category']
    urban_df = urban_df[['complaint_type', 'resolution_description']]
    urban_df = urban_df.rename(columns = {
        "resolution_description" : "text",
        "complaint_type" : "label"
    })

    return urban_df 


def split_data(texts, labels):

    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=0.3,
        random_state=42,
        stratify = labels)

    return X_train, X_val, y_train, y_val

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_text(examples):

    return tokenizer(
        examples['text'],
        padding = "max_length",
        truncation = True,
        max_length = 128,
    )

def label_encode(y_train, y_val):
    
    le = LabelEncoder()

    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    id2label = {i: label for i, label in enumerate(le.classes_)}
    label2id = {label: i for i, label in enumerate(le.classes_)}

    return y_train_encoded, y_val_encoded, le, id2label, label2id

def convert_hf_dataset(X__train, X_val, y_train_encoded, y_val_encoded):

    train_df = pd.DataFrame({
        "text": X_train.tolist(),
        "label": y_train_encoded
    })

    val_df = pd.DataFrame({
        "text": X_val.tolist(),
        "label": y_val_encoded
    })

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    train_dataset = train_dataset.map(tokenize_text, batched=True)
    val_dataset = val_dataset.map(tokenize_text, batched=True)

    train_dataset = train_dataset.remove_columns(["text"])
    val_dataset = val_dataset.remove_columns(["text"])

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    return train_dataset, val_dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "macro_f1": f1_score(labels, preds, average="macro")
    }

def get_class_weights(y_train_encoded):
    class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_encoded),
    y=y_train_encoded
    )

    return torch.tensor(class_weights, dtype=torch.float)



from torch.nn import CrossEntropyLoss

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )

        loss = loss_fct(
            logits.view(-1, model.config.num_labels),
            labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss
    
config = DistilBertConfig.from_pretrained(
    "distilbert-base-uncased",
    num_labels = 6,
    dropout = 0.3,
    attention_dropout = 0.3
)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    config = config)

def train_model(train_dataset, val_dataset, model, tokenizer, y_train_encoded):

    data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

    for param in model.distilbert.parameters():
        param.requires_grad = False
    for param in model.distilbert.transformer.layer[-3:].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    training_args = TrainingArguments(
        num_train_epochs = 5,
        per_device_train_batch_size = 32,
        per_device_eval_batch_size = 32,
        learning_rate = 3e-5,
        weight_decay = 0.02,
        warmup_ratio = 0.1,
        lr_scheduler_type = 'linear',
        evaluation_strategy = 'epoch',
        save_strategy = 'epoch',
        logging_strategy = 'epoch',
        load_best_model_at_end = True,
        metric_for_best_model = 'weighted_f1',
        greater_is_better = True
    )

    class_weights = get_class_weights(y_train_encoded)

    trainer = WeightedTrainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        tokenizer = tokenizer,
        data_collator = data_collator,
        compute_metrics = compute_metrics,
        class_weights = class_weights
    )

    trainer.train()
    
    return model, trainer

def evaluate_model(trainer, val_dataset, label_encoder):

    predictions = trainer.predict(val_dataset)

    logits = predictions.predictions
    y_true = predictions.label_ids
    y_pred = np.argmax(logits, axis=1)

    y_true_labels = label_encoder.inverse_transform(y_true)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    print("Weighted F1:")
    print(f1_score(y_true_labels, y_pred_labels, average="weighted"))

    print("\nMacro F1:")
    print(f1_score(y_true_labels, y_pred_labels, average="macro"))

    print("\nClassification Report:")
    print(classification_report(y_true_labels, y_pred_labels))

    print("\nConfusion Matrix:")
    labels = label_encoder.classes_
    print(confusion_matrix(y_true_labels, y_pred_labels, labels=labels))


def save_model(trainer):
    
    trainer.save_model(SAVED_MODEL_DIR)
    trainer.tokenizer.save_pretrained(SAVED_MODEL_DIR)
    
    with open(SAVED_MODEL_DIR / "id2label.json", "w") as f:

        json.dump(id2label, f)

    print("Model Saved Successfully")
    print("Model Saved Successfully")


def main(): 

    df = load_data()

    texts = df['text']
    labels = df['label']

    X_train, X_val, y_train, y_val = split_data(texts, labels)

    y_train_encoded, y_val_encoded, le, id2label, label2id = label_encode(y_train, y_val)

    train_dataset, val_dataset = convert_hf_dataset(X_train, X_val, y_train_encoded, y_val_encoded)

    model, trainer = train_model(train_dataset, val_dataset, model, tokenizer, y_train_encoded)

    evaluate_model(trainer, val_dataset, le)

    save_model(trainer, id2label)

    print("Training Complete, Model Saved Successfully!")


if __name__ == "__main__":
    main()


