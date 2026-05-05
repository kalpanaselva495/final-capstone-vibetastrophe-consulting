
from pathlib import Path
from sre_parse import Tokenizer
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import joblib
    

from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gensim.downloader as api
 

PROCESSED_DATA = Path("data/processed")
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
    """Load preprocessed data from data/processed/.

    Use the shared pipeline:
        from pipelines.data_pipeline import load_processed_data
        df = load_processed_data()
    """
    # TODO: Load your preprocessed dataset
    df = pd.read_csv(PROCESSED_DATA / "processed_urbanpulse_311_complaints.csv")
    return df

def preprocess_text(texts):

    texts = texts.fillna("").astype(str)
    texts = texts.str.lower()
    texts = texts.str.replace(r'[^\w\s]', '', regex=True)
    texts = texts.str.replace(r'\s+', ' ', regex=True).str.strip()
    return texts


def create_embedding_matrix(vectorizer, glove_model, embedding_dim=100):
    vocab = vectorizer.get_vocabulary()
    
    embedding_matrix = np.zeros((len(vocab), embedding_dim))

    for i, word in enumerate(vocab):
        if word in glove_model:
            embedding_matrix[i] = glove_model[word]

    return embedding_matrix

def vectorize_text(X_train, X_val):

    from keras.layers import TextVectorization

    vectorizer = TextVectorization(
        max_tokens = 30000,
        output_mode = 'int',
        output_sequence_length = 250
    )

    vectorizer.adapt(X_train)

    X_train_vec = vectorizer(X_train)
    X_val_vec = vectorizer(X_val)

    return X_train_vec, X_val_vec, vectorizer


def train_model(X_train_vec, y_train, X_val_vec, y_val, glove_model, vectorizer):

    import tensorflow as tf
    from keras.layers import Embedding, GRU, Bidirectional, Dense, Dropout, Conv1D, BatchNormalization, GlobalMaxPooling1D
    from keras.models import Sequential
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping

    embedding_dim = 100

    embedding_matrix = create_embedding_matrix(
        vectorizer,
        glove_model,
        embedding_dim 
    )

    vocab_size = len(vectorizer.get_vocabulary())

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim = vocab_size,
                                   output_dim = embedding_dim,
                                   weights = [embedding_matrix],
                                   trainable = True),
        tf.keras.layers.Bidirectional(GRU(128, return_sequences = True)),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation = 'relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(6, activation = 'softmax')
    ])

    early_stopping = EarlyStopping(
        monitor = "val_accuracy",
        patience = 3,
        restore_best_weights = True
    )

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    history = model.fit(
        X_train_vec,
        y_train,
        validation_data = (X_val_vec, y_val),
        epochs = 10,
        batch_size = 128,
        callbacks = [early_stopping]
    )

    return model

def evaluate_model(model, vectorizer, X_val, y_val_encoded, label_encoder, num_examples=10):

    X_val_vec = vectorizer(X_val)
    y_probs = model.predict(X_val_vec)
    y_pred = np.argmax(y_probs, axis = 1)

    f1_weighted = f1_score(y_val_encoded, y_pred, average = 'weighted')
    print('\nWeighted F1:')
    print(f1_weighted)

    print('\nClassification Report:')
    print(classification_report(y_val_encoded, y_pred, target_names=label_encoder.classes_))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val_encoded, y_pred))

    examples_df = pd.DataFrame({
        "text": X_val.iloc[:num_examples].values,
        "actual": label_encoder.inverse_transform(y_val_encoded[:num_examples]),
        "predicted": label_encoder.inverse_transform(y_pred[:num_examples]),
        "confidence": np.max(y_probs[:num_examples], axis=1)
    })

    print("\nExample Predictions:")
    print(examples_df)

    return {
        "weighted_f1": f1_weighted,
        "classification_report": classification_report(
            y_val_encoded,
            y_pred,
            target_names=label_encoder.classes_,
            output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_val_encoded, y_pred),
        "example_predictions": examples_df
    }

def save_model(model, vectorizer, label_encoder):


    SAVED_MODEL_DIR = Path("./models/model4_nlp_classification/saved_model")
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model.save(SAVED_MODEL_DIR / "gru_model.keras")
    joblib.dump(vectorizer, SAVED_MODEL_DIR / "vectorizer.joblib")
    joblib.dump(label_encoder, SAVED_MODEL_DIR / "label_encoder.joblib")

    print(f"Model Saved to {SAVED_MODEL_DIR}")


def main():

    df = pd.read_csv(Path("data/raw/smart_city_csvs/urbanpulse_311_complaints.csv"))

    df = create_complaint_categories(df)
    urban_df = df.copy()
    urban_df['complaint_type'] = df['complaint_category']

    X = urban_df['resolution_description']
    y = urban_df['complaint_type']

    X = preprocess_text(X)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3,
    random_state=42, stratify=y)

    X_train_vec, X_val_vec, vectorizer = vectorize_text(X_train, X_val)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    glove_model = api.load("glove-wiki-gigaword-100")
    
    model = train_model(X_train_vec, y_train_encoded, X_val_vec, y_val_encoded, glove_model, vectorizer)

    evaluate_model(model, vectorizer, X_val, y_val_encoded, label_encoder)

    save_model(model, vectorizer, label_encoder)

    print(f"Training Complete, model saved!")


if __name__ == "__main__":
    main()


