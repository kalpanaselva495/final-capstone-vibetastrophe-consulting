"""
train.py — Data loading, cleaning, feature engineering, and ML pipeline
for the NYC 311 complaints dataset.

Usage:
    python train.py
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Descriptor keyword rules (shared with predict.py via import)
# ---------------------------------------------------------------------------
DESCRIPTOR_RULES = [
    ('Noise',             ['LOUD','NOISE','BANGING','POUNDING','HORN',
                           'MUSIC','ALARM','BARKING','TELEVISION',
                           'PARTY','SHRIEKING','TALKING','CAR/TRUCK']),
    ('Heat_HotWater',     ['ENTIRE BUILDING','APARTMENT ONLY','RADIATOR',
                           'BOILER','NO HEAT','INADEQUATE','HEAT',
                           'HOT WATER','STEAM']),
    ('Plumbing_Water',    ['WATER SUPPLY','BASIN','SINK','BATHTUB',
                           'SHOWER','TOILET','LEAK','NO WATER',
                           'HYDRANT','SEWER','DIRTY WATER','SEWAGE',
                           'DRAIN','DAMP','SLOW LEAK','HEAVY FLOW',
                           'WATER METER','CATCH BASIN','WATER MAIN',
                           'PLUMBING']),
    ('Parking_Traffic',   ['PARKING','DOUBLE PARKED','LICENSE PLATE',
                           'BIKE LANE','BUS LAYOVER','DERELICT',
                           'DRIVEWAY','SIGNAL','CONE','CROSSWALK',
                           'BLOCKED HYDRANT','PEDESTRIAN','TRAFFIC',
                           'VEHICLE','WITH LICENSE']),
    ('Building_Structure',['CEILING','WALL','FLOOR','DOOR','WINDOW',
                           'CABINET','WIRING','OUTLET','INTERCOM',
                           'VENTILATION','ELECTRIC','REFRIGERATOR',
                           'COOKING GAS','SMOKE','CARBON MONOXIDE',
                           'LIGHTING','POWER','PAINT','PLASTER',
                           'MOLD','STRUCTURAL','BELL/BUZZER',
                           'GARBAGE/RECYCLING STORAGE',
                           'ILLEGAL CONVERSION']),
    ('Street_Sidewalk',   ['POTHOLE','SIDEWALK','ROADWAY','CAVE-IN',
                           'STREET LIGHT','SNOW','ICE','FAILED STREET',
                           'ROUGH','BROKEN SIDEWALK','ROAD',
                           'STREET REPAIR','PITTED']),
    ('Sanitation_Waste',  ['TRASH','GARBAGE','RECYCLING','GRAFFITI',
                           'DOG WASTE','ILLEGAL DUMP','DIRTY',
                           'LITTER','UNSANITARY','WASTE']),
    ('Pest_Animal',       ['RAT','PEST','MICE','MOUSE','ROACH',
                           'BED BUG','ANIMAL','INSECT','RODENT',
                           'MOSQUITO','BIRD','DOG']),
    ('Access_Elevator',   ['NO ACCESS','PARTIAL ACCESS','ACCESS','ELEVATOR']),
    ('Tree_Vegetation',   ['BRANCH','TREE','LIMB','FALLEN','TRUNK','ROOT']),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_customer_data(filepath):
    """Load 311 complaints CSV and return a DataFrame."""
    return pd.read_csv(filepath)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_data(df):
    """
    Clean the 311 complaints data.

    Steps:
    1. Remove duplicate rows by unique_key.
    2. Fill missing numeric values with column median.
    3. Fill missing categorical values with column mode.

    Returns a cleaned DataFrame with no duplicates and no missing values.
    """
    cleaned = df.copy()

    before = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset='unique_key', keep='first')
    print(f"Step 1 — Duplicates removed : {before - len(cleaned):,}  "
          f"(rows remaining: {len(cleaned):,})")

    numeric_cols = (
        cleaned.select_dtypes(include=[np.number])
               .columns
               .difference(['unique_key'])
               .tolist()
    )
    filled_numeric = 0
    for col in numeric_cols:
        n_missing = cleaned[col].isnull().sum()
        if n_missing > 0:
            median_val = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(median_val)
            filled_numeric += n_missing
    print(f"Step 2 — Numeric cells filled  : {filled_numeric:,}")

    categorical_cols = cleaned.select_dtypes(include='object').columns.tolist()
    filled_categorical = 0
    for col in categorical_cols:
        n_missing = cleaned[col].isnull().sum()
        if n_missing > 0:
            mode_val = cleaned[col].mode()[0]
            cleaned[col] = cleaned[col].fillna(mode_val)
            filled_categorical += n_missing
    print(f"Step 3 — Categorical cells filled: {filled_categorical:,}")

    print(f"\nCleaning complete — shape: {cleaned.shape}  "
          f"nulls remaining: {cleaned.isnull().sum().sum()}")
    return cleaned


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------
def _map_descriptor(desc):
    """Map a raw descriptor string to one of the DESCRIPTOR_RULES categories."""
    if not isinstance(desc, str):
        return 'Other'
    d = desc.upper()
    for category, keywords in DESCRIPTOR_RULES:
        if any(k in d for k in keywords):
            return category
    return 'Other'


def prepare_features(df, top_n_complaint_types=10, scaler=None):
    """
    Engineer, encode, and scale features from a cleaned 311 DataFrame.

    Args:
        df:                    Cleaned DataFrame from clean_data().
        top_n_complaint_types: Keep this many top complaint types; rest → 'Other'.
        scaler:                Fitted StandardScaler to reuse, or None to fit fresh.

    Returns:
        X (pd.DataFrame), y (pd.Series), scaler (StandardScaler)
    """
    data = df.copy()

    data.drop(columns=['unique_key', 'agency_name', 'resolution_description'],
              inplace=True)

    data['created_date'] = pd.to_datetime(data['created_date'], errors='coerce')
    data['closed_date']  = pd.to_datetime(data['closed_date'],  errors='coerce')
    data['resolution_hours'] = (
        (data['closed_date'] - data['created_date']).dt.total_seconds() / 3600
    )
    data.loc[
        (data['resolution_hours'] < 0) | (data['resolution_hours'] > 720),
        'resolution_hours'
    ] = np.nan

    data['hour_of_day'] = data['created_date'].dt.hour
    data['day_of_week'] = data['created_date'].dt.dayofweek
    data['month']       = data['created_date'].dt.month
    data['is_weekend']  = (data['day_of_week'] >= 5).astype(int)
    data.drop(columns=['created_date', 'closed_date'], inplace=True)

    for col in data.select_dtypes(include=[np.number]).columns:
        if data[col].isnull().any():
            data[col].fillna(data[col].median(), inplace=True)

    data['descriptor'] = data['descriptor'].apply(_map_descriptor)

    top_types = data['complaint_type'].value_counts().head(top_n_complaint_types).index
    data['complaint_type'] = data['complaint_type'].apply(
        lambda x: x if x in top_types else 'Other'
    )

    y        = data['complaint_type'].copy()
    cat_cols = data.select_dtypes(include='object').columns.tolist()
    X        = pd.get_dummies(data, columns=cat_cols, drop_first=True)

    numeric_cols = ['resolution_hours', 'hour_of_day', 'day_of_week', 'month', 'is_weekend']
    if scaler is None:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    else:
        X[numeric_cols] = scaler.transform(X[numeric_cols])

    return X, y, scaler


def prepare_ml_dataframe(df, top_n_complaint_types=10, test_size=0.2,
                         random_state=42):
    """
    Full end-to-end ML preparation pipeline.

    Steps: clean → drop columns → parse dates → temporal features →
           descriptor mapping → group rare types → one-hot encode →
           stratified split → scale numerics.

    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    data = df.copy()

    # Clean
    data = data.drop_duplicates(subset='unique_key', keep='first')
    for col in data.select_dtypes(include=[np.number]).columns.difference(['unique_key']):
        if data[col].isnull().any():
            data[col].fillna(data[col].median(), inplace=True)
    for col in data.select_dtypes(include='object').columns:
        if data[col].isnull().any():
            data[col].fillna(data[col].mode()[0], inplace=True)

    data.drop(columns=['unique_key', 'agency_name', 'resolution_description'],
              inplace=True)

    # Dates
    data['created_date'] = pd.to_datetime(data['created_date'], errors='coerce')
    data['closed_date']  = pd.to_datetime(data['closed_date'],  errors='coerce')
    data['resolution_hours'] = (
        (data['closed_date'] - data['created_date']).dt.total_seconds() / 3600
    )
    data.loc[
        (data['resolution_hours'] < 0) | (data['resolution_hours'] > 720),
        'resolution_hours'
    ] = np.nan
    data.drop(columns=['created_date', 'closed_date'], inplace=True)

    tmp = pd.to_datetime(df['created_date'], errors='coerce')
    data['hour_of_day'] = tmp.dt.hour
    data['day_of_week'] = tmp.dt.dayofweek
    data['month']       = tmp.dt.month
    data['is_weekend']  = (tmp.dt.dayofweek >= 5).astype(int)

    numeric_cols = ['resolution_hours', 'hour_of_day', 'day_of_week', 'month', 'is_weekend']
    for col in numeric_cols:
        if data[col].isnull().any():
            data[col].fillna(data[col].median(), inplace=True)

    data['descriptor'] = data['descriptor'].apply(_map_descriptor)

    top_types = data['complaint_type'].value_counts().head(top_n_complaint_types).index
    data['complaint_type'] = data['complaint_type'].apply(
        lambda x: x if x in top_types else 'Other'
    )

    y        = data['complaint_type'].copy()
    cat_cols = data.select_dtypes(include='object').columns.tolist()
    X        = pd.get_dummies(data, columns=cat_cols, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

    print(f"Pipeline complete — X_train {X_train.shape} | X_test {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler, X_train.columns.tolist()


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def train_model(X_train, y_train, model_type='random_forest', random_state=42):
    """
    Train a classifier on the prepared training data.

    Args:
        X_train:      Feature matrix.
        y_train:      Target labels.
        model_type:   'random_forest' or 'gradient_boosting'.
        random_state: Random seed.

    Returns:
        Fitted model.
    """
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100, max_depth=15,
            n_jobs=-1, random_state=random_state,
            class_weight='balanced'
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=5,
            learning_rate=0.1, random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"Training {model_type} ...")
    model.fit(X_train, y_train)
    print("Training complete.")
    return model


def evaluate_model(model, X_test, y_test):
    """Print accuracy and classification report for a fitted model."""
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {acc:.4f}")
    print(classification_report(y_test, y_pred))
    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import pathlib
    _ROOT = pathlib.Path(__file__).resolve().parents[2]
    DATA_PATH   = str(_ROOT / 'data' / 'raw' / 'urbanpulse_311_complaints.csv')
    MODEL_PATH  = str(pathlib.Path(__file__).parent / 'model.joblib')
    SCALER_PATH = str(pathlib.Path(__file__).parent / 'scaler.joblib')

    print("=== Loading data ===")
    df = load_customer_data(DATA_PATH)
    print(f"Shape: {df.shape}")

    print("\n=== Preparing ML dataframe ===")
    X_train, X_test, y_train, y_test, scaler, feature_names = \
        prepare_ml_dataframe(df)

    print("\n=== Training model ===")
    model = train_model(X_train, y_train, model_type='random_forest')

    print("\n=== Evaluation ===")
    evaluate_model(model, X_test, y_test)

    print("\n=== Saving artifacts ===")
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    import json
    with open('feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    print(f"Saved: {MODEL_PATH}, {SCALER_PATH}, feature_names.json")
