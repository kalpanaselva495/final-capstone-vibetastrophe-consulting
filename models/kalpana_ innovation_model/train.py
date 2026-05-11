"""
train.py - Data Loading, Preprocessing, and Model Training

This module handles:
- Loading 311 complaint data
- Feature engineering and preprocessing
- Training XGBoost classification and regression models
"""

import joblib
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Classification
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# ============================================================================
# PART 1: DATA LOADING
# ============================================================================

def load_customer_data(filepath):
    """
    Load customer data from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: The loaded data
    """
    return pd.read_csv(filepath)


# ============================================================================
# PART 2: DATA PREPARATION
# ============================================================================

def preprocess_311_text(text: str) -> str:
    """
    Clean and normalize 311 complaint text for classification.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs and emails BEFORE punctuation stripping
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)

    # 3. Remove ordinal numbers whole (5th, 42nd, 1st, 3rd, etc.)
    text = re.sub(r'\b\d+(?:st|nd|rd|th)\b', '', text)

    # 4. Strip non-ASCII characters (handles Chinese, Spanish, etc.)
    text = text.encode('ascii', errors='ignore').decode('ascii')

    # 5. Remove all punctuation except apostrophes
    text = re.sub(r"[^\w\s']", ' ', text)

    # 6. Remove remaining digit sequences
    text = re.sub(r'\d+', '', text)

    # 7. Collapse repeated characters ("sooooo" -> "soo")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # 8. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 9. Truncate to 100 words
    words = text.split()
    if len(words) > 100:
        text = ' '.join(words[:100])

    return text


def prepare_features(df):
    """
    Prepare features for machine learning on the urbanpulse 311 complaints dataset.

    Steps:
    1. Drop non-predictive columns (unique_key, resolution_description, agency_name)
    2. Engineer time-based features from created_date and closed_date
    3. Fill missing values
    4. Encode status ordinally
    5. One-hot encode borough, open_data_channel_type, and agency
    6. Scale numeric features using StandardScaler
    7. Clean descriptor text using preprocess_311_text

    Args:
        df (pd.DataFrame): The raw DataFrame

    Returns:
        pd.DataFrame: The prepared DataFrame with all numeric columns (scaled)
                      and a cleaned descriptor column.
    """
    df_prep = df.copy()

    # Step 1: Drop non-predictive / high-cardinality columns
    df_prep = df_prep.drop(
        columns=['unique_key', 'resolution_description', 'agency_name'],
        errors='ignore'
    )

    # Step 2: Engineer time-based features from dates
    df_prep['created_date'] = pd.to_datetime(df_prep['created_date'])
    df_prep['closed_date']  = pd.to_datetime(df_prep['closed_date'])

    df_prep['created_hour']      = df_prep['created_date'].dt.hour
    df_prep['created_dayofweek'] = df_prep['created_date'].dt.dayofweek
    df_prep['created_month']     = df_prep['created_date'].dt.month
    df_prep['resolution_hours']  = (
        df_prep['closed_date'] - df_prep['created_date']
    ).dt.total_seconds() / 3600

    # is_resolved: 1 = ticket closed, 0 = still open (closed_date was missing)
    df_prep['is_resolved'] = df_prep['closed_date'].notna().astype(int)

    df_prep = df_prep.drop(columns=['created_date', 'closed_date'])

    # Step 2b: Derived time features
    df_prep['is_weekend'] = (df_prep['created_dayofweek'] >= 5).astype(int)
    df_prep['season'] = df_prep['created_month'].map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    ).fillna(0).astype(int)
    df_prep['hour_bucket'] = pd.cut(
        df_prep['created_hour'], bins=[-1, 5, 11, 17, 23],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # Step 3: Fill missing resolution_hours using per-complaint-type median
    # Open tickets (is_resolved=0) had no closed_date — fill with type-level median
    # so the imputed value is realistic for that complaint category
    type_medians = df_prep.groupby('complaint_type')['resolution_hours'].transform('median')
    global_median = df_prep['resolution_hours'].median()
    df_prep['resolution_hours'] = df_prep['resolution_hours'].fillna(type_medians).fillna(global_median)

    # Step 4: Encode status ordinally
    status_mapping = {
        'Open': 0, 'Assigned': 1, 'Started': 2,
        'In Progress': 3, 'Pending': 4, 'Closed': 5, 'Unspecified': 0
    }
    df_prep['status'] = df_prep['status'].map(status_mapping).fillna(0).astype(int)

    # Step 5: One-hot encode categorical columns
    df_prep = pd.get_dummies(
        df_prep, columns=['borough', 'open_data_channel_type', 'agency'], dtype=int
    )

    # Step 6: Scale numeric features
    scaler = StandardScaler()
    exclude_cols = {'status', 'complaint_type', 'descriptor', 'resolution_hours'}
    onehot_prefixes = ('borough_', 'open_data_channel_type_', 'agency_')
    numeric_cols = [
        col for col in df_prep.select_dtypes(include='number').columns
        if col not in exclude_cols and not col.startswith(onehot_prefixes)
    ]
    if numeric_cols:
        df_prep[numeric_cols] = scaler.fit_transform(df_prep[numeric_cols])

    df_prep = df_prep.fillna(0)

    # Step 7: Clean descriptor text — fill missing with 'unspecified' not empty string
    df_prep['descriptor'] = df_prep['descriptor'].fillna('unspecified').apply(preprocess_311_text)

    return df_prep


def create_complaint_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map complaint types to the top 5 categories + "Other" (6 classes total).

    The top 5 categories are:
    - Illegal Parking
    - HEAT/HOT WATER
    - Noise - Residential
    - Snow or Ice
    - Blocked Driveway

    Everything else maps to "Other". This gives you 6 classes total.
    """
    top_5 = {
        "Illegal Parking",
        "HEAT/HOT WATER",
        "Noise - Residential",
        "Snow or Ice",
        "Blocked Driveway",
    }
    
    df = df.copy()
    df["complaint_type"] = df["complaint_type"].where(
        df["complaint_type"].isin(top_5), other="Other"
    )
    return df


# ============================================================================
# PART 3: CLASSIFICATION MODEL TRAINING
# ============================================================================

def train_classification_model(X_train, X_test, y_train, y_test, le):
    """
    Train XGBoost classifier for 311 complaint type prediction.
    
    Args:
        X_train, X_test: Training and test feature matrices
        y_train, y_test: Training and test labels (integer-encoded)
        le: LabelEncoder for complaint types
        
    Returns:
        tuple: (trained_model, predictions, accuracy, f1_score)
    """
    print("Training XGBoost Classification Model...")
    
    xgb_clf = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(le.classes_),
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        tree_method='hist',
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )

    xgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    
    y_pred = xgb_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average='macro')
    
    print(f"\n[Classification Results]")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1 (macro): {f1_mac:.4f}")
    
    return xgb_clf, y_pred, acc, f1_mac


# ============================================================================
# PART 4: REGRESSION MODEL TRAINING
# ============================================================================

def train_regression_model(X_reg_train, X_reg_test, y_reg_train, y_reg_test):
    """
    Train XGBoost regressor for 311 complaint resolution time prediction.
    
    Args:
        X_reg_train, X_reg_test: Training and test feature matrices
        y_reg_train, y_reg_test: Training and test resolution times (hours)
        
    Returns:
        tuple: (trained_model, predictions, mae, rmse, r2)
    """
    print("Training XGBoost Regression Model...")
    
    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        tree_method='hist',
        eval_metric='rmse',
        random_state=42,
        n_jobs=-1
    )

    xgb_reg.fit(
        X_reg_train, y_reg_train,
        eval_set=[(X_reg_test, y_reg_test)],
        verbose=50
    )
    
    y_reg_pred = xgb_reg.predict(X_reg_test)
    
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    r2 = r2_score(y_reg_test, y_reg_pred)
    
    print(f"\n[Regression Results]")
    print(f"MAE: {mae:.1f} hours ({mae/24:.1f} days)")
    print(f"RMSE: {rmse:.1f} hours ({rmse/24:.1f} days)")
    print(f"R²: {r2:.4f} ({r2*100:.1f}%)")
    
    return xgb_reg, y_reg_pred, mae, rmse, r2

# ============================================================================
# MODEL SAVING
# ============================================================================

def save_models(clf_model, clf_le, reg_model, reg_le, reg_feature_cols):
    """Save trained models and encoders."""
    # os.makedirs("saved_models", exist_ok=True)
    joblib.dump(clf_model, "models/kalpana_ innovation_model/saved_model/clf_model.joblib")
    joblib.dump(clf_le, "models/kalpana_ innovation_model/saved_model/clf_le.joblib")
    joblib.dump(reg_model, "models/kalpana_ innovation_model/saved_model/reg_model.joblib")
    joblib.dump(reg_le, "models/kalpana_ innovation_model/saved_model/reg_le.joblib")
    joblib.dump(reg_feature_cols, "models/kalpana_ innovation_model/saved_model/reg_feature_cols.joblib")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Run the complete training pipeline."""
    
    # 1. Load data
    print("Loading data...")
    df = load_customer_data('data/raw/smart_city_csvs/urbanpulse_311_complaints.csv')
    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns\n")
    
    # 2. Prepare features
    print("Preparing features...")
    df_prepared = prepare_features(df)
    print(f"Prepared: {df_prepared.shape[0]:,} rows, {df_prepared.shape[1]} columns\n")
    
    # 3. Create complaint categories
    print("Creating complaint categories...")
    df_categorized = create_complaint_categories(df_prepared)
    print(f"Complaint types: {df_categorized['complaint_type'].nunique()}")
    print(df_categorized['complaint_type'].value_counts())
    print()
    
    # 4. Train classification model
    print("=" * 70)
    print("CLASSIFICATION: Predicting complaint type")
    print("=" * 70)
    
    feature_cols = [c for c in df_categorized.columns
                    if c not in ('complaint_type', 'descriptor')]
    X = df_categorized[feature_cols]
    y_raw = df_categorized['complaint_type']
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    clf_model, clf_pred, clf_acc, clf_f1 = train_classification_model(
        X_train, X_test, y_train, y_test, le
    )
    
    # 5. Train regression model
    print("\n" + "=" * 70)
    print("REGRESSION: Predicting resolution time")
    print("=" * 70)
    
    df_reg = df_categorized.copy()
    le_type = LabelEncoder()
    df_reg['complaint_type_enc'] = le_type.fit_transform(df_reg['complaint_type'])
    
    reg_feature_cols = [c for c in df_reg.columns
                        if c not in ('complaint_type', 'descriptor', 'resolution_hours')]
    
    X_reg = df_reg[reg_feature_cols]
    y_reg_raw = df_reg['resolution_hours'].copy()
    
    # Filter valid resolution times
    valid_mask = (
        y_reg_raw.notna() &
        (y_reg_raw >= 0) &
        (y_reg_raw <= y_reg_raw.quantile(0.98))
    )
    
    X_reg = X_reg[valid_mask].reset_index(drop=True)
    y_reg = np.log1p(y_reg_raw[valid_mask].reset_index(drop=True))

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.20, random_state=42
    )
    
    reg_model, reg_pred, reg_mae, reg_rmse, reg_r2 = train_regression_model(
        X_reg_train, X_reg_test, y_reg_train, y_reg_test
    )
    
    save_models(clf_model, le, reg_model, le_type, reg_feature_cols)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    return {
        'clf_model': clf_model,
        'clf_le': le,
        'reg_model': reg_model,
        'reg_le': le_type,
        'reg_feature_cols': reg_feature_cols,
        'df_categorized': df_categorized,
        'valid_mask': valid_mask
    }


if __name__ == '__main__':
    models = main()
    print("\nModels trained and ready for prediction!")
