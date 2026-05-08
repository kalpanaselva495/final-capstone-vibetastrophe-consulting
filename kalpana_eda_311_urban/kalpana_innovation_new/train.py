"""
train.py — Load data, preprocess, train XGBoost classifier and regressor,
           and save all artifacts to disk.

Usage:
    python train.py
"""

import re
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_FILE = "urbanpulse_311_complaints.csv"
TOP_5_CATEGORIES = {
    "Illegal Parking",
    "HEAT/HOT WATER",
    "Noise - Residential",
    "Snow or Ice",
    "Blocked Driveway",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_customer_data(filepath: str) -> pd.DataFrame:
    """Load customer data from a CSV file."""
    return pd.read_csv(filepath)


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
def preprocess_311_text(text: str) -> str:
    """Clean and normalize 311 complaint text for classification."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\b\d+(?:st|nd|rd|th)\b", "", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    if len(words) > 100:
        text = " ".join(words[:100])

    return text


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for machine learning on the 311 complaints dataset.

    Steps:
    1. Drop non-predictive columns
    2. Engineer time-based features
    3. Fill missing values
    4. Encode status ordinally
    5. One-hot encode borough, channel, and agency
    6. Scale numeric features with StandardScaler
    7. Clean descriptor text
    """
    df_prep = df.copy()

    df_prep = df_prep.drop(
        columns=["unique_key", "resolution_description", "agency_name"],
        errors="ignore",
    )

    df_prep["created_date"] = pd.to_datetime(df_prep["created_date"])
    df_prep["closed_date"] = pd.to_datetime(df_prep["closed_date"])

    df_prep["created_hour"] = df_prep["created_date"].dt.hour
    df_prep["created_dayofweek"] = df_prep["created_date"].dt.dayofweek
    df_prep["created_month"] = df_prep["created_date"].dt.month
    df_prep["resolution_hours"] = (
        df_prep["closed_date"] - df_prep["created_date"]
    ).dt.total_seconds() / 3600

    df_prep = df_prep.drop(columns=["created_date", "closed_date"])

    df_prep["resolution_hours"] = df_prep["resolution_hours"].fillna(
        df_prep["resolution_hours"].median()
    )

    status_mapping = {
        "Open": 0, "Assigned": 1, "Started": 2,
        "In Progress": 3, "Pending": 4, "Closed": 5, "Unspecified": 0,
    }
    df_prep["status"] = df_prep["status"].map(status_mapping).fillna(0).astype(int)

    df_prep = pd.get_dummies(
        df_prep, columns=["borough", "open_data_channel_type", "agency"], dtype=int
    )

    scaler = StandardScaler()
    exclude_cols = {"status", "complaint_type", "descriptor"}
    onehot_prefixes = ("borough_", "open_data_channel_type_", "agency_")
    numeric_cols = [
        col
        for col in df_prep.select_dtypes(include=["int64", "float64"]).columns
        if col not in exclude_cols and not col.startswith(onehot_prefixes)
    ]
    df_prep[numeric_cols] = scaler.fit_transform(df_prep[numeric_cols])
    df_prep = df_prep.fillna(0)

    df_prep["descriptor"] = (
        df_prep["descriptor"].fillna("").apply(preprocess_311_text)
    )

    return df_prep, scaler


def create_complaint_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Map 151 complaint types down to top 5 + 'Other' (6 classes total)."""
    df = df.copy()
    df["complaint_type"] = df["complaint_type"].where(
        df["complaint_type"].isin(TOP_5_CATEGORIES), other="Other"
    )
    return df


# ---------------------------------------------------------------------------
# Training — classification
# ---------------------------------------------------------------------------
def train_classifier(df_categorized: pd.DataFrame):
    """Train XGBoost multi-class classifier. Returns clf, le, feature_cols, splits."""
    feature_cols = [
        c for c in df_categorized.columns
        if c not in ("complaint_type", "descriptor")
    ]
    X = df_categorized[feature_cols].copy()
    y_raw = df_categorized["complaint_type"].copy()

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Classifier — training on {len(X_train):,} rows, {len(le.classes_)} classes")

    clf = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=len(le.classes_),
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    print("Classifier training complete!\n")
    return clf, le, feature_cols, (X_train, X_test, y_train, y_test)


# ---------------------------------------------------------------------------
# Training — regression
# ---------------------------------------------------------------------------
def train_regressor(df_categorized: pd.DataFrame, df_raw: pd.DataFrame):
    """Train XGBoost regressor for resolution time. Returns reg, le_type, feature_cols, splits."""
    y_reg_raw = df_raw["resolution_hours"].copy()

    df_reg = df_categorized.copy()
    le_type = LabelEncoder()
    df_reg["complaint_type_enc"] = le_type.fit_transform(df_reg["complaint_type"])

    reg_feature_cols = [
        c for c in df_reg.columns
        if c not in ("complaint_type", "descriptor", "resolution_hours")
    ]
    X_reg = df_reg[reg_feature_cols].copy()
    y_reg = y_reg_raw.copy()

    valid_mask = (
        y_reg.notna()
        & (y_reg >= 0)
        & (y_reg <= y_reg.quantile(0.98))
    )
    X_reg = X_reg[valid_mask].reset_index(drop=True)
    y_reg = y_reg[valid_mask].reset_index(drop=True)

    print(f"Regressor — training on {len(X_reg):,} rows after filtering")

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.20, random_state=42
    )

    reg = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        tree_method="hist",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(
        X_reg_train, y_reg_train,
        eval_set=[(X_reg_test, y_reg_test)],
        verbose=50,
    )
    print("Regressor training complete!\n")
    return reg, le_type, reg_feature_cols, (X_reg_train, X_reg_test, y_reg_train, y_reg_test)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== Loading data ===")
    df = load_customer_data(DATA_FILE)
    print(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns\n")

    print("=== Preparing features ===")
    df_prepared, scaler = prepare_features(df)
    df_categorized = create_complaint_categories(df_prepared)
    print(f"Prepared shape: {df_categorized.shape}\n")

    print("=== Training classifier ===")
    clf, le, clf_feature_cols, clf_splits = train_classifier(df_categorized)

    print("=== Training regressor ===")
    reg, le_type, reg_feature_cols, reg_splits = train_regressor(df_categorized, df)

    print("=== Saving artifacts ===")
    clf.save_model("xgb_classifier.ubj")
    reg.save_model("xgb_regressor.ubj")
    joblib.dump(le,              "label_encoder_classes.pkl")
    joblib.dump(le_type,         "label_encoder_complaint_type.pkl")
    joblib.dump(scaler,          "scaler.pkl")
    joblib.dump(clf_feature_cols, "clf_feature_cols.pkl")
    joblib.dump(reg_feature_cols, "reg_feature_cols.pkl")
    joblib.dump(clf_splits,      "clf_splits.pkl")
    joblib.dump(reg_splits,      "reg_splits.pkl")
    print("Artifacts saved.")


if __name__ == "__main__":
    main()
