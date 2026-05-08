"""
predict.py — Load saved XGBoost models and run inference on new complaints.

Run train.py first to generate the model artifacts.

Usage:
    python predict.py
"""

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb


# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
def load_artifacts():
    """Load all saved model artifacts from disk."""
    clf = xgb.XGBClassifier()
    clf.load_model("xgb_classifier.ubj")

    reg = xgb.XGBRegressor()
    reg.load_model("xgb_regressor.ubj")

    le              = joblib.load("label_encoder_classes.pkl")
    le_type         = joblib.load("label_encoder_complaint_type.pkl")
    clf_feature_cols = joblib.load("clf_feature_cols.pkl")
    reg_feature_cols = joblib.load("reg_feature_cols.pkl")

    return clf, reg, le, le_type, clf_feature_cols, reg_feature_cols


# ---------------------------------------------------------------------------
# Complaint category prediction
# ---------------------------------------------------------------------------
def predict_complaint_category(
    clf,
    le,
    clf_feature_cols: list,
    agency: str,
    borough: str,
    channel: str,
    hour: int,
    month: int,
    status: int = 3,
    resolution_hours: float = 0.0,
) -> str:
    """
    Predict the complaint category (6-class) for a single complaint.

    Args:
        clf: Trained XGBClassifier.
        le: Fitted LabelEncoder for class labels.
        clf_feature_cols: Feature column names used during training.
        agency: Agency code (e.g. 'HPD', 'NYPD').
        borough: Borough name (e.g. 'BRONX', 'BROOKLYN').
        channel: Submission channel (e.g. 'ONLINE', 'MOBILE').
        hour: Hour of the day the complaint was created (0–23).
        month: Month the complaint was created (1–12).
        status: Ordinal status code (default 3 = In Progress).
        resolution_hours: Known resolution hours if available (default 0).

    Returns:
        Predicted complaint category as a string.
    """
    row = pd.DataFrame([{col: 0 for col in clf_feature_cols}])

    row["created_hour"]      = hour
    row["created_dayofweek"] = 0
    row["created_month"]     = month
    row["status"]            = status
    row["resolution_hours"]  = resolution_hours

    borough_col = f"borough_{borough.upper()}"
    channel_col = f"open_data_channel_type_{channel.upper()}"
    agency_col  = f"agency_{agency.upper()}"
    if borough_col in row.columns: row[borough_col] = 1
    if channel_col in row.columns: row[channel_col] = 1
    if agency_col  in row.columns: row[agency_col]  = 1

    pred_label = clf.predict(row)[0]
    return le.inverse_transform([pred_label])[0]


# ---------------------------------------------------------------------------
# Resolution time prediction
# ---------------------------------------------------------------------------
def predict_resolution_time(
    reg,
    le_type,
    reg_feature_cols: list,
    complaint_type: str,
    agency: str,
    borough: str,
    channel: str,
    hour: int,
    month: int,
) -> tuple[float, float]:
    """
    Predict how many hours a complaint will take to resolve.

    Args:
        reg: Trained XGBRegressor.
        le_type: Fitted LabelEncoder for complaint_type_enc.
        reg_feature_cols: Feature column names used during regression training.
        complaint_type: Complaint category string.
        agency: Agency code (e.g. 'HPD').
        borough: Borough name (e.g. 'BRONX').
        channel: Submission channel (e.g. 'ONLINE').
        hour: Hour of the day (0–23).
        month: Month (1–12).

    Returns:
        Tuple of (predicted_hours, predicted_days).
    """
    row = pd.DataFrame([{col: 0 for col in reg_feature_cols}])

    row["created_hour"]      = hour
    row["created_dayofweek"] = 0
    row["created_month"]     = month
    row["status"]            = 3

    borough_col = f"borough_{borough.upper()}"
    channel_col = f"open_data_channel_type_{channel.upper()}"
    agency_col  = f"agency_{agency.upper()}"
    if borough_col in row.columns: row[borough_col] = 1
    if channel_col in row.columns: row[channel_col] = 1
    if agency_col  in row.columns: row[agency_col]  = 1

    if complaint_type in le_type.classes_:
        row["complaint_type_enc"] = le_type.transform([complaint_type])[0]

    hours = float(reg.predict(row)[0])
    return hours, hours / 24


# ---------------------------------------------------------------------------
# Main — demo predictions
# ---------------------------------------------------------------------------
def main():
    print("Loading model artifacts ...")
    clf, reg, le, le_type, clf_feature_cols, reg_feature_cols = load_artifacts()
    print("Models loaded.\n")

    # --- Resolution time demo ---
    scenarios = [
        ("HEAT/HOT WATER",       "HPD",  "BRONX",     "ONLINE", 22, 1),
        ("Illegal Parking",      "NYPD", "BROOKLYN",  "MOBILE",  9, 3),
        ("Noise - Residential",  "NYPD", "MANHATTAN", "PHONE",  23, 6),
        ("Snow or Ice",          "DSNY", "QUEENS",    "ONLINE",  7, 2),
        ("Blocked Driveway",     "NYPD", "BROOKLYN",  "MOBILE",  8, 4),
    ]

    print("=== Predicted Resolution Times ===")
    print(f"{'Complaint Type':<25} {'Agency':<6} {'Borough':<12} {'Hour':>4} {'Month':>5}  →  Predicted Time")
    print("-" * 75)
    for comp, agency, borough, channel, hour, month in scenarios:
        hrs, days = predict_resolution_time(
            reg, le_type, reg_feature_cols,
            comp, agency, borough, channel, hour, month,
        )
        print(
            f"{comp:<25} {agency:<6} {borough:<12} {hour:>4}   {month:>3}"
            f"   →  {hrs:>6.1f} hrs  ({days:.1f} days)"
        )

    # --- Category prediction demo ---
    print("\n=== Predicted Complaint Categories ===")
    cat_scenarios = [
        ("HPD",  "BRONX",     "ONLINE", 22, 1),
        ("NYPD", "BROOKLYN",  "MOBILE",  9, 3),
        ("DSNY", "QUEENS",    "ONLINE",  7, 2),
    ]
    print(f"{'Agency':<6} {'Borough':<12} {'Hour':>4} {'Month':>5}  →  Predicted Category")
    print("-" * 60)
    for agency, borough, channel, hour, month in cat_scenarios:
        cat = predict_complaint_category(
            clf, le, clf_feature_cols,
            agency, borough, channel, hour, month,
        )
        print(f"{agency:<6} {borough:<12} {hour:>4}   {month:>3}   →  {cat}")


if __name__ == "__main__":
    main()
