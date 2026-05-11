"""
predict.py — Road Deterioration Prediction using XGBoost
=========================================================
Filters road/street-related complaints from the NYC 311 dataset,
engineers a deterioration severity target, trains an XGBoost classifier,
evaluates it, and exposes a prediction function for new data.

Target classes (deterioration_level):
    0 – Low       : minor issues, resolved quickly
    1 – Medium    : moderate road damage, moderate resolution time
    2 – High      : serious damage (potholes, cave-ins), slow resolution
    3 – Critical  : severe/repeated structural complaints, very slow resolution

Usage:
    python predict.py                        # full train + evaluate pipeline
    python predict.py --input new_data.csv   # predict on new raw CSV
"""

import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, ConfusionMatrixDisplay,
)

from train import load_customer_data, clean_data, _map_descriptor

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Road-related complaint types and descriptor keywords
# ---------------------------------------------------------------------------
ROAD_COMPLAINT_TYPES = {
    'Street Condition',
    'Snow or Ice',
    'Traffic Signal Condition',
    'Blocked Driveway',
    'Street Light Condition',
    'Highway Sign - Damaged',
    'Highway Sign - Missing',
    'Sidewalk Condition',
    'Curb Condition',
    'Pothole',
}

# Severity weight per complaint type (used in target engineering)
COMPLAINT_SEVERITY = {
    'Street Condition'         : 3,
    'Pothole'                  : 3,
    'Sidewalk Condition'       : 2,
    'Curb Condition'           : 2,
    'Snow or Ice'              : 2,
    'Traffic Signal Condition' : 2,
    'Blocked Driveway'         : 1,
    'Street Light Condition'   : 1,
    'Highway Sign - Damaged'   : 2,
    'Highway Sign - Missing'   : 1,
}

# Descriptor keywords that signal road structural damage
HIGH_SEVERITY_DESCRIPTOR_KWS = [
    'POTHOLE', 'CAVE-IN', 'STRUCTURAL', 'BROKEN', 'FAILED',
    'COLLAPSED', 'SINK', 'ROUGH', 'PITTED', 'CRACK',
]


# ---------------------------------------------------------------------------
# Step 1: Filter road-related complaints
# ---------------------------------------------------------------------------
def filter_road_complaints(df):
    """
    Keep only rows whose complaint_type or descriptor indicates a road issue.
    Also keeps rows where descriptor_cat == 'Street_Sidewalk'.

    Returns a filtered copy of the DataFrame.
    """
    df2 = df.copy()

    # Map descriptor to category
    df2['descriptor_cat'] = df2['descriptor'].apply(_map_descriptor)

    mask = (
        df2['complaint_type'].isin(ROAD_COMPLAINT_TYPES) |
        (df2['descriptor_cat'] == 'Street_Sidewalk')
    )
    filtered = df2[mask].reset_index(drop=True)
    print(f"Road complaints filtered: {len(filtered):,} rows "
          f"({len(filtered)/len(df)*100:.1f}% of dataset)")
    print("Complaint type breakdown:")
    print(filtered['complaint_type'].value_counts().head(15).to_string())
    return filtered


# ---------------------------------------------------------------------------
# Step 2: Engineer deterioration severity target
# ---------------------------------------------------------------------------
def create_deterioration_label(df):
    """
    Build a 4-level deterioration severity label from complaint features.

    Scoring logic:
        base_score  = COMPLAINT_SEVERITY[complaint_type]   (1-3)
        + 1 if descriptor contains a high-severity keyword
        + resolution_hours bucket (0 / 1 / 2 based on percentile)

    Final score → label:
        0-1  → 0 (Low)
        2    → 1 (Medium)
        3    → 2 (High)
        4+   → 3 (Critical)
    """
    df2 = df.copy()

    # Parse resolution time
    df2['created_dt'] = pd.to_datetime(df2['created_date'], errors='coerce')
    df2['closed_dt']  = pd.to_datetime(df2['closed_date'],  errors='coerce')
    df2['resolution_hours'] = (
        (df2['closed_dt'] - df2['created_dt']).dt.total_seconds() / 3600
    )
    df2.loc[
        (df2['resolution_hours'] < 0) | (df2['resolution_hours'] > 720),
        'resolution_hours'
    ] = np.nan
    df2['resolution_hours'] = df2['resolution_hours'].fillna(df2['resolution_hours'].median())

    # Base severity from complaint type
    df2['base_score'] = df2['complaint_type'].map(COMPLAINT_SEVERITY).fillna(1).astype(int)

    # Bonus point for structural descriptor keywords
    def _desc_severity(desc):
        if not isinstance(desc, str):
            return 0
        d = desc.upper()
        return 1 if any(kw in d for kw in HIGH_SEVERITY_DESCRIPTOR_KWS) else 0

    df2['desc_bonus'] = df2['descriptor'].apply(_desc_severity)

    # Resolution-time bucket  (0 = fast, 1 = moderate, 2 = slow)
    p33, p66 = df2['resolution_hours'].quantile([0.33, 0.66])
    df2['res_bucket'] = pd.cut(
        df2['resolution_hours'],
        bins=[-np.inf, p33, p66, np.inf],
        labels=[0, 1, 2],
    ).astype(int)

    # Total score → 4-level label
    df2['score'] = df2['base_score'] + df2['desc_bonus'] + df2['res_bucket']
    bins   = [-np.inf, 1, 2, 3, np.inf]
    labels = [0, 1, 2, 3]          # Low / Medium / High / Critical
    df2['deterioration_level'] = pd.cut(df2['score'], bins=bins, labels=labels).astype(int)

    dist = df2['deterioration_level'].value_counts().sort_index()
    level_names = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}
    print("\nDeterioration level distribution:")
    for lvl, cnt in dist.items():
        print(f"  {lvl} ({level_names[lvl]:<8}): {cnt:6,}  ({cnt/len(df2)*100:.1f}%)")

    return df2


# ---------------------------------------------------------------------------
# Step 3: Feature engineering for road deterioration
# ---------------------------------------------------------------------------
def build_road_features(df):
    """
    Build the feature matrix X and target y for road deterioration prediction.

    Features used:
        - Temporal : hour_of_day, day_of_week, month, is_weekend
        - Location : borough (one-hot)
        - Channel  : open_data_channel_type (one-hot)
        - Status   : status (one-hot)
        - Descriptor category (one-hot)
        - Complaint severity weight (numeric)
        - Resolution hours (numeric, capped at 720)

    Returns:
        X (pd.DataFrame), y (pd.Series)
    """
    data = df.copy()

    # Temporal features
    data['created_dt'] = pd.to_datetime(data['created_date'], errors='coerce')
    data['hour_of_day'] = data['created_dt'].dt.hour
    data['day_of_week'] = data['created_dt'].dt.dayofweek
    data['month']       = data['created_dt'].dt.month
    data['is_weekend']  = (data['day_of_week'] >= 5).astype(int)

    # Complaint severity weight (numeric signal)
    data['severity_weight'] = data['complaint_type'].map(COMPLAINT_SEVERITY).fillna(1)

    # Resolution hours (already computed in create_deterioration_label)
    if 'resolution_hours' not in data.columns:
        data['created_dt2'] = pd.to_datetime(data['created_date'], errors='coerce')
        data['closed_dt2']  = pd.to_datetime(data['closed_date'],  errors='coerce')
        data['resolution_hours'] = (
            (data['closed_dt2'] - data['created_dt2']).dt.total_seconds() / 3600
        )
        data.loc[
            (data['resolution_hours'] < 0) | (data['resolution_hours'] > 720),
            'resolution_hours'
        ] = np.nan
    data['resolution_hours'] = data['resolution_hours'].fillna(data['resolution_hours'].median())

    y = data['deterioration_level'].copy()

    # Keep only the columns we need
    feature_cols = [
        'hour_of_day', 'day_of_week', 'month', 'is_weekend',
        'severity_weight', 'resolution_hours',
        'borough', 'open_data_channel_type', 'status', 'descriptor_cat',
    ]
    X_raw = data[feature_cols].copy()

    # One-hot encode categoricals
    cat_cols = ['borough', 'open_data_channel_type', 'status', 'descriptor_cat']
    X = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features: {X.columns.tolist()}")
    return X, y


# ---------------------------------------------------------------------------
# Step 4: Scale numeric features
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    'hour_of_day', 'day_of_week', 'month', 'is_weekend',
    'severity_weight', 'resolution_hours',
]

def scale_features(X_train, X_test):
    """Fit StandardScaler on X_train, apply to both splits. Returns scaled copies + scaler."""
    scaler = StandardScaler()
    num_cols = [c for c in NUMERIC_FEATURES if c in X_train.columns]
    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])
    return X_train, X_test, scaler


# ---------------------------------------------------------------------------
# Step 5: Train XGBoost
# ---------------------------------------------------------------------------
def train_xgboost(X_train, y_train, random_state=42):
    """
    Train an XGBoost multi-class classifier for road deterioration levels.

    Returns the fitted XGBClassifier.
    """
    n_classes = y_train.nunique()
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='multi:softprob',
        num_class=n_classes,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=random_state,
        n_jobs=-1,
    )
    print(f"\nTraining XGBoost ({n_classes} classes, {X_train.shape[1]} features, "
          f"{len(X_train):,} samples) ...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=50,
    )
    print("Training complete.")
    return model


# ---------------------------------------------------------------------------
# Step 6: Evaluate
# ---------------------------------------------------------------------------
LEVEL_NAMES = ['Low', 'Medium', 'High', 'Critical']

def evaluate_model(model, X_test, y_test, plot=True):
    """
    Print accuracy, classification report, and optionally plot
    confusion matrix + feature importance.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc     = accuracy_score(y_test, y_pred)

    present_labels = sorted(y_test.unique())
    target_names   = [LEVEL_NAMES[i] for i in present_labels]

    print(f"\n{'='*55}")
    print(f"  XGBoost Road Deterioration — Evaluation")
    print(f"{'='*55}")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")

    try:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        print(f"  ROC-AUC  : {auc:.4f}  (weighted OvR)")
    except Exception:
        pass

    print(f"\n{classification_report(y_test, y_pred, target_names=target_names)}")

    if plot:
        _plot_confusion_matrix(y_test, y_pred, target_names)
        _plot_feature_importance(model, X_test.columns.tolist())

    return acc


def _plot_confusion_matrix(y_test, y_pred, target_names):
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Road Deterioration — Confusion Matrix')
    plt.tight_layout()
    plt.savefig('road_confusion_matrix.png', dpi=120)
    plt.show()
    print("Saved: road_confusion_matrix.png")


def _plot_feature_importance(model, feature_names, top_n=20):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    top_names   = [feature_names[i] for i in indices]
    top_scores  = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_names[::-1], top_scores[::-1], color='steelblue')
    ax.set_xlabel('Feature Importance (gain)')
    ax.set_title(f'XGBoost — Top {top_n} Features')
    plt.tight_layout()
    plt.savefig('road_feature_importance.png', dpi=120)
    plt.show()
    print("Saved: road_feature_importance.png")


# ---------------------------------------------------------------------------
# Step 7: Cross-validation
# ---------------------------------------------------------------------------
def cross_validate_model(X, y, n_splits=5, random_state=42):
    """Run stratified k-fold CV and report mean ± std accuracy."""
    n_classes = y.nunique()
    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective='multi:softprob', num_class=n_classes,
        eval_metric='mlogloss', use_label_encoder=False,
        random_state=random_state, n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"\nCross-validation ({n_splits}-fold):")
    for i, s in enumerate(scores, 1):
        print(f"  Fold {i}: {s:.4f}")
    print(f"  Mean  : {scores.mean():.4f}  ±  {scores.std():.4f}")
    return scores


# ---------------------------------------------------------------------------
# Prediction on new raw data
# ---------------------------------------------------------------------------
def predict_new(df_raw, model, scaler, feature_names, output_path='road_predictions.csv'):
    """
    Apply the full pipeline to a new raw DataFrame and return predictions.

    Args:
        df_raw:        Raw DataFrame (same schema as training CSV).
        model:         Fitted XGBClassifier.
        scaler:        Fitted StandardScaler.
        feature_names: List of feature column names from training.
        output_path:   Where to save predictions CSV.

    Returns:
        pd.DataFrame with unique_key, predicted_level, level_name, confidence.
    """
    df_clean = clean_data(df_raw)
    df_road  = filter_road_complaints(df_clean)

    if df_road.empty:
        print("No road-related complaints found in input.")
        return pd.DataFrame()

    df_road  = create_deterioration_label(df_road)
    X, _     = build_road_features(df_road)

    # Align columns to training feature set
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names].copy()

    # Scale numerics
    num_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    X[num_cols] = scaler.transform(X[num_cols])

    preds = model.predict(X)
    proba = model.predict_proba(X)

    results = df_road[['unique_key']].copy() if 'unique_key' in df_road.columns \
              else pd.DataFrame(index=range(len(preds)))
    results['predicted_level'] = preds
    results['level_name']      = [LEVEL_NAMES[p] for p in preds]
    results['confidence']      = proba.max(axis=1).round(4)

    results.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    print(results['level_name'].value_counts().to_string())
    return results


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------
def run_training_pipeline(data_path='urbanpulse_311_complaints.csv',
                           model_out='road_xgb_model.joblib',
                           scaler_out='road_xgb_scaler.joblib',
                           features_out='road_xgb_features.json',
                           test_size=0.2, random_state=42, run_cv=True):
    """
    End-to-end pipeline:
    load → clean → filter road → label → features →
    split → scale → train XGBoost → evaluate → save.
    """
    print("=" * 55)
    print("  Road Deterioration Prediction — XGBoost Pipeline")
    print("=" * 55)

    # 1. Load & clean
    print(f"\n[1] Loading data from {data_path} ...")
    df_raw   = load_customer_data(data_path)
    df_clean = clean_data(df_raw)

    # 2. Filter road complaints
    print("\n[2] Filtering road-related complaints ...")
    df_road = filter_road_complaints(df_clean)

    # 3. Create target
    print("\n[3] Engineering deterioration labels ...")
    df_road = create_deterioration_label(df_road)

    # 4. Build features
    print("\n[4] Building feature matrix ...")
    X, y = build_road_features(df_road)

    # 5. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\n[5] Split — Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # 6. Scale
    print("\n[6] Scaling numeric features ...")
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # 7. Cross-validation (optional)
    if run_cv:
        print("\n[7] Cross-validation ...")
        X_cv_scaled = X_train.copy()
        cross_validate_model(X_cv_scaled, y_train, random_state=random_state)

    # 8. Train XGBoost
    print("\n[8] Training XGBoost ...")
    model = train_xgboost(X_train, y_train, random_state=random_state)

    # 9. Evaluate
    print("\n[9] Evaluating on test set ...")
    evaluate_model(model, X_test, y_test, plot=True)

    # 10. Save artifacts
    print("\n[10] Saving artifacts ...")
    joblib.dump(model,  model_out)
    joblib.dump(scaler, scaler_out)
    feature_names = X_train.columns.tolist()
    with open(features_out, 'w') as f:
        json.dump(feature_names, f)
    print(f"  Saved: {model_out}")
    print(f"  Saved: {scaler_out}")
    print(f"  Saved: {features_out}")

    print("\n" + "=" * 55)
    print("  Pipeline complete.")
    print("=" * 55)
    return model, scaler, feature_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Road Deterioration Prediction with XGBoost'
    )
    parser.add_argument('--mode',     default='train',
                        choices=['train', 'predict'],
                        help='"train" runs the full pipeline; '
                             '"predict" applies a saved model to --input')
    parser.add_argument('--input',    default='urbanpulse_311_complaints.csv')
    parser.add_argument('--model',    default='road_xgb_model.joblib')
    parser.add_argument('--scaler',   default='road_xgb_scaler.joblib')
    parser.add_argument('--features', default='road_xgb_features.json')
    parser.add_argument('--output',   default='road_predictions.csv')
    parser.add_argument('--no-cv',    action='store_true',
                        help='Skip cross-validation (faster)')
    args = parser.parse_args()

    if args.mode == 'train':
        run_training_pipeline(
            data_path=args.input,
            model_out=args.model,
            scaler_out=args.scaler,
            features_out=args.features,
            run_cv=not args.no_cv,
        )
    else:
        print("Loading saved model artifacts ...")
        model  = joblib.load(args.model)
        scaler = joblib.load(args.scaler)
        with open(args.features) as f:
            feature_names = json.load(f)

        df_raw = load_customer_data(args.input)
        predict_new(df_raw, model, scaler, feature_names, output_path=args.output)
