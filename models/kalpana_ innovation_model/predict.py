"""
predict.py - Model Predictions and Evaluation

This module handles:
- Making predictions with trained models
- Evaluating model performance
- Practical prediction scenarios
- Confusion matrices and performance metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)


# ============================================================================
# CLASSIFICATION EVALUATION
# ============================================================================

def evaluate_classification(y_test, y_pred, le, clf_model, feature_cols):
    """
    Comprehensive evaluation of classification model.
    
    Args:
        y_test: True labels (integer-encoded)
        y_pred: Predicted labels
        le: LabelEncoder for complaint types
        clf_model: Trained XGBoost classifier
        feature_cols: List of feature column names
        
    Returns:
        dict: Evaluation metrics and results
    """
    print("\n" + "=" * 70)
    print("CLASSIFICATION MODEL EVALUATION")
    print("=" * 70)
    
    acc = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average='macro')
    f1_wt = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nAccuracy          : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"F1 (macro)        : {f1_mac:.4f}")
    print(f"F1 (weighted)     : {f1_wt:.4f}")
    
    print("\nPer-Class Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Feature importance
    importances = clf_model.feature_importances_
    feat_df = pd.DataFrame(
        {'feature': feature_cols, 'importance': importances}
    ).sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(feat_df.head(15).to_string(index=False))
    
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': acc,
        'f1_macro': f1_mac,
        'f1_weighted': f1_wt,
        'confusion_matrix': cm,
        'feature_importance': feat_df,
        'predictions': y_pred
    }


def plot_confusion_matrix(cm, le, title="Confusion Matrix"):
    """Plot confusion matrix heatmap."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
    axes[0].set_title('Raw Counts')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].tick_params(axis='x', rotation=30)
    
    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[1],
                vmin=0, vmax=1)
    axes[1].set_title('Row-Normalized (Recall)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# REGRESSION EVALUATION
# ============================================================================

def evaluate_regression(y_test, y_pred, reg_model, reg_feature_cols):
    """
    Comprehensive evaluation of regression model.
    
    Args:
        y_test: True resolution hours
        y_pred: Predicted resolution hours
        reg_model: Trained XGBoost regressor
        reg_feature_cols: List of feature column names
        
    Returns:
        dict: Evaluation metrics and results
    """
    print("\n" + "=" * 70)
    print("REGRESSION MODEL EVALUATION")
    print("=" * 70)
    
    # Inverse-transform from log1p space back to real hours for interpretable metrics
    y_test_orig = np.expm1(y_test.values)
    y_pred_orig = np.maximum(np.expm1(y_pred), 0)

    mae  = mean_absolute_error(y_test_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    r2   = r2_score(y_test_orig, y_pred_orig)
    mape = np.mean(np.abs((y_test_orig - y_pred_orig) / (y_test_orig + 1))) * 100

    baseline_pred = np.full_like(y_pred_orig, y_test_orig.mean())
    baseline_mae  = mean_absolute_error(y_test_orig, baseline_pred)

    print(f"\nMAE (Mean Absolute Error)  : {mae:.1f} hours  ({mae/24:.1f} days)")
    print(f"RMSE (Root Mean Sq. Error) : {rmse:.1f} hours  ({rmse/24:.1f} days)")
    print(f"R² (Variance explained)    : {r2:.4f}  ({r2*100:.1f}%)")
    print(f"MAPE (Mean Abs % Error)    : {mape:.1f}%")
    print(f"\nBaseline (mean prediction) : {baseline_mae:.1f} hours")
    print(f"Improvement over baseline  : {(baseline_mae - mae):.1f} hours ({(baseline_mae - mae)/baseline_mae*100:.1f}%)")
    
    # Feature importance
    importances = reg_model.feature_importances_
    feat_df = pd.DataFrame(
        {'feature': reg_feature_cols, 'importance': importances}
    ).sort_values('importance', ascending=False)
    
    print("\nTop 15 Features Driving Resolution Time:")
    print(feat_df.head(15).to_string(index=False))
    
    residuals = y_test_orig - y_pred_orig

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'baseline_mae': baseline_mae,
        'residuals': residuals,
        'feature_importance': feat_df,
        'predictions': y_pred_orig
    }


def plot_regression_diagnostics(y_test, y_pred, residuals):
    """Plot regression diagnostic plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Predicted vs Actual
    cap = np.percentile(y_test, 95)
    mask = (y_test <= cap) & (y_pred <= cap)
    axes[0].scatter(y_test[mask], y_pred[mask], alpha=0.15, s=5, color='steelblue')
    axes[0].plot([0, cap], [0, cap], 'r--', linewidth=1.5, label='Perfect prediction')
    axes[0].set_title('Predicted vs Actual Resolution Time')
    axes[0].set_xlabel('Actual Hours')
    axes[0].set_ylabel('Predicted Hours')
    axes[0].legend()
    
    # Residual distribution
    pd.Series(residuals).clip(-200, 200).hist(bins=60, ax=axes[1], 
                                               color='coral', edgecolor='white')
    axes[1].axvline(0, color='navy', linestyle='--', linewidth=1.5)
    axes[1].set_title('Residuals (Actual − Predicted)')
    axes[1].set_xlabel('Error (hours)')
    axes[1].set_ylabel('Count')
    
    # Residual vs Predicted
    axes[2].scatter(y_pred, residuals, alpha=0.15, s=5, color='mediumseagreen')
    axes[2].axhline(0, color='navy', linestyle='--', linewidth=1.5)
    axes[2].set_title('Residuals vs Predicted Values')
    axes[2].set_xlabel('Predicted Hours')
    axes[2].set_ylabel('Residuals (hours)')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_complaint_type(clf_model, le, X_new):
    """
    Predict complaint type for new complaint(s).
    
    Args:
        clf_model: Trained XGBoost classifier
        le: LabelEncoder for classes
        X_new: Feature matrix for new complaints
        
    Returns:
        list: Predicted complaint type names
    """
    y_pred_encoded = clf_model.predict(X_new)
    return le.inverse_transform(y_pred_encoded)


def predict_resolution_time(reg_model, X_new):
    """
    Predict resolution time for new complaint(s).
    
    Args:
        reg_model: Trained XGBoost regressor
        X_new: Feature matrix for new complaints
        
    Returns:
        np.ndarray: Predicted resolution hours
    """
    return np.maximum(np.expm1(reg_model.predict(X_new)), 0)


# ============================================================================
# PRACTICAL PREDICTION SCENARIOS
# ============================================================================

def demo_predictions(reg_model, reg_feature_cols, le_type):
    """
    Run practical demo scenarios - predict resolution time for realistic complaints.
    
    Args:
        reg_model: Trained regression model
        reg_feature_cols: List of feature columns
        le_type: LabelEncoder for complaint types
    """
    
    print("\n" + "=" * 80)
    print("PRACTICAL PREDICTION DEMO: Real Complaint Scenarios")
    print("=" * 80)
    
    def predict_scenario(complaint_type, agency, borough, channel, hour, month):
        """Predict for a specific scenario."""
        # Build feature row
        row = pd.DataFrame([{col: 0 for col in reg_feature_cols}])
        
        # Set time features
        row['created_hour']      = hour
        row['created_dayofweek'] = 0  # Monday as default
        row['created_month']     = month
        row['status']            = 3  # In Progress

        # Derived time features (must match prepare_features)
        row['is_weekend']  = 0  # Monday
        row['season']      = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                               6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}.get(month, 0)
        row['hour_bucket'] = 0 if hour < 6 else (1 if hour < 12 else (2 if hour < 18 else 3))
        
        # Set one-hot flags
        borough_col  = f'borough_{borough.upper()}'
        channel_col  = f'open_data_channel_type_{channel.upper()}'
        agency_col   = f'agency_{agency.upper()}'
        
        if borough_col  in row.columns: row[borough_col]  = 1
        if channel_col  in row.columns: row[channel_col]  = 1
        if agency_col   in row.columns: row[agency_col]   = 1
        
        # Encode complaint type
        if complaint_type in le_type.classes_:
            row['complaint_type_enc'] = le_type.transform([complaint_type])[0]
        
        hours = max(float(np.expm1(reg_model.predict(row)[0])), 0)
        days = hours / 24
        return hours, days
    
    # Realistic scenarios
    scenarios = [
        ("HEAT/HOT WATER",      "HPD",   "BRONX",     "ONLINE", 22, 1),
        ("Illegal Parking",     "NYPD",  "BROOKLYN",  "MOBILE",  9, 3),
        ("Noise - Residential", "NYPD",  "MANHATTAN", "PHONE",  23, 6),
        ("Snow or Ice",         "DSNY",  "QUEENS",    "ONLINE",  7, 2),
        ("Blocked Driveway",    "NYPD",  "BROOKLYN",  "MOBILE",  8, 4),
    ]
    
    print(f"\n{'Complaint':<25} {'Agency':<6} {'Borough':<12} {'Hour':>4} {'Mo':>2}  →  Prediction")
    print("-" * 80)
    
    for comp, agency, borough, channel, hour, month in scenarios:
        hrs, days = predict_scenario(comp, agency, borough, channel, hour, month)
        print(f"{comp:<25} {agency:<6} {borough:<12} {hour:>4}   {month:>2}  →  "
              f"{hrs:>6.1f} hrs  ({days:.1f} days)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run evaluation and demo predictions."""
    print("Use this module after training models in train.py")
    print("\nKey functions:")
    print("  - evaluate_classification(y_test, y_pred, le, clf_model, feature_cols)")
    print("  - evaluate_regression(y_test, y_pred, reg_model, reg_feature_cols)")
    print("  - predict_complaint_type(clf_model, le, X_new)")
    print("  - predict_resolution_time(reg_model, X_new)")
    print("  - demo_predictions(reg_model, reg_feature_cols, le_type)")


if __name__ == '__main__':
    main()
