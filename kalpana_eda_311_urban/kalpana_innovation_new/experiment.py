"""
experiment.py — EDA visualizations and model evaluation plots.

Run train.py first to generate the model artifacts.

Usage:
    python experiment.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
%matplotlib inline


# ---------------------------------------------------------------------------
# Load data and artifacts
# ---------------------------------------------------------------------------
def load_all():
    """Load the raw dataset and all trained artifacts."""
    from train import load_customer_data, prepare_features, create_complaint_categories

    df = load_customer_data("urbanpulse_311_complaints.csv")
    df_prepared, _ = prepare_features(df)
    df_categorized = create_complaint_categories(df_prepared)

    clf = xgb.XGBClassifier()
    clf.load_model("xgb_classifier.ubj")

    reg = xgb.XGBRegressor()
    reg.load_model("xgb_regressor.ubj")

    le               = joblib.load("label_encoder_classes.pkl")
    le_type          = joblib.load("label_encoder_complaint_type.pkl")
    clf_feature_cols = joblib.load("clf_feature_cols.pkl")
    reg_feature_cols = joblib.load("reg_feature_cols.pkl")
    clf_splits       = joblib.load("clf_splits.pkl")
    reg_splits       = joblib.load("reg_splits.pkl")

    return (df, df_categorized, clf, reg, le, le_type,
            clf_feature_cols, reg_feature_cols, clf_splits, reg_splits)


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
def eda_overview(df: pd.DataFrame):
    """Print basic dataset info and missing value summary."""
    print(f"Shape: {df.shape}")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(f"\nMissing values:\n{missing}")
    print(f"\nTotal missing: {missing.sum():,} ({missing.sum()/df.size*100:.2f}%)")


def plot_complaint_distribution(df: pd.DataFrame):
    """Bar chart of top 15 complaint types and pie chart of 6-category split."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    df["complaint_type"].value_counts().head(15).plot(
        kind="barh", ax=axes[0], color="steelblue"
    )
    axes[0].set_title("Top 15 Complaint Types (raw 151 classes)")
    axes[0].set_xlabel("Count")
    axes[0].invert_yaxis()

    top_5 = ["Illegal Parking", "HEAT/HOT WATER", "Noise - Residential",
              "Snow or Ice", "Blocked Driveway"]
    cat_counts = df["complaint_type"].apply(
        lambda x: x if x in top_5 else "Other"
    )
    cat_counts.value_counts().plot(
        kind="pie", autopct="%1.1f%%", ax=axes[1],
        colors=["#2196F3", "#FF5722", "#4CAF50", "#FFC107", "#9C27B0", "#607D8B"],
    )
    axes[1].set_title("After Collapsing to 6 Categories")
    axes[1].set_ylabel("")
    plt.tight_layout()
    plt.show()


def plot_temporal_patterns(df: pd.DataFrame):
    """Complaint counts by hour, day of week, and month."""
    df = df.copy()
    df["created_date"] = pd.to_datetime(df["created_date"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    df["created_date"].dt.hour.value_counts().sort_index().plot(
        ax=axes[0], color="steelblue", marker="o"
    )
    axes[0].set_title("Complaints by Hour of Day")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Count")

    day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    day_counts = (
        df["created_date"].dt.dayofweek.map(day_map)
        .value_counts()
        .reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    )
    day_counts.plot(ax=axes[1], kind="bar", color="coral")
    axes[1].set_title("Complaints by Day of Week")
    axes[1].tick_params(axis="x", rotation=45)

    month_map = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    month_counts = (
        df["created_date"].dt.month.map(month_map)
        .value_counts()
        .reindex(list(month_map.values()))
    )
    month_counts.plot(ax=axes[2], kind="bar", color="mediumseagreen")
    axes[2].set_title("Complaints by Month")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def plot_borough_distribution(df: pd.DataFrame):
    """Bar chart of complaint counts by borough."""
    fig, ax = plt.subplots(figsize=(9, 4))
    df["borough"].value_counts().plot(kind="bar", ax=ax, color="slateblue")
    ax.set_title("Complaints by Borough")
    ax.set_xlabel("Borough")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():,.0f}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center", va="bottom", fontsize=9,
        )
    plt.tight_layout()
    plt.show()


def plot_resolution_time(df: pd.DataFrame):
    """Resolution time histogram and median by status."""
    df = df.copy()
    df["created_date"] = pd.to_datetime(df["created_date"])
    df["closed_date"]  = pd.to_datetime(df["closed_date"])
    df["resolution_hours"] = (
        df["closed_date"] - df["created_date"]
    ).dt.total_seconds() / 3600

    rt = df["resolution_hours"].dropna()
    rt = rt[(rt >= 0) & (rt <= rt.quantile(0.95))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    rt.hist(bins=60, ax=axes[0], color="tomato", edgecolor="white")
    axes[0].set_title("Resolution Time Distribution (hrs, 95th pct)")
    axes[0].set_xlabel("Hours")

    df.groupby("status")["resolution_hours"].median().sort_values().plot(
        kind="barh", ax=axes[1], color="teal"
    )
    axes[1].set_title("Median Resolution Time by Status")
    axes[1].set_xlabel("Median Hours")
    plt.tight_layout()
    plt.show()

    print("Resolution time stats (hours):")
    print(df["resolution_hours"].describe().round(1))


def plot_complaint_borough_heatmap(df: pd.DataFrame):
    """Heatmap of top 5 complaint types across boroughs."""
    top_5 = ["Illegal Parking", "HEAT/HOT WATER", "Noise - Residential",
              "Snow or Ice", "Blocked Driveway"]
    heat_df = df[df["complaint_type"].isin(top_5)].copy()
    pivot = heat_df.groupby(["borough", "complaint_type"]).size().unstack(fill_value=0)

    plt.figure(figsize=(12, 5))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, cbar_kws={"label": "Count"})
    plt.title("Top 5 Complaint Types by Borough")
    plt.tight_layout()
    plt.show()


def plot_channel_distribution(df: pd.DataFrame):
    """Bar chart of complaint submission channels."""
    fig, ax = plt.subplots(figsize=(9, 4))
    df["open_data_channel_type"].value_counts().plot(
        kind="bar", ax=ax, color="darkorange"
    )
    ax.set_title("How Complaints Are Submitted")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():,.0f}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center", va="bottom", fontsize=9,
        )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Classifier evaluation
# ---------------------------------------------------------------------------
def evaluate_classifier(clf, le, clf_feature_cols, clf_splits):
    """Print metrics and plot confusion matrix + feature importances."""
    X_train, X_test, y_train, y_test = clf_splits
    y_pred = clf.predict(X_test)

    acc    = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro")
    f1_wt  = f1_score(y_test, y_pred, average="weighted")

    print("=" * 55)
    print("  XGBoost Classifier Test Results")
    print("=" * 55)
    print(f"  Accuracy      : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 (macro)    : {f1_mac:.4f}")
    print(f"  F1 (weighted) : {f1_wt:.4f}")
    print("=" * 55)
    print()
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
    axes[0].set_title("Confusion Matrix — Raw Counts", fontsize=13)
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    axes[0].tick_params(axis="x", rotation=30)

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=le.classes_, yticklabels=le.classes_,
                ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix — Row-Normalised (Recall)", fontsize=13)
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.show()

    # Feature importance
    importances = clf.feature_importances_
    feat_df = (
        pd.DataFrame({"feature": clf_feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
    )
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    top20 = feat_df.head(20)
    axes[0].barh(top20["feature"][::-1], top20["importance"][::-1], color="steelblue")
    axes[0].set_title("XGBoost Classifier — Top 20 Feature Importances", fontsize=13)
    axes[0].set_xlabel("Importance")
    axes[0].tick_params(axis="y", labelsize=9)

    cum_imp = feat_df["importance"].cumsum() / feat_df["importance"].sum()
    axes[1].plot(range(1, len(cum_imp) + 1), cum_imp, color="tomato", linewidth=2)
    axes[1].axhline(0.8, color="grey",  linestyle="--", label="80%")
    axes[1].axhline(0.9, color="navy",  linestyle="--", label="90%")
    n80 = (cum_imp < 0.8).sum() + 1
    n90 = (cum_imp < 0.9).sum() + 1
    axes[1].axvline(n80, color="grey", linestyle=":")
    axes[1].axvline(n90, color="navy", linestyle=":")
    axes[1].set_title("Cumulative Feature Importance", fontsize=13)
    axes[1].set_xlabel("Number of Features"); axes[1].set_ylabel("Cumulative Importance")
    axes[1].legend(); axes[1].set_xlim(0, len(cum_imp))
    plt.tight_layout(); plt.show()

    print(f"\nTop 20 features:\n{feat_df.head(20).to_string(index=False)}")
    print(f"\n{n80} features explain 80%  |  {n90} features explain 90%")


def plot_learning_curve(clf, le, clf_feature_cols, clf_splits):
    """Retrain with early stopping and plot log-loss learning curve."""
    X_train, X_test, y_train, y_test = clf_splits

    xgb_es = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=len(le.classes_),
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        tree_method="hist",
        eval_metric="mlogloss",
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1,
    )
    xgb_es.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )

    evals      = xgb_es.evals_result()
    train_loss = evals["validation_0"]["mlogloss"]
    val_loss   = evals["validation_1"]["mlogloss"]
    best_round = xgb_es.best_iteration

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_loss, label="Train log-loss", color="steelblue")
    ax.plot(val_loss,   label="Val log-loss",   color="tomato")
    ax.axvline(best_round, color="grey", linestyle="--",
               label=f"Best round: {best_round}")
    ax.set_title("XGBoost Learning Curve (log-loss vs boosting rounds)")
    ax.set_xlabel("Boosting Round"); ax.set_ylabel("Log-Loss")
    ax.legend()
    plt.tight_layout(); plt.show()

    print(f"Best iteration  : {best_round}")
    print(f"Best val loss   : {val_loss[best_round]:.4f}")
    print(f"Final train loss: {train_loss[best_round]:.4f}")


# ---------------------------------------------------------------------------
# Regressor evaluation
# ---------------------------------------------------------------------------
def evaluate_regressor(reg, reg_feature_cols, reg_splits, df_categorized, df_raw):
    """Print MAE/RMSE/R² and plot prediction diagnostics."""
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = reg_splits
    y_reg_pred = reg.predict(X_reg_test)

    mae  = mean_absolute_error(y_reg_test, y_reg_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    r2   = r2_score(y_reg_test, y_reg_pred)
    mape = np.mean(np.abs((y_reg_test - y_reg_pred) / (y_reg_test + 1))) * 100
    baseline_mae = mean_absolute_error(
        y_reg_test, [y_reg_train.mean()] * len(y_reg_test)
    )

    print("=" * 55)
    print("  XGBoost Regressor — Response Time Results")
    print("=" * 55)
    print(f"  MAE   : {mae:.1f} hrs  ({mae/24:.1f} days)")
    print(f"  RMSE  : {rmse:.1f} hrs  ({rmse/24:.1f} days)")
    print(f"  R²    : {r2:.4f}  ({r2*100:.1f}%)")
    print(f"  MAPE  : {mape:.1f}%")
    print(f"  Baseline MAE : {baseline_mae:.1f} hrs")
    print("=" * 55)

    residuals = y_reg_test.values - y_reg_pred

    # Recover valid_mask to get category labels aligned with test set
    y_reg_raw = df_raw["resolution_hours"].copy()
    valid_mask = (
        y_reg_raw.notna()
        & (y_reg_raw >= 0)
        & (y_reg_raw <= y_reg_raw.quantile(0.98))
    )

    le_type = joblib.load("label_encoder_complaint_type.pkl")
    df_reg = df_categorized.copy()
    df_reg["complaint_type_enc"] = le_type.transform(df_reg["complaint_type"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cap = np.percentile(y_reg_test, 95)
    mask_cap = (y_reg_test <= cap) & (y_reg_pred <= cap)
    axes[0].scatter(y_reg_test[mask_cap], y_reg_pred[mask_cap],
                    alpha=0.15, s=5, color="steelblue")
    axes[0].plot([0, cap], [0, cap], "r--", linewidth=1.5, label="Perfect prediction")
    axes[0].set_title("Predicted vs Actual Resolution Time")
    axes[0].set_xlabel("Actual Hours"); axes[0].set_ylabel("Predicted Hours")
    axes[0].legend()

    pd.Series(residuals).clip(-200, 200).hist(
        bins=60, ax=axes[1], color="coral", edgecolor="white"
    )
    axes[1].axvline(0, color="navy", linestyle="--", linewidth=1.5)
    axes[1].set_title("Residuals (Actual − Predicted)")
    axes[1].set_xlabel("Error (hours)"); axes[1].set_ylabel("Count")

    test_idx = y_reg_test.index
    cat_labels = df_reg.loc[valid_mask].reset_index(drop=True).loc[test_idx, "complaint_type"]
    mae_by_cat = (
        pd.DataFrame({
            "actual": y_reg_test.values,
            "predicted": y_reg_pred,
            "category": cat_labels.values,
        })
        .groupby("category")
        .apply(lambda g: mean_absolute_error(g["actual"], g["predicted"]))
        .sort_values()
    )
    mae_by_cat.plot(kind="barh", ax=axes[2], color="teal", edgecolor="white")
    axes[2].set_title("MAE by Complaint Category (hours)")
    axes[2].set_xlabel("Mean Absolute Error (hours)")

    plt.tight_layout(); plt.show()

    print("\nMAE per complaint category:")
    for cat, err in mae_by_cat.items():
        print(f"  {cat:<25} {err:.1f} hrs  ({err/24:.1f} days)")

    # Feature importance
    imp = reg.feature_importances_
    feat_reg_df = (
        pd.DataFrame({"feature": reg_feature_cols, "importance": imp})
        .sort_values("importance", ascending=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    top20 = feat_reg_df.head(20)
    axes[0].barh(top20["feature"][::-1], top20["importance"][::-1], color="darkorange")
    axes[0].set_title("Top 20 Features Driving Resolution Time", fontsize=13)
    axes[0].set_xlabel("Importance")
    axes[0].tick_params(axis="y", labelsize=9)

    borough_cols = [c for c in df_reg.columns if c.startswith("borough_")]
    borough_avg = {}
    for col in borough_cols:
        borough_name = col.replace("borough_", "")
        m = valid_mask & (df_categorized[col] == 1)
        if m.sum() > 0:
            borough_avg[borough_name] = df_raw["resolution_hours"][m].mean()

    pd.Series(borough_avg).sort_values().plot(
        kind="barh", ax=axes[1], color="mediumpurple", edgecolor="white"
    )
    axes[1].set_title("Avg Actual Resolution Time by Borough (hours)", fontsize=13)
    axes[1].set_xlabel("Average Hours")
    plt.tight_layout(); plt.show()

    print("\nTop 15 features driving resolution time:")
    print(feat_reg_df.head(15).to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== Loading data and artifacts ===")
    (df, df_categorized, clf, reg, le, le_type,
     clf_feature_cols, reg_feature_cols,
     clf_splits, reg_splits) = load_all()

    print("\n=== EDA Overview ===")
    eda_overview(df)
    plot_complaint_distribution(df)
    plot_temporal_patterns(df)
    plot_borough_distribution(df)
    plot_resolution_time(df)
    plot_complaint_borough_heatmap(df)
    plot_channel_distribution(df)

    print("\n=== Classifier Evaluation ===")
    evaluate_classifier(clf, le, clf_feature_cols, clf_splits)

    print("\n=== Classifier Learning Curve ===")
    plot_learning_curve(clf, le, clf_feature_cols, clf_splits)

    print("\n=== Regressor Evaluation ===")
    evaluate_regressor(reg, reg_feature_cols, reg_splits, df_categorized, df)


if __name__ == "__main__":
    main()
