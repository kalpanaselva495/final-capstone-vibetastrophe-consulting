"""
experiment.py — Exploratory Data Analysis for the NYC 311 complaints dataset.

Runs all EDA plots from the notebook:
  - Missing value analysis
  - Top complaint types, boroughs, agencies
  - Temporal patterns (hour, day, resolution time)
  - Descriptor category analysis
  - Complaint type vs borough heatmap

Usage:
    python experiment.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from train import load_customer_data, clean_data, _map_descriptor, DESCRIPTOR_RULES

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


# ---------------------------------------------------------------------------
# Part 1: Missing value analysis
# ---------------------------------------------------------------------------
def plot_missing_values(df):
    missing     = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df  = (
        pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
        .sort_values('Missing %', ascending=False)
    )

    print('=== Missing Value Summary ===')
    print(missing_df[missing_df['Missing Count'] > 0])

    cols_with_missing = missing_df[missing_df['Missing Count'] > 0]
    if not cols_with_missing.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(cols_with_missing.index, cols_with_missing['Missing %'], color='salmon')
        ax.set_xlabel('Missing %')
        ax.set_title('Missing Values by Column')
        for i, v in enumerate(cols_with_missing['Missing %']):
            ax.text(v + 0.2, i, f'{v}%', va='center')
        plt.tight_layout()
        plt.show()
    else:
        print('No missing values found.')


# ---------------------------------------------------------------------------
# Part 2: Distribution of key categorical variables
# ---------------------------------------------------------------------------
def plot_categorical_distributions(df):
    # Top 15 complaint types
    top15 = df['complaint_type'].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    top15.plot(kind='barh', ax=ax, color='steelblue')
    ax.invert_yaxis()
    ax.set_xlabel('Number of Complaints')
    ax.set_title('Top 15 Complaint Types')
    for i, v in enumerate(top15.values):
        ax.text(v + 200, i, f'{v:,}', va='center', fontsize=8)
    plt.tight_layout()
    plt.show()
    print(f'Total unique complaint types: {df["complaint_type"].nunique()}')

    # Borough / Status / Channel
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    borough_counts = df['borough'].value_counts()
    axes[0].bar(borough_counts.index, borough_counts.values, color='teal')
    axes[0].set_title('Complaints by Borough')
    axes[0].set_xlabel('Borough'); axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=30)

    status_counts = df['status'].value_counts()
    axes[1].bar(status_counts.index, status_counts.values, color='coral')
    axes[1].set_title('Complaint Status')
    axes[1].set_xlabel('Status'); axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=20)

    channel_counts = df['open_data_channel_type'].value_counts()
    axes[2].bar(channel_counts.index, channel_counts.values, color='mediumpurple')
    axes[2].set_title('Submission Channel')
    axes[2].set_xlabel('Channel'); axes[2].set_ylabel('Count')
    axes[2].tick_params(axis='x', rotation=20)

    plt.tight_layout()
    plt.show()

    # Top 10 agencies
    top_agencies = df['agency'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    top_agencies.plot(kind='bar', ax=ax, color='darkorange')
    ax.set_title('Top 10 Agencies by Complaint Volume')
    ax.set_xlabel('Agency'); ax.set_ylabel('Number of Complaints')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.show()
    print('\nTop 10 agencies:\n', top_agencies)


# ---------------------------------------------------------------------------
# Part 3: Date & time analysis
# ---------------------------------------------------------------------------
def plot_temporal_patterns(df_clean):
    df_eda = df_clean.copy()
    df_eda['created_date'] = pd.to_datetime(df_eda['created_date'], errors='coerce')
    df_eda['closed_date']  = pd.to_datetime(df_eda['closed_date'],  errors='coerce')

    df_eda['resolution_hours'] = (
        (df_eda['closed_date'] - df_eda['created_date']).dt.total_seconds() / 3600
    )
    df_eda['hour_of_day'] = df_eda['created_date'].dt.hour
    df_eda['day_of_week'] = df_eda['created_date'].dt.day_name()
    df_eda['month']       = df_eda['created_date'].dt.month
    df_eda['is_weekend']  = df_eda['created_date'].dt.dayofweek >= 5

    print('Temporal features:')
    print(df_eda[['created_date','closed_date','resolution_hours',
                  'hour_of_day','day_of_week','is_weekend']].head())
    print('\nResolution time stats (hours):')
    print(df_eda['resolution_hours'].describe().round(2))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    hour_counts = df_eda['hour_of_day'].value_counts().sort_index()
    axes[0, 0].bar(hour_counts.index, hour_counts.values, color='steelblue')
    axes[0, 0].set_title('Complaints by Hour of Day')
    axes[0, 0].set_xlabel('Hour'); axes[0, 0].set_ylabel('Count')

    day_order  = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    day_counts = df_eda['day_of_week'].value_counts().reindex(day_order)
    axes[0, 1].bar(day_counts.index, day_counts.values, color='teal')
    axes[0, 1].set_title('Complaints by Day of Week')
    axes[0, 1].set_xlabel('Day'); axes[0, 1].set_ylabel('Count')
    axes[0, 1].tick_params(axis='x', rotation=30)

    res_valid = df_eda['resolution_hours'].dropna()
    res_valid = res_valid[(res_valid >= 0) & (res_valid <= 500)]
    axes[1, 0].hist(res_valid, bins=50, color='coral', edgecolor='white')
    axes[1, 0].set_title('Resolution Time Distribution (<=500 hrs)')
    axes[1, 0].set_xlabel('Hours to Resolve'); axes[1, 0].set_ylabel('Count')

    top10_types = df_clean['complaint_type'].value_counts().head(10).index
    median_res  = (
        df_eda[df_eda['complaint_type'].isin(top10_types)]
        .groupby('complaint_type')['resolution_hours']
        .median().sort_values()
    )
    axes[1, 1].barh(median_res.index, median_res.values, color='mediumpurple')
    axes[1, 1].set_title('Median Resolution Time — Top 10 Types')
    axes[1, 1].set_xlabel('Median Hours')

    plt.tight_layout()
    plt.show()

    # Heatmap: complaint type vs borough
    pivot = (
        df_clean[df_clean['complaint_type'].isin(top10_types)]
        .groupby(['complaint_type', 'borough'])
        .size()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd', ax=ax, linewidths=0.5)
    ax.set_title('Complaint Type vs Borough (Top 10 Types)')
    ax.set_xlabel('Borough'); ax.set_ylabel('Complaint Type')
    plt.tight_layout()
    plt.show()

    return df_eda


# ---------------------------------------------------------------------------
# Part 4: Descriptor category analysis
# ---------------------------------------------------------------------------
def plot_descriptor_analysis(df_clean):
    df_desc = df_clean.copy()
    df_desc['descriptor_cat'] = df_desc['descriptor'].apply(_map_descriptor)

    df_desc['created_date'] = pd.to_datetime(df_desc['created_date'], errors='coerce')
    df_desc['closed_date']  = pd.to_datetime(df_desc['closed_date'],  errors='coerce')
    df_desc['resolution_hours'] = (
        (df_desc['closed_date'] - df_desc['created_date']).dt.total_seconds() / 3600
    )
    df_desc.loc[
        (df_desc['resolution_hours'] < 0) | (df_desc['resolution_hours'] > 720),
        'resolution_hours'
    ] = np.nan

    cat_order = df_desc['descriptor_cat'].value_counts().index.tolist()
    print("Descriptor category counts:")
    print(df_desc['descriptor_cat'].value_counts().to_string())

    # Plot 1: Category distribution
    counts = df_desc['descriptor_cat'].value_counts().reindex(cat_order)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color='steelblue')
    ax.set_xlabel('Number of Complaints')
    ax.set_title('Descriptor Category Distribution')
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + 300, bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', fontsize=8)
    plt.tight_layout()
    plt.show()

    # Plot 2: Descriptor category vs top 10 complaint types (heatmap)
    top10 = df_desc['complaint_type'].value_counts().head(10).index
    pivot = (
        df_desc[df_desc['complaint_type'].isin(top10)]
        .groupby(['descriptor_cat', 'complaint_type'])
        .size()
        .unstack(fill_value=0)
        .reindex(cat_order)
    )
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd',
                ax=ax, linewidths=0.4, cbar_kws={'label': 'Count'})
    ax.set_title('Descriptor Category vs Complaint Type (Top 10)')
    ax.set_xlabel('Complaint Type'); ax.set_ylabel('Descriptor Category')
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    plt.show()

    # Plot 3: Descriptor category vs borough (stacked %)
    borough_pivot = (
        df_desc.groupby(['descriptor_cat', 'borough'])
        .size().unstack(fill_value=0).reindex(cat_order)
    )
    borough_pct = borough_pivot.div(borough_pivot.sum(axis=1), axis=0) * 100
    colors = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B2','#937860']
    borough_pct.plot(kind='barh', stacked=True, figsize=(12, 6),
                     color=colors[:len(borough_pct.columns)])
    plt.title('Descriptor Category vs Borough (% share)')
    plt.xlabel('Percentage (%)'); plt.ylabel('Descriptor Category')
    plt.legend(title='Borough', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Plot 4: Resolution time by descriptor category
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    res_data = [
        df_desc.loc[(df_desc['descriptor_cat'] == cat) &
                    (df_desc['resolution_hours'] <= 200),
                    'resolution_hours'].dropna().values
        for cat in cat_order
    ]
    axes[0].boxplot(res_data, vert=False, patch_artist=True,
                    boxprops=dict(facecolor='lightsteelblue'),
                    medianprops=dict(color='red', linewidth=2))
    axes[0].set_yticks(range(1, len(cat_order) + 1))
    axes[0].set_yticklabels(cat_order)
    axes[0].set_xlabel('Resolution Hours (capped at 200)')
    axes[0].set_title('Resolution Time by Descriptor Category')

    medians = (
        df_desc.groupby('descriptor_cat')['resolution_hours']
        .median().reindex(cat_order).sort_values()
    )
    axes[1].barh(medians.index, medians.values, color='coral')
    axes[1].set_xlabel('Median Resolution Hours')
    axes[1].set_title('Median Resolution Time by Descriptor Category')
    for i, v in enumerate(medians.values):
        axes[1].text(v + 0.3, i, f'{v:.1f}h', va='center', fontsize=8)

    plt.tight_layout()
    plt.show()
    print("\nMedian resolution hours by category:")
    print(medians.round(2).to_string())


# ---------------------------------------------------------------------------
# Main — run all EDA
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    DATA_PATH = 'urbanpulse_311_complaints.csv'

    print("=== Loading data ===")
    df = load_customer_data(DATA_PATH)
    print(f"Shape: {df.shape}")

    print("\n=== Missing Value Analysis ===")
    plot_missing_values(df)

    print("\n=== Categorical Distributions ===")
    plot_categorical_distributions(df)

    print("\n=== Cleaning data ===")
    df_clean = clean_data(df)

    print("\n=== Temporal Patterns ===")
    plot_temporal_patterns(df_clean)

    print("\n=== Descriptor Category Analysis ===")
    plot_descriptor_analysis(df_clean)

    print("\nEDA complete.")
