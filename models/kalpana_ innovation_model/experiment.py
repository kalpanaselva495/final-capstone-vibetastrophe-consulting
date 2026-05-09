"""
experiment.py - Exploratory Data Analysis & Experimentation

This module handles:
- Exploratory Data Analysis (EDA)
- Data visualizations
- Experimentation and analysis
- Reflection questions and insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PART 1: BASIC DATA EXPLORATION
# ============================================================================

def explore_data(df):
    """
    Perform initial data exploration.
    
    Args:
        df (pd.DataFrame): The raw dataset
    """
    print("\n" + "=" * 70)
    print("DATA EXPLORATION")
    print("=" * 70)
    
    print(f"\nDataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nData types:\n{df.dtypes}")
    
    print(f"\n\nFirst few rows:")
    print(df.head())
    
    print(f"\n\nMissing values per column:")
    missing = df.isnull().sum()
    missing_pct = missing / len(df) * 100
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing': missing.values,
        'Percent': missing_pct.values
    }).sort_values('Missing', ascending=False)
    print(missing_df[missing_df['Missing'] > 0])
    
    print(f"\n\nComplaint type distribution (raw):")
    print(f"Unique values: {df['complaint_type'].nunique()}")
    print(df['complaint_type'].value_counts().head(15))


# ============================================================================
# PART 2: MISSING VALUES & CLASS DISTRIBUTION
# ============================================================================

def plot_missing_values(df):
    """Visualize missing values and target class distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Missing values
    missing = df.isnull().sum()
    missing_top = missing[missing > 0].sort_values(ascending=False).head(10)
    missing_top.plot(kind='barh', ax=axes[0], color='steelblue')
    axes[0].set_title('Missing Values (Top 10)')
    axes[0].set_xlabel('Count')
    
    # Target class distribution
    df['complaint_type'].value_counts().head(15).plot(kind='barh', ax=axes[1], 
                                                       color='coral')
    axes[1].set_title('Top 15 Complaint Types (raw 151 classes)')
    axes[1].set_xlabel('Count')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()


def plot_class_distribution(df):
    """Compare raw vs collapsed complaint categories."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Raw complaint types (top 15)
    df['complaint_type'].value_counts().head(15).plot(kind='barh', ax=axes[0], 
                                                       color='steelblue')
    axes[0].set_title('Top 15 Complaint Types (raw 151 classes)')
    axes[0].set_xlabel('Count')
    axes[0].invert_yaxis()
    
    # After collapsing to 6 categories
    top_5 = ['Illegal Parking', 'HEAT/HOT WATER', 'Noise - Residential',
             'Snow or Ice', 'Blocked Driveway']
    cat_counts = df['complaint_type'].apply(
        lambda x: x if x in top_5 else 'Other'
    )
    cat_counts.value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[1],
        colors=['#2196F3', '#FF5722', '#4CAF50', '#FFC107', '#9C27B0', '#607D8B'])
    axes[1].set_title('After Collapsing to 6 Categories')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.show()
    
    print('\nTop 5 categories as % of total:')
    print(cat_counts.value_counts(normalize=True).mul(100).round(1).to_string())


# ============================================================================
# PART 3: TEMPORAL PATTERNS
# ============================================================================

def plot_temporal_patterns(df):
    """Analyze complaints by time of day, day of week, and month."""
    df['created_date'] = pd.to_datetime(df['created_date'])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    
    # By hour of day
    df['created_date'].dt.hour.value_counts().sort_index().plot(
        ax=axes[0], color='steelblue', marker='o')
    axes[0].set_title('Complaints by Hour of Day')
    axes[0].set_xlabel('Hour')
    axes[0].set_ylabel('Count')
    
    # By day of week
    day_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    day_counts = (df['created_date'].dt.dayofweek.map(day_map)
                  .value_counts().reindex(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']))
    day_counts.plot(ax=axes[1], kind='bar', color='coral')
    axes[1].set_title('Complaints by Day of Week')
    axes[1].tick_params(axis='x', rotation=45)
    
    # By month
    month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    month_counts = (df['created_date'].dt.month.map(month_map)
                    .value_counts().reindex(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']))
    month_counts.plot(ax=axes[2], kind='bar', color='mediumseagreen')
    axes[2].set_title('Complaints by Month')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# PART 4: GEOGRAPHIC PATTERNS
# ============================================================================

def plot_geographic_patterns(df):
    """Analyze complaints by borough and agency."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # By borough
    df['borough'].value_counts().plot(kind='bar', ax=axes[0], color='slateblue')
    axes[0].set_title('Complaints by Borough')
    axes[0].set_xlabel('Borough')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    for p in axes[0].patches:
        axes[0].annotate(f'{p.get_height():,.0f}', 
                        (p.get_x() + p.get_width()/2, p.get_height()),
                        ha='center', va='bottom', fontsize=9)
    
    # By top agencies
    df['agency'].value_counts().head(10).plot(kind='barh', ax=axes[1], 
                                              color='teal')
    axes[1].set_title('Top 10 Agencies by Complaint Count')
    axes[1].set_xlabel('Count')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# PART 5: RESOLUTION TIME ANALYSIS
# ============================================================================

def analyze_resolution_time(df):
    """Analyze complaint resolution time patterns."""
    df['closed_date'] = pd.to_datetime(df['closed_date'])
    df['created_date'] = pd.to_datetime(df['created_date'])
    df['resolution_hours'] = (
        df['closed_date'] - df['created_date']
    ).dt.total_seconds() / 3600
    
    rt = df['resolution_hours'].dropna()
    rt = rt[(rt >= 0) & (rt <= rt.quantile(0.95))]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Histogram
    rt.hist(bins=60, ax=axes[0], color='tomato', edgecolor='white')
    axes[0].set_title('Resolution Time Distribution (hrs, 95th pct)')
    axes[0].set_xlabel('Hours')
    
    # By status
    df.groupby('status')['resolution_hours'].median().sort_values().plot(
        kind='barh', ax=axes[1], color='teal')
    axes[1].set_title('Median Resolution Time by Status')
    axes[1].set_xlabel('Median Hours')
    
    plt.tight_layout()
    plt.show()
    
    print('\nResolution time stats (hours):')
    print(df['resolution_hours'].describe().round(1))
    
    return df['resolution_hours']


# ============================================================================
# PART 6: COMPLAINT TYPE HEATMAP
# ============================================================================

def plot_complaint_heatmap(df):
    """Plot heatmap of top complaint types by borough."""
    top_5 = ['Illegal Parking', 'HEAT/HOT WATER', 'Noise - Residential',
             'Snow or Ice', 'Blocked Driveway']
    heat_df = df[df['complaint_type'].isin(top_5)].copy()
    pivot = heat_df.groupby(['borough', 'complaint_type']).size().unstack(fill_value=0)
    
    plt.figure(figsize=(12, 5))
    sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd',
                linewidths=0.5, cbar_kws={'label': 'Count'})
    plt.title('Top 5 Complaint Types by Borough')
    plt.tight_layout()
    plt.show()


# ============================================================================
# PART 7: SUBMISSION CHANNEL ANALYSIS
# ============================================================================

def plot_submission_channels(df):
    """Analyze how complaints are submitted."""
    fig, ax = plt.subplots(figsize=(9, 4))
    df['open_data_channel_type'].value_counts().plot(kind='bar', ax=ax, 
                                                      color='darkorange')
    ax.set_title('How Complaints Are Submitted')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():,.0f}', 
                   (p.get_x() + p.get_width()/2, p.get_height()),
                   ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.show()


# ============================================================================
# PART 8: DESCRIPTOR TEXT ANALYSIS
# ============================================================================

def analyze_descriptor_text(df):
    """Analyze the length and content of complaint descriptions."""
    df['text_length'] = df['descriptor'].fillna('').apply(lambda x: len(x.split()))
    
    top_5 = ['Illegal Parking', 'HEAT/HOT WATER', 'Noise - Residential',
             'Snow or Ice', 'Blocked Driveway']
    cat_label = df['complaint_type'].apply(lambda x: x if x in top_5 else 'Other')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Histogram
    df['text_length'].clip(upper=100).hist(bins=50, ax=axes[0], 
                                           color='mediumpurple', edgecolor='white')
    axes[0].set_title('Descriptor Word Count (clipped at 100)')
    axes[0].set_xlabel('Word Count')
    
    # By category
    df.groupby(cat_label)['text_length'].mean().sort_values().plot(
        kind='barh', ax=axes[1], color='mediumpurple')
    axes[1].set_title('Avg Descriptor Length by Category')
    axes[1].set_xlabel('Avg Words')
    
    plt.tight_layout()
    plt.show()
    
    print('Descriptor word count stats:')
    print(df['text_length'].describe().round(1))


# ============================================================================
# REFLECTION QUESTIONS
# ============================================================================

def print_reflection_questions():
    """Print key reflection questions for analysis."""
    
    print("\n" + "=" * 70)
    print("REFLECTION QUESTIONS")
    print("=" * 70)
    
    print("""
PART 1: PROBLEM DEFINITION & DATA LOADING
-------------------------------------------

1. What is the size and scope of the NYC 311 dataset?
   - How many records? What time period?
   - How many complaint types initially?

2. Why is the data imbalanced?
   - Which complaint types are most common?
   - What are the top 5 complaint types?


PART 3: CLASSIFICATION MODEL
------------------------------

1. What accuracy did XGBoost achieve?
   - How does this compare to a baseline (random guessing)?
   - Is it good enough for production?

2. Which features were most important?
   - Look at the top 20 feature importance chart
   - Are the top drivers what you expected from EDA?

3. Where does the model struggle?
   - Examine the confusion matrix
   - Which complaint category has the lowest recall?
   - Why might that be?

4. What does the learning curve tell you?
   - Did the model converge early or late?
   - Does this suggest simple or complex patterns?


PART 4: REGRESSION MODEL (RESOLUTION TIME)
---------------------------------------------

1. What was the MAE (Mean Absolute Error)?
   - In plain English: on average, how many hours off are predictions?
   - Is that acceptable for city operations?

2. Which complaint category was hardest to predict?
   - Look at "MAE by Complaint Category"
   - Why might some types be harder than others?

3. What were the top 3 features driving resolution time?
   - Are they what you expected?
   - Do they make operational sense?

4. How much better is XGBoost than the baseline?
   - Baseline = predicting the mean every time
   - What % improvement does the model achieve?

5. Look at the practical demo predictions
   - Does HEAT/HOT WATER at 10pm in January get a longer time?
   - Does Illegal Parking in the morning get a shorter time?
   - Does the model make operational sense?


OVERALL INSIGHTS
-----------------

1. What could you improve next?
   - Add TF-IDF features from descriptor text?
   - Try hyperparameter tuning?
   - Handle class imbalance with SMOTE?

2. How would you deploy these models?
   - Real-time prediction API?
   - Batch processing?
   - What data freshness do you need?

3. What operational changes would you recommend?
   - Which complaint types are slowest?
   - Which boroughs have longest delays?
   - What should the city prioritize?
    """)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all exploratory analysis."""
    print("Exploratory Data Analysis Module")
    print("\nKey functions:")
    print("  - explore_data(df)")
    print("  - plot_missing_values(df)")
    print("  - plot_class_distribution(df)")
    print("  - plot_temporal_patterns(df)")
    print("  - plot_geographic_patterns(df)")
    print("  - analyze_resolution_time(df)")
    print("  - plot_complaint_heatmap(df)")
    print("  - plot_submission_channels(df)")
    print("  - analyze_descriptor_text(df)")
    print("  - print_reflection_questions()")
    
    print_reflection_questions()


if __name__ == '__main__':
    main()
