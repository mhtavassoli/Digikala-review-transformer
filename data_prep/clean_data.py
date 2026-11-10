# data_prep/clean_data.py
"""
Phase 1: Data Cleaning based on the "Separating Foam from Water" principle.
- rate=0 is treated as MISSING for rating_band, not as 'low'.
- Duplicates are handled with a clear policy.
- Inconsistent labels are dropped.
"""
import pandas as pd
import numpy as np

def clean_rate_zero(df: pd.DataFrame) -> pd.DataFrame:
    """
    Treat rate==0 as NaN for the purpose of rating_band.
    This aligns with the consensus: rate=0 means "not rated", not "very bad".
    However, keep the original 'rate' column for auditing.
    """
    df = df.copy()
    # Create a new column for rate_band calculation
    df['rate_for_band'] = df['rate'].replace(0, np.nan)
    # Apply the project's band thresholds (0-2.5 low, 2.5-4 medium, 4-5 high)
    # But we shift low to start from >0, because 0 is now NaN
    conditions = [
        (df['rate_for_band'] > 0) & (df['rate_for_band'] < 2.5),
        (df['rate_for_band'] >= 2.5) & (df['rate_for_band'] < 4.0),
        (df['rate_for_band'] >= 4.0) & (df['rate_for_band'] <= 5.0)
    ]
    choices = ['low', 'medium', 'high']
    df['rating_band'] = np.select(conditions, choices, default=np.nan)
    return df

def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Policy:
    1. If all columns are identical, keep one.
    2. If only metadata (likes, seller, etc.) differs, keep the first (or latest).
    3. If body, rate, or recommendation_status differs, drop ALL versions
       (because we don't know which label is correct).
    """
    # Identify duplicates based on 'id'
    dup_ids = df[df.duplicated('id', keep=False)]['id'].unique()
    rows_to_drop = []
    
    for rid in dup_ids:
        group = df[df['id'] == rid]
        # Check if critical columns are identical
        critical_cols = ['body', 'rate', 'recommendation_status']
        if group[critical_cols].nunique().max() > 1:
            # Inconsistent labels/text -> drop all
            rows_to_drop.extend(group.index.tolist())
        else:
            # Only metadata differs -> keep the first occurrence
            rows_to_drop.extend(group.index[1:].tolist())
    
    return df.drop(index=rows_to_drop)

def filter_empty_bodies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where body is literally empty, 'nan', or only whitespace.
    Keep emojis and meaningless text; they are part of the real world.
    """
    # Replace the literal string 'nan' (from bad CSV reading) with actual NaN
    df['body'] = df['body'].replace('nan', np.nan)
    # Drop only truly empty
    df = df[df['body'].notna() & (df['body'].str.strip() != '')]
    return df

# --- Pipeline ---
if __name__ == "__main__":
    # Assume df is loaded from your raw CSV
    # df = pd.read_csv('...', dtype=str)  # Read all as string to avoid NaN coercion
    # df = filter_empty_bodies(df)
    # df = handle_duplicates(df)
    # df = clean_rate_zero(df)
    # Now proceed to split...
    pass