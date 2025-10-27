from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

def time_based_split(df: pd.DataFrame, date_col: str = "issueDate", test_size=0.2, val_size=0.2, random_state=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if date_col not in df.columns or df[date_col].isna().all():
        # fallback: random split
        train_df, temp = train_test_split(df, test_size=test_size+val_size, random_state=random_state, shuffle=True)
        val_ratio = val_size / (test_size + val_size)
        val_df, test_df = train_test_split(temp, test_size=1 - val_ratio, random_state=random_state, shuffle=True)
        return train_df, val_df, test_df

    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)
    n = len(df_sorted)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    test_df = df_sorted.iloc[-n_test:]
    val_df = df_sorted.iloc[-n_test-n_val:-n_test]
    train_df = df_sorted.iloc[:-n_test-n_val]
    return train_df, val_df, test_df
