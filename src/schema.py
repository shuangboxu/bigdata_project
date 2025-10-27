from typing import List
from .config import NUMERIC_CANDIDATES, CATEGORICAL_CANDIDATES, DATE_COLUMNS, ID_COLUMNS

REQUIRED_ANY: List[str] = ID_COLUMNS + DATE_COLUMNS  # must exist if present in your file
SUGGESTED_NUMERIC = NUMERIC_CANDIDATES
SUGGESTED_CATEG = CATEGORICAL_CANDIDATES

def check_columns(df):
    # only soft checks; we don't hard fail because user data may vary
    missing = []
    for c in ID_COLUMNS + DATE_COLUMNS:
        if c not in df.columns:
            missing.append(c)
    return missing
