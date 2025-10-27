from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from .config import (
    NUMERIC_CANDIDATES, CATEGORICAL_CANDIDATES, DATE_COLUMNS,
    LEAKY_FOR_INTEREST
)

# --- Custom transformers ---
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper
        self.lq_ = None
        self.uq_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lq_ = np.nanquantile(X, self.lower, axis=0)
        self.uq_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lq_, self.uq_)

class FreqEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.maps_: Dict[int, Dict[str, float]] = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X).astype(str)
        self.maps_ = {}
        for i, col in enumerate(X.columns):
            vc = X[col].value_counts(dropna=False, normalize=True)
            self.maps_[i] = vc.to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).astype(str)
        out = []
        for i, col in enumerate(X.columns):
            m = self.maps_[i]
            out.append(X[col].map(lambda v: m.get(v, 0.0)).values.reshape(-1, 1))
        return np.hstack(out)

# --- Feature engineering ---
def month_diff(d1: pd.Series, d2: pd.Series) -> pd.Series:
    # months between two datetime series
    d1 = pd.to_datetime(d1, errors="coerce")
    d2 = pd.to_datetime(d2, errors="coerce")
    months = (d1.dt.year - d2.dt.year) * 12 + (d1.dt.month - d2.dt.month)
    return months

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Ensure numeric types for key columns ---
    for col in ["loanAmnt", "installment", "annualIncome", "dti", "revolUtil"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # credit history length in months
    if {"issueDate", "earliesCreditLine"}.issubset(df.columns):
        df["creditHistoryMonths"] = month_diff(df["issueDate"], df["earliesCreditLine"])
    # ratios and logs
    if {"installment", "loanAmnt"}.issubset(df.columns):
        df["installment_ratio"] = df["installment"] / (df["loanAmnt"].replace(0, np.nan))
    if "annualIncome" in df.columns:
        df["log_income"] = np.log1p(df["annualIncome"])
    if "loanAmnt" in df.columns:
        df["log_loanAmnt"] = np.log1p(df["loanAmnt"])
    # time parts
    if "issueDate" in df.columns:
        df["issueYear"] = df["issueDate"].dt.year
        df["issueMonth"] = df["issueDate"].dt.month
    return df

def select_columns(df: pd.DataFrame, target: str = None, task: str = "reg") -> Tuple[List[str], List[str]]:
    # 只挑选真正是数值类型的列
    num_candidates = NUMERIC_CANDIDATES + [
        "creditHistoryMonths", "installment_ratio",
        "log_income", "log_loanAmnt", "issueYear", "issueMonth"
    ]
    num_cols = [
        c for c in num_candidates
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    # 类别列：必须存在而且不是数值
    cat_cols = [
        c for c in CATEGORICAL_CANDIDATES
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])
    ]

    # 防止泄露：预测利率时去掉 grade/subGrade
    if task == "reg" and target == "interestRate":
        cat_cols = [c for c in cat_cols if c not in LEAKY_FOR_INTEREST]

    # 删除目标列
    num_cols = [c for c in num_cols if c != target]
    cat_cols = [c for c in cat_cols if c != target]

    return num_cols, cat_cols

def build_preprocess_pipeline(df: pd.DataFrame, target: str = None, task: str = "reg"):
    num_cols, cat_cols = select_columns(df, target, task)

    low_card = [c for c in cat_cols if df[c].nunique(dropna=False) <= 30]
    high_card = [c for c in cat_cols if c not in low_card]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("winsor", Winsorizer(0.01, 0.99)),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])

    cat_low_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),

    ])

    cat_high_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("freq", FreqEncoder()),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if low_card:
        transformers.append(("cat_low", cat_low_pipe, low_card))
    if high_card:
        transformers.append(("cat_high", cat_high_pipe, high_card))

    pre = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    return pre, num_cols, low_card, high_card

def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Basic NA normalization
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip().replace({"": np.nan})
    return df
