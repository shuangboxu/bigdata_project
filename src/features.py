# src/features.py
from __future__ import annotations
import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"ficoRangeLow", "ficoRangeHigh"}.issubset(df.columns):
        df["fico_mid"] = (df["ficoRangeLow"] + df["ficoRangeHigh"]) / 2.0
    if "earliesCreditLine" in df.columns and pd.api.types.is_datetime64_any_dtype(df["earliesCreditLine"]):
        ref = df["earliesCreditLine"].dt.to_period("M").dt.to_timestamp()
        now = df["issueDate"] if ("issueDate" in df.columns and pd.api.types.is_datetime64_any_dtype(df["issueDate"])) else pd.Timestamp.today()
        df["credit_age_years"] = (now - ref).dt.days / 365.25
    if "revolUtil" in df.columns:
        ru = df["revolUtil"]
        if pd.api.types.is_string_dtype(ru):
            ru = ru.str.rstrip("%")
        df["revolUtil_clean"] = np.clip(pd.to_numeric(ru, errors="coerce"), 0, 150)
    if "dti" in df.columns:
        df["dti_cap"] = np.clip(pd.to_numeric(df["dti"], errors="coerce"), 0, 100)
    if "fico_mid" in df.columns and "dti_cap" in df.columns:
        df["fico_over_dti"] = df["fico_mid"] / (df["dti_cap"] + 1.0)
    if "annualIncome" in df.columns:
        df["annualIncome_log"] = np.log1p(pd.to_numeric(df["annualIncome"], errors="coerce"))
    return df
