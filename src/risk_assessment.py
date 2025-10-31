"""Utilities for credit risk assessment on tabular lending data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskWeights:
    """Weights that balance the contribution of each risk factor."""

    interest_rate: float = 0.30
    debt_to_income: float = 0.15
    revol_util: float = 0.10
    delinq_2yrs: float = 0.10
    term: float = 0.05
    fico: float = 0.20
    annual_income: float = 0.10

    def as_dict(self) -> Dict[str, float]:
        return {
            "interest_rate": self.interest_rate,
            "debt_to_income": self.debt_to_income,
            "revol_util": self.revol_util,
            "delinq_2yrs": self.delinq_2yrs,
            "term": self.term,
            "fico": self.fico,
            "annual_income": self.annual_income,
        }


RISK_WEIGHTS = RiskWeights()


def _robust_minmax(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
    """Scale a numeric series to the [0, 1] range using robust quantiles.

    Compared with plain min-max scaling this approach is less sensitive to
    outliers, which is important for financial ratios such as DTI or revolving
    utilization.
    """

    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.empty:
        return pd.Series(dtype="float64")

    lo = s.quantile(lower)
    hi = s.quantile(upper)
    if not np.isfinite(lo):
        lo = s.min(skipna=True)
    if not np.isfinite(hi):
        hi = s.max(skipna=True)

    denom = hi - lo
    if not np.isfinite(denom) or np.isclose(denom, 0):
        scaled = pd.Series(0.5, index=s.index, dtype="float64")
    else:
        scaled = (s - lo) / denom
    scaled = scaled.clip(0, 1)
    return scaled.fillna(scaled.median(skipna=True)).fillna(0.5)


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Dataframe is missing required columns: {missing}")


def assess_risk(df: pd.DataFrame, weights: RiskWeights = RISK_WEIGHTS) -> pd.DataFrame:
    """Return a dataframe with risk score and risk level for every borrower.

    Parameters
    ----------
    df:
        Raw lending data similar to ``data/数据示例.xlsx``.
    weights:
        Contribution of each risk factor. The defaults were hand-picked based on
        domain heuristics: repayment burden and historical delinquencies are
        strong risk amplifiers, while higher FICO and income mitigate risk.

    Returns
    -------
    pd.DataFrame
        ``id``, ``risk_score``, ``risk_level`` plus individual factor
        contributions.
    """

    required = {
        "id",
        "interestRate",
        "dti",
        "revolUtil",
        "delinquency_2years",
        "term",
        "ficoRangeLow",
        "ficoRangeHigh",
        "annualIncome",
    }
    _ensure_columns(df, required)

    data = df.copy()
    data["term"] = pd.to_numeric(data["term"], errors="coerce")
    # Convert term values such as 3/5 (years) to months for a linear scale.
    term = data["term"].where(data["term"] > 10, data["term"] * 12)

    fico = (
        pd.to_numeric(data["ficoRangeLow"], errors="coerce")
        + pd.to_numeric(data["ficoRangeHigh"], errors="coerce")
    ) / 2.0

    components = {
        "interest_rate": _robust_minmax(data["interestRate"]),
        "debt_to_income": _robust_minmax(data["dti"]),
        "revol_util": _robust_minmax(data["revolUtil"]),
        "delinq_2yrs": _robust_minmax(data["delinquency_2years"]),
        "term": _robust_minmax(term),
        # Higher fico/income lower the risk so we flip the scale.
        "fico": 1 - _robust_minmax(fico),
        "annual_income": 1 - _robust_minmax(np.log1p(data["annualIncome"])),
    }

    weight_dict = weights.as_dict()
    risk_score = sum(weight_dict[name] * components[name] for name in components)

    risk_level = pd.cut(
        risk_score,
        bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=["低风险", "中风险", "高风险"],
        right=False,
    )

    out = pd.DataFrame({
        "id": data["id"],
        "risk_score": risk_score.round(4),
        "risk_level": risk_level.astype(str),
    })

    # Attach individual component scores for explainability.
    for name, values in components.items():
        out[f"component_{name}"] = values.round(4)

    return out


def summarize_risk(risk_df: pd.DataFrame) -> Dict[str, int]:
    """Provide a simple count of customers in each risk bucket."""

    if "risk_level" not in risk_df:
        raise KeyError("Expected column 'risk_level' in risk assessment results")
    counts = risk_df["risk_level"].value_counts().sort_index()
    return counts.to_dict()

