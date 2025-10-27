# src/pipeline_reg.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

from .features import build_features

ART_DIR = Path("artifacts/regression")
ART_DIR.mkdir(parents=True, exist_ok=True)

def _split_time_aware(df: pd.DataFrame, time_col="issueDate", val_ratio=0.2, test_ratio=0.2):
    if time_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df = df.sort_values(time_col)
    n = len(df)
    n_test = int(n * test_ratio)
    n_val  = int((n - n_test) * val_ratio)
    train = df.iloc[: n - n_test - n_val]
    val   = df.iloc[n - n_test - n_val : n - n_test]
    test  = df.iloc[n - n_test :]
    return train, val, test

def _save_figs(y_true, y_pred, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.scatter(y_true, y_pred, s=8, alpha=0.5)
    m, M = float(np.min(y_true)), float(np.max(y_true))
    ax.plot([m, M], [m, M], "k--", lw=1)
    ax.set_xlabel("True"); ax.set_ylabel("Pred"); ax.set_title("Pred vs True")
    fig.tight_layout(); fig.savefig(outdir / "pred_vs_true.png", dpi=150); plt.close(fig)

    res = y_pred - y_true
    fig, ax = plt.subplots(figsize=(4,3))
    ax.hist(res, bins=30); ax.set_title("Residuals")
    fig.tight_layout(); fig.savefig(outdir / "residuals.png", dpi=150); plt.close(fig)

def train_regression(df: pd.DataFrame, target="interestRate") -> Dict[str, Any]:
    df = build_features(df)
    y = pd.to_numeric(df[target], errors="coerce").astype(float)
    X = df.drop(columns=[target])

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if (not pd.api.types.is_numeric_dtype(X[c])) and (X[c].dtype.name != "datetime64[ns]")]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01))]), cat_cols),
        ],
        remainder="drop"
    )

    hgb = HistGradientBoostingRegressor(learning_rate=0.06, max_depth=8, max_iter=300, l2_regularization=0.1, min_samples_leaf=40, random_state=42)
    lasso = Lasso(alpha=0.01, max_iter=10000, random_state=42)

    pipe_hgb = Pipeline([("pre", pre), ("model", hgb)])
    pipe_las = Pipeline([("pre", pre), ("model", lasso)])

    data = pd.concat([X, y.rename("target")], axis=1)
    train, val, test = _split_time_aware(data)
    X_tr, y_tr = train.drop(columns=["target"]), train["target"].astype(float)
    X_va, y_va = val.drop(columns=["target"]), val["target"].astype(float)
    X_te, y_te = test.drop(columns=["target"]), test["target"].astype(float)

    pipe_hgb.fit(X_tr, y_tr)
    pipe_las.fit(X_tr, y_tr)

    def eval_blend(X_, y_):
        p1 = pipe_hgb.predict(X_); p2 = pipe_las.predict(X_)
        p  = 0.7 * p1 + 0.3 * p2
        return {"MAE": float(mean_absolute_error(y_, p)), "RMSE": float(np.sqrt(mean_squared_error(y_, p))), "R2": float(r2_score(y_, p))}, p

    val_metrics, val_pred = eval_blend(X_va, y_va)
    test_metrics, test_pred = eval_blend(X_te, y_te)

    outdir = ART_DIR / "hgb_reg"
    _save_figs(y_va.values, val_pred, outdir)

    with open(ART_DIR / "regression_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2, ensure_ascii=False)

    return {"val": val_metrics, "test": test_metrics, "model_dir": str(outdir)}
