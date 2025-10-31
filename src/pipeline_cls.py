# src/pipeline_cls.py
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt

from .features import build_features

ART_DIR = Path("artifacts/classification")
ART_DIR.mkdir(parents=True, exist_ok=True)

LEAKY = {"subGrade", "interestRate"}

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

def _save_cm(y_true, y_pred, labels, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5,4))
    label_ids = list(range(len(labels)))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=label_ids,
        display_labels=labels,
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(outdir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

def train_classification(df: pd.DataFrame, target="grade") -> Dict[str, Any]:
    cols = [c for c in df.columns if c not in LEAKY]
    df = df[cols]
    df = build_features(df)

    y_raw = df[target].astype(str)
    labels_sorted = sorted(y_raw.unique().tolist())
    label2id = {lab: i for i, lab in enumerate(labels_sorted)}
    y = y_raw.map(label2id).astype(int)
    X = df.drop(columns=[target])

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if (not pd.api.types.is_numeric_dtype(X[c])) and (X[c].dtype.name != "datetime64[ns]")]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01))]), cat_cols),
        ],
        remainder="drop"
    )

    base = HistGradientBoostingClassifier(learning_rate=0.08, max_depth=8, max_iter=400, l2_regularization=0.2, min_samples_leaf=60, class_weight="balanced", random_state=42)
    clf = Pipeline([("pre", pre), ("model", CalibratedClassifierCV(base, method="isotonic", cv=3))])

    data = pd.concat([X, y.rename("target")], axis=1)
    train, val, test = _split_time_aware(data)
    X_tr, y_tr = train.drop(columns=["target"]), train["target"].astype(int)
    X_va, y_va = val.drop(columns=["target"]), val["target"].astype(int)
    X_te, y_te = test.drop(columns=["target"]), test["target"].astype(int)

    clf.fit(X_tr, y_tr)

    def eval_set(X_, y_):
        p = clf.predict(X_)
        proba = clf.predict_proba(X_)
        acc = accuracy_score(y_, p)
        f1_macro = f1_score(y_, p, average="macro")
        f1_weighted = f1_score(y_, p, average="weighted")
        try:
            auc_ovr = roc_auc_score(y_, proba, multi_class="ovr")
        except Exception:
            auc_ovr = float("nan")
        return {"accuracy": float(acc), "f1_macro": float(f1_macro), "f1_weighted": float(f1_weighted), "auc_ovr": float(auc_ovr)}, p

    val_metrics, val_pred = eval_set(X_va, y_va)
    test_metrics, test_pred = eval_set(X_te, y_te)

    outdir = ART_DIR / "hgb_cls"
    _save_cm(y_va, val_pred, labels_sorted, outdir)

    with open(ART_DIR / "classification_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": val_metrics, "labels": labels_sorted}, f, indent=2, ensure_ascii=False)

    return {"metrics": val_metrics, "labels": labels_sorted, "model_dir": str(outdir)}

def run_classification_pipeline(df):
    """
    Robust wrapper for agent_flow:
    - Auto-detect binary label column: try ['pd_label','target','label','default','y'] (case-insensitive),
      fallback to the last column.
    - Coerce label to {0,1}: supports numeric/boolean/strings (yes/no, true/false, default/good, 逾期/正常, 是/否).
    - Drop rows with missing/unknown labels; clean infs in features.
    - Train a simple sklearn classifier; return metrics incl. y_true, y_pred, auc and the model.
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import GradientBoostingClassifier

    if not isinstance(df, pd.DataFrame):
        raise TypeError("run_classification_pipeline expects a pandas DataFrame")

    # 1) pick label column (case-insensitive)
    lower_map = {c: c.lower() for c in df.columns}
    inv_lower = {v: k for k, v in lower_map.items()}
    preferred = ["pd_label", "target", "label", "default", "y"]
    target_col = None
    for name in preferred:
        if name in inv_lower:
            target_col = inv_lower[name]
            break
    if target_col is None:
        target_col = df.columns[-1]

    y_raw = df[target_col]

    # 2) coerce label to {0,1}
    def _coerce_binary_label(s: pd.Series) -> pd.Series:
        # numeric -> 0/1
        if pd.api.types.is_bool_dtype(s):
            return s.astype(int)
        # try numeric-like strings first
        s_str = s.astype(str).str.strip().str.lower()
        # map common tokens
        pos_tokens = {"1", "y", "yes", "true", "t", "default", "bad", "overdue", "逾期", "是", "违约", "坏"}
        neg_tokens = {"0", "n", "no", "false", "f", "non-default", "good", "正常", "否", "未违约", "好"}

        mapped = pd.Series(np.nan, index=s.index, dtype="float64")
        # numeric strings
        is_numlike = s_str.str.match(r"^[+-]?(\d+(\.\d+)?|\.?\d+)$", na=False)
        mapped.loc[is_numlike] = pd.to_numeric(s_str[is_numlike], errors="coerce")
        # tokens
        mapped.loc[s_str.isin(pos_tokens)] = 1.0
        mapped.loc[s_str.isin(neg_tokens)] = 0.0

        # if original is numeric dtype, prefer it (handles 0/1 already)
        if pd.api.types.is_numeric_dtype(s):
            mapped_num = pd.to_numeric(s, errors="coerce")
            mapped = mapped.fillna(mapped_num)

        # clamp to 0/1 if values are near-binary
        mapped = mapped.replace([np.inf, -np.inf], np.nan)
        # round exact 0/1 values
        mapped.loc[mapped == 0] = 0
        mapped.loc[mapped == 1] = 1

        # anything not 0/1 remains NaN and will be dropped
        return mapped.astype("float64")

    y_mapped = _coerce_binary_label(y_raw)

    # 3) assemble features and clean
    X = df.drop(columns=[target_col]).copy()

    # try normalising object columns that actually hold numbers
    obj_like = X.select_dtypes(include=["object", "string"]).columns.tolist()
    if obj_like:
        # remove common non-numeric symbols before coercion
        cleanup = (
            X[obj_like]
            .apply(lambda col: col.astype(str).str.replace(r"[,\s%￥¥]", "", regex=True))
        )
        coerced = cleanup.apply(pd.to_numeric, errors="coerce")
        # keep coerced column when most values converted successfully
        for col in obj_like:
            valid_ratio = coerced[col].notna().mean()
            if valid_ratio >= 0.5:
                X[col] = coerced[col]

    # only numeric features to keep it simple/robust
    X = X.select_dtypes(include=["number", "bool"]).copy()
    if X.empty:
        raise ValueError("No numeric features found for classification.")

    # replace infs, then impute medians
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    # 4) drop rows with NaN label; align X/y
    valid = y_mapped.isin([0.0, 1.0])
    X = X.loc[valid].copy()
    y = y_mapped.loc[valid].astype(int)

    if X.shape[0] < 4 or y.nunique() < 2:
        raise ValueError(
            f"Not enough valid labeled rows or only one class present after cleaning: n={X.shape[0]}, classes={y.unique().tolist()}"
        )

    # 5) split (use stratify only if each class has at least 2 samples)
    class_counts = y.value_counts()
    use_stratify = class_counts.min() >= 2
    strat = y if use_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=strat
    )

    # 6) model
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 7) predictions + metrics
    try:
        y_pred = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_score = model.decision_function(X_test)
        y_pred = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-12)

    try:
        auc = roc_auc_score(y_test, y_pred)
    except Exception:
        auc = float("nan")

    metrics = {
        "auc": float(auc) if auc == auc else 0.0,  # NaN-safe
        "y_true": y_test.to_numpy(),
        "y_pred": y_pred,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "target_col": target_col,
        "class_counts": class_counts.to_dict()
    }
    return metrics, model
