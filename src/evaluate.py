from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
)
from .io_utils import save_fig

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    # RMSE 用 MSE 的平方根
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}

def classification_metrics(y_true, y_pred, y_prob=None, labels=None) -> Dict[str, Any]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist() if labels is not None else None,
        "labels": labels
    }
    if y_prob is not None and len(np.unique(y_true)) > 2:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
            metrics["auc_ovr"] = float(auc)
        except Exception:
            pass
    return metrics

def plot_pred_vs_true(y_true, y_pred, out_path: str):
    fig, ax = plt.subplots(figsize=(5,5), dpi=140)
    ax.scatter(y_true, y_pred, alpha=0.4)
    minv, maxv = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    ax.plot([minv, maxv], [minv, maxv])
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs True")
    save_fig(fig, out_path)
    plt.close(fig)

def plot_residuals(y_true, y_pred, out_path: str):
    res = y_pred - y_true
    fig, ax = plt.subplots(figsize=(5,4), dpi=140)
    ax.hist(res, bins=30)
    ax.set_title("Residuals")
    save_fig(fig, out_path)
    plt.close(fig)

def plot_confusion_matrix(cm, labels, out_path: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,5), dpi=140)
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    # annotate
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i][j], ha="center", va="center", color="black")
    save_fig(fig, out_path)
    plt.close(fig)
