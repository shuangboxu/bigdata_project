# src/cluster.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from .features import build_features

ART_DIR = Path("artifacts/clustering")
ART_DIR.mkdir(parents=True, exist_ok=True)

def _cluster_profiles(df_num: pd.DataFrame, labels: np.ndarray, out: Path, topn: int = 10):
    prof = df_num.groupby(labels).agg(["mean","median"])
    var_rank = df_num.groupby(labels).mean().var().sort_values(ascending=False).head(topn).index
    prof_small = prof.loc[:, prof.columns.get_level_values(0).isin(var_rank)]
    out.write_text(prof_small.to_markdown(), encoding="utf-8")

def run_clustering(df: pd.DataFrame) -> Dict[str, Any]:
    df = build_features(df)
    num_df = df.select_dtypes(include=[np.number]).copy()

    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("pca", PCA(n_components=8, random_state=42))])
    Z = pipe.fit_transform(num_df)

    best_k, best_score, best_model = None, -1.0, None
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        yk = km.fit_predict(Z)
        score = silhouette_score(Z, yk)
        if score > best_score:
            best_k, best_score, best_model = k, score, km

    # Plot silhouette vs k
    ks, ss = [], []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        yk = km.fit_predict(Z)
        ks.append(k); ss.append(silhouette_score(Z, yk))
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(ks, ss, marker="o"); ax.set_xlabel("k"); ax.set_ylabel("silhouette"); ax.set_title("Silhouette vs k (PCA)")
    (ART_DIR / "kmeans").mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(ART_DIR / "kmeans" / "silhouette.png", dpi=150); plt.close(fig)

    labels = best_model.fit_predict(Z)
    prof_md = ART_DIR / "kmeans" / "cluster_profile.md"
    _cluster_profiles(num_df, labels, prof_md)

    return {"model_dir": str(ART_DIR / "kmeans"), "best_k": int(best_k), "best_score": float(best_score)}
