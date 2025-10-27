# src/eda.py
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

ART_DIR = Path("artifacts/eda")
ART_DIR.mkdir(parents=True, exist_ok=True)

def eda_basic(df: pd.DataFrame) -> Dict[str, Any]:
    info = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "na_counts": df.isna().sum().to_dict(),
    }
    if "issueDate" in df.columns and pd.api.types.is_datetime64_any_dtype(df["issueDate"]):
        ts = df.set_index("issueDate").sort_index()
        fig, ax = plt.subplots(figsize=(8, 3))
        ts.resample("ME").size().plot(ax=ax)
        ax.set_title("Loans per Month")
        ax.set_xlabel("Month"); ax.set_ylabel("Count")
        out = ART_DIR / "loans_per_month.png"
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return info
