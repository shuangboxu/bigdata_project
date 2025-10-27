# src/io_utils.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Union, List

# 可能是日期的字段
DATE_COLS_CANDIDATES = ["issueDate", "earliesCreditLine"]

# 常见的日期格式
DATE_FORMATS: List[str] = [
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%Y/%m/%d",
    "%b-%Y",
    "%b-%y",
    "%Y%m%d",
]

def _parse_dates_series(s: pd.Series) -> pd.Series:
    """稳健日期解析：先排除明显不是日期的，再尝试常见格式，最后兜底。"""
    # 如果全是数字或0，当作非日期列
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(pd.Series([pd.NaT] * len(s), index=s.index))

    # 初始化为空
    out = pd.Series([pd.NaT] * len(s), index=s.index)

    # 先尝试常见格式
    for fmt in DATE_FORMATS:
        try:
            parsed = pd.to_datetime(s, format=fmt, errors="coerce")
            mask = out.isna() & parsed.notna()
            out.loc[mask] = parsed.loc[mask]
        except Exception:
            continue

    # 最后兜底
    mask = out.isna() & s.notna()
    if mask.any():
        out.loc[mask] = pd.to_datetime(s[mask], errors="coerce")

    return out

def read_table(path: Union[str, Path]) -> pd.DataFrame:
    """读取 Excel/CSV，并自动尝试解析候选日期列。"""
    p = Path(path)
    if p.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(p)
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

    for dc in DATE_COLS_CANDIDATES:
        if dc in df.columns:
            df[dc] = _parse_dates_series(df[dc])
    return df
import os
import matplotlib.pyplot as plt

def save_fig(fig, path: str, tight: bool = True):
    """
    Save a matplotlib figure to the specified path.
    Creates directories automatically if they don't exist.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[IO] Figure saved to {path}")
