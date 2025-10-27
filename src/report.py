# src/report.py
from __future__ import annotations
from pathlib import Path
import re, json

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACTS = {
    "regression": "artifacts/regression/regression_metrics.json",
    "classification": "artifacts/classification/classification_metrics.json",
    "clustering": "artifacts/clustering/kmeans/silhouette.png",
}

def _to_rel(md_text: str) -> str:
    """把 Windows 绝对路径改为相对路径，方便跨机展示。"""
    return re.sub(
        r'!\]\((?:[A-Za-z]:\\\\|[A-Za-z]:/)[^)]*(artifacts[/\\\\][^)]+)\)',
        lambda m: f'!]({Path(m.group(1)).as_posix()})',
        md_text,
    )

def _load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def compile_report() -> str:
    """汇总回归 / 分类 / 聚类的结果，写入 report.md"""
    md_path = REPORT_DIR / "report.md"

    sections = ["# Data Analysis Report\n\n> Auto-generated report (Markdown).\n\n"]

    # 回归
    reg = _load_json(ARTIFACTS["regression"])
    if reg:
        val, test = reg.get("val", {}), reg.get("test", {})
        sections.append("## 1. Regression (Predicting `interestRate`)\n\n")
        sections.append("**Validation**:\n")
        for k, v in val.items():
            sections.append(f"- {k}: {v:.4f}\n")
        sections.append("\n**Test**:\n")
        for k, v in test.items():
            sections.append(f"- {k}: {v:.4f}\n")
        sections.append("\n**Figures**:\n")
        sections.append("![](artifacts/regression/hgb_reg/pred_vs_true.png)\n")
        sections.append("![](artifacts/regression/hgb_reg/residuals.png)\n\n---\n\n")

    # 分类
    cls = _load_json(ARTIFACTS["classification"])
    if cls:
        m, labels = cls.get("metrics", {}), cls.get("labels", [])
        sections.append("## 2. Classification (Predicting `grade`)\n\n")
        for k, v in m.items():
            sections.append(f"- {k}: {v:.4f}\n")
        sections.append("\n**Figure**:\n")
        sections.append("![](artifacts/classification/hgb_cls/confusion_matrix.png)\n\n---\n\n")

    # 聚类
    clu_fig = ARTIFACTS["clustering"]
    if Path(clu_fig).exists():
        sections.append("## 3. Clustering\n\n")
        try:
            with open("artifacts/clustering/kmeans/cluster_profile.md", encoding="utf-8") as f:
                profile = f.read()
            sections.append("**Cluster Profiles (excerpt)**:\n\n")
            sections.append(profile[:500] + "...\n\n")
        except Exception:
            pass
        sections.append("![](artifacts/clustering/kmeans/silhouette.png)\n\n")

    # 写入文件
    txt = "".join(sections)
    txt = _to_rel(txt)
    md_path.write_text(txt, encoding="utf-8")
    return str(md_path)
