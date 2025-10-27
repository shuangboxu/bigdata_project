# llm_insights.py
from __future__ import annotations
import os, json, pathlib, datetime, re
from typing import Dict, Any, List, Tuple, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

DEFAULT_SYSTEM_PROMPT = """You are a senior data scientist and analytics writer.
Given raw metrics and figure file paths from a student project,
produce a crisp, business-facing narrative with bullet points and short paragraphs.
Always include: (1) Problem framing; (2) Data quality; (3) Key results with numbers;
(4) Interpretation/limitations; (5) Actionable recommendations; (6) Next steps.
Keep it compact but insightful.
"""

USER_TEMPLATE = """PROJECT CONTEXT
- Artifacts directory: {artifacts_dir}
- Collected metrics keys: {metric_keys}
- Figure files (as references): {fig_refs}

RAW METRICS (JSON)
{metrics_pretty}

WRITING REQUIREMENTS
- Output in **Markdown**, section title: "## LLM Insights"
- Reference figures by filename only when helpful, e.g., `artifacts/.../roc_curve.png`.
- If multiple tasks exist (regression / classification / clustering), organize by task.
- If information is missing, say "not provided".
"""

def _read_json_safely(p: str) -> Optional[Dict[str, Any]]:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _collect_metrics(artifacts_dir: str) -> Tuple[Dict[str, Any], List[str]]:
    metrics: Dict[str, Any] = {}
    ads = pathlib.Path(artifacts_dir)
    if not ads.exists():
        return metrics, []
    for pat in ["**/*metrics*.json", "**/*_report.json", "**/metrics.json"]:
        for p in ads.glob(pat):
            obj = _read_json_safely(str(p))
            if obj:
                metrics[p.stem] = obj
    figs: List[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.svg"):
        figs.extend([str(p) for p in ads.glob(f"**/{ext}")])
    return metrics, figs

def _format_metrics_for_prompt(metrics: Dict[str, Any]) -> str:
    try:
        return json.dumps(metrics, indent=2, ensure_ascii=False) if metrics else "{}"
    except Exception:
        return str(metrics)

def generate_llm_insights(artifacts_dir: str, model: str = "gpt-4o-mini", api_key: Optional[str] = None, language: str = "en", max_tokens: int = 900) -> str:
    metrics, figs = _collect_metrics(artifacts_dir)
    metrics_pretty = _format_metrics_for_prompt(metrics)
    fig_refs = [os.path.relpath(f, start=artifacts_dir) if os.path.isabs(f) else f for f in figs]

    sys_prompt = DEFAULT_SYSTEM_PROMPT + ("\nWrite in Chinese." if language.lower().startswith("zh") else "")
    user_prompt = USER_TEMPLATE.format(
        artifacts_dir=artifacts_dir,
        metric_keys=list(metrics.keys()),
        fig_refs=fig_refs[:20],
        metrics_pretty=metrics_pretty
    )

    if OpenAI is None or not (api_key or os.getenv("OPENAI_API_KEY")):
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"""## LLM Insights

> Placeholder generated at {stamp}.
OpenAI SDK or API key not available in this environment.
Please set OPENAI_API_KEY and re-run.
"""

    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content.strip()
    if not content.lstrip().lower().startswith("##"):
        content = "## LLM Insights\n\n" + content
    return content

def append_to_report(report_md_path: str, section_md: str) -> str:
    path = pathlib.Path(report_md_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# Data Analysis Report\n\n> Auto-generated report (Markdown).\n\n", encoding="utf-8")
    txt = path.read_text(encoding="utf-8")
    new = re.sub(r"\n## LLM Insights[\s\S]*$", "", txt, flags=re.IGNORECASE)
    new = new.rstrip() + "\n\n" + section_md + "\n"
    path.write_text(new, encoding="utf-8")
    return str(path)
