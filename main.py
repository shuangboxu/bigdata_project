import argparse, os
from pathlib import Path

from src.io_utils import read_table
from src.eda import eda_basic
from src.pipeline_reg import train_regression
from src.pipeline_cls import train_classification
from src.cluster import run_clustering
from src.report import compile_report
from src.risk_assessment import assess_risk, summarize_risk
from src.config import DEFAULT_INPUT

try:
    from llm_insights import generate_llm_insights, append_to_report
    _LLM_OK = True
except Exception:
    _LLM_OK = False


def main():
    parser = argparse.ArgumentParser(description="Big Data Project CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # EDA
    p_eda = sub.add_parser("eda")
    p_eda.add_argument("--in", dest="input_path", default=str(DEFAULT_INPUT))

    # 回归
    p_reg = sub.add_parser("train-reg")
    p_reg.add_argument("--in", dest="input_path", default=str(DEFAULT_INPUT))
    p_reg.add_argument("--target", default="interestRate")

    # 分类
    p_cls = sub.add_parser("train-cls")
    p_cls.add_argument("--in", dest="input_path", default=str(DEFAULT_INPUT))
    p_cls.add_argument("--target", default="grade")

    # 聚类
    p_clu = sub.add_parser("cluster")
    p_clu.add_argument("--in", dest="input_path", default=str(DEFAULT_INPUT))

    # 报告
    sub.add_parser("report")

    # 风险评估
    p_risk = sub.add_parser("risk-score")
    p_risk.add_argument("--in", dest="input_path", default=str(DEFAULT_INPUT))
    p_risk.add_argument("--out", dest="output_path", default="artifacts/risk_scores.csv")

    # LLM 洞见
    p_llm = sub.add_parser("report-llm")
    p_llm.add_argument("--artifacts-dir", type=str, default="artifacts")
    p_llm.add_argument("--report-md", type=str, default="reports/report.md")
    p_llm.add_argument("--model", type=str, default="gpt-4o-mini")
    p_llm.add_argument("--language", type=str, default="en")
    p_llm.add_argument("--max-tokens", type=int, default=900)

    args = parser.parse_args()

    # ========== 普通报告 ==========
    if args.cmd == "report":
        out = compile_report()
        print(f"Report written to: {out}")
        return

    # ========== LLM 洞见 ==========
    if args.cmd == "report-llm":
        if not _LLM_OK:
            print("[WARN] llm_insights not available. Place llm_insights.py next to main.py.")
            return

        # 先生成/刷新基础报告
        base_report = compile_report()
        print(f"[OK] Base report refreshed at: {base_report}")

        # 再追加 AI 洞见
        section = generate_llm_insights(
            artifacts_dir=args.artifacts_dir,
            model=args.model,
            api_key=os.getenv("OPENAI_API_KEY"),
            language=args.language,
            max_tokens=args.max_tokens,
        )
        out_path = append_to_report(args.report_md, section)
        print(f"[OK] LLM insights appended to: {out_path}")
        return

    # ========== 需要数据的任务 ==========
    input_path = Path(getattr(args, "input_path"))
    df = read_table(input_path)

    if args.cmd == "eda":
        info = eda_basic(df)
        print("EDA Summary:", info)
    elif args.cmd == "train-reg":
        res = train_regression(df, target=args.target)
        print("Regression:", res)
    elif args.cmd == "train-cls":
        res = train_classification(df, target=args.target)
        print("Classification:", res)
    elif args.cmd == "cluster":
        res = run_clustering(df)
        print("Clustering:", res)
    elif args.cmd == "risk-score":
        risk_df = assess_risk(df)
        summary = summarize_risk(risk_df)
        out_path = Path(getattr(args, "output_path"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        risk_df.to_csv(out_path, index=False)
        print("Risk distribution:", summary)
        print(f"Risk assessment saved to: {out_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
