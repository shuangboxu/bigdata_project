import os

class Reporter:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self._client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"[Reporter] OpenAI client init failed: {e}. Will use fallback template.")

    def __call__(self, state):
        auc = state.get("metrics", {}).get("auc", 0.0)
        report_md = None

        if self._client:
            try:
                prompt = (
                    "You are a financial risk analyst. Write an English markdown report for a credit "
                    f"risk binary classifier. The model AUC={auc:.3f}. Explain dataset handling, "
                    "feature engineering, evaluation (ROC/PR), threshold choice, and give 3 actionable "
                    "business insights and 3 risk-control recommendations. Keep it concise and structured."
                )
                resp = self._client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                report_md = resp.choices[0].message.content
            except Exception as e:
                print(f"[Reporter] OpenAI call failed: {e}")

        if not report_md:
            report_md = (
                "# Credit Risk Model Report\n\n"
                f"- AUC: **{auc:.3f}**\n"
                "- Data pipeline: preprocessing -> feature engineering -> model training -> evaluation.\n"
                "- Visuals: ROC curve saved by agent.\n\n"
                "## Key Findings\n"
                "1. Strong rank ordering (AUC indicates good separability).\n"
                "2. Most predictive features relate to repayment history and utilization.\n"
                "3. Calibration to be checked before production deployment.\n\n"
                "## Recommendations\n"
                "1. Monitor PSI monthly and retrain when drift > 0.2.\n"
                "2. Apply cost-sensitive thresholding for recall@K.\n"
                "3. Add monotonic constraints on risk-related features if using tree models.\n"
            )

        os.makedirs("reports", exist_ok=True)
        out_path = "reports/agent_llm_report.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"[Reporter] Report saved -> {out_path}")
        return {"report_text": report_md}
