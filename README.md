# LangGraph Credit Risk Workflow

A reproducible end-to-end project for credit risk data analysis with LangGraph agents, classic CLI utilities, automated charts, and LLM-generated reports.

## 1. Environment Setup （环境准备）
1. **Clone & enter the repo**
   ```bash
   git clone <this-repo-url>
   cd bigdata_project
   ```
2. **(Optional) Create a virtualenv**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure environment variables**
   - Copy `.env.example` to `.env` and fill in values (e.g. `OPENAI_API_KEY`).
   - You can also export variables in your shell before running commands.

## 2. Prepare Input Data （准备数据）
- Supported formats: Excel (`.xlsx`, `.xls`) and CSV.
- Place your dataset inside the `data/` folder. The workflow will try to load, in order:
  1. `--data <path>` passed on the command line.
  2. `DATA_FILE` or `WORKFLOW_DATA` environment variables.
  3. Default sample `data/数据示例.xlsx` (already included for quick testing).
- Ensure the dataset contains a binary target column (e.g. `pd_label`, `target`, `label`, `default`, `y`). Non-binary values will be coerced automatically when possible.

## 3. Run the Full Agent Workflow （运行多智能体流程）
1. Execute the pipeline end-to-end:
   ```bash
   python -m agent_flow.run_workflow --data data/sample.xlsx
   # 如果已在 .env 中设置 DATA_FILE，可直接运行：
   # python -m agent_flow.run_workflow
   ```
2. What happens during the run:
   - **Planner**: Generates the high-level execution plan and passes along the raw dataframe.
   - **Executor**: Cleans the data, trains a Gradient Boosting classifier, and collects metrics/predictions.
   - **Visualizer**: Produces the ROC curve and stores it under `artifacts/classification/`.
   - **Reporter**: Uses OpenAI (when `OPENAI_API_KEY` is set) or a fallback template to create a Markdown report.
   - **HTML Summary**: After the graph finishes, `reports/workflow_overview.html` is rendered summarizing plan steps, metrics, ROC image, and the Markdown report content.
3. Key outputs to inspect:
   - `artifacts/classification/roc_curve_agent.png`
   - `reports/agent_llm_report.md`
   - `reports/workflow_overview.html`

## 4. Optional LLM Enhancements （可选的 LLM 报告）
- Set the API key via `.env` or shell: `export OPENAI_API_KEY=sk-...`.
- With the key, the reporter node will call OpenAI for a tailored Markdown report. Without it, a deterministic fallback report is created so the workflow always completes.

## 5. Classic CLI Entrypoints （传统命令行工具）
Besides the LangGraph workflow you can run individual stages through the legacy CLI:
```bash
# Exploratory data analysis
python main.py eda --in data/sample.xlsx

# Supervised learning
python main.py train-cls --in data/sample.xlsx --target grade
python main.py train-reg --in data/sample.xlsx --target interestRate

# Unsupervised clustering
python main.py cluster --in data/sample.xlsx

# Credit risk scoring on tabular data
python main.py risk-score --in data/数据示例.xlsx --out artifacts/risk_scores.csv

# Build the static report bundle
python main.py report
```
Run `python main.py --help` for the complete list of switches (including LLM add-ons under `report-llm`).
The risk scoring command saves a CSV with borrower-level scores and prints the
distribution across the low/medium/high risk buckets.

## 6. 风险分析可视化仪表盘
使用最新的风险评分数据生成一个静态网页仪表盘，帮助决策者快速识别低风险用户。

1. **生成仪表盘 JSON 数据**
   ```bash
   python scripts/generate_risk_dashboard_data.py
   ```
   - 默认会读取 `artifacts/risk_scores.csv` 并在同一目录写入 `risk_scores.json`。
   - 如需指定其他数据源或输出路径，可使用 `--input` 与 `--output` 参数，例如：
     ```bash
     python scripts/generate_risk_dashboard_data.py --input my_scores.csv --output artifacts/custom_scores.json
     ```

2. **打开网页仪表盘**
   - 直接用浏览器打开仓库根目录下的 `index.html`。
   - 页面会自动加载 `artifacts/risk_scores.json`，展示风险概览、用户排名、分布图表，并提供风险等级、评分阈值与用户 ID 搜索筛选。

当原始 CSV 更新后，重新运行脚本即可刷新网页所用的数据文件。

## 7. Repository Layout （项目结构）
```
agent_flow/        LangGraph agent nodes (planner, executor, visualizer, reporter)
artifacts/         Persisted models, metrics, and plots
reports/           Markdown and HTML reports
src/               Data preparation and model code used by the executors
data/              Input Excel/CSV files (ignored from version control)
```

## 8. Troubleshooting （常见问题）
- **Input file not found?** Pass `--data` explicitly or set `DATA_FILE`/`WORKFLOW_DATA` before running.
- **AUC is zero or NaN?** Confirm the target column really is binary and that both classes remain after cleaning.
- **Images missing in HTML?** Open `reports/workflow_overview.html` from the repository root so relative paths resolve correctly.
- **Matplotlib/GUI warnings?** The project saves plots to disk only; no GUI backend is required.

Happy modelling! （祝使用愉快！）
