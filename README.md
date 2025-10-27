# LangGraph Credit Risk Workflow

A reproducible end-to-end project for credit risk data analysis with LangGraph agents, classic CLI utilities, automated charts, and LLM-generated reports.

## Setup
- Python >= 3.9 (validated on 3.13)
- Install dependencies: `pip install -r requirements.txt`
- Copy `.env.example` to `.env` and adjust values as needed

## Quickstart
1. Place your Excel/CSV file under `data/` (for example `data/sample.xlsx`).
2. Run the multi-agent workflow:
   ```bash
   python -m agent_flow.run_workflow --data data/sample.xlsx
   ```
3. Inspect the generated artifacts:
   - `artifacts/classification/roc_curve_agent.png` - ROC curve from the visualizer
   - `reports/agent_llm_report.md` - Markdown report (fallback template when no API key)
   - `reports/workflow_overview.html` - HTML overview of the pipeline, metrics, and visuals

The workflow executes the following stages automatically:
- Planner: produce an execution plan
- Executor: train and evaluate a binary classifier
- Visualizer: export evaluation graphics
- Reporter: write or fallback-generate the narrative report

## Optional LLM Report
Set the API key in `.env` or your shell before running the workflow:
```
OPENAI_API_KEY=sk-...
```
Without a key the reporter writes a concise template report so the run still completes.

## Classic CLI Utilities
You can continue to use the original CLI entry points for individual tasks:
```bash
python main.py eda --in data/sample.xlsx
python main.py train-cls --in data/sample.xlsx --target grade
python main.py report
```
Run `python main.py --help` to see all commands.

## Repository Layout
```
agent_flow/        LangGraph agent nodes (planner, executor, visualizer, reporter)
artifacts/         Persisted models, metrics, and plots
reports/           Markdown and HTML reports
src/               Data preparation and model code used by the executors
data/              Input Excel/CSV files (ignored from version control)
```

## FAQ
- **Input file not found?** Specify it with `DATA_FILE` in `.env` or pass `--data` to the workflow command.
- **AUC is zero or NaN?** Check that the label really is binary and that both classes are present after cleaning.
- **Images missing in the HTML page?** Open `reports/workflow_overview.html` from the project root so the relative paths resolve.

---

```bash
pip install -r requirements.txt
python -m agent_flow.run_workflow --data data/sample.xlsx
```

Artifacts live in `artifacts/`; Markdown and HTML outputs live in `reports/`.

