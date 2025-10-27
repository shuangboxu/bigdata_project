# 大数据小项目（Excel 风险数据）

本仓库包含一个完整、可复现的数据流程：**EDA → 特征工程 → 回归 / 分类 / 聚类 → 评估 → Markdown 报告生成 →（可选）LLM 洞见追加**。

> Excel 数据文件请放在 `data/` 目录（默认：`data/数据示例.xlsx`）。

---

## 运行环境
- Python ≥ 3.9（已在 3.13 上测试）
- 依赖请见 `requirements.txt`（含 `langgraph`, `openai`, `scikit-learn`, `matplotlib`, `pandas` 等）

安装依赖
```bash
pip install -r requirements.txt
```

---

## 快速开始（CLI）
```bash
# 1) （可选）创建虚拟环境并安装依赖
pip install -r requirements.txt

# 2) 运行 EDA（生成基础图表与数据概览到 artifacts/）
python main.py eda --in data/数据示例.xlsx

# 3) 回归：预测利率 interestRate
python main.py train-reg --in data/数据示例.xlsx --target interestRate

# 4) 分类：预测等级 grade
python main.py train-cls --in data/数据示例.xlsx --target grade

# 5) 聚类：发现客户/贷款分群
python main.py cluster --in data/数据示例.xlsx

# 6) 生成 Markdown 报告（输出至 reports/report.md）
python main.py report

# 7) （可选/推荐）追加 LLM 洞见（使用 ChatGPT API，在报告末尾写入“## LLM Insights”）
#    先确保已安装 openai 并设置 OPENAI_API_KEY（见下方“ChatGPT API 配置”）
python main.py report-llm --artifacts-dir artifacts --report-md reports/report.md --model gpt-4o-mini --language en
# 或中文洞见
python main.py report-llm --language zh
```

---

## ChatGPT API 配置（用于 LLM 洞见）
1) 安装额外依赖：
```bash
pip install openai>=1.30.0
```

2) 设置 API Key：
- Windows（PowerShell）
  ```powershell
  setx OPENAI_API_KEY "<你的API Key>"
  ```
  重新打开终端后可通过 `echo $env:OPENAI_API_KEY` 检查。

- Linux/macOS（bash）
  ```bash
  export OPENAI_API_KEY="<你的API Key>"
  ```

注意：必须是形如 `sk-...` 的 Secret Key；`sess-...` 的浏览器 Session Key 无法用于 SDK。

---

## 多智能体工作流（LangGraph）
除了传统 CLI，本项目提供 LangGraph 多节点有向图（DAG）执行流：

```
Planner → Executor → Visualizer → Reporter
```

- Planner：规划执行步骤（清洗/建模/可视化/报告）
- Executor：训练并评估分类模型，产出 `metrics` 与 `model`
- Visualizer：生成 ROC 曲线等图表 → `artifacts/...`
- Reporter：调用 LLM 生成英文报告（未配置 API Key 时使用回退模板）→ `reports/agent_llm_report.md`

运行
```bash
python -m agent_flow.run_workflow
```

示例输出
- `artifacts/classification/roc_curve_agent.png`
- `reports/agent_llm_report.md`（未配置 API Key 时照常生成回退版）

说明：Agent 执行器会自动寻找二分类标签列（优先 `pd_label/target/label/default/y`，否则取最后一列），并对标签做稳健映射（支持 0/1、Yes/No、逾期/正常、是/否 等）。

---

## 目录结构（关键路径）
```
bigdata_project/
├── agent_flow/                 # LangGraph 多节点工作流
│   ├── planner.py
│   ├── executor.py
│   ├── visualizer.py
│   ├── reporter.py
│   └── run_workflow.py
├── src/                        # 传统管线核心逻辑
│   ├── eda.py  features.py  preprocess.py  ...
│   └── pipeline_cls.py  pipeline_reg.py
├── artifacts/                  # 模型、图表、指标
├── reports/                    # Markdown 报告
├── data/                       # Excel/CSV 数据
├── main.py                     # 传统 CLI 入口
└── requirements.txt
```

---

## 说明与最佳实践
- 时间切分优先使用 `issueDate`，以减少信息泄漏。
- 为避免泄漏：
  - 回归任务（预测 `interestRate`）默认剔除 `grade/subGrade`。
  - 仅使用“申请时可见”的特征训练模型。
- 所有模型、编码器、图表与指标均保存在 `artifacts/`；最终 Markdown 报告在 `reports/`。
- 报告内图片使用相对路径便于跨平台浏览（如 `artifacts/.../roc.png`）。

---

## 常见问题
- Q: 没有配置 OpenAI Key，会报错吗？  
  A: 不会。Reporter 会使用回退模板，依然生成 `reports/agent_llm_report.md`。  
- Q: 标签列不是 0/1 怎么办？  
  A: Agent 会自动映射常见字符串（Yes/No、逾期/正常、是/否等）到 0/1，并丢弃无法判断的样本。  
- Q: 还能继续用原来的 `main.py` 吗？  
  A: 可以。LangGraph 只是“包了一层智能调度”，原 CLI 完全保留可用。

---


---

# Big Data Mini-Project (Excel Risk Data)

This repository provides a reproducible pipeline for **EDA → Feature Engineering → Regression / Classification / Clustering → Evaluation → Markdown Report generation → (optional) LLM insights**.

> Place your Excel file under `data/` (default: `data/数据示例.xlsx`).

---

## Environment
- Python ≥ 3.9 (tested on 3.13)
- See `requirements.txt` for dependencies (`langgraph`, `openai`, `scikit-learn`, `matplotlib`, `pandas`, etc.)

Install
```bash
pip install -r requirements.txt
```

---

## Quickstart (CLI)
```bash
# 1) (Optional) Create a virtual environment and install dependencies
pip install -r requirements.txt

# 2) Run EDA (figures and basic profile saved under artifacts/)
python main.py eda --in data/数据示例.xlsx

# 3) Regression: predict interestRate
python main.py train-reg --in data/数据示例.xlsx --target interestRate

# 4) Classification: predict grade
python main.py train-cls --in data/数据示例.xlsx --target grade

# 5) Clustering: discover loan/customer segments
python main.py cluster --in data/数据示例.xlsx

# 6) Compile the Markdown report (outputs to reports/report.md)
python main.py report

# 7) (Optional/Recommended) Append LLM insights (ChatGPT API) to the end of the report
#    Make sure openai is installed and OPENAI_API_KEY is set (see “ChatGPT API Setup” below)
python main.py report-llm --artifacts-dir artifacts --report-md reports/report.md --model gpt-4o-mini --language en
# Or generate Chinese insights
python main.py report-llm --language zh
```

---

## ChatGPT API Setup (for LLM insights)
1) Install the extra dependency:
```bash
pip install openai>=1.30.0
```

2) Set your API key:
- Windows (PowerShell)
  ```powershell
  setx OPENAI_API_KEY "<YOUR_API_KEY>"
  ```
  Reopen the terminal and verify with `echo $env:OPENAI_API_KEY`.

- Linux/macOS (bash)
  ```bash
  export OPENAI_API_KEY="<YOUR_API_KEY>"
  ```

Note: You must use a secret key of the form `sk-...`. Browser session keys (`sess-...`) do not work with the SDK.

---

## Multi-Agent Workflow (LangGraph)
Beyond the classic CLI, a LangGraph multi-node DAG workflow is available:

```
Planner → Executor → Visualizer → Reporter
```

- Planner: plans steps (cleaning/modeling/visualization/report)
- Executor: trains and evaluates the classifier, returns `metrics` and `model`
- Visualizer: generates ROC and other plots → `artifacts/...`
- Reporter: calls an LLM to write an English report (fallback template when API key is missing) → `reports/agent_llm_report.md`

Run
```bash
python -m agent_flow.run_workflow
```

Sample outputs
- `artifacts/classification/roc_curve_agent.png`
- `reports/agent_llm_report.md` (fallback version is written if no API key)

Note: The executor auto-detects a binary target (prefers `pd_label/target/label/default/y`, else last column) and robustly maps strings to 0/1 (e.g., Yes/No, 逾期/正常, 是/否).

---

## Project Layout (Key paths)
```
bigdata_project/
├── agent_flow/                 # LangGraph multi-agent workflow
│   ├── planner.py
│   ├── executor.py
│   ├── visualizer.py
│   ├── reporter.py
│   └── run_workflow.py
├── src/                        # Classic pipeline core
│   ├── eda.py  features.py  preprocess.py  ...
│   └── pipeline_cls.py  pipeline_reg.py
├── artifacts/                  # Models, plots, metrics
├── reports/                    # Markdown reports
├── data/                       # Excel/CSV data
├── main.py                     # Classic CLI entry
└── requirements.txt
```

---

## Notes & Best Practices
- Prefer `issueDate` for time-based splits to reduce leakage.
- To avoid leakage:
  - For regression on `interestRate`, exclude `grade/subGrade` by default.
  - Use only features available at application time.
- All artifacts (models, encoders, plots, metrics) are under `artifacts/`; the final Markdown report under `reports/`.
- Prefer relative paths for images in reports (e.g., `artifacts/.../roc.png`).

---

## FAQ
- Q: Will it fail without an OpenAI key?  
  A: No. The Reporter uses a fallback template and still writes `reports/agent_llm_report.md`.  
- Q: What if my label is not 0/1?  
  A: The agent maps common strings (Yes/No, 逾期/正常, 是/否) to 0/1 and drops undecidable rows.  
- Q: Can I still use the original `main.py`?  
  A: Yes. LangGraph simply wraps an intelligent scheduler; the classic CLI remains available.
