English_Project_Design_Report# LangGraph-based Credit Risk Workflow

## Abstract
This report documents the design, implementation, and evaluation of a LangGraph-driven workflow for credit risk modelling. The workflow orchestrates data ingestion, preprocessing, feature engineering, model training, automated visualization, and report generation. Using a Gradient Boosting classifier on a Lending Club-style portfolio, the system achieved an area under the ROC curve (AUC) of 0.936, demonstrating effective discrimination between default and non-default accounts despite severe class imbalance.

**Keywords:** credit risk modelling; LangGraph agents; gradient boosting; automated analytics; imbalanced classification

## Introduction
Consumer lending platforms must quantify default risk accurately to support underwriting and portfolio management. Traditional notebooks or monolithic scripts make it difficult to ensure repeatable, auditable model development. This project delivers an end-to-end workflow that combines conventional Python data science components with LangGraph agents to produce traceable experimentation, automated documentation, and reusable artifacts. By structuring the pipeline into planner, executor, visualizer, and reporter roles, the solution captures both the decision logic and operational steps required for credit risk analytics.

## Dataset & Data Preprocessing
The workflow processes a portfolio extract containing 10,722 loan records and 44 variables (`data/数据示例.xlsx`). Core features include contractual terms (`loanAmnt`, `term`, `interestRate`, `installment`), borrower capacity indicators (`annualIncome`, `dti`, `revolUtil`), credit history metrics (`ficoRangeLow`/`High`, `openAcc`, `totalAcc`), and categorical descriptors such as `grade`, `subGrade`, `employmentLength`, and `homeOwnership`. The dependent variable corresponds to column `n12` after automatic target detection, representing a binary default flag with only 28 positive cases versus 10,096 negatives.

Data ingestion relies on a resilient loader that first attempts Excel parsing and falls back to CSV when necessary. Subsequent preprocessing occurs inside the classification pipeline: label coercion harmonises textual and numeric encodings into {0,1}; rows with missing or invalid labels are discarded; only numeric predictors are retained to preserve robustness across heterogeneous sources. Infinite values are replaced, and remaining gaps are imputed with feature-wise medians. These safeguards ensure the downstream estimator receives clean matrices even when users supply partially formatted datasets.

## Methodology
The LangGraph orchestration defines four collaborating agents. The `DataPlanner` proposes a deterministic execution plan spanning preprocessing, feature engineering, training, evaluation, visualisation, and narrative reporting. The `DataExecutor` invokes the classification pipeline, which trains a scikit-learn `GradientBoostingClassifier` on a stratified 70/30 split. To counter class imbalance, the training routine enforces valid splits only when both classes retain at least two examples and records class counts for auditability. The `Visualizer` node renders a ROC curve using matplotlib, persisting the artefact for dashboards. Finally, the `Reporter` generates a Markdown summary via the OpenAI API when available, otherwise falling back to a template to guarantee completeness.

Inside the classifier pipeline, numerical predictors are standardised while labels are kept in their native coding to maintain interpretability. Gradient boosting was selected for its robustness on tabular credit data and ability to capture nonlinear interactions without extensive manual feature crafting. The modular architecture makes it trivial to substitute alternative estimators or expand preprocessing (for example, adding categorical encoders) while preserving the orchestration contract.

## Experimental Results & Analysis
Running the workflow on the bundled dataset produced a test AUC of 0.936, confirming strong rank-ordering power. The agent also reports training and test sample sizes (7,086 and 3,038 rows respectively) together with the automatically detected target column and class distribution. Given the pronounced imbalance (default rate ≈0.28%), recall-oriented metrics must be interpreted carefully; however, the ROC curve indicates that acceptable true-positive rates can be achieved at low false-positive levels. Visual inspection of the exported ROC image corroborates the numerical findings.

The minimal feature engineering combined with gradient boosting handles heterogeneous credit attributes effectively. Nevertheless, the scarcity of positive samples raises the risk of variance in tail scenarios. Future work should evaluate cost-sensitive losses or ensemble calibration strategies (e.g., isotonic post-processing) to stabilise threshold selection under shifting economic conditions.

## Conclusion
The project demonstrates that agentic orchestration can streamline conventional credit risk modelling. LangGraph coordinates deterministic preprocessing, robust model training, and automated reporting into a reproducible workflow that achieves competitive discrimination despite limited defaults. Extending the system with additional explainability artefacts, drift monitoring hooks, and human-in-the-loop reviews would further enhance deployment readiness.

## References
1. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research.
2. LangGraph Documentation. (2024). LangChain AI. https://python.langchain.com/docs/langgraph
3. Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics.
