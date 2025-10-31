# Credit Risk Model Report

- AUC: **0.641**
- Data pipeline: preprocessing -> feature engineering -> model training -> evaluation.
- Visuals: ROC curve saved by agent.

## Key Findings
1. Strong rank ordering (AUC indicates good separability).
2. Most predictive features relate to repayment history and utilization.
3. Calibration to be checked before production deployment.

## Recommendations
1. Monitor PSI monthly and retrain when drift > 0.2.
2. Apply cost-sensitive thresholding for recall@K.
3. Add monotonic constraints on risk-related features if using tree models.
