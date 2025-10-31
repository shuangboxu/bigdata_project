# Data Analysis Report

> Auto-generated report (Markdown).

## 1. Regression (Predicting `interestRate`)

**Validation**:
- MAE: 0.7048
- RMSE: 1.1176
- R2: 0.9510

**Test**:
- MAE: 1.1076
- RMSE: 1.7789
- R2: 0.8966

**Figures**:
![](artifacts/regression/hgb_reg/pred_vs_true.png)
![](artifacts/regression/hgb_reg/residuals.png)

---

## 2. Classification (Predicting `grade`)

- accuracy: 0.4487
- f1_macro: 0.2959
- f1_weighted: 0.4225
- auc_ovr: 0.8080

**Figure**:
![](artifacts/classification/hgb_cls/confusion_matrix.png)

---

## 3. Clustering

**Cluster Profiles (excerpt)**:

|    |   ('id', 'mean') |   ('id', 'median') |   ('loanAmnt', 'mean') |   ('loanAmnt', 'median') |   ('installment', 'mean') |   ('installment', 'median') |   ('employmentTitle', 'mean') |   ('employmentTitle', 'median') |   ('annualIncome', 'mean') |   ('annualIncome', 'median') |   ('revolBal', 'mean') |   ('revolBal', 'median') |   ('totalAcc', 'mean') |   ('totalAcc', 'median') |   ('title', 'mean') |   ('title', 'median') |   ('n8', 'mean') |   ('n8', 'median') |   ('fico_over_dti', 'mean')...

![](artifacts/clustering/kmeans/silhouette.png)

