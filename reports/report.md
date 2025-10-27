# Data Analysis Report

> Auto-generated report (Markdown).

## 1. Regression (Predicting `interestRate`)

**Validation**:
- MAE: 0.7326
- RMSE: 1.1527
- R2: 0.9472

**Test**:
- MAE: 1.1081
- RMSE: 1.7732
- R2: 0.8961

**Figures**:
![](artifacts/regression/hgb_reg/pred_vs_true.png)
![](artifacts/regression/hgb_reg/residuals.png)

---

## 2. Classification (Predicting `grade`)

- accuracy: 0.5738
- f1_macro: 0.3831
- f1_weighted: 0.5513
- auc_ovr: 0.8475

**Figure**:
![](artifacts/classification/hgb_cls/confusion_matrix.png)

---

## 3. Clustering

**Cluster Profiles (excerpt)**:

|    |   ('id', 'mean') |   ('id', 'median') |   ('loanAmnt', 'mean') |   ('loanAmnt', 'median') |   ('installment', 'mean') |   ('installment', 'median') |   ('employmentTitle', 'mean') |   ('employmentTitle', 'median') |   ('annualIncome', 'mean') |   ('annualIncome', 'median') |   ('revolBal', 'mean') |   ('revolBal', 'median') |   ('totalAcc', 'mean') |   ('totalAcc', 'median') |   ('title', 'mean') |   ('title', 'median') |   ('n8', 'mean') |   ('n8', 'median') |   ('fico_over_dti', 'mean')...

![](artifacts/clustering/kmeans/silhouette.png)

## LLM Insights

### 问题框架
本项目旨在评估分类和回归模型的性能，以便为后续的决策提供数据支持。通过分析贷款数据，我们希望识别出影响贷款批准的关键因素，并优化模型以提高预测准确性。

### 数据质量
所使用的数据集经过初步清洗，包含分类和回归的相关指标。尽管数据质量较高，但仍需注意样本量和特征选择可能影响模型的表现。

### 关键结果

#### 分类模型
- 准确率（Accuracy）：57.38%
- F1宏观值（F1 Macro）：38.31%
- F1加权值（F1 Weighted）：55.13%
- AUC（曲线下面积）：84.75%

#### 回归模型
- 验证集 MAE：0.73
- 验证集 RMSE：1.15
- 验证集 R²：94.72%
- 测试集 MAE：1.11
- 测试集 RMSE：1.77
- 测试集 R²：89.61%

### 解释/局限性
分类模型的准确率和F1值相对较低，表明模型在区分不同类别方面存在一定困难。尽管AUC值较高，显示出模型在区分正负样本方面的潜力，但仍需进一步优化。回归模型表现良好，R²值接近1，表明模型能够解释大部分的变异性，但MAE和RMSE的值显示出一定的预测误差。

### 可行建议
- 对分类模型进行特征工程，尝试不同的特征组合，以提高模型的准确性。
- 考虑使用集成学习方法（如随机森林或XGBoost）来提升分类性能。
- 对回归模型进行超参数调优，以进一步减少预测误差。

### 下一步
- 进行分类模型的特征选择和优化实验。
- 评估不同模型的表现，并选择最佳模型进行部署。
- 进行更深入的数据分析，识别潜在的影响因素，以支持业务决策。
