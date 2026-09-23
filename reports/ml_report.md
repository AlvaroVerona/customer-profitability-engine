# Machine Learning Model Report

All targets use a time-based train/validation/test split (see `models.evaluation.time_based_split`) and predict month t+1 from features observed as of month t -- no target uses same-month information about itself.

## 6.1 Churn

### churn (champion: random_forest)

| model | roc_auc | pr_auc | precision_at_0.5 | recall_at_0.5 | brier_score | base_rate | n |
|---|---|---|---|---|---|---|---|
| logistic_regression | 0.7456 | 0.06602 | 0.04507 | 0.7087 | 0.2083 | 0.02221 | 7.312e+04 |
| random_forest | 0.7467 | 0.06845 | 0.04661 | 0.6644 | 0.2009 | 0.02221 | 7.312e+04 |
| xgboost | 0.7412 | 0.06759 | 0.04973 | 0.6201 | 0.1766 | 0.02221 | 7.312e+04 |

**Calibration (random_forest, test set):** raw Brier score = 0.2009, post-hoc Platt-scaled Brier score = 0.0212. Class rebalancing (needed for usable ranking on a rare event) distorts predict_proba away from true probabilities; the calibrated version, not the raw model, is what should feed Phase 7's CLV survival term. Deciles of predicted probability vs. actual event rate, calibrated model:

 predicted_mean  actual_rate    n
         0.0035       0.0031 7313
         0.0050       0.0055 7312
         0.0070       0.0082 7312
         0.0091       0.0089 7312
         0.0115       0.0127 7312
         0.0146       0.0167 7312
         0.0190       0.0234 7312
         0.0257       0.0254 7312
         0.0360       0.0401 7312
         0.0686       0.0781 7312

## 6.2 Future revenue

### revenue (champion: xgboost)

| model | mae | rmse | mape | r_squared | n |
|---|---|---|---|---|---|
| linear_regression | 8.502 | 22.37 | 0.749 | 0.8645 | 7.312e+04 |
| random_forest | 7.599 | 21.36 | 0.4992 | 0.8765 | 7.312e+04 |
| xgboost | 7.482 | 21.15 | 0.5209 | 0.8789 | 7.312e+04 |

## 6.3 Future balances / activity

### deposit_balance (champion: random_forest)

| model | mae | rmse | mape | r_squared | n |
|---|---|---|---|---|---|
| linear_regression | 817.7 | 1.011e+04 | 0.7862 | -0.7376 | 7.24e+04 |
| random_forest | 626.8 | 1052 | 0.1564 | 0.9812 | 7.24e+04 |
| xgboost | 637.1 | 1113 | 0.2498 | 0.9789 | 7.24e+04 |

Note: linear regression's negative R^2 on deposit balance is a real finding, not a bug -- monetary balances are heavy-tailed, and a handful of high-balance customers dominate squared error for a model that can't flex to outliers the way a tree ensemble can.

### loan_balance (champion: linear_regression)

| model | mae | rmse | mape | r_squared | n |
|---|---|---|---|---|---|
| linear_regression | 1072 | 2121 | 0.1628 | 0.9582 | 1.968e+04 |
| random_forest | 1084 | 2127 | 0.172 | 0.9579 | 1.968e+04 |
| xgboost | 1149 | 2235 | 0.1804 | 0.9535 | 1.968e+04 |

### transaction_volume (champion: random_forest)

| model | mae | rmse | mape | r_squared | n |
|---|---|---|---|---|---|
| linear_regression | 296 | 435.4 | 0.5121 | 0.6863 | 7.312e+04 |
| random_forest | 271.8 | 416 | 0.4016 | 0.7136 | 7.312e+04 |
| xgboost | 273.3 | 418.4 | 0.4007 | 0.7102 | 7.312e+04 |

## 6.4 Product adoption

### adoption_savings_account (champion: random_forest)

| model | roc_auc | pr_auc | precision_at_0.5 | recall_at_0.5 | brier_score | base_rate | n |
|---|---|---|---|---|---|---|---|
| logistic_regression | 0.8272 | 0.05863 | 0.04477 | 0.8559 | 0.2035 | 0.01735 | 3.36e+04 |
| random_forest | 0.8287 | 0.05836 | 0.04178 | 0.8988 | 0.1848 | 0.01735 | 3.36e+04 |
| xgboost | 0.8185 | 0.055 | 0.04644 | 0.7324 | 0.1501 | 0.01735 | 3.36e+04 |

### adoption_consumer_loan (champion: logistic_regression)

| model | roc_auc | pr_auc | precision_at_0.5 | recall_at_0.5 | brier_score | base_rate | n |
|---|---|---|---|---|---|---|---|
| logistic_regression | 0.884 | 0.05162 | 0.02413 | 0.8557 | 0.1708 | 0.007418 | 5.325e+04 |
| random_forest | 0.8706 | 0.04333 | 0.025 | 0.8405 | 0.1494 | 0.007418 | 5.325e+04 |
| xgboost | 0.8597 | 0.04167 | 0.02931 | 0.7063 | 0.1094 | 0.007418 | 5.325e+04 |

### adoption_investment_account (champion: logistic_regression)

| model | roc_auc | pr_auc | precision_at_0.5 | recall_at_0.5 | brier_score | base_rate | n |
|---|---|---|---|---|---|---|---|
| logistic_regression | 0.8694 | 0.0549 | 0.03536 | 0.8439 | 0.1759 | 0.01093 | 4.75e+04 |
| random_forest | 0.8588 | 0.05218 | 0.0356 | 0.8285 | 0.1565 | 0.01093 | 4.75e+04 |
| xgboost | 0.853 | 0.05322 | 0.03989 | 0.6994 | 0.1186 | 0.01093 | 4.75e+04 |

## 6.5 Explainability (SHAP)

### Churn -- global feature importance (mean |SHAP|, top 10)

- `tenure_months`: 0.0768
- `login_frequency`: 0.0417
- `average_balance`: 0.0343
- `income`: 0.0204
- `transaction_volume`: 0.0162
- `rate_differential`: 0.0134
- `atm_usage_share`: 0.0110
- `transaction_count`: 0.0088
- `average_transaction_value`: 0.0057
- `balance_growth_1m`: 0.0055

### Revenue -- global feature importance (mean |SHAP|, top 10)

- `outstanding_balance`: 32.1738
- `expected_loss`: 6.8231
- `income`: 3.6239
- `customer_segment_Premium`: 2.9326
- `utilization`: 1.7049
- `atm_usage_share`: 1.2633
- `product_count`: 1.0187
- `login_frequency`: 0.6130
- `tenure_months`: 0.5549
- `age`: 0.3848

