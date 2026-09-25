# Model Validation Report

PROJECT_SPEC.md PHASE 15: for every Phase 6 target, compare the sophisticated models (random forest, XGBoost) against a simple linear/logistic-regression baseline on a held-out, *future* time slice (never a random split, since every target here is a forecast of month t+1) -- reusing the metrics already computed by `make models` (`reports/ml_report.json`), not refitting anything.

## Baseline vs. champion (test set)

| target | metric | baseline (logistic/linear) | champion | champion value | improvement vs. baseline |
|---|---|---|---|---|---|
| churn | roc_auc | 0.7456 | random_forest | 0.7467 | +0.1% |
| revenue | rmse | 22.37 | xgboost | 21.15 | +5.4% |
| deposit_balance | rmse | 1.011e+04 | random_forest | 1052 | +89.6% |
| loan_balance | rmse | 2121 | linear_regression (= baseline) | 2121 | +0.0% |
| transaction_volume | rmse | 435.4 | random_forest | 416 | +4.5% |
| adoption_savings_account | roc_auc | 0.8272 | random_forest | 0.8287 | +0.2% |
| adoption_consumer_loan | roc_auc | 0.884 | logistic_regression (= baseline) | 0.884 | +0.0% |
| adoption_investment_account | roc_auc | 0.8694 | logistic_regression (= baseline) | 0.8694 | +0.0% |

The baseline (linear/logistic regression) is itself the champion on 3 of 8 targets. This is a genuine finding about this dataset, not an oversight: `loan_balance`, `adoption_consumer_loan`, and `adoption_investment_account` all have close-to-linear, low-noise generating mechanisms in the synthetic data generator, so a linear model's extra bias (relative to a tree ensemble's flexibility) buys nothing, and its lower variance wins on held-out data. This is exactly why PROJECT_SPEC.md's modeling philosophy (section 25) ranks predictive performance below economic coherence and interpretability -- a simpler, equally accurate model is preferred, not treated as a consolation prize.

## Time-based validation: actual split ranges used per target

Train/validation/test are split by calendar month (never by row), and the split point is computed independently per target dataset because each target drops a different subset of customer-months (e.g. product-adoption datasets only keep rows where the customer doesn't yet own the product) -- see `models.evaluation.time_based_split` and `models.report._split_summary`. Configured split sizes: test = last 6 months, validation = the 3 months before that.

| target | train | validation | test |
|---|---|---|---|
| churn | 2023-01..2025-02 (n=260,592) | 2025-03..2025-05 (n=34,451) | 2025-06..2025-11 (n=73,121) |
| revenue | 2023-01..2025-02 (n=260,592) | 2025-03..2025-05 (n=34,451) | 2025-06..2025-11 (n=73,121) |
| deposit_balance | 2023-01..2025-02 (n=257,955) | 2025-03..2025-05 (n=34,071) | 2025-06..2025-11 (n=72,405) |
| loan_balance | 2023-01..2025-02 (n=68,877) | 2025-03..2025-05 (n=9,254) | 2025-06..2025-11 (n=19,677) |
| transaction_volume | 2023-01..2025-02 (n=260,592) | 2025-03..2025-05 (n=34,451) | 2025-06..2025-11 (n=73,121) |
| adoption_savings_account | 2023-01..2025-02 (n=120,487) | 2025-03..2025-05 (n=15,840) | 2025-06..2025-11 (n=33,600) |
| adoption_consumer_loan | 2023-01..2025-02 (n=191,059) | 2025-03..2025-05 (n=25,095) | 2025-06..2025-11 (n=53,251) |
| adoption_investment_account | 2023-01..2025-02 (n=170,874) | 2025-03..2025-05 (n=22,424) | 2025-06..2025-11 (n=47,497) |

