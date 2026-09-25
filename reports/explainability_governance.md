# Explainability & Governance

PROJECT_SPEC.md Phase 14: how every stage's output can be interpreted, and a worked customer-level decision explanation for a real customer from the current pipeline run -- not a hypothetical example, so this section cannot silently drift out of sync with the code.

## How economic profit is calculated (Profitability, Phase 4)

Every customer-month's economic profit follows one waterfall (`profitability/engine.py`), traceable back to the raw transactional data at each step:

```
gross_contribution         = total_revenue + deposit_contribution - operating_cost
risk_adjusted_contribution = gross_contribution - expected_loss
economic_profit             = risk_adjusted_contribution - acquisition_cost
```

- `total_revenue` = interest revenue (`AvgLoanBalance x LoanRate`) + interchange revenue (`CardVolume x InterchangeRate`) + two documented synthetic fee streams (low-balance fee, Premium subscription) -- see `profitability/revenue.py`.
- `deposit_contribution` = `AvgDepositBalance x (FTPRate - CustomerDepositRate) / 12` -- positive when the bank's internal funding value exceeds what it pays the customer (`profitability/ftp.py`'s sign convention).
- `expected_loss` = `PD x LGD x EAD`, summed over every loan the customer holds that month.
- `operating_cost` = support cost (`contacts x cost_per_contact`) + card payment-processing cost + a flat per-account servicing cost.
- `acquisition_cost` is recognized exactly once, in the customer's acquisition month, and only if that month falls inside the observed data window (a back-book customer's CAC was already spent in the past, so it is not re-charged).

Every one of these formulas has a dedicated unit test with a hand-calculated example (`tests/test_profitability.py`, `tests/test_ftp.py`, `tests/test_risk_cost.py`).

## How future value is forecast (CLV, Phase 7)

`CLV_i = HistoricalEconomicProfit_i + sum_t [P(Survival_t) x ExpectedProfit_t] / (1+r)^t` (`clv/calculator.py`), kept as three separate columns -- `historical_economic_profit`, `future_clv`, `total_customer_economic_value` -- so a reader is never left guessing which one a plain "CLV" figure means.
- `P(Survival_t)` comes from Phase 6's *calibrated* churn model, evaluated once on the customer's latest snapshot and held as a **constant monthly hazard** over the horizon -- a documented simplification, not a month-by-month re-forecast of the customer's evolving state.
- `ExpectedProfit` is the customer's own trailing 3-month run-rate across the Phase 4 waterfall, held **flat** over the horizon -- consistent with the flat-hazard assumption above.
- Phase 7.5's Monte Carlo band (`clv_p5`/`clv_p50`/`clv_p95`) exists specifically so CLV is never presented as one exact number: it simulates stochastic churn timing (geometric survival draw) and profit-level uncertainty (log-normal multiplier, sized by each customer's own historical profit volatility).

## Why a model produces a particular prediction (ML, Phase 6)

Top SHAP drivers for the random_forest churn model (mean |SHAP|, portfolio-wide):

- `tenure_months`: 0.0768
- `login_frequency`: 0.0417
- `average_balance`: 0.0343
- `income`: 0.0204
- `transaction_volume`: 0.0162

Every classification model's raw probabilities are additionally **calibrated** (Platt scaling, `models/evaluation.calibrate_probabilities`) before being used downstream, because the class-rebalancing needed for usable ranking on a rare event (~2%/month churn) distorts `predict_proba` away from true probabilities -- a raw-vs-calibrated Brier score comparison is in `reports/ml_report.md`.
Per-customer SHAP explanations (not just the global importance above) are available via `models.evaluation.explain_customer`, applied identically for any of the eight Phase 6 targets' XGBoost/random-forest champions.

## Why a particular customer/action combination was selected (Optimization, Phase 11)

Solver status on the current run: **OPTIMAL**, 5,000 of 44,406 candidates selected.

Two worked examples (`optimization/explanation.py`, a rank-based approximation -- see its module docstring for why this is not a literal trace of the CP-SAT solver's internal reasoning):

- `CUST_011797`: Selected: CREDIT_PRODUCT ranks #1 of 44,406 candidates by incremental profit (EUR 2,019.41) and fit within budget/capacity/risk headroom.
- `CUST_009962`: Not selected: this customer's best candidate (PREMIUM_SUBSCRIPTION, EUR 54.97) ranks #17985 of 44,406 candidates by incremental profit. The optimizer selected 5,000 candidates that, jointly, fit the budget/capacity/risk constraints better -- not evidence this candidate's value was negative, just that the constrained resources were allocated elsewhere first.

## Customer-Level Decision Explanations (worked examples)

### A high-value customer

```
Customer CUST_011797

Recommended Action:
Credit Product

Why:
- High expected CLV (top 23% of active customers, EUR 2,095)
- Customer does not currently hold a consumer credit line
- Positive incremental value (+EUR 2,019.41 over doing nothing)

Expected impact:
- Cost: EUR 15.00
- Incremental profit: EUR 2,019.41
- CLV impact: EUR 2,023.91
```

### A typical (unremarkable) customer

```
Customer CUST_009962

Recommended Action:
Premium Subscription

Why:
- Elevated churn probability (top 18% churn risk, 3.2%/month)
- Customer does not currently hold the Premium tier
- Positive incremental value (+EUR 54.97 over doing nothing)

Expected impact:
- Cost: EUR 8.00
- Incremental profit: EUR 54.97
- CLV impact: EUR 56.97
```

### A customer for whom no action is recommended

```
Customer CUST_003302

Recommended Action:
No Action

Why:
- No eligible action provides positive incremental value over doing nothing for this customer.

Expected impact:
- Cost: EUR 0.00
- Incremental profit: EUR 0.00
- CLV impact: EUR 0.00
```

## Responsible decision-making (PROJECT_SPEC.md section 26)

- The entire dataset is synthetic (`data/README.md`); no real customer, transaction, or credit data is used anywhere in this project.
- Every profitability, CLV, and action-value figure is model-based and carries the assumptions documented in this report and in each phase's own module docstrings -- none of it is a verified real-world business outcome.
- ML predictions carry quantified uncertainty (calibration, Monte Carlo bands); Phase 6/7 explicitly avoid presenting a point estimate as if it were certain.
- The optimization engine's output is a decision-support ranking under Phase 10's simulated effects, not a causal guarantee -- `optimization/solver.py`'s own docstring says so explicitly, and repeats it in every generated report.
- No sensitive personal characteristic (e.g. protected demographic attributes) is used as an eligibility or treatment criterion anywhere in `actions/definitions.py`; eligibility rules are behavioral/financial (product ownership, income, employment status for credit risk).
- Any real-world deployment of a system like this would need additional regulatory, fair-lending/fair-treatment, legal, and model-risk-management controls well beyond what this portfolio project implements.

