# Customer Profitability & Action Optimization Engine

A customer-level economic decision engine for a **synthetic neobank**,
combining profitability accounting, Funds Transfer Pricing, credit risk,
econometrics, machine learning, Customer Lifetime Value, simulation, and
mathematical optimization to determine where customer value comes from and
how a bank can allocate limited resources to maximize incremental economic
profit.

Full specification: [`PROJECT_SPEC.md`](PROJECT_SPEC.md).

> **Status: Phase 0 (project setup) through Phase 6 (machine learning) are
> complete.** Everything from Phase 7 onward (CLV, segmentation, action
> simulation, optimization, dashboard) is not yet implemented. This README
> will be replaced by the full portfolio-quality version in Phase 18, once
> those results actually exist — nothing below is a business finding, only
> a description of what runs today.

## What exists today

- A month-by-month synthetic data simulation for 20,000 customers over 36
  months (`src/customer_profitability/data/generator.py`), producing
  `customers`, `accounts`, `deposits`, `cards`, `loans`, `customer_service`,
  and `customer_monthly` parquet tables under `data/raw/`.
- Deliberately realistic correlation structure (income ↔ age ↔ segment ↔
  balances ↔ credit limits), a documented multi-factor churn hazard, and
  deliberately injected missing values / duplicate rows for the Phase 2 data
  quality engine to catch. See [`data/README.md`](data/README.md) for full
  generation and churn-mechanism documentation.
- A Phase 2 Data Quality Engine (`src/customer_profitability/data/quality.py`)
  running structural, missingness (expected/suspicious/critical),
  duplicate, financial-validity, temporal-consistency, and
  referential-integrity checks on every raw table. It never silently drops a
  row: every table is split into `data/processed/<table>_validated.parquet`
  and `data/processed/<table>_quarantined.parquet`, and a scored report is
  written to `reports/data_quality_report.{md,json}`.
- A Phase 3 Customer 360 pipeline (`src/customer_profitability/features/`)
  that builds demographic, deposit, transaction, credit, engagement, and
  behavioral (growth/volatility) features into one customer-month panel at
  `data/features/customer_360.parquet`, from the Phase 2 *validated* tables.
  All rolling/growth features are trailing (never look at a future month),
  and "no exposure to a product" (e.g. no loan) is filled as 0 for amount
  columns but left `NaN` for ratio columns (e.g. utilization) rather than
  conflated with "measured as zero".
- A Phase 4 Historical Customer Profitability Engine
  (`src/customer_profitability/profitability/`) computing, per customer per
  month: interest/interchange/fee revenue, deposit funding contribution via
  a documented FTP mechanism, expected credit loss (PD x LGD x EAD),
  operating cost, and acquisition cost (recognized once, in the
  acquisition month, only for customers acquired inside the observed
  window) -- rolled up into the
  `Revenue -> Gross Contribution -> Risk-Adjusted Contribution -> Economic Profit`
  waterfall from `PROJECT_SPEC.md` section 3, plus per-customer summary
  metrics (profit margin, contribution margin, revenue/cost per month) and
  monthly/quarterly/annual aggregation. Output at
  `data/processed/profitability_{monthly,customer_summary}.parquet`.
  On the full synthetic portfolio: 94.3% of customers are profitable,
  total economic profit ≈ 11.3M, ranked Premium > Affluent > Mass > Student
  as expected from the underlying income/balance correlations.
- A Phase 5 Econometric Analysis (`src/customer_profitability/econometrics/`):
  an OLS revenue-drivers model (`total_revenue ~ average_balance + transaction_count
  + product_count + income`, fit with HC3 heteroskedasticity-robust standard
  errors after `diagnostics.breusch_pagan_test` confirmed heteroskedasticity)
  and a logistic churn-drivers model (`P(churn) ~ rate_gap + activity + tenure
  + profitability`). Both are deliberately *explanatory* (same-month
  regressors, not lagged) -- they exist to recover and interpret the
  synthetic churn/revenue generating mechanism, not to forecast; Phase 6
  builds the point-in-time-correct predictive models. `diagnostics.py`
  provides VIF, residual normality, Breusch-Pagan, and goodness-of-fit
  shared across both models, plus `flag_economic_significance`, which keeps
  "statistically significant" (p < 0.05) and "economically significant"
  (effect size vs. an analyst-set threshold) as separate columns rather than
  conflating them. On the full data: the churn model correctly recovers
  every driver's true sign from the Phase 1 generating mechanism (wider
  rate gap → higher churn odds; more activity/tenure/profitability → lower),
  all VIFs < 5. Output at `reports/econometrics_report.{md,json}`.
- A Phase 6 Machine Learning suite (`src/customer_profitability/models/`):
  churn (`P(Churn_{t+1}=1)`), future revenue, three balance/activity targets
  (deposit balance, loan balance, transaction volume), and three product-
  adoption models (savings, consumer loan, investment) -- every target
  built by forward-shifting the label one month within each customer's
  series (features at t predict the outcome at t+1; a customer's last
  observed row, churn month or not, is dropped since its future is
  genuinely unknown), and every classification/regression target compared
  across a baseline (logistic/linear regression) and two tree ensembles
  (random forest, XGBoost) on a calendar-based train/validation/test split
  (`models.evaluation.time_based_split`, driven by `config.models`).
  Churn probabilities are additionally post-hoc Platt-scaled
  (`calibrate_probabilities`) after the raw class-rebalanced models turned
  out well-ranked (ROC-AUC ~0.75) but badly miscalibrated in absolute terms
  (Brier score 0.20 raw -> 0.02 calibrated) -- exactly the failure mode
  PROJECT_SPEC.md 6.1 flags as consequential for Phase 7's CLV survival
  term. SHAP (`TreeExplainer`) provides global feature importance and a
  per-customer explanation for the churn and revenue champions; on the
  full data, churn's top SHAP driver is `tenure_months`, matching the
  Phase 1 generating mechanism. Output at `reports/ml_report.{md,json}`.
- 69 passing unit tests (`tests/test_data_generation.py`,
  `tests/test_data_quality.py`, `tests/test_features.py`,
  `tests/test_ftp.py`, `tests/test_risk_cost.py`,
  `tests/test_profitability.py`, `tests/test_econometrics.py`,
  `tests/test_models.py`) covering reproducibility, financial validity,
  temporal/referential integrity, correlation structure, every
  quality-engine check against hand-crafted defective rows, the feature
  pipeline's NaN-propagation and no-leakage guarantees, every
  profitability formula against hand-calculated examples, both
  econometric models' ability to recover *known* true coefficients from
  simulated data, and every Phase 6 target's forward-shift/eligibility
  logic, time-based split boundaries, and calibration.

## Reproducibility

```bash
make install         # uv sync
make generate-data    # regenerate data/raw/*.parquet (seed 42)
make quality           # run the data quality engine -> data/processed/, reports/
make features           # build Customer 360 -> data/features/customer_360.parquet
make profitability      # run the profitability engine -> data/processed/profitability_*.parquet
make econometrics       # fit revenue/churn driver models -> reports/econometrics_report.{md,json}
make models             # fit all Phase 6 models -> reports/ml_report.{md,json}
make test              # run the test suite
```

Everything is seeded (`seed: 42` in `config/settings.yaml`); rerunning
`make generate-data` reproduces byte-identical output.

## Architecture (target, per `PROJECT_SPEC.md`)

```text
SYNTHETIC RAW DATA → DATA QUALITY → CUSTOMER 360
   → HISTORICAL PROFITABILITY + BEHAVIORAL FEATURES/ML
   → FUTURE PROFITABILITY → CLV → ECONOMIC SEGMENTATION
   → ACTION SIMULATION → OPTIMIZATION ENGINE
   → RECOMMENDED ACTIONS → STREAMLIT APP
```

## Repository structure

See `PROJECT_SPEC.md` §7 for the full target layout. Implemented so far:
`config/`, `data/`, `reports/`, `src/customer_profitability/{data,features,profitability,econometrics,models,utils}/`, `tests/`.

## Tech stack

Python, pandas, NumPy, DuckDB, PyArrow, statsmodels, SciPy, scikit-learn,
XGBoost, SHAP, OR-Tools, Plotly, Matplotlib, Streamlit, pytest, uv.
