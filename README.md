# Customer Profitability & Action Optimization Engine

A customer-level economic decision engine for a **synthetic neobank**,
combining profitability accounting, Funds Transfer Pricing, credit risk,
econometrics, machine learning, Customer Lifetime Value, simulation, and
mathematical optimization to determine where customer value comes from and
how a bank can allocate limited resources to maximize incremental economic
profit.

Full specification: [`PROJECT_SPEC.md`](PROJECT_SPEC.md).

> **Status: Phase 0 (project setup) through Phase 4 (historical
> profitability engine) are complete.** Everything from Phase 5 onward
> (econometrics, ML, CLV, segmentation, action simulation, optimization,
> dashboard) is not yet implemented. This README will be replaced by the
> full portfolio-quality version in Phase 18, once those results actually
> exist — nothing below is a business finding, only a description of what
> runs today.

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
- 45 passing unit tests (`tests/test_data_generation.py`,
  `tests/test_data_quality.py`, `tests/test_features.py`,
  `tests/test_ftp.py`, `tests/test_risk_cost.py`,
  `tests/test_profitability.py`) covering reproducibility, financial
  validity, temporal/referential integrity, correlation structure, every
  quality-engine check against hand-crafted defective rows, the feature
  pipeline's NaN-propagation and no-leakage guarantees, and every
  profitability formula against hand-calculated examples (interest
  revenue, interchange revenue, FTP/deposit contribution, expected loss,
  operating cost, and the full economic-profit waterfall).

## Reproducibility

```bash
make install         # uv sync
make generate-data    # regenerate data/raw/*.parquet (seed 42)
make quality           # run the data quality engine -> data/processed/, reports/
make features           # build Customer 360 -> data/features/customer_360.parquet
make profitability      # run the profitability engine -> data/processed/profitability_*.parquet
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
`config/`, `data/`, `reports/`, `src/customer_profitability/{data,features,profitability,utils}/`, `tests/`.

## Tech stack

Python, pandas, NumPy, DuckDB, PyArrow, statsmodels, SciPy, scikit-learn,
XGBoost, SHAP, OR-Tools, Plotly, Matplotlib, Streamlit, pytest, uv.
