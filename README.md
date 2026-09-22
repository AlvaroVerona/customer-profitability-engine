# Customer Profitability & Action Optimization Engine

A customer-level economic decision engine for a **synthetic neobank**,
combining profitability accounting, Funds Transfer Pricing, credit risk,
econometrics, machine learning, Customer Lifetime Value, simulation, and
mathematical optimization to determine where customer value comes from and
how a bank can allocate limited resources to maximize incremental economic
profit.

Full specification: [`PROJECT_SPEC.md`](PROJECT_SPEC.md).

> **Status: Phase 0 (project setup) and Phase 1 (synthetic data generation)
> are complete.** Everything from Phase 2 onward (data quality, profitability
> engine, econometrics, ML, CLV, segmentation, action simulation,
> optimization, dashboard) is not yet implemented. This README will be
> replaced by the full portfolio-quality version in Phase 18, once those
> results actually exist — nothing below is a business finding, only a
> description of what runs today.

## What exists today

- A month-by-month synthetic data simulation for 20,000 customers over 36
  months (`src/customer_profitability/data/generator.py`), producing
  `customers`, `accounts`, `deposits`, `cards`, `loans`, `customer_service`,
  and `customer_monthly` parquet tables under `data/raw/`.
- Deliberately realistic correlation structure (income ↔ age ↔ segment ↔
  balances ↔ credit limits), a documented multi-factor churn hazard, and
  deliberately injected missing values / duplicate rows for the upcoming
  Phase 2 data quality engine. See [`data/README.md`](data/README.md) for
  full generation and churn-mechanism documentation.
- 14 passing unit tests (`tests/test_data_generation.py`) covering
  reproducibility, financial validity, temporal/referential integrity, and
  correlation structure.

## Reproducibility

```bash
make install         # uv sync
make generate-data    # regenerate data/raw/*.parquet (seed 42)
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
`config/`, `data/`, `src/customer_profitability/{data,utils}/`, `tests/`.

## Tech stack

Python, pandas, NumPy, DuckDB, PyArrow, statsmodels, SciPy, scikit-learn,
XGBoost, SHAP, OR-Tools, Plotly, Matplotlib, Streamlit, pytest, uv.
