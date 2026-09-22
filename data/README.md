# Data

All data in this project is **synthetic**, generated locally by
`src/customer_profitability/data/generator.py` with global seed `42`
(see `config/settings.yaml`). No proprietary or real banking data is used
anywhere in this repository.

## Layout

- `raw/` — direct output of the generator (Phase 1). Not committed to git
  (see `.gitignore`); regenerate with `make generate-data`.
- `processed/` — output of the Phase 2 data quality engine (`VALIDATED` /
  `QUARANTINED` splits).
- `features/` — output of the Phase 3 Customer 360 feature engineering.

## Tables (`raw/`)

| File | Grain | Description |
|---|---|---|
| `customers.parquet` | 1 row / customer | Demographics, acquisition, latent traits, churn outcome |
| `accounts.parquet` | 1 row / customer-product | Product adoption and open/close dates |
| `deposits.parquet` | 1 row / customer-month | Current account balances, flows, deposit rate |
| `cards.parquet` | 1 row / customer-month (card holders only) | Card transaction activity and interchange revenue |
| `loans.parquet` | 1 row / loan-month | Consumer credit line: balance, utilization, PD/LGD/EAD |
| `customer_service.parquet` | 1 row / customer-month (with contact) | Support contacts and servicing cost |
| `customer_monthly.parquet` | 1 row / customer-month | Consolidated monthly snapshot used as the base panel for later phases |

## Generation approach

Customer attributes are drawn with deliberate cross-correlations rather than
independently (e.g. income follows an age curve modulated by employment
status; `customer_segment` is derived from the income percentile; deposit
balances and credit limits scale with income and segment; PD is driven by a
latent `risk_score` correlated with employment status and income).

Monthly behaviour (balances, transactions, support contacts, satisfaction,
and churn) is produced by a **single month-by-month state simulation**,
vectorized across all customers, rather than sampled independently per
row. This is what gives the panel realistic temporal structure: a
customer's balance, engagement and satisfaction this month depend on their
own state last month, plus the shared macro deposit-rate path.

## Churn-generating mechanism (documented per Phase 1.7 / Phase 5.2 needs)

Every month, each still-active customer has a churn probability from a
logistic hazard:

```
logit = base
      + w_sat      * (0.65 - satisfaction)                     # unhappier -> more likely
      + w_eng      * (0.55 - engagement)                       # less engaged -> more likely
      + w_rategap  * rate_sensitivity * max(market_rate - deposit_rate, 0)
      - w_tenure   * log1p(tenure_months)                      # loyalty effect
      - w_profit   * (profit_rank - 0.5)                       # more profitable -> stickier
      + w_balance  * max(-recent_balance_change_pct, 0)        # withdrawal trend -> more likely
p_churn = clip(sigmoid(logit), 0, 0.25)
```

Coefficients live in `_CHURN_COEF` in `generator.py`. `satisfaction` and
`engagement` are evaluated against typical-customer baselines (0.65 / 0.55)
so an average customer sits near the base hazard, and customers who deviate
from that baseline are pushed up or down from there. This mechanism is used
directly by the generator (there is no separate "ground truth" churn model
to compare against) — it exists so that Phase 5/6 churn models have a
non-trivial, multi-factor, temporally consistent target to recover.

Once a customer churns, no further rows are generated for them in any
monthly table (`deposits`, `cards`, `loans`, `customer_service`,
`customer_monthly`) — this is enforced structurally by the simulation loop,
not filtered after the fact.

## Deliberate data quality issues

Per the Phase 1 acceptance criteria, the generator injects:

- **Missing values** at `missing_rate` (default 1%) into a subset of
  economically meaningful columns per table (e.g. `deposits.average_balance`,
  `loans.interest_rate`).
- **Duplicate rows** at `duplicate_rate` (default 0.3%) by re-appending a
  random sample of existing rows.

These exist specifically so the Phase 2 Data Quality Engine has real issues
to detect — they are not artifacts of a bug.

## Sign conventions

- All balances, volumes, and cost fields are non-negative.
- `deposit_rate` / `market_rate` are annualized decimal rates (e.g. `0.035`
  = 3.5%/year).
- `DepositContribution = balance × (FTP_rate − deposit_rate)`: positive when
  the bank's internal funding value exceeds what it pays the customer.
