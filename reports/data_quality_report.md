# Data Quality Report

**Overall quality score: 99.6 / 100**

| Table | Records | Duplicates | Quarantined | Validated | Score |
|---|---:|---:|---:|---:|---:|
| customers | 20060 | 60 | 60 | 20000 | 99.7 |
| accounts | 75367 | 225 | 225 | 75142 | 99.7 |
| deposits | 389328 | 1164 | 1164 | 388164 | 99.6 |
| cards | 379812 | 1136 | 1136 | 378676 | 99.6 |
| loans | 105105 | 314 | 1332 | 103773 | 98.7 |
| customer_service | 118627 | 354 | 354 | 118273 | 99.7 |
| customer_monthly | 389328 | 1164 | 1164 | 388164 | 99.7 |

## customers
- Missingness:
  - `churn_date`: 12692 (63.27%) -- expected
- Invalid values:

## accounts
- Missingness:
  - `account_close_date`: 49278 (65.38%) -- expected
- Referential violations:

## deposits
- Missingness:
  - `average_balance`: 3994 (1.03%) -- suspicious
  - `deposit_rate`: 3869 (0.99%) -- suspicious
- Invalid values:
- Temporal violations:
- Referential violations:

## cards
- Missingness:
  - `transaction_volume`: 3696 (0.97%) -- suspicious
- Invalid values:
- Temporal violations:
- Referential violations:

## loans
- Missingness:
  - `interest_rate`: 1020 (0.97%) -- critical
- Invalid values:
- Temporal violations:
- Referential violations:

## customer_service
- Missingness:
  - `average_handling_time`: 1205 (1.02%) -- expected
- Invalid values:
- Temporal violations:
- Referential violations:

## customer_monthly
- Missingness:
  - `satisfaction_proxy`: 3928 (1.01%) -- expected
- Invalid values:
- Temporal violations:
- Referential violations:
