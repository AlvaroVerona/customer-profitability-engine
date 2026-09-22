"""Phase 3 tests for Customer 360 feature engineering.

Not in PROJECT_SPEC.md's explicit test list, but the rolling/growth/NaN
handling logic here is exactly the kind of thing that fails silently
(as `average_balance` briefly did -- a NaN "suspicious missing" value was
getting summed to 0.0 by pandas' default groupby behaviour instead of
staying NaN), so it gets the same "every important calculation has a test"
treatment as the other phases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from customer_profitability.features.behavioral_features import build_behavioral_features
from customer_profitability.features.credit_features import build_credit_features
from customer_profitability.features.customer_features import (
    build_customer_360,
    build_deposit_features,
)
from customer_profitability.features.transaction_features import build_transaction_features
from customer_profitability.utils.metrics import nansum, rolling_trailing_stats


def test_rolling_trailing_stats_never_uses_future_months() -> None:
    df = pd.DataFrame(
        {
            "customer_id": ["C1"] * 4,
            "month": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"]),
            "x": [10.0, 20.0, 30.0, 999999.0],
        }
    )
    truncated = rolling_trailing_stats(df.iloc[:3], ["x"], windows=(3,))
    full = rolling_trailing_stats(df, ["x"], windows=(3,))
    march_from_truncated = truncated.loc[truncated["month"] == "2023-03-01", "x_avg_3m"].iloc[0]
    march_from_full = full.loc[full["month"] == "2023-03-01", "x_avg_3m"].iloc[0]
    assert march_from_truncated == march_from_full == 20.0  # mean(10, 20, 30), unaffected by April


def test_nansum_keeps_all_nan_group_as_nan_not_zero() -> None:
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [np.nan, np.nan, 5.0]})
    out = df.groupby("g")["v"].apply(nansum)
    assert pd.isna(out["a"])
    assert out["b"] == 5.0


def test_deposit_features_rate_differential_sign() -> None:
    deposits = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "month": pd.to_datetime(["2023-01-01"]),
            "product": ["current_account"],
            "opening_balance": [100.0],
            "closing_balance": [100.0],
            "average_balance": [100.0],
            "inflows": [10.0],
            "outflows": [5.0],
            "deposit_rate": [0.02],
            "market_rate": [0.035],
        }
    )
    out = build_deposit_features(deposits)
    # DepositContribution = balance * (FTP - deposit_rate) is positive when the
    # bank pays the customer less than its internal funding value, i.e. when
    # deposit_rate < market/FTP rate -- so rate_differential should be negative here.
    assert out.loc[0, "rate_differential"] < 0
    assert out.loc[0, "rate_differential"] == 0.02 - 0.035


def test_deposit_features_propagates_missing_average_balance() -> None:
    deposits = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "month": pd.to_datetime(["2023-01-01"]),
            "product": ["current_account"],
            "opening_balance": [100.0],
            "closing_balance": [100.0],
            "average_balance": [np.nan],
            "inflows": [10.0],
            "outflows": [5.0],
            "deposit_rate": [0.02],
            "market_rate": [0.035],
        }
    )
    out = build_deposit_features(deposits)
    assert pd.isna(out.loc[0, "average_balance"])
    assert pd.isna(out.loc[0, "average_balance_avg_3m"])  # must not silently become 0


def test_credit_features_expected_loss_formula() -> None:
    loans = pd.DataFrame(
        {
            "loan_id": ["L1"],
            "customer_id": ["C1"],
            "month": pd.to_datetime(["2023-01-01"]),
            "origination_date": pd.to_datetime(["2022-06-01"]),
            "outstanding_balance": [1000.0],
            "credit_limit": [2000.0],
            "utilization": [0.5],
            "interest_rate": [0.12],
            "interest_income": [10.0],
            "pd": [0.04],
            "lgd": [0.4],
            "ead": [1000.0],
        }
    )
    out = build_credit_features(loans)
    assert out.loc[0, "expected_loss"] == 0.04 * 0.4 * 1000.0
    assert out.loc[0, "utilization"] == 0.5


def test_transaction_features_ratios_and_average_value() -> None:
    cards = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "month": pd.to_datetime(["2023-01-01"]),
            "transaction_count": [10],
            "transaction_volume": [500.0],
            "interchange_revenue": [3.5],
            "atm_withdrawal_volume": [50.0],
            "international_transaction_volume": [25.0],
            "payment_processing_cost": [1.0],
        }
    )
    out = build_transaction_features(cards)
    assert out.loc[0, "average_transaction_value"] == 50.0
    assert out.loc[0, "atm_usage_share"] == 0.1
    assert out.loc[0, "international_share"] == 0.05


def test_behavioral_growth_clips_infinite_to_nan() -> None:
    deposits = pd.DataFrame(
        {
            "customer_id": ["C1", "C1"],
            "month": pd.to_datetime(["2023-01-01", "2023-02-01"]),
            "product": ["current_account"] * 2,
            "opening_balance": [0.0, 100.0],
            "closing_balance": [0.0, 100.0],
            "average_balance": [0.0, 100.0],
            "inflows": [0.0, 10.0],
            "outflows": [0.0, 5.0],
            "deposit_rate": [0.02, 0.02],
            "market_rate": [0.035, 0.035],
        }
    )
    cards = pd.DataFrame(
        columns=[
            "customer_id",
            "month",
            "transaction_count",
            "transaction_volume",
            "interchange_revenue",
            "atm_withdrawal_volume",
            "international_transaction_volume",
            "payment_processing_cost",
        ]
    )
    out = build_behavioral_features(deposits, cards)
    growth = out.loc[out["month"] == "2023-02-01", "balance_growth_1m"].iloc[0]
    assert pd.isna(growth)  # 0 -> 100 is undefined growth, not +inf


def test_customer_360_fills_no_exposure_with_zero_not_nan_for_amounts() -> None:
    customers = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "age": [30],
            "income": [40000.0],
            "customer_segment": ["Mass"],
        }
    )
    customer_monthly = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "month": pd.to_datetime(["2023-01-01"]),
            "tenure_months": [0],
            "product_count": [1],
            "support_contacts": [0],
            "login_frequency": [5.0],
            "churned_this_month": [False],
        }
    )
    deposits = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "month": pd.to_datetime(["2023-01-01"]),
            "product": ["current_account"],
            "opening_balance": [100.0],
            "closing_balance": [100.0],
            "average_balance": [100.0],
            "inflows": [10.0],
            "outflows": [5.0],
            "deposit_rate": [0.02],
            "market_rate": [0.035],
        }
    )
    empty_cards = pd.DataFrame(
        columns=[
            "customer_id",
            "month",
            "transaction_count",
            "transaction_volume",
            "interchange_revenue",
            "atm_withdrawal_volume",
            "international_transaction_volume",
            "payment_processing_cost",
        ]
    )
    empty_loans = pd.DataFrame(
        columns=[
            "loan_id",
            "customer_id",
            "month",
            "origination_date",
            "outstanding_balance",
            "credit_limit",
            "utilization",
            "interest_rate",
            "interest_income",
            "pd",
            "lgd",
            "ead",
        ]
    )
    panel = build_customer_360(
        {
            "customers": customers,
            "customer_monthly": customer_monthly,
            "deposits": deposits,
            "cards": empty_cards,
            "loans": empty_loans,
        }
    )
    row = panel.iloc[0]
    assert row["outstanding_balance"] == 0.0
    assert row["expected_loss"] == 0.0
    assert pd.isna(row["utilization"])  # 0% of a nonexistent credit line is not "0"
