"""Phase 4.1 -- Revenue.

Computed directly from the validated raw tables (not from the generator's
own precomputed columns) so the formulas are visible, testable, and
traceable here, independent of how the synthetic data happened to be built.

Fee revenue is deliberately limited to two clearly-defined synthetic
streams (a low-balance account fee and a Premium subscription fee) rather
than fabricating loan/transfer fee schedules with no underlying driver in
the data -- see PROJECT_SPEC.md 4.1 "Include only clearly defined synthetic
revenue streams" and Development Principle #2 ("do not fabricate final
business results"). This is a documented simplification, not an omission.
"""

from __future__ import annotations

import pandas as pd

from customer_profitability.utils.config import RevenueConfig
from customer_profitability.utils.metrics import nansum


def calculate_interest_revenue(loans: pd.DataFrame) -> pd.DataFrame:
    """InterestRevenue = AverageLoanBalance x LoanRate, allocated monthly.

    `loans.interest_income` already stores `outstanding_balance x interest_rate / 12`
    per loan-month; this aggregates it to the customer-month grain.
    """
    return (
        loans.groupby(["customer_id", "month"], as_index=False)
        .agg(interest_revenue=("interest_income", nansum))
    )


def calculate_interchange_revenue(cards: pd.DataFrame, revenue_config: RevenueConfig) -> pd.DataFrame:
    """InterchangeRevenue = CardTransactionVolume x InterchangeRate."""
    df = cards.groupby(["customer_id", "month"], as_index=False).agg(
        transaction_volume=("transaction_volume", nansum)
    )
    df["interchange_revenue"] = df["transaction_volume"] * revenue_config.interchange_rate
    return df.drop(columns=["transaction_volume"])


def calculate_fee_revenue(
    deposits: pd.DataFrame, customers: pd.DataFrame, revenue_config: RevenueConfig
) -> pd.DataFrame:
    """Two synthetic fee streams:

    - `account_fee`: a low-balance maintenance fee, charged when a
      customer's average current-account balance falls below
      `low_balance_threshold` (a standard retail-banking practice).
    - `premium_subscription_fee`: a flat monthly fee for customers in the
      Premium segment (representing a paid account tier).
    """
    balances = deposits.groupby(["customer_id", "month"], as_index=False).agg(
        average_balance=("average_balance", nansum)
    )
    df = balances.merge(customers[["customer_id", "customer_segment"]], on="customer_id", how="left")

    below_threshold = df["average_balance"] < revenue_config.low_balance_threshold
    df["account_fee"] = (below_threshold.fillna(False) * revenue_config.low_balance_fee).astype(float)
    is_premium = df["customer_segment"] == "Premium"
    df["premium_subscription_fee"] = (is_premium * revenue_config.premium_monthly_fee).astype(float)
    df["fee_revenue"] = df["account_fee"] + df["premium_subscription_fee"]

    return df[["customer_id", "month", "account_fee", "premium_subscription_fee", "fee_revenue"]]


def calculate_total_revenue(
    interest_revenue: pd.DataFrame, interchange_revenue: pd.DataFrame, fee_revenue: pd.DataFrame
) -> pd.DataFrame:
    df = interest_revenue.merge(interchange_revenue, on=["customer_id", "month"], how="outer").merge(
        fee_revenue, on=["customer_id", "month"], how="outer"
    )
    amount_cols = ["interest_revenue", "interchange_revenue", "account_fee", "premium_subscription_fee", "fee_revenue"]
    df[amount_cols] = df[amount_cols].fillna(0.0)
    df["total_revenue"] = df["interest_revenue"] + df["interchange_revenue"] + df["fee_revenue"]
    return df
