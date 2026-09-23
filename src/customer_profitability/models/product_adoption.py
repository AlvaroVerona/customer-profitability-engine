"""Phase 6.4 -- product adoption models: P(customer adopts product X by
month t+1 | doesn't have it as of month t), for each of the products named
in PROJECT_SPEC.md 6.4 (savings, consumer loan, investment).

Customer 360 doesn't carry per-product ownership flags (Phase 3 only keeps
an aggregate `product_count`), so this module derives them here from the
validated `accounts` table: a customer owns a product in month t if
`account_open_date <= t` and (`account_close_date` is null or `>= t`).
"""

from __future__ import annotations

import pandas as pd

from customer_profitability.models.evaluation import fit_and_evaluate_classification

ADOPTION_PRODUCTS = ["savings_account", "consumer_loan", "investment_account"]


def _ownership_flags(accounts: pd.DataFrame, product: str, df: pd.DataFrame) -> pd.Series:
    """Boolean Series, same length/order as `df`: does this customer own
    `product` as of this row's month?"""
    prod_accounts = accounts.loc[
        accounts["product"] == product, ["customer_id", "account_open_date", "account_close_date"]
    ]
    merged = df[["customer_id", "month"]].merge(prod_accounts, on="customer_id", how="left")
    open_month = merged["account_open_date"].dt.to_period("M").dt.to_timestamp()
    close_month = merged["account_close_date"].dt.to_period("M").dt.to_timestamp()
    owns = merged["account_open_date"].notna() & (merged["month"] >= open_month)
    owns &= close_month.isna() | (merged["month"] <= close_month)
    return owns.fillna(False)


def build_adoption_dataset(customer_360: pd.DataFrame, accounts: pd.DataFrame, product: str) -> pd.DataFrame:
    df = customer_360.sort_values(["customer_id", "month"]).reset_index(drop=True).copy()
    df["has_product"] = _ownership_flags(accounts, product, df).to_numpy()
    df["has_product_next_month"] = df.groupby("customer_id")["has_product"].shift(-1)

    eligible = df[(~df["has_product"]) & df["has_product_next_month"].notna()].copy()
    eligible["target"] = eligible["has_product_next_month"].astype(int)
    return eligible.reset_index(drop=True)


def fit_and_evaluate(df: pd.DataFrame, train_mask: pd.Series, val_mask: pd.Series, test_mask: pd.Series) -> dict:
    return fit_and_evaluate_classification(df, train_mask, val_mask, test_mask)
