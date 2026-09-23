"""Phase 6.2 -- future revenue model: predict ExpectedRevenue_{i,t+1} from
features observed as of month t.

Same forward-shift-within-customer construction as churn.py: the target for
row t is `total_revenue` (Phase 4 output) read from row t+1.
"""

from __future__ import annotations

import pandas as pd

from customer_profitability.models.evaluation import fit_and_evaluate_regression


def build_revenue_dataset(customer_360: pd.DataFrame, profitability_monthly: pd.DataFrame) -> pd.DataFrame:
    df = customer_360.merge(
        profitability_monthly[["customer_id", "month", "total_revenue"]], on=["customer_id", "month"], how="left"
    ).sort_values(["customer_id", "month"])
    df["target"] = df.groupby("customer_id")["total_revenue"].shift(-1)
    return df.dropna(subset=["target"]).reset_index(drop=True)


def fit_and_evaluate(df: pd.DataFrame, train_mask: pd.Series, val_mask: pd.Series, test_mask: pd.Series) -> dict:
    return fit_and_evaluate_regression(df, train_mask, val_mask, test_mask)
