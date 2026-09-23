"""Phase 6.1 -- churn model: predict P(Churn_{i,t+1}=1) from features
observed as of month t.

Unlike Phase 5's explanatory churn model (same-month regressors, recovering
the generating mechanism), this is a *forecaster*: the target for row t is
`churned_this_month` read from row t+1 of the same customer (a forward
shift within each customer's time-sorted series). A customer's last
observed row -- whether because they churned that month or the observation
window simply ends -- has no t+1 to read a label from and is dropped, since
the future outcome is genuinely unknown, not zero.
"""

from __future__ import annotations

import pandas as pd

from customer_profitability.models.evaluation import fit_and_evaluate_classification


def build_churn_dataset(customer_360: pd.DataFrame) -> pd.DataFrame:
    df = customer_360.sort_values(["customer_id", "month"]).copy()
    df["target"] = df.groupby("customer_id")["churned_this_month"].shift(-1)
    return df.dropna(subset=["target"]).reset_index(drop=True)


def fit_and_evaluate(df: pd.DataFrame, train_mask: pd.Series, val_mask: pd.Series, test_mask: pd.Series) -> dict:
    return fit_and_evaluate_classification(df, train_mask, val_mask, test_mask)
