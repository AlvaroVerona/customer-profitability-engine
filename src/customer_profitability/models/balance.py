"""Phase 6.3 -- future balance/volume models: deposit balance, loan
balance, and transaction volume, each predicted for month t+1 from
features observed as of month t (same forward-shift construction as
churn.py / revenue.py).

Loan balance is restricted to customer-months where the customer already
holds an outstanding loan balance this month: for the ~85% of
customer-months with no loan, "predict 0" is trivial and would dominate
the fit statistics without saying anything about forecasting skill on the
population that actually matters for this target.
"""

from __future__ import annotations

import pandas as pd

from customer_profitability.models.evaluation import fit_and_evaluate_regression

BALANCE_TARGETS: dict[str, dict] = {
    "deposit_balance": {"column": "average_balance", "restrict_to_active": False},
    "loan_balance": {"column": "outstanding_balance", "restrict_to_active": True},
    "transaction_volume": {"column": "transaction_volume", "restrict_to_active": False},
}


def build_balance_dataset(customer_360: pd.DataFrame, target_name: str) -> pd.DataFrame:
    spec = BALANCE_TARGETS[target_name]
    column = spec["column"]
    df = customer_360.sort_values(["customer_id", "month"]).copy()
    df["target"] = df.groupby("customer_id")[column].shift(-1)
    df = df.dropna(subset=["target"])
    if spec["restrict_to_active"]:
        df = df[df[column] > 0]
    return df.reset_index(drop=True)


def fit_and_evaluate(df: pd.DataFrame, train_mask: pd.Series, val_mask: pd.Series, test_mask: pd.Series) -> dict:
    return fit_and_evaluate_regression(df, train_mask, val_mask, test_mask)
