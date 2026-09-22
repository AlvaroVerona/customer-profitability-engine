"""Phase 3.4 -- Credit features, from the validated `loans` table.

`expected_loss = PD x LGD x EAD` per loan; PD/LGD are aggregated to the
customer-month grain as an EAD-weighted average (a customer with several
loans is riskier in proportion to how much exposure sits behind each PD).
"""

from __future__ import annotations

import pandas as pd

from customer_profitability.utils.metrics import rolling_trailing_stats, safe_divide

ROLLING_WINDOWS = (3, 6, 12)


def build_credit_features(loans: pd.DataFrame, windows: tuple[int, ...] = ROLLING_WINDOWS) -> pd.DataFrame:
    df = loans.copy()
    df["expected_loss"] = df["pd"] * df["lgd"] * df["ead"]
    df["_pd_x_ead"] = df["pd"] * df["ead"]
    df["_lgd_x_ead"] = df["lgd"] * df["ead"]

    agg = df.groupby(["customer_id", "month"], as_index=False).agg(
        outstanding_balance=("outstanding_balance", "sum"),
        credit_limit=("credit_limit", "sum"),
        expected_loss=("expected_loss", "sum"),
        ead=("ead", "sum"),
        _pd_x_ead=("_pd_x_ead", "sum"),
        _lgd_x_ead=("_lgd_x_ead", "sum"),
    )
    agg["utilization"] = safe_divide(agg["outstanding_balance"], agg["credit_limit"], fill=float("nan"))
    agg["estimated_pd"] = safe_divide(agg["_pd_x_ead"], agg["ead"], fill=float("nan"))
    agg["estimated_lgd"] = safe_divide(agg["_lgd_x_ead"], agg["ead"], fill=float("nan"))
    agg = agg.drop(columns=["_pd_x_ead", "_lgd_x_ead"])

    roll = rolling_trailing_stats(agg, ["outstanding_balance", "expected_loss"], windows)
    return agg.merge(roll, on=["customer_id", "month"])
