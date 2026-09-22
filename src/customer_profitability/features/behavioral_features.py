"""Phase 3 -- Behavioral features: growth rates, recent activity, and
volatility, built from the validated `deposits` and `cards` tables.

Growth rates use `pct_change` within each customer's own time-sorted
series, so they only ever compare a month to *earlier* months of that same
customer -- consistent with the "no leakage" rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from customer_profitability.utils.metrics import nansum, safe_divide

GROWTH_LAGS = (1, 3)


def _clean_growth(s: pd.Series) -> pd.Series:
    """A pct_change from a base of 0 is mathematically +/-inf; that's not a
    usable feature (most ML code chokes on inf), so treat it as undefined,
    same as a pct_change with no prior observation (NaN)."""
    return s.replace([np.inf, -np.inf], np.nan)


def build_behavioral_features(deposits: pd.DataFrame, cards: pd.DataFrame) -> pd.DataFrame:
    dep = (
        deposits.groupby(["customer_id", "month"], as_index=False)
        .agg(average_balance=("average_balance", nansum), inflows=("inflows", nansum))
        .sort_values(["customer_id", "month"])
    )
    dep_grouped = dep.groupby("customer_id", sort=False)
    for lag in GROWTH_LAGS:
        dep[f"balance_growth_{lag}m"] = _clean_growth(dep_grouped["average_balance"].pct_change(lag))
    dep["deposit_growth_1m"] = _clean_growth(dep_grouped["inflows"].pct_change(1))
    dep = dep.drop(columns=["average_balance", "inflows"])

    card = (
        cards.groupby(["customer_id", "month"], as_index=False)
        .agg(transaction_volume=("transaction_volume", nansum))
        .sort_values(["customer_id", "month"])
    )
    card_grouped = card.groupby("customer_id", sort=False)
    for lag in GROWTH_LAGS:
        card[f"spending_growth_{lag}m"] = _clean_growth(card_grouped["transaction_volume"].pct_change(lag))
    roll3_mean = card_grouped["transaction_volume"].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    roll3_std = card_grouped["transaction_volume"].rolling(window=3, min_periods=1).std().reset_index(level=0, drop=True)
    card["activity_volatility_3m"] = roll3_std
    card["recent_activity_ratio"] = safe_divide(card["transaction_volume"], roll3_mean, fill=float("nan"))
    card = card.drop(columns=["transaction_volume"])

    return dep.merge(card, on=["customer_id", "month"], how="outer")
