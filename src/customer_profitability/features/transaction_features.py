"""Phase 3.3 -- Transaction features (card activity), from the validated
`cards` table."""

from __future__ import annotations

import pandas as pd

from customer_profitability.utils.metrics import nansum, rolling_trailing_stats, safe_divide

ROLLING_WINDOWS = (3, 6, 12)


def build_transaction_features(cards: pd.DataFrame, windows: tuple[int, ...] = ROLLING_WINDOWS) -> pd.DataFrame:
    df = (
        cards.groupby(["customer_id", "month"], as_index=False)
        .agg(
            transaction_count=("transaction_count", "sum"),
            transaction_volume=("transaction_volume", nansum),
            atm_withdrawal_volume=("atm_withdrawal_volume", "sum"),
            international_transaction_volume=("international_transaction_volume", "sum"),
        )
    )
    df["average_transaction_value"] = safe_divide(
        df["transaction_volume"], df["transaction_count"], fill=float("nan")
    )
    df["atm_usage_share"] = safe_divide(df["atm_withdrawal_volume"], df["transaction_volume"], fill=float("nan"))
    df["international_share"] = safe_divide(
        df["international_transaction_volume"], df["transaction_volume"], fill=float("nan")
    )

    roll = rolling_trailing_stats(df, ["transaction_count", "transaction_volume"], windows)
    return df.merge(roll, on=["customer_id", "month"])
