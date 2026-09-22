"""Phase 3 -- Customer 360 feature engineering.

Builds deposit-level features and the top-level `build_customer_360`
orchestrator that assembles demographic, deposit, transaction, credit,
engagement, and behavioral features into a single customer-month panel.

Reads from `data/processed/*_validated.parquet` (Phase 2 output), not raw
data -- Customer 360 sits downstream of the Data Quality Engine per the
architecture in PROJECT_SPEC.md section 5. `customer_monthly_validated` is
the anchor panel: it already has the correct customer-month grain (no rows
before acquisition or after churn), so every other feature table is left-
joined onto it and missing product usage is filled as "no exposure" (0),
not imputed as an unknown value.

All rolling/growth features are trailing (see
`utils.metrics.rolling_trailing_stats`), so a row's features only use that
row's own month and earlier ones -- required for Phase 6 models to predict
month t+1 from features observed as of month t without leakage.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from customer_profitability.features.behavioral_features import build_behavioral_features
from customer_profitability.features.credit_features import build_credit_features
from customer_profitability.features.transaction_features import build_transaction_features
from customer_profitability.utils.config import Settings, load_settings
from customer_profitability.utils.logging import get_logger
from customer_profitability.utils.metrics import nansum, rolling_trailing_stats

logger = get_logger(__name__)

ROLLING_WINDOWS = (3, 6, 12)

# Columns that mean "no exposure to this product" when a customer has no
# deposit/card/loan rows for a month, and so are safe to fill with 0 rather
# than leave as NaN. Ratio columns (utilization, growth rates, ...) are
# deliberately left as NaN -- 0% utilization on a nonexistent credit line is
# a different (wrong) statement than "not applicable".
_ZERO_FILL_COLS = [
    "transaction_count",
    "transaction_volume",
    "atm_withdrawal_volume",
    "international_transaction_volume",
    "outstanding_balance",
    "credit_limit",
    "expected_loss",
    "ead",
] + [f"{c}_avg_{w}m" for c in ("transaction_count", "transaction_volume", "outstanding_balance", "expected_loss") for w in ROLLING_WINDOWS]


def build_deposit_features(deposits: pd.DataFrame, windows: tuple[int, ...] = ROLLING_WINDOWS) -> pd.DataFrame:
    df = deposits.groupby(["customer_id", "month"], as_index=False).agg(
        average_balance=("average_balance", nansum),
        inflows=("inflows", nansum),
        outflows=("outflows", nansum),
        deposit_rate=("deposit_rate", "mean"),
        market_rate=("market_rate", "mean"),
    )
    df["rate_differential"] = df["deposit_rate"] - df["market_rate"]

    means = rolling_trailing_stats(df, ["average_balance", "inflows", "outflows"], windows, stats=("mean",))
    vol = rolling_trailing_stats(df, ["average_balance"], windows, stats=("std",))
    vol = vol.rename(columns={f"average_balance_std_{w}m": f"balance_volatility_{w}m" for w in windows})

    return df.merge(means, on=["customer_id", "month"]).merge(vol, on=["customer_id", "month"])


def build_engagement_features(customer_monthly: pd.DataFrame) -> pd.DataFrame:
    """Login frequency, product count, transaction frequency, and support
    interactions are already produced at the customer-month grain by the
    generator; this just selects and renames them to the Customer 360
    vocabulary (`transaction_count` *is* transaction frequency here, and
    `support_contacts` *is* support interactions)."""
    return customer_monthly[
        ["customer_id", "month", "login_frequency", "product_count", "support_contacts", "tenure_months"]
    ].rename(columns={"support_contacts": "support_interactions"})


def build_customer_360(tables: dict[str, pd.DataFrame], windows: tuple[int, ...] = ROLLING_WINDOWS) -> pd.DataFrame:
    customer_monthly = tables["customer_monthly"]
    customers = tables["customers"][["customer_id", "age", "income", "customer_segment"]].drop_duplicates("customer_id")

    base = customer_monthly[["customer_id", "month", "churned_this_month"]]
    engagement = build_engagement_features(customer_monthly)
    deposit_feats = build_deposit_features(tables["deposits"], windows)
    txn_feats = build_transaction_features(tables["cards"], windows)
    credit_feats = build_credit_features(tables["loans"], windows)
    behavioral_feats = build_behavioral_features(tables["deposits"], tables["cards"])

    panel = (
        base.merge(customers, on="customer_id", how="left")
        .merge(engagement, on=["customer_id", "month"], how="left")
        .merge(deposit_feats, on=["customer_id", "month"], how="left")
        .merge(txn_feats, on=["customer_id", "month"], how="left")
        .merge(credit_feats, on=["customer_id", "month"], how="left")
        .merge(behavioral_feats, on=["customer_id", "month"], how="left")
    )

    fill_cols = [c for c in _ZERO_FILL_COLS if c in panel.columns]
    panel[fill_cols] = panel[fill_cols].fillna(0.0)
    return panel


def _read_validated(processed_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in ["customers", "deposits", "cards", "loans", "customer_monthly"]:
        tables[name] = pd.read_parquet(processed_dir / f"{name}_validated.parquet")
    return tables


def main() -> None:
    settings: Settings = load_settings()
    tables = _read_validated(settings.processed_dir)
    panel = build_customer_360(tables)

    settings.features_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.features_dir / "customer_360.parquet"
    panel.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows, %d columns)", out_path, len(panel), panel.shape[1])


if __name__ == "__main__":
    main()
