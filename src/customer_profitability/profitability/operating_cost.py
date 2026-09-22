"""Phase 4.4 -- Operating costs.

Three components:

- `support_cost`: from `customer_service.estimated_service_cost`
  (= Contacts x CostPerContact, computed at generation time).
- `payment_processing_cost`: from `cards.payment_processing_cost`.
- `account_servicing_cost`: a flat per-active-account monthly cost
  (statements, infra, compliance overhead) from config -- charged to every
  customer-month in the panel, since it applies regardless of activity.
"""

from __future__ import annotations

import pandas as pd

from customer_profitability.utils.config import OperatingConfig
from customer_profitability.utils.metrics import nansum


def calculate_operating_cost(
    customer_service: pd.DataFrame,
    cards: pd.DataFrame,
    active_customer_months: pd.DataFrame,
    operating_config: OperatingConfig,
) -> pd.DataFrame:
    support = customer_service.groupby(["customer_id", "month"], as_index=False).agg(
        support_cost=("estimated_service_cost", nansum)
    )
    processing = cards.groupby(["customer_id", "month"], as_index=False).agg(
        payment_processing_cost=("payment_processing_cost", nansum)
    )

    df = active_customer_months[["customer_id", "month"]].merge(
        support, on=["customer_id", "month"], how="left"
    ).merge(processing, on=["customer_id", "month"], how="left")
    df[["support_cost", "payment_processing_cost"]] = df[["support_cost", "payment_processing_cost"]].fillna(0.0)
    df["account_servicing_cost"] = operating_config.account_servicing_cost_monthly
    df["operating_cost"] = df["support_cost"] + df["payment_processing_cost"] + df["account_servicing_cost"]
    return df
