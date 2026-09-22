"""Phase 4.3 -- Credit risk cost.

ExpectedLoss = PD x LGD x EAD. `loans.pd` is already a monthly (not annual)
default probability, and `ead` is the loan's month-end exposure, so no
further monthly allocation is needed here -- summing per loan-month per
customer already gives the monthly expected loss.
"""

from __future__ import annotations

import pandas as pd

from customer_profitability.utils.metrics import nansum


def calculate_expected_loss(loans: pd.DataFrame) -> pd.DataFrame:
    df = loans.copy()
    df["expected_loss"] = df["pd"] * df["lgd"] * df["ead"]
    return df.groupby(["customer_id", "month"], as_index=False).agg(expected_loss=("expected_loss", nansum))
