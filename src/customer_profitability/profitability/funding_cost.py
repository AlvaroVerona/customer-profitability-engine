"""Phase 4.2 -- Deposit funding contribution (FTP).

See `ftp.py` for the sign convention. Rates in the data are annualized
decimals (e.g. 0.035 = 3.5%/year), so the monthly contribution divides the
annualized spread by 12.
"""

from __future__ import annotations

import pandas as pd

from customer_profitability.profitability.ftp import annual_to_monthly, get_ftp_rate_annual
from customer_profitability.utils.config import FtpConfig
from customer_profitability.utils.metrics import nansum


def calculate_deposit_contribution(deposits: pd.DataFrame, ftp_config: FtpConfig) -> pd.DataFrame:
    df = deposits.groupby(["customer_id", "month"], as_index=False).agg(
        average_balance=("average_balance", nansum),
        deposit_rate=("deposit_rate", "mean"),
    )
    ftp_rate_annual = get_ftp_rate_annual(ftp_config)
    monthly_spread = annual_to_monthly(ftp_rate_annual - df["deposit_rate"])
    df["deposit_contribution"] = df["average_balance"] * monthly_spread
    return df[["customer_id", "month", "deposit_contribution"]]
