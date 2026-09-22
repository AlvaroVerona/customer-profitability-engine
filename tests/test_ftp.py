"""Phase 4.2 -- Funds Transfer Pricing tests, with a hand-calculated example."""

from __future__ import annotations

import pandas as pd
import pytest

from customer_profitability.profitability.ftp import annual_to_monthly, get_ftp_rate_annual
from customer_profitability.profitability.funding_cost import calculate_deposit_contribution
from customer_profitability.utils.config import FtpConfig


def test_get_ftp_rate_annual_reads_base_rate() -> None:
    cfg = FtpConfig(base_ftp_rate_annual=0.045, market_rate_annual=0.035)
    assert get_ftp_rate_annual(cfg) == 0.045


def test_annual_to_monthly_divides_by_twelve() -> None:
    assert annual_to_monthly(0.06) == 0.005


def test_deposit_contribution_hand_calculated_example() -> None:
    # balance=1000, FTP=4.5%/yr, customer paid 2%/yr
    # DepositContribution = 1000 x (0.045 - 0.02) / 12 = 2.083333...
    deposits = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "month": pd.to_datetime(["2023-01-01"]),
            "product": ["current_account"],
            "opening_balance": [1000.0],
            "closing_balance": [1000.0],
            "average_balance": [1000.0],
            "inflows": [0.0],
            "outflows": [0.0],
            "deposit_rate": [0.02],
            "market_rate": [0.035],
        }
    )
    cfg = FtpConfig(base_ftp_rate_annual=0.045, market_rate_annual=0.035)
    out = calculate_deposit_contribution(deposits, cfg)
    assert out.loc[0, "deposit_contribution"] == pytest.approx(1000.0 * (0.045 - 0.02) / 12)


def test_deposit_contribution_negative_when_customer_rate_exceeds_ftp() -> None:
    # A customer paid more than the FTP rate produces a negative contribution
    # -- the bank is paying out more than the funding is internally worth.
    deposits = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "month": pd.to_datetime(["2023-01-01"]),
            "product": ["current_account"],
            "opening_balance": [1000.0],
            "closing_balance": [1000.0],
            "average_balance": [1000.0],
            "inflows": [0.0],
            "outflows": [0.0],
            "deposit_rate": [0.06],
            "market_rate": [0.035],
        }
    )
    cfg = FtpConfig(base_ftp_rate_annual=0.045, market_rate_annual=0.035)
    out = calculate_deposit_contribution(deposits, cfg)
    assert out.loc[0, "deposit_contribution"] < 0
