"""Funds Transfer Pricing (FTP) rate mechanism.

Sign convention (documented here because `funding_cost.py` depends on it):

    DepositContribution = AverageDepositBalance x (FTPRate - CustomerDepositRate)

`FTPRate` is the internal rate the bank's treasury credits the deposit-
gathering business for the funding value of a deposit; `CustomerDepositRate`
is what is actually paid out to the customer. A *positive* DepositContribution
means the bank earns positive net interest margin on that deposit -- the
normal case, since `FTPRate` (set close to the market reference rate) is
almost always higher than what a retail customer is paid. A *negative*
DepositContribution would mean the bank is paying the customer more than the
funding is internally worth, which only happens for a small number of
highly rate-sensitive customers on generous, above-market pricing.

The FTP rate here is a single flat annual rate from config (a simplification
documented in the final report's Limitations section -- a production FTP
framework would use a full maturity curve).
"""

from __future__ import annotations

from customer_profitability.utils.config import FtpConfig


def get_ftp_rate_annual(ftp_config: FtpConfig) -> float:
    return ftp_config.base_ftp_rate_annual


def annual_to_monthly(annual_rate: float) -> float:
    return annual_rate / 12
