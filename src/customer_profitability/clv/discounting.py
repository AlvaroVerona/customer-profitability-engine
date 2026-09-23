"""Phase 7.3 -- discounting.

    PV_t = ExpectedProfit_t / (1 + r)^t

`r` here is a *monthly* effective rate derived from the configured annual
rate via `(1 + r_annual)^(1/12) - 1` (compound conversion), not the naive
`r_annual / 12` -- the difference matters once compounded over a 36-month
horizon and this is the financially correct way to convert an annual rate
to a monthly one for a monthly cash-flow stream.
"""

from __future__ import annotations

import numpy as np

from customer_profitability.utils.config import ClvConfig


def monthly_discount_rate(annual_rate: float) -> float:
    return (1 + annual_rate) ** (1 / 12) - 1


def annuity_factor(monthly_rate: float, n_months: np.ndarray | int) -> np.ndarray:
    """Present value of $1 received at the end of each of the next
    `n_months` months: sum_{t=1}^{n} 1/(1+r)^t = (1 - (1+r)^-n) / r.
    Vectorized over an array of `n_months` (e.g. simulated survival
    lengths); `n_months == 0` correctly gives a factor of 0."""
    n = np.asarray(n_months, dtype=float)
    if monthly_rate == 0:
        return n
    return (1 - (1 + monthly_rate) ** (-n)) / monthly_rate


def survival_weighted_annuity_factor(monthly_rate: float, p_churn_monthly: np.ndarray, horizon_months: int) -> np.ndarray:
    """Closed-form value of sum_{t=1}^{T} S(t) / (1+r)^t, where S(t) is the
    probability a customer -- active right now -- is *still* active during
    forecast month t, under a constant monthly hazard p:

        S(t) = (1 - p)^(t-1)   [S(1) = 1: they are active this month by
                                 construction, since they haven't churned yet]

    This lines up exactly with `rng.geometric(p)` in `calculator.monte_carlo_clv`
    (number of trials to first success, support {1, 2, ...}, P(K=k) =
    (1-p)^(k-1) p): a customer earns profit in months 1..min(K, T), and
    P(t <= K) = P(K >= t) = (1-p)^(t-1) = S(t) -- so the deterministic
    formula here and the Monte Carlo simulation converge to the same
    expectation. (Getting this exponent wrong -- (1-p)^t instead of
    (1-p)^(t-1) -- was caught exactly by checking that the two methods
    agree; see test_clv.py.)

    Derivation: sum_{t=1}^{T} (1-p)^(t-1)/(1+r)^t is a geometric series in
    q = (1-p)/(1+r) that simplifies to (1 - q^T) / (r + p).

    The "constant hazard" assumption (survival based on the single latest
    observed monthly churn probability, held flat over the whole horizon,
    rather than re-forecast month by month) is documented in
    `forecasting.py` and in the final report's Limitations section -- a
    full month-by-month re-simulation of each customer's evolving state
    over a 3-year horizon is out of scope for this phase.
    """
    p = np.asarray(p_churn_monthly, dtype=float)
    q = (1 - p) / (1 + monthly_rate)
    T = horizon_months
    denom = monthly_rate + p
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(np.isclose(denom, 0.0), T / (1 + monthly_rate), (1 - q**T) / denom)
    return factor


def clv_config_monthly_rate(clv_config: ClvConfig) -> float:
    return monthly_discount_rate(clv_config.annual_discount_rate)
