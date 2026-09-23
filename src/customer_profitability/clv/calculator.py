"""Phase 7.4 / 7.5 -- CLV assembly and uncertainty.

    CLV_i = HistoricalEconomicProfit_i + sum_{t=1}^{T} [P(Survival_t) x ExpectedProfit_t] / (1+r)^t

Per PROJECT_SPEC.md 7.4, historical and future value are kept as separate
columns (`historical_economic_profit`, `future_clv`) as well as summed into
`total_customer_economic_value` -- callers should be explicit about which
one they mean rather than assuming "CLV" always includes the past.

`compute_base_clv` is the exact expectation of the survival-weighted,
discounted flat profit stream under this phase's modeling assumptions
(closed form, no simulation). `monte_carlo_clv` (7.5) instead simulates a
*distribution* of outcomes -- stochastic churn timing plus profit-level
uncertainty -- to produce P5/P50/P95 bands, because "CLV is not an exact
number" is stated explicitly in the spec. The two should agree
approximately at the P50 (see test_clv.py); they will not agree exactly,
because a median differs from a mean under the right-skewed distribution
that stochastic churn timing produces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from customer_profitability.clv.discounting import (
    annuity_factor,
    clv_config_monthly_rate,
    survival_weighted_annuity_factor,
)
from customer_profitability.utils.config import ClvConfig

CLV_INPUT_COLUMNS = ["customer_id", "historical_economic_profit", "expected_profit", "p_churn_monthly"]


def compute_base_clv(df: pd.DataFrame, clv_config: ClvConfig) -> pd.DataFrame:
    monthly_rate = clv_config_monthly_rate(clv_config)
    factor = survival_weighted_annuity_factor(monthly_rate, df["p_churn_monthly"].to_numpy(), clv_config.horizon_months)
    out = df.copy()
    out["future_clv"] = out["expected_profit"] * factor
    out["total_customer_economic_value"] = out["historical_economic_profit"] + out["future_clv"]
    return out


def monte_carlo_clv(
    df: pd.DataFrame,
    clv_config: ClvConfig,
    profit_sigma: pd.Series,
    n_simulations: int = 1000,
    seed: int = 42,
    batch_size: int = 2000,
) -> pd.DataFrame:
    """Vectorized Monte Carlo, batched over customers to bound memory.

    Two sources of uncertainty per simulation draw:
      - stochastic churn timing: K ~ Geometric(p_churn_monthly), months of
        activity = min(K, horizon) (see discounting.py docstring for why
        this lines up with the closed-form survival formula)
      - profit-level uncertainty: a mean-1 log-normal multiplier on the
        flat expected-profit estimate, with each customer's own sigma from
        `forecasting.estimate_profit_volatility` -- a customer whose
        historical profit has been volatile gets a wider band than one
        whose has been stable.

    `n_simulations` defaults far below `config.simulation.n_simulations`
    (10,000) on purpose: that setting is sized for Phase 12's portfolio-
    level scenario Monte Carlo, which runs a handful of scenarios rather
    than one simulation per customer; running 10,000 x 20,000 here would
    cost much more compute for a per-customer uncertainty band that
    doesn't need that much precision. This is a documented scope choice,
    not an oversight.
    """
    rng = np.random.default_rng(seed)
    monthly_rate = clv_config_monthly_rate(clv_config)
    horizon = clv_config.horizon_months

    # Explicit rename_axis + reset_index rather than relying on the caller's
    # Series already having an index named "customer_id" (fragile -- e.g.
    # estimate_profit_volatility's groupby output happens to have it, but
    # nothing should be assumed of an arbitrary caller-supplied Series).
    sigma_df = profit_sigma.rename("profit_sigma").rename_axis("customer_id").reset_index()
    df = df.merge(sigma_df, on="customer_id", how="left")
    df["profit_sigma"] = df["profit_sigma"].fillna(profit_sigma.mean())

    customer_ids = df["customer_id"].to_numpy()
    p_churn = df["p_churn_monthly"].to_numpy()
    expected_profit = df["expected_profit"].to_numpy()
    historical = df["historical_economic_profit"].to_numpy()
    sigma = df["profit_sigma"].to_numpy()
    n = len(df)

    chunks = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b = end - start
        p_safe = np.clip(p_churn[start:end], 1e-6, 1 - 1e-9)[:, None]
        k = rng.geometric(np.broadcast_to(p_safe, (b, n_simulations)))
        months_active = np.minimum(k, horizon)
        factor = annuity_factor(monthly_rate, months_active)

        mu = -0.5 * sigma[start:end][:, None] ** 2
        noise = rng.normal(0, 1, size=(b, n_simulations))
        multiplier = np.exp(mu + sigma[start:end][:, None] * noise)

        simulated_future = expected_profit[start:end][:, None] * multiplier * factor
        simulated_total = historical[start:end][:, None] + simulated_future

        chunks.append(
            pd.DataFrame(
                {
                    "customer_id": customer_ids[start:end],
                    "clv_p5": np.percentile(simulated_total, 5, axis=1),
                    "clv_p50": np.percentile(simulated_total, 50, axis=1),
                    "clv_p95": np.percentile(simulated_total, 95, axis=1),
                }
            )
        )
    return pd.concat(chunks, ignore_index=True)
