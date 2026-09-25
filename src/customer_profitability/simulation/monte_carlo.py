"""Phase 12 -- portfolio-level Monte Carlo simulation.

Two stochastic pieces, simulated independently and summed:

1. **Baseline portfolio value** (every active customer, whether or not
   they receive an action): the same survival-timing + profit-level-noise
   mechanism as Phase 7's `clv.calculator.monte_carlo_clv` (geometric
   churn-timing draw, closed-form annuity, log-normal profit multiplier),
   batched over customers for memory -- but aggregated as a *portfolio
   total per simulation draw* here, rather than a per-customer percentile
   across draws. Already-churned customers contribute their (certain,
   already-realized) historical profit with no simulated future.

2. **Action outcomes** (Phase 11's selected customer/action pairs): each
   selected pair's acceptance is redrawn as an independent Bernoulli trial
   per simulation (`acceptance_probability`), rather than assumed at its
   expected value -- this is what "quantify uncertainty in ... action
   outcomes" (PROJECT_SPEC.md 12) means concretely. Conditional on
   acceptance, the customer's future value is recomputed with the action's
   *full* effect applied (churn hazard reduced by `churn_reduction_pct`,
   profit adjusted by the full additional revenue/risk/operational-cost,
   not a blended fraction of it) and compared against their scenario-
   adjusted baseline future value; conditional on non-acceptance, they
   contribute nothing beyond baseline (delta 0, cost 0). Realized cost is
   the full `incentive_cost` when accepted, 0 otherwise.

   This is deliberately *not* implemented by dividing Phase 10's
   already-blended `incremental_profit`/`expected_action_cost` back out by
   `acceptance_probability`: that was tried first and is wrong, because
   Phase 10 built its "expected profit with action" by blending the
   accept/reject outcomes *inside* a nonlinear formula (the churn-survival
   annuity factor), so `incremental_profit` is not simply
   `acceptance_probability x (conditional value)` -- dividing back out
   silently inflates the conditional value. The two-branch recomputation
   here is the version that is actually consistent with a literal Bernoulli
   accept/reject draw, and is what makes "probability of budget overrun"
   answerable (the budget was set against Phase 10's *expected*, not
   realized, cost).

Both pieces apply the scenario's `churn_multiplier`/`profit_multiplier`
(scenarios.py) before simulating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from customer_profitability.clv.discounting import (
    annuity_factor,
    clv_config_monthly_rate,
    survival_weighted_annuity_factor,
)
from customer_profitability.simulation.scenarios import Scenario
from customer_profitability.utils.config import ClvConfig

DEFAULT_BATCH_SIZE = 2000


def simulate_baseline_portfolio(
    clv_df: pd.DataFrame,
    profit_sigma: pd.Series,
    scenario: Scenario,
    clv_config: ClvConfig,
    n_simulations: int,
    seed: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """`clv_df`: Phase 7's clv table (all customers, active or not), with
    `is_active`, `p_churn_monthly`, `expected_profit`, `historical_economic_profit`.
    Returns an `(n_simulations,)` array: total portfolio value per draw."""
    active = clv_df[clv_df["is_active"]].merge(
        profit_sigma.rename("profit_sigma").rename_axis("customer_id").reset_index(), on="customer_id", how="left"
    )
    active["profit_sigma"] = active["profit_sigma"].fillna(profit_sigma.mean())
    churned_historical_total = float(clv_df.loc[~clv_df["is_active"], "historical_economic_profit"].sum())

    rng = np.random.default_rng(seed)
    monthly_rate = clv_config_monthly_rate(clv_config)
    horizon = clv_config.horizon_months

    p_churn = np.clip(active["p_churn_monthly"].to_numpy() * scenario.churn_multiplier, 1e-6, 0.9)
    expected_profit = active["expected_profit"].to_numpy() * scenario.profit_multiplier
    historical = active["historical_economic_profit"].to_numpy()
    sigma = active["profit_sigma"].to_numpy()
    n = len(active)

    portfolio_totals = np.zeros(n_simulations)
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
        portfolio_totals += simulated_total.sum(axis=0)

    return portfolio_totals + churned_historical_total


def simulate_action_outcomes(
    selected: pd.DataFrame,
    scenario: Scenario,
    clv_config: ClvConfig,
    n_simulations: int,
    seed: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """`selected`: Phase 11's selected (customer, action) rows (NO_ACTION
    excluded), with Phase 10's raw per-customer columns merged back in
    (`p_churn_monthly`, `expected_profit`, `future_clv`, `churn_reduction_pct`,
    `additional_revenue_monthly`, `additional_risk_monthly`,
    `operational_cost_monthly`, `acceptance_probability`, `incentive_cost`
    -- the last from the action catalog, not back-divided from an expected
    value). Returns `(incremental_profit_samples, realized_cost_samples)`,
    each `(n_simulations,)`.
    """
    if len(selected) == 0:
        zeros = np.zeros(n_simulations)
        return zeros, zeros

    monthly_rate = clv_config_monthly_rate(clv_config)
    horizon = clv_config.horizon_months

    p_churn_accept = np.clip(
        selected["p_churn_monthly"].to_numpy() * scenario.churn_multiplier * (1 - selected["churn_reduction_pct"].to_numpy()),
        1e-6,
        0.9,
    )
    profit_accept = (
        selected["expected_profit"].to_numpy()
        + selected["additional_revenue_monthly"].to_numpy()
        - selected["additional_risk_monthly"].to_numpy()
        - selected["operational_cost_monthly"].to_numpy()
    ) * scenario.profit_multiplier
    factor_accept = survival_weighted_annuity_factor(monthly_rate, p_churn_accept, horizon)
    future_clv_accept = profit_accept * factor_accept

    p_churn_reject = np.clip(selected["p_churn_monthly"].to_numpy() * scenario.churn_multiplier, 1e-6, 0.9)
    profit_reject = selected["expected_profit"].to_numpy() * scenario.profit_multiplier
    factor_reject = survival_weighted_annuity_factor(monthly_rate, p_churn_reject, horizon)
    future_clv_reject = profit_reject * factor_reject

    incentive_cost = selected["incentive_cost"].to_numpy()
    delta_if_accept = future_clv_accept - future_clv_reject - incentive_cost

    rng = np.random.default_rng(seed)
    acceptance = selected["acceptance_probability"].to_numpy()
    n = len(selected)

    profit_total = np.zeros(n_simulations)
    cost_total = np.zeros(n_simulations)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b = end - start
        accepted = rng.random((b, n_simulations)) < acceptance[start:end][:, None]
        profit_total += (accepted * delta_if_accept[start:end][:, None]).sum(axis=0)
        cost_total += (accepted * incentive_cost[start:end][:, None]).sum(axis=0)

    return profit_total, cost_total


def summarize_samples(samples: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(samples.mean()),
        "p5": float(np.percentile(samples, 5)),
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, 95)),
        "std": float(samples.std()),
    }


def run_scenario(
    clv_df: pd.DataFrame,
    selected_actions: pd.DataFrame,
    profit_sigma: pd.Series,
    scenario: Scenario,
    clv_config: ClvConfig,
    budget: float,
    n_simulations: int,
    seed: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    baseline_samples = simulate_baseline_portfolio(
        clv_df, profit_sigma, scenario, clv_config, n_simulations, seed, batch_size
    )
    action_profit_samples, action_cost_samples = simulate_action_outcomes(
        selected_actions, scenario, clv_config, n_simulations, seed + 1, batch_size
    )
    total_samples = baseline_samples + action_profit_samples

    return {
        "scenario": scenario.name,
        "total_portfolio_value": summarize_samples(total_samples),
        "incremental_action_profit": summarize_samples(action_profit_samples),
        "realized_action_cost": summarize_samples(action_cost_samples),
        "probability_negative_incremental_profit": float((action_profit_samples < 0).mean()),
        "probability_budget_overrun": float((action_cost_samples > budget).mean()),
    }
