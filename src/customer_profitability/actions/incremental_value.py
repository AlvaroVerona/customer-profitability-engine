"""Phase 10 -- incremental value of each customer/action pair.

    IncrementalProfit_{i,a} = ExpectedProfit_{i,a} - ExpectedProfit_{i,NoAction} - ActionCost_{i,a}

Both `ExpectedProfit` terms are *future* values -- Phase 7's CLV machinery
(survival-weighted, discounted over `config.clv.horizon_months`), applied
to the customer's baseline churn hazard and expected monthly profit versus
the same, modified by the action's simulated effects (`simulator.py`). So
this measures the incremental *discounted future value* of offering an
action versus doing nothing, not a single month's profit delta.

`ActionCost_{i,a}` is the *expected* incentive cost --
`acceptance_probability x incentive_cost` -- since the incentive is only
paid if the customer actually accepts the offer; likewise every effect
(revenue, balance, risk, churn reduction) is blended by acceptance
probability before it reaches the CLV formula, since a declined offer
changes nothing. `NO_ACTION` nets to (approximately) zero by construction
-- every one of its effect columns is 0 -- rather than by a special case;
see test_actions.py for the check.

Two output columns per PROJECT_SPEC.md's Page 6 mockup ("Action / Cost /
ΔProfit / ΔCLV"):
    delta_clv           = simulated future value minus baseline, before cost
    incremental_profit  = delta_clv minus the expected action cost (the
                           formula above; this is what optimization ranks on)
"""

from __future__ import annotations

import pandas as pd

from customer_profitability.actions.definitions import build_action_catalog
from customer_profitability.clv.discounting import (
    clv_config_monthly_rate,
    survival_weighted_annuity_factor,
)
from customer_profitability.utils.config import Settings


def compute_incremental_value(baseline: pd.DataFrame, effects: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """`baseline`: customer_id, p_churn_monthly, expected_profit, future_clv
    (Phase 7's CLV table, still-active customers only). `effects`: Phase
    10's simulator output (long format, one row per eligible customer/action
    pair)."""
    catalog = {a.action_type.value: a.incentive_cost for a in build_action_catalog(settings.actions)}
    monthly_rate = clv_config_monthly_rate(settings.clv)
    horizon = settings.clv.horizon_months

    df = effects.merge(
        baseline[["customer_id", "p_churn_monthly", "expected_profit", "future_clv"]], on="customer_id", how="left"
    )

    p_churn_with_action = df["p_churn_monthly"] * (1 - df["acceptance_probability"] * df["churn_reduction_pct"])
    expected_profit_with_action = df["expected_profit"] + df["acceptance_probability"] * (
        df["additional_revenue_monthly"] - df["additional_risk_monthly"] - df["operational_cost_monthly"]
    )
    factor = survival_weighted_annuity_factor(monthly_rate, p_churn_with_action.to_numpy(), horizon)

    df["simulated_future_clv"] = expected_profit_with_action.to_numpy() * factor
    df["delta_clv"] = df["simulated_future_clv"] - df["future_clv"]
    df["expected_action_cost"] = df["acceptance_probability"] * df["action_type"].map(catalog)
    df["incremental_profit"] = df["delta_clv"] - df["expected_action_cost"]
    return df


def recommend_action(incremental_value: pd.DataFrame, customer_id: str, settings: Settings) -> pd.DataFrame:
    """All eligible actions for one customer, ranked by incremental profit
    -- the "Recommended Actions" table from PROJECT_SPEC.md's Page 6 mockup.
    NO_ACTION is always included as the comparison baseline (row order is
    not filtered to only positive-incremental actions -- that filtering is
    Phase 11's job, under budget/capacity constraints)."""
    names = {a.action_type.value: a.name for a in build_action_catalog(settings.actions)}
    rows = incremental_value[incremental_value["customer_id"] == customer_id].copy()
    rows["action_name"] = rows["action_type"].map(names)
    return rows.sort_values("incremental_profit", ascending=False).reset_index(drop=True)
