"""Phase 11 -- optimization constraints and candidate preparation.

Every candidate is one (customer_id, action_type) pair from Phase 10's
incremental-value table. `NO_ACTION` is never a decision variable: it is
the implicit default for any customer the optimizer doesn't pick an action
for (the "one action per customer" constraint allows zero), so including
it as a candidate would be redundant, not wrong.

Budget and risk are both evaluated in *expected* terms (acceptance-
probability-weighted), consistent with how Phase 10 built `expected_action_cost`
and every other effect -- a customer who is unlikely to accept doesn't
meaningfully draw down the budget or the risk appetite.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from customer_profitability.actions.definitions import ActionType
from customer_profitability.utils.config import OptimizationConfig
from customer_profitability.utils.metrics import safe_divide


@dataclass(frozen=True)
class OptimizationConstraints:
    budget: float
    capacity: int
    max_incremental_risk_monthly: float
    min_expected_roi: float


def constraints_from_config(optimization_config: OptimizationConfig) -> OptimizationConstraints:
    return OptimizationConstraints(
        budget=optimization_config.default_budget,
        capacity=optimization_config.default_capacity,
        max_incremental_risk_monthly=optimization_config.default_max_incremental_risk_monthly,
        min_expected_roi=optimization_config.default_min_expected_roi,
    )


def prepare_candidates(incremental_value: pd.DataFrame, constraints: OptimizationConstraints) -> pd.DataFrame:
    """Drops NO_ACTION rows and any pair whose gross ROI
    (`delta_clv / expected_action_cost`) falls below `min_expected_roi`,
    before the pair is even offered to the solver as a candidate."""
    df = incremental_value[incremental_value["action_type"] != ActionType.NO_ACTION.value].copy()
    df["expected_incremental_risk_monthly"] = df["acceptance_probability"] * df["additional_risk_monthly"]
    gross_roi = safe_divide(df["delta_clv"].to_numpy(), df["expected_action_cost"].to_numpy(), fill=0.0)
    df["gross_roi"] = gross_roi
    return df[df["gross_roi"] >= constraints.min_expected_roi].reset_index(drop=True)
