"""Phase 11 -- orchestrates candidate preparation, solving, and the
counterfactual comparison PROJECT_SPEC.md 11 asks for ("No optimization
vs. Optimized allocation").

The optimized selection is a decision-support output, not a causal claim:
it ranks candidates by Phase 10's *simulated* incremental profit, which is
itself built from documented heuristic acceptance/effect parameters (see
`actions/simulator.py`), not from a causally-identified, experimentally
validated action-effect model. Per PROJECT_SPEC.md 11's own instruction
("do not claim the optimized strategy is causal unless the underlying
action-effect estimates support such a conclusion"), the reports produced
here describe this as an optimal allocation *under the simulated effects*,
not a guaranteed real-world outcome.
"""

from __future__ import annotations

import pandas as pd
from ortools.sat.python import cp_model

from customer_profitability.actions.definitions import ActionType
from customer_profitability.optimization.constraints import (
    OptimizationConstraints,
    prepare_candidates,
)
from customer_profitability.optimization.model import SCALE, solve, to_cents
from customer_profitability.utils.logging import get_logger

logger = get_logger(__name__)


def _cent_consistent_sum(series: pd.Series) -> float:
    """Sums a monetary/risk column the same way the CP-SAT model enforced
    it -- rounded to integer cents per row, then summed -- rather than
    summing the original unrounded floats. Without this, a constraint that
    the solver satisfied *exactly* in integer-cent terms can appear to be
    violated by a fraction of a percent in the report, purely from
    re-deriving the sum in floats afterward."""
    return sum(to_cents(series)) / SCALE

_STATUS_NAMES = {
    cp_model.OPTIMAL: "OPTIMAL",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.MODEL_INVALID: "MODEL_INVALID",
    cp_model.UNKNOWN: "UNKNOWN",
}


def optimize(incremental_value: pd.DataFrame, constraints: OptimizationConstraints) -> dict:
    candidates = prepare_candidates(incremental_value, constraints)
    selected, _cp_solver, status = solve(candidates, constraints)
    status_name = _STATUS_NAMES.get(status, str(status))
    logger.info(
        "Solver status: %s, %d candidates considered, %d selected", status_name, len(candidates), len(selected)
    )

    retention = selected[selected["action_type"] == ActionType.RETENTION_INCENTIVE.value]
    expected_churns_averted = float(
        (retention["acceptance_probability"] * retention["churn_reduction_pct"] * retention["p_churn_monthly"]).sum()
    )

    total_cost = _cent_consistent_sum(selected["expected_action_cost"])
    total_risk = _cent_consistent_sum(selected["expected_incremental_risk_monthly"])
    return {
        "status": status_name,
        "candidates": candidates,
        "selected": selected,
        "n_candidates": len(candidates),
        "n_selected": len(selected),
        "total_cost": total_cost,
        "remaining_budget": constraints.budget - total_cost,
        "total_incremental_profit": _cent_consistent_sum(selected["incremental_profit"]),
        "total_delta_clv": float(selected["delta_clv"].sum()),
        "total_expected_incremental_risk_monthly": total_risk,
        "expected_monthly_churns_averted": expected_churns_averted,
        "action_distribution": selected["action_type"].value_counts().to_dict(),
        "constraints": constraints,
    }


def counterfactual_summary(result: dict) -> dict:
    """No optimization (every customer gets NO_ACTION, the true baseline
    with zero cost and zero incremental profit) vs. the optimized
    allocation actually selected."""
    return {
        "no_optimization_incremental_profit": 0.0,
        "no_optimization_cost": 0.0,
        "optimized_incremental_profit": result["total_incremental_profit"],
        "optimized_cost": result["total_cost"],
        "value_added_by_optimization": result["total_incremental_profit"],
    }
