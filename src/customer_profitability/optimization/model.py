"""Phase 11 -- the optimization model itself.

    max  sum_i sum_a x_{i,a} * IncrementalProfit_{i,a}
    s.t. sum_a x_{i,a} <= 1                                    (one action per customer)
         sum_{i,a} x_{i,a} * ExpectedCost_{i,a} <= Budget       (budget)
         sum_{i,a} x_{i,a} <= Capacity                          (operational capacity)
         sum_{i,a} x_{i,a} * ExpectedRisk_{i,a} <= MaxRisk       (risk)
         x_{i,a} in {0,1}

Built with OR-Tools CP-SAT (`ortools.sat.python.cp_model`), which is exact
(not a heuristic) for this problem size and needs no external solver
license. CP-SAT works in integers, so every monetary/risk quantity is
scaled to integer cents (`SCALE`) before being handed to the model; the
returned selection is exact regardless (the rounding only affects the
model's internal arithmetic by fractions of a cent per candidate).
"""

from __future__ import annotations

import pandas as pd
from ortools.sat.python import cp_model

from customer_profitability.optimization.constraints import OptimizationConstraints

SCALE = 100  # work in integer cents


def to_cents(series: pd.Series) -> list[int]:
    return (series * SCALE).round().astype(int).tolist()


def solve(
    candidates: pd.DataFrame, constraints: OptimizationConstraints, time_limit_seconds: float = 30.0
) -> tuple[pd.DataFrame, cp_model.CpSolver, int]:
    """`candidates`: Phase 10's incremental-value rows (NO_ACTION excluded,
    `optimization.constraints.prepare_candidates` already applied). Returns
    (selected_rows, solver, status) -- callers should check `status` is
    OPTIMAL or FEASIBLE before trusting the selection."""
    model = cp_model.CpModel()
    n = len(candidates)
    x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

    for idx in candidates.groupby("customer_id").indices.values():
        model.Add(sum(x[i] for i in idx) <= 1)

    cost_cents = to_cents(candidates["expected_action_cost"])
    model.Add(sum(cost_cents[i] * x[i] for i in range(n)) <= round(constraints.budget * SCALE))

    model.Add(sum(x) <= constraints.capacity)

    risk_cents = to_cents(candidates["expected_incremental_risk_monthly"])
    model.Add(sum(risk_cents[i] * x[i] for i in range(n)) <= round(constraints.max_incremental_risk_monthly * SCALE))

    profit_cents = to_cents(candidates["incremental_profit"])
    model.Maximize(sum(profit_cents[i] * x[i] for i in range(n)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return candidates.iloc[0:0], solver, status

    selected_idx = [i for i in range(n) if solver.Value(x[i]) == 1]
    return candidates.iloc[selected_idx].reset_index(drop=True), solver, status
