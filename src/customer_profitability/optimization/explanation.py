"""Phase 14 -- explains why a particular customer/action combination was,
or was not, selected by the Phase 11 optimizer.

This is a rank-based *approximation*, not a literal trace of the CP-SAT
solver's internal decision process: the ILP jointly optimizes under
budget, capacity, *and* risk simultaneously, so "ranked #47 by incremental
profit" is a useful, honest summary of competitive position, not a claim
that the solver evaluated candidates strictly in that order. A candidate
can rank well by profit alone and still lose out once the risk or budget
constraint is the binding one -- the explanation says so rather than
overstating precision the method doesn't have.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SelectionExplanation:
    customer_id: str
    was_selected: bool
    action_type: str | None
    reason: str


def explain_selection(customer_id: str, result: dict) -> SelectionExplanation:
    """`result`: the dict returned by `optimization.solver.optimize`
    (`candidates` = every pair considered after the ROI pre-filter,
    `selected` = the ILP's chosen subset)."""
    candidates: pd.DataFrame = result["candidates"]
    selected: pd.DataFrame = result["selected"]

    own_selected = selected[selected["customer_id"] == customer_id]
    if len(own_selected):
        row = own_selected.iloc[0]
        rank = int((candidates["incremental_profit"] > row["incremental_profit"]).sum()) + 1
        reason = (
            f"Selected: {row['action_type']} ranks #{rank} of {len(candidates):,} candidates by incremental "
            f"profit (EUR {row['incremental_profit']:,.2f}) and fit within budget/capacity/risk headroom."
        )
        return SelectionExplanation(customer_id, True, row["action_type"], reason)

    own_candidates = candidates[candidates["customer_id"] == customer_id]
    if own_candidates.empty:
        reason = (
            "No eligible action for this customer cleared the minimum gross-ROI pre-filter, "
            "so none was even offered to the optimizer as a candidate."
        )
        return SelectionExplanation(customer_id, False, None, reason)

    best = own_candidates.sort_values("incremental_profit", ascending=False).iloc[0]
    rank = int((candidates["incremental_profit"] > best["incremental_profit"]).sum()) + 1
    reason = (
        f"Not selected: this customer's best candidate ({best['action_type']}, "
        f"EUR {best['incremental_profit']:,.2f}) ranks #{rank} of {len(candidates):,} candidates by "
        f"incremental profit. The optimizer selected {len(selected):,} candidates that, jointly, fit the "
        "budget/capacity/risk constraints better -- not evidence this candidate's value was negative, "
        "just that the constrained resources were allocated elsewhere first."
    )
    return SelectionExplanation(customer_id, False, None, reason)
