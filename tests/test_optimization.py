"""Phase 11 -- optimization engine tests."""

from __future__ import annotations

import pandas as pd
import pytest

from customer_profitability.actions.definitions import ActionType
from customer_profitability.optimization.constraints import (
    OptimizationConstraints,
    constraints_from_config,
    prepare_candidates,
)
from customer_profitability.optimization.model import solve
from customer_profitability.optimization.report import build_full_action_plan
from customer_profitability.optimization.solver import counterfactual_summary, optimize

from .conftest import small_settings


def test_constraints_from_config_maps_fields() -> None:
    settings = small_settings()
    constraints = constraints_from_config(settings.optimization)
    assert constraints.budget == settings.optimization.default_budget
    assert constraints.capacity == settings.optimization.default_capacity
    assert constraints.max_incremental_risk_monthly == settings.optimization.default_max_incremental_risk_monthly
    assert constraints.min_expected_roi == settings.optimization.default_min_expected_roi


def _incremental_value_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C2", "C2"],
            "action_type": ["NO_ACTION", "SAVINGS_CROSS_SELL", "NO_ACTION", "CREDIT_PRODUCT"],
            "acceptance_probability": [1.0, 0.4, 1.0, 0.3],
            "churn_reduction_pct": [0.0, 0.1, 0.0, 0.1],
            "p_churn_monthly": [0.02, 0.02, 0.05, 0.05],
            "additional_risk_monthly": [0.0, 0.0, 0.0, 10.0],
            "delta_clv": [0.0, 20.0, 0.0, -5.0],  # C2's credit product has a negative gross benefit
            "expected_action_cost": [0.0, 2.0, 0.0, 4.5],
            "incremental_profit": [0.0, 18.0, 0.0, -9.5],
        }
    )


def test_prepare_candidates_drops_no_action() -> None:
    constraints = OptimizationConstraints(budget=1000, capacity=100, max_incremental_risk_monthly=1000, min_expected_roi=0.0)
    out = prepare_candidates(_incremental_value_fixture(), constraints)
    assert "NO_ACTION" not in set(out["action_type"])


def test_prepare_candidates_computes_expected_risk() -> None:
    # min_expected_roi set low enough not to filter out C2/CREDIT_PRODUCT (whose gross ROI is
    # negative) -- this test isolates the risk computation, not the ROI filter.
    constraints = OptimizationConstraints(budget=1000, capacity=100, max_incremental_risk_monthly=1000, min_expected_roi=-999.0)
    out = prepare_candidates(_incremental_value_fixture(), constraints).set_index(["customer_id", "action_type"])
    # C2/CREDIT_PRODUCT: acceptance 0.3 x additional_risk 10.0 = 3.0
    assert out.loc[("C2", "CREDIT_PRODUCT"), "expected_incremental_risk_monthly"] == pytest.approx(3.0)


def test_prepare_candidates_filters_by_min_roi() -> None:
    # C2/CREDIT_PRODUCT has gross_roi = delta_clv / cost = -5.0 / 4.5 < 0 -- filtered out at min_expected_roi=0.0.
    constraints = OptimizationConstraints(budget=1000, capacity=100, max_incremental_risk_monthly=1000, min_expected_roi=0.0)
    out = prepare_candidates(_incremental_value_fixture(), constraints)
    assert not ((out["customer_id"] == "C2") & (out["action_type"] == "CREDIT_PRODUCT")).any()
    assert ((out["customer_id"] == "C1") & (out["action_type"] == "SAVINGS_CROSS_SELL")).any()


def _two_customer_candidates() -> pd.DataFrame:
    # C1 can take action A (profit 10, cost 5) or action B (profit 8, cost 1).
    # C2 can only take action A-equivalent (profit 6, cost 5).
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C2"],
            "action_type": ["RETENTION_INCENTIVE", "SAVINGS_CROSS_SELL", "RETENTION_INCENTIVE"],
            "incremental_profit": [10.0, 8.0, 6.0],
            "expected_action_cost": [5.0, 1.0, 5.0],
            "expected_incremental_risk_monthly": [0.0, 0.0, 0.0],
        }
    )


def test_solve_respects_one_action_per_customer() -> None:
    constraints = OptimizationConstraints(budget=1000, capacity=100, max_incremental_risk_monthly=1000, min_expected_roi=0.0)
    selected, _solver, _status = solve(_two_customer_candidates(), constraints)
    assert selected["customer_id"].value_counts().max() == 1


def test_solve_maximizes_profit_with_ample_budget() -> None:
    # No budget/capacity pressure -> should take both C1's best option (A, profit 10) and C2's only option.
    constraints = OptimizationConstraints(budget=1000, capacity=100, max_incremental_risk_monthly=1000, min_expected_roi=0.0)
    selected, _solver, _status = solve(_two_customer_candidates(), constraints)
    c1_action = selected.loc[selected["customer_id"] == "C1", "action_type"].iloc[0]
    assert c1_action == "RETENTION_INCENTIVE"  # profit 10 beats profit 8
    assert set(selected["customer_id"]) == {"C1", "C2"}
    assert selected["incremental_profit"].sum() == pytest.approx(16.0)


def test_solve_respects_tight_budget() -> None:
    # Budget only allows cost <= 5: cannot afford both C1's action A (5) and C2's action (5) -> picks
    # whichever combination maximizes profit within budget. Best feasible: C1 action A alone (profit 10, cost 5).
    constraints = OptimizationConstraints(budget=5, capacity=100, max_incremental_risk_monthly=1000, min_expected_roi=0.0)
    selected, _solver, _status = solve(_two_customer_candidates(), constraints)
    assert selected["expected_action_cost"].sum() <= 5.0 + 1e-6
    assert selected["incremental_profit"].sum() == pytest.approx(10.0)


def test_solve_respects_capacity() -> None:
    constraints = OptimizationConstraints(budget=1000, capacity=1, max_incremental_risk_monthly=1000, min_expected_roi=0.0)
    selected, _solver, _status = solve(_two_customer_candidates(), constraints)
    assert len(selected) <= 1


def test_solve_respects_risk_constraint() -> None:
    candidates = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "action_type": ["CREDIT_PRODUCT", "CREDIT_PRODUCT"],
            "incremental_profit": [10.0, 10.0],
            "expected_action_cost": [1.0, 1.0],
            "expected_incremental_risk_monthly": [8.0, 8.0],
        }
    )
    constraints = OptimizationConstraints(budget=1000, capacity=100, max_incremental_risk_monthly=10.0, min_expected_roi=0.0)
    selected, _solver, _status = solve(candidates, constraints)
    assert selected["expected_incremental_risk_monthly"].sum() <= 10.0 + 1e-6
    assert len(selected) == 1  # can't afford both (8+8=16 > 10)


def test_optimize_end_to_end_and_counterfactual() -> None:
    incremental_value = _incremental_value_fixture()
    constraints = OptimizationConstraints(budget=1000, capacity=100, max_incremental_risk_monthly=1000, min_expected_roi=0.0)
    result = optimize(incremental_value, constraints)
    assert result["status"] in {"OPTIMAL", "FEASIBLE"}
    assert result["total_incremental_profit"] >= 0

    counterfactual = counterfactual_summary(result)
    assert counterfactual["no_optimization_incremental_profit"] == 0.0
    assert counterfactual["optimized_incremental_profit"] == result["total_incremental_profit"]


def test_build_full_action_plan_covers_every_customer_with_no_action_default() -> None:
    settings = small_settings()
    incremental_value = _incremental_value_fixture()
    selected = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "action_type": ["SAVINGS_CROSS_SELL"],
            "expected_action_cost": [2.0],
            "delta_clv": [20.0],
            "incremental_profit": [18.0],
        }
    )
    plan = build_full_action_plan(incremental_value, selected, settings)
    assert set(plan["customer_id"]) == {"C1", "C2"}
    c2_row = plan[plan["customer_id"] == "C2"].iloc[0]
    assert c2_row["action_type"] == ActionType.NO_ACTION.value
    assert c2_row["incremental_profit"] == 0.0
    c1_row = plan[plan["customer_id"] == "C1"].iloc[0]
    assert c1_row["action_type"] == "SAVINGS_CROSS_SELL"
    assert c1_row["incremental_profit"] == 18.0
