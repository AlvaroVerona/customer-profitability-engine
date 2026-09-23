"""Phase 11 orchestrator: solves the optimization on the full portfolio
and writes a report.
"""

from __future__ import annotations

import json

import pandas as pd

from customer_profitability.actions.definitions import ActionType, build_action_catalog
from customer_profitability.optimization.constraints import constraints_from_config
from customer_profitability.optimization.solver import counterfactual_summary, optimize
from customer_profitability.utils.config import Settings, load_settings
from customer_profitability.utils.logging import get_logger

logger = get_logger(__name__)


def build_full_action_plan(incremental_value: pd.DataFrame, selected: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """One row per active customer: the selected action if the optimizer
    picked one for them, otherwise the NO_ACTION default -- PROJECT_SPEC.md's
    "Recommended Actions" output, for every customer, not just the ones
    the optimizer intervened on."""
    names = {a.action_type.value: a.name for a in build_action_catalog(settings.actions)}
    all_customers = incremental_value[["customer_id"]].drop_duplicates()
    plan = all_customers.merge(
        selected[["customer_id", "action_type", "expected_action_cost", "delta_clv", "incremental_profit"]],
        on="customer_id",
        how="left",
    )
    plan["action_type"] = plan["action_type"].fillna(ActionType.NO_ACTION.value)
    for col in ["expected_action_cost", "delta_clv", "incremental_profit"]:
        plan[col] = plan[col].fillna(0.0)
    plan["action_name"] = plan["action_type"].map(names)
    return plan


def to_markdown(result: dict, counterfactual: dict, plan: pd.DataFrame) -> str:
    constraints = result["constraints"]
    dist = pd.Series(result["action_distribution"], name="n_selected").rename_axis("action_type").reset_index()
    lines = [
        "# Optimization Engine Report",
        "",
        (
            "Selects the portfolio of customer actions maximizing total "
            "incremental economic profit under budget, capacity, risk, and "
            "one-action-per-customer constraints (OR-Tools CP-SAT, exact "
            "solve). This is an optimal allocation *under Phase 10's "
            "simulated action effects* -- a decision-support ranking, not a "
            "causal guarantee (PROJECT_SPEC.md 11)."
        ),
        "",
        f"**Solver status: {result['status']}**",
        "",
        "## Constraints",
        "",
        f"- Budget: {constraints.budget:,.2f} (expected cost basis)",
        f"- Operational capacity: {constraints.capacity:,} actions",
        f"- Max incremental monthly credit risk: {constraints.max_incremental_risk_monthly:,.2f}",
        f"- Min gross ROI (delta_clv / cost): {constraints.min_expected_roi:.2f}",
        "",
        "## Outputs",
        "",
        f"- Candidates considered: {result['n_candidates']:,}",
        f"- Actions selected: {result['n_selected']:,}",
        f"- Total expected cost: {result['total_cost']:,.2f} (remaining budget: {result['remaining_budget']:,.2f})",
        f"- Total incremental profit: {result['total_incremental_profit']:,.2f}",
        f"- Total delta CLV: {result['total_delta_clv']:,.2f}",
        f"- Total expected incremental monthly credit risk: {result['total_expected_incremental_risk_monthly']:,.2f}",
        f"- Expected monthly churns averted (selected retention offers): {result['expected_monthly_churns_averted']:.2f}",
        "",
        "## Action distribution (selected)",
        "",
        dist.to_string(index=False),
        "",
        "## Counterfactual: no optimization vs. optimized allocation",
        "",
        f"- No optimization (everyone gets NO_ACTION): incremental profit = {counterfactual['no_optimization_incremental_profit']:,.2f}, cost = {counterfactual['no_optimization_cost']:,.2f}",
        f"- Optimized allocation: incremental profit = {counterfactual['optimized_incremental_profit']:,.2f}, cost = {counterfactual['optimized_cost']:,.2f}",
        f"- Value added by running the optimization: {counterfactual['value_added_by_optimization']:,.2f}",
        "",
        f"## Full recommended action plan: {len(plan):,} active customers, {(plan['action_type'] != 'NO_ACTION').sum():,} receive an intervention",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    settings = load_settings()
    incremental_value = pd.read_parquet(settings.processed_dir / "action_incremental_value.parquet")

    constraints = constraints_from_config(settings.optimization)
    result = optimize(incremental_value, constraints)
    counterfactual = counterfactual_summary(result)
    plan = build_full_action_plan(incremental_value, result["selected"], settings)

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    plan_path = settings.processed_dir / "optimized_action_plan.parquet"
    plan.to_parquet(plan_path, index=False)
    logger.info("Wrote %s", plan_path)

    reports_dir = settings.raw_dir.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "optimization_report.md").write_text(to_markdown(result, counterfactual, plan))
    json_payload = {
        "status": result["status"],
        "n_candidates": result["n_candidates"],
        "n_selected": result["n_selected"],
        "total_cost": result["total_cost"],
        "remaining_budget": result["remaining_budget"],
        "total_incremental_profit": result["total_incremental_profit"],
        "total_delta_clv": result["total_delta_clv"],
        "total_expected_incremental_risk_monthly": result["total_expected_incremental_risk_monthly"],
        "expected_monthly_churns_averted": result["expected_monthly_churns_averted"],
        "action_distribution": result["action_distribution"],
        "counterfactual": counterfactual,
    }
    (reports_dir / "optimization_report.json").write_text(json.dumps(json_payload, indent=2))
    logger.info("Wrote reports/optimization_report.{md,json}")


if __name__ == "__main__":
    main()
