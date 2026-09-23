"""Phase 10 orchestrator: simulates every eligible customer/action pair,
computes incremental value against NO_ACTION, and writes a report.
"""

from __future__ import annotations

import json

import pandas as pd

from customer_profitability.actions.definitions import build_action_catalog
from customer_profitability.actions.incremental_value import (
    compute_incremental_value,
    recommend_action,
)
from customer_profitability.actions.simulator import simulate_action_effects
from customer_profitability.clv.forecasting import latest_snapshot
from customer_profitability.utils.config import Settings, load_settings
from customer_profitability.utils.logging import get_logger

logger = get_logger(__name__)


def build_customer_inputs(clv: pd.DataFrame, customers: pd.DataFrame, customer_360: pd.DataFrame) -> pd.DataFrame:
    active = clv[clv["is_active"]][["customer_id", "p_churn_monthly", "expected_profit", "future_clv"]]
    snapshot = latest_snapshot(customer_360)[["customer_id", "average_balance_avg_3m"]]
    income = customers[["customer_id", "income"]]
    return active.merge(income, on="customer_id", how="left").merge(snapshot, on="customer_id", how="left")


def run_action_simulation(settings: Settings) -> pd.DataFrame:
    clv = pd.read_parquet(settings.processed_dir / "clv.parquet")
    customers = pd.read_parquet(settings.processed_dir / "customers_validated.parquet").drop_duplicates("customer_id")
    customer_360 = pd.read_parquet(settings.features_dir / "customer_360.parquet")
    eligibility = pd.read_parquet(settings.processed_dir / "action_eligibility.parquet")

    customer_inputs = build_customer_inputs(clv, customers, customer_360)
    logger.info("Simulating actions for %d active customers", len(customer_inputs))

    effects = simulate_action_effects(customer_inputs, eligibility, settings)
    incremental = compute_incremental_value(customer_inputs, effects, settings)
    logger.info("Computed incremental value for %d (customer, action) pairs", len(incremental))
    return incremental


def _action_summary(incremental: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    names = {a.action_type.value: a.name for a in build_action_catalog(settings.actions)}
    summary = incremental.groupby("action_type").agg(
        n_eligible=("customer_id", "size"),
        mean_acceptance_probability=("acceptance_probability", "mean"),
        mean_delta_clv=("delta_clv", "mean"),
        mean_expected_cost=("expected_action_cost", "mean"),
        mean_incremental_profit=("incremental_profit", "mean"),
        total_incremental_profit=("incremental_profit", "sum"),
        pct_beats_no_action=("incremental_profit", lambda s: float((s > 0).mean())),
    )
    summary["action_name"] = summary.index.map(names)
    return summary.reset_index()


def to_markdown(incremental: pd.DataFrame, summary: pd.DataFrame, example: pd.DataFrame, example_id: str) -> str:
    lines = [
        "# Action Impact Simulation Report",
        "",
        (
            "Every action's effect is simulated *conditional on acceptance* "
            "(simulator.py) and blended by acceptance probability into the "
            "expected value of *offering* it (incremental_value.py) -- "
            "acceptance/effect parameters are documented heuristics "
            "(`config.action_simulation`), not fitted models, since no "
            "historical offer/acceptance data exists in this project."
        ),
        "",
        "## Summary by action",
        "",
        summary[
            [
                "action_name",
                "n_eligible",
                "mean_acceptance_probability",
                "mean_delta_clv",
                "mean_expected_cost",
                "mean_incremental_profit",
                "total_incremental_profit",
                "pct_beats_no_action",
            ]
        ]
        .round(3)
        .to_string(index=False),
        "",
        f"## Example: recommended actions for customer `{example_id}`",
        "",
        example[["action_name", "expected_action_cost", "delta_clv", "incremental_profit"]].round(2).to_string(index=False),
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    settings = load_settings()
    incremental = run_action_simulation(settings)

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.processed_dir / "action_incremental_value.parquet"
    incremental.to_parquet(out_path, index=False)
    logger.info("Wrote %s", out_path)

    summary = _action_summary(incremental, settings)

    example_id = incremental["customer_id"].iloc[0]
    example = recommend_action(incremental, example_id, settings)

    reports_dir = settings.raw_dir.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "action_simulation_report.md").write_text(to_markdown(incremental, summary, example, example_id))
    (reports_dir / "action_simulation_report.json").write_text(
        json.dumps(summary.set_index("action_type").to_dict(orient="index"), indent=2)
    )
    logger.info("Wrote reports/action_simulation_report.{md,json}")

    for _, row in summary.iterrows():
        logger.info(
            "%s: n=%d mean_incremental_profit=%.2f pct_beats_no_action=%.1f%%",
            row["action_name"],
            row["n_eligible"],
            row["mean_incremental_profit"],
            100 * row["pct_beats_no_action"],
        )


if __name__ == "__main__":
    main()
