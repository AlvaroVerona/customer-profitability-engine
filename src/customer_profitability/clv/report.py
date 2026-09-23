"""Phase 7 orchestrator: assembles CLV for every customer and writes a report.

Customers who have already churned within the observed window have no
future relationship to forecast -- their `future_clv` is forced to 0 and
their Monte Carlo band collapses to their (certain, already-realized)
historical value, regardless of what the churn model or run-rate would
otherwise say about their long-dormant final snapshot.
"""

from __future__ import annotations

import json

import pandas as pd

from customer_profitability.clv.calculator import compute_base_clv, monte_carlo_clv
from customer_profitability.clv.forecasting import (
    build_expected_profit,
    compute_run_rate,
    estimate_profit_volatility,
    latest_snapshot,
    predict_churn_probability,
)
from customer_profitability.utils.config import Settings, load_settings
from customer_profitability.utils.logging import get_logger

logger = get_logger(__name__)

MC_SIMULATIONS = 1000


def build_clv_table(
    customer_360: pd.DataFrame,
    profitability_monthly: pd.DataFrame,
    profitability_summary: pd.DataFrame,
    customers: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    still_active_ids = set(customers.loc[~customers["churn_flag"], "customer_id"])

    snapshot = latest_snapshot(customer_360)
    snapshot_active = snapshot[snapshot["customer_id"].isin(still_active_ids)]
    p_churn = predict_churn_probability(snapshot_active, settings.processed_dir / "models").reset_index()

    run_rate = compute_run_rate(profitability_monthly)
    expected_profit_df = build_expected_profit(run_rate)[["customer_id", "expected_profit"]]

    historical = profitability_summary[["customer_id", "economic_profit"]].rename(
        columns={"economic_profit": "historical_economic_profit"}
    )

    base_input = (
        historical.merge(expected_profit_df, on="customer_id", how="left")
        .merge(p_churn, on="customer_id", how="left")
    )
    base_input["is_active"] = base_input["customer_id"].isin(still_active_ids)
    base_input["expected_profit"] = base_input["expected_profit"].fillna(0.0)
    base_input["p_churn_monthly"] = base_input["p_churn_monthly"].fillna(1.0)

    base_clv = compute_base_clv(base_input, settings.clv)
    base_clv.loc[~base_clv["is_active"], "future_clv"] = 0.0
    base_clv["total_customer_economic_value"] = base_clv["historical_economic_profit"] + base_clv["future_clv"]

    sigma = estimate_profit_volatility(profitability_monthly)
    mc_input = base_clv[["customer_id", "historical_economic_profit", "expected_profit", "p_churn_monthly"]]
    mc = monte_carlo_clv(mc_input, settings.clv, sigma, n_simulations=MC_SIMULATIONS, seed=settings.seed)

    mc = mc.merge(base_clv[["customer_id", "is_active", "historical_economic_profit"]], on="customer_id")
    for col in ["clv_p5", "clv_p50", "clv_p95"]:
        mc.loc[~mc["is_active"], col] = mc.loc[~mc["is_active"], "historical_economic_profit"]
    mc = mc.drop(columns=["is_active", "historical_economic_profit"])

    return base_clv.merge(mc, on="customer_id")


def _read_inputs(settings: Settings) -> dict[str, pd.DataFrame]:
    return {
        "customer_360": pd.read_parquet(settings.features_dir / "customer_360.parquet"),
        "profitability_monthly": pd.read_parquet(settings.processed_dir / "profitability_monthly.parquet"),
        "profitability_summary": pd.read_parquet(settings.processed_dir / "profitability_customer_summary.parquet"),
        "customers": pd.read_parquet(settings.processed_dir / "customers_validated.parquet").drop_duplicates(
            "customer_id"
        ),
    }


def to_markdown(clv: pd.DataFrame) -> str:
    active = clv[clv["is_active"]]
    lines = [
        "# Customer Lifetime Value Report",
        "",
        (
            "`historical_economic_profit` and `future_clv` are kept separate "
            "(PROJECT_SPEC.md 7.4); `total_customer_economic_value` sums them. "
            "`clv_p5`/`clv_p50`/`clv_p95` are a Monte Carlo uncertainty band "
            "(1,000 simulations/customer) around `total_customer_economic_value` "
            "-- CLV is not presented as a single exact number (7.5)."
        ),
        "",
        f"- Customers: {len(clv)} total, {len(active)} still active as of the observation window end",
        f"- Total historical economic profit: {clv['historical_economic_profit'].sum():,.0f}",
        f"- Total future CLV (active customers only): {active['future_clv'].sum():,.0f}",
        f"- Total customer economic value: {clv['total_customer_economic_value'].sum():,.0f}",
        f"- Mean total CEV (active): {active['total_customer_economic_value'].mean():,.2f}",
        f"- Median total CEV, Monte Carlo (active): {active['clv_p50'].mean():,.2f}",
        "",
        "## Distribution (active customers)",
        "",
        active[["historical_economic_profit", "future_clv", "total_customer_economic_value", "clv_p5", "clv_p50", "clv_p95"]]
        .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        .round(2)
        .to_string(),
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    settings = load_settings()
    inputs = _read_inputs(settings)

    logger.info("Building CLV table")
    clv = build_clv_table(
        inputs["customer_360"],
        inputs["profitability_monthly"],
        inputs["profitability_summary"],
        inputs["customers"],
        settings,
    )

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.processed_dir / "clv.parquet"
    clv.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(clv))

    reports_dir = settings.raw_dir.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "clv_report.md").write_text(to_markdown(clv))
    summary = {
        "n_customers": len(clv),
        "n_active": int(clv["is_active"].sum()),
        "total_historical_economic_profit": float(clv["historical_economic_profit"].sum()),
        "total_future_clv": float(clv.loc[clv["is_active"], "future_clv"].sum()),
        "total_customer_economic_value": float(clv["total_customer_economic_value"].sum()),
    }
    (reports_dir / "clv_report.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote reports/clv_report.{md,json}")


if __name__ == "__main__":
    main()
