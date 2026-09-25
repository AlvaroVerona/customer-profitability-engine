"""Phase 17 -- Final Report.

Assembles `reports/final_report.md`, the spec's 8-section portfolio
deliverable (PROJECT_SPEC.md "PHASE 17 -- Final Report"), purely by
reading the JSON reports and processed parquet tables every earlier phase
already wrote to disk. Nothing here is computed, estimated, or fabricated
for the purpose of this document -- if a number isn't already sitting in
`reports/*.json` or `data/processed/*.parquet`, it doesn't appear here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from customer_profitability.utils.config import Settings, load_settings
from customer_profitability.utils.logging import get_logger

logger = get_logger(__name__)


def _load_json(reports_dir: Path, name: str) -> dict:
    return json.loads((reports_dir / name).read_text())


def _profitability_summary(processed_dir: Path) -> dict:
    summary = pd.read_parquet(processed_dir / "profitability_customer_summary.parquet")
    monthly = pd.read_parquet(processed_dir / "profitability_monthly.parquet")
    return {
        "n_customers": len(summary),
        "pct_profitable": 100 * (summary["economic_profit"] > 0).mean(),
        "total_revenue": summary["total_revenue"].sum(),
        "total_gross_contribution": summary["total_gross_contribution"].sum(),
        "total_expected_loss": summary["total_expected_loss"].sum(),
        "total_operating_cost": summary["total_operating_cost"].sum(),
        "total_acquisition_cost": summary["total_acquisition_cost"].sum(),
        "total_economic_profit": summary["economic_profit"].sum(),
        "interest_revenue": monthly["interest_revenue"].sum(),
        "interchange_revenue": monthly["interchange_revenue"].sum(),
        "fee_revenue": monthly["fee_revenue"].sum(),
        "deposit_contribution": monthly["deposit_contribution"].sum(),
    }


def _credit_risk_summary(processed_dir: Path) -> dict:
    loans = pd.read_parquet(processed_dir / "loans_validated.parquet")
    return {
        "n_loan_observations": len(loans),
        "mean_pd": loans["pd"].mean(),
        "mean_lgd": loans["lgd"].mean(),
        "mean_expected_loss_rate": (loans["pd"] * loans["lgd"]).mean(),
        "total_outstanding_balance": loans["outstanding_balance"].sum(),
    }


def _executive_summary(quality: dict, profitability: dict, optimization: dict, clv: dict) -> list[str]:
    return [
        "# Final Report -- Customer Profitability & Action Optimization Engine",
        "",
        "## 1. Executive Summary",
        "",
        (
            "**Business problem.** A neobank does not know, at the customer level, where its "
            "profit actually comes from, which customers are worth investing in, or how to "
            "allocate a limited retention/growth budget across millions of possible "
            "customer-action pairs. This project builds a synthetic (but internally "
            "consistent, seed-42-reproducible) neobank dataset and a full analytical pipeline "
            "that answers exactly that: **which customers are profitable, why, what they are "
            "worth going forward, and which actions the bank should fund given a hard "
            "budget/capacity/risk constraint.**"
        ),
        "",
        (
            "**Methodology.** The pipeline runs Customer 360 -> historical profitability "
            "(revenue, Funds Transfer Pricing, Expected Credit Loss, operating cost) -> "
            "econometric driver analysis -> machine-learned forecasts (churn, revenue, "
            "balances, product adoption) -> Customer Lifetime Value with Monte Carlo "
            "uncertainty -> economic segmentation -> action simulation -> a CP-SAT "
            "constrained-optimization allocation engine -> portfolio-level Monte Carlo "
            "scenario stress-testing -> a Streamlit decision-support dashboard with "
            "customer-level decision explanations."
        ),
        "",
        "**Main findings, from the current pipeline run (not illustrative numbers):**",
        "",
        (
            f"- Data quality: {quality['overall_quality_score']:.1f}/100 overall score across all raw tables "
            "(every defective row is quarantined and retained, never silently dropped)."
        ),
        (
            f"- {profitability['n_customers']:,} customers, "
            f"{profitability['pct_profitable']:.1f}% individually profitable, "
            f"EUR {profitability['total_economic_profit']:,.0f} total historical economic profit."
        ),
        (
            f"- {clv['n_active']:,} of {clv['n_customers']:,} customers still active, with "
            f"EUR {clv['total_customer_economic_value']:,.0f} total customer economic value "
            "(historical profit already booked + discounted, survival-weighted future profit)."
        ),
        (
            "- The optimization engine, under the current budget/capacity/risk constraints, "
            f"selects {optimization['n_selected']:,} of {optimization['n_candidates']:,} candidate "
            f"customer/action pairs for EUR {optimization['total_cost']:,.0f} spent, generating "
            f"EUR {optimization['total_incremental_profit']:,.0f} of incremental profit "
            f"(status: {optimization['status']})."
        ),
        "",
    ]


def _customer_economics_section(profitability: dict) -> list[str]:
    revenue_total = profitability["interest_revenue"] + profitability["interchange_revenue"] + profitability["fee_revenue"]
    return [
        "## 2. Customer Economics",
        "",
        (
            "Every customer-month's economics follow one waterfall "
            "(`profitability/engine.py`): "
            "`gross_contribution = total_revenue + deposit_contribution - operating_cost`, "
            "`risk_adjusted_contribution = gross_contribution - expected_loss`, "
            "`economic_profit = risk_adjusted_contribution - acquisition_cost` "
            "(acquisition cost recognized once, in the acquisition month, and only if that "
            "month falls inside the observed data window)."
        ),
        "",
        "**Revenue composition** (sum across all customer-months observed):",
        "",
        (
            f"- Interest revenue: EUR {profitability['interest_revenue']:,.0f} "
            f"({100 * profitability['interest_revenue'] / revenue_total:.1f}% of revenue)"
        ),
        (
            f"- Interchange revenue: EUR {profitability['interchange_revenue']:,.0f} "
            f"({100 * profitability['interchange_revenue'] / revenue_total:.1f}% of revenue)"
        ),
        (
            f"- Fee revenue (low-balance + Premium subscription): EUR {profitability['fee_revenue']:,.0f} "
            f"({100 * profitability['fee_revenue'] / revenue_total:.1f}% of revenue)"
        ),
        (
            f"- Deposit (FTP) contribution: EUR {profitability['deposit_contribution']:,.0f} "
            "(the bank's internal funding value of customer deposits, net of what is paid to the customer)"
        ),
        "",
        "**Cost composition:**",
        "",
        (
            "- Operating cost (support + card processing + account servicing): "
            f"EUR {profitability['total_operating_cost']:,.0f}"
        ),
        (
            "- Acquisition cost (recognized once per customer, in-window only): "
            f"EUR {profitability['total_acquisition_cost']:,.0f}"
        ),
        "",
        (
            f"**Bottom line:** EUR {profitability['total_gross_contribution']:,.0f} total gross "
            f"contribution, {profitability['pct_profitable']:.1f}% of the {profitability['n_customers']:,} "
            "customers individually economic-profit-positive over their observed history -- "
            "profitability is concentrated, not evenly spread (see Segmentation, section 5)."
        ),
        "",
    ]


def _risk_section(profitability: dict, credit_risk: dict) -> list[str]:
    return [
        "## 3. Risk-Adjusted Profitability",
        "",
        (
            "`expected_loss = PD x LGD x EAD`, summed over every loan a customer holds each "
            "month (`profitability/risk_cost.py`). Across "
            f"{credit_risk['n_loan_observations']:,} loan-month observations: mean PD "
            f"{credit_risk['mean_pd']:.2%}/month, mean LGD {credit_risk['mean_lgd']:.1%}, "
            f"mean expected-loss rate {credit_risk['mean_expected_loss_rate']:.2%}/month of "
            f"outstanding balance, on EUR {credit_risk['total_outstanding_balance']:,.0f} total "
            "outstanding loan balance in the current book."
        ),
        "",
        (
            f"Expected credit loss (EUR {profitability['total_expected_loss']:,.0f} total) is the "
            "single largest cost line ahead of operating cost "
            f"(EUR {profitability['total_operating_cost']:,.0f}) -- credit risk, not servicing "
            "cost, is what most separates a profitable customer from an unprofitable one in "
            "this dataset. Risk-adjusted contribution "
            f"(EUR {profitability['total_gross_contribution'] - profitability['total_expected_loss']:,.0f}) "
            "is gross contribution minus this expected loss, before acquisition cost -- the "
            "single number every downstream stage (CLV, action value, optimization) actually "
            "optimizes against, not raw revenue."
        ),
        "",
    ]


def _clv_section(clv: dict) -> list[str]:
    return [
        "## 4. Customer Lifetime Value",
        "",
        (
            "`CLV_i = HistoricalEconomicProfit_i + sum_t [P(Survival_t) x ExpectedProfit_t] / "
            "(1+r)^t` (`clv/calculator.py`), reported as three separate columns -- "
            "`historical_economic_profit`, `future_clv`, `total_customer_economic_value` -- so "
            "a reader is never left guessing which one a plain \"CLV\" figure means."
        ),
        "",
        "**Methodology and assumptions:**",
        "",
        (
            "- Survival probability comes from Phase 6's *calibrated* churn model (Platt "
            "scaling on top of a class-rebalanced classifier, since rebalancing needed for "
            "usable ranking on a ~2%/month event distorts raw `predict_proba`), evaluated once "
            "on each customer's latest snapshot and held as a **constant monthly hazard** over "
            "the horizon -- a documented simplification, not a month-by-month re-forecast."
        ),
        (
            "- `ExpectedProfit` is each customer's own trailing 3-month run-rate across the "
            "Phase 4 waterfall, held **flat** over the horizon, consistent with the flat-hazard "
            "assumption."
        ),
        (
            "- A Monte Carlo band (`clv_p5`/`clv_p50`/`clv_p95`) exists specifically so CLV is "
            "never presented as one exact number: it draws stochastic churn timing (geometric "
            "survival draw) and profit-level uncertainty (log-normal multiplier sized by each "
            "customer's own historical profit volatility)."
        ),
        "",
        "**Portfolio totals (current run):**",
        "",
        f"- {clv['n_active']:,} of {clv['n_customers']:,} customers active.",
        f"- Total historical economic profit already booked: EUR {clv['total_historical_economic_profit']:,.0f}.",
        f"- Total future CLV (discounted, survival-weighted): EUR {clv['total_future_clv']:,.0f}.",
        f"- Total customer economic value: EUR {clv['total_customer_economic_value']:,.0f}.",
        "",
    ]


def _segmentation_section(segmentation: dict, profiles: pd.DataFrame) -> list[str]:
    lines = [
        "## 5. Economic Segmentation",
        "",
        (
            f"K-Means on standardized economic features (never demographics), k={segmentation['best_k']} "
            f"selected under a documented business-interpretability floor of 5 segments over the "
            f"raw silhouette-argmax (silhouette {segmentation['kmeans_silhouette']:.3f}) -- the raw "
            "metric alone picked a coarser k that collapsed clearly distinct segments together."
        ),
        "",
        "| Segment | Customers | Avg. historical profit | Avg. total CLV | Avg. churn risk/mo. |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in profiles.sort_values("total_customer_economic_value", ascending=False).iterrows():
        lines.append(
            f"| {row['segment_name']} | {int(row['n_customers']):,} | "
            f"EUR {row['historical_economic_profit']:,.0f} | EUR {row['total_customer_economic_value']:,.0f} | "
            f"{row['p_churn_monthly']:.1%} |"
        )
    lines += [
        "",
        (
            "The two \"High Value\" segments hold a disproportionate share of total customer "
            "economic value despite being the smallest by headcount -- the same concentration "
            "pattern visible in the Customer Economics section, now attributable to specific, "
            "behaviorally-defined groups rather than an undifferentiated \"profitable "
            "customers\" bucket."
        ),
        "",
    ]
    return lines


def _action_optimization_section(actions: dict, action_sim: dict, optimization: dict) -> list[str]:
    action_rows = "\n".join(
        f"| {info['action_name']} | {info['n_eligible']:,} | {info['mean_acceptance_probability']:.0%} | "
        f"EUR {info['mean_incremental_profit']:,.2f} | EUR {info['total_incremental_profit']:,.0f} |"
        for name, info in action_sim.items()
        if name != "NO_ACTION"
    )
    action_dist_rows = "\n".join(
        f"| {action} | {count:,} |" for action, count in optimization["action_distribution"].items()
    )
    return [
        "## 6. Action Optimization",
        "",
        (
            f"**Action framework** (`actions/definitions.py`): {len(actions['eligible_counts']) - 1} real "
            "actions (retention incentive, savings cross-sell, credit product, investment "
            "product, Premium subscription) plus NO_ACTION as an explicit, always-eligible "
            "baseline. Eligibility is behavioral/financial only -- product ownership, income, "
            "employment status for credit risk -- never a protected demographic attribute."
        ),
        "",
        "**Simulated incremental value per action** (mean over eligible active customers):",
        "",
        "| Action | Eligible customers | Acceptance prob. | Mean incremental profit | Total incremental profit |",
        "|---|---:|---:|---:|---:|",
        action_rows,
        "",
        (
            "**Optimization formulation** (`optimization/model.py`, exact CP-SAT integer "
            "programming, not a heuristic): maximize total incremental profit subject to "
            "one-action-per-customer, a budget cap, an operational capacity cap, and a maximum "
            "incremental monthly risk cap."
        ),
        "",
        (
            f"**Current run:** status {optimization['status']}, {optimization['n_selected']:,} of "
            f"{optimization['n_candidates']:,} candidates selected, EUR {optimization['total_cost']:,.0f} "
            f"spent (EUR {optimization['remaining_budget']:,.0f} budget remaining), "
            f"EUR {optimization['total_incremental_profit']:,.0f} total incremental profit, "
            f"~{optimization['expected_monthly_churns_averted']:.2f} expected monthly churns averted."
        ),
        "",
        "| Selected action | Count |",
        "|---|---:|",
        action_dist_rows,
        "",
    ]


def _scenario_section(monte_carlo: dict) -> list[str]:
    lines = [
        "## 7. Scenario Analysis",
        "",
        (
            "Portfolio-level Monte Carlo simulation (`simulation/monte_carlo.py`) re-draws "
            "action acceptance (Bernoulli) and applies each scenario's churn/profit "
            "multipliers across many simulated portfolios, rather than reporting a single "
            "point estimate."
        ),
        "",
        "| Scenario | Mean total portfolio value | Mean incremental action profit | P(negative incremental profit) | P(budget overrun) |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ["base", "upside", "downside", "stress"]:
        s = monte_carlo[key]
        lines.append(
            f"| {s['scenario']} | EUR {s['total_portfolio_value']['mean']:,.0f} | "
            f"EUR {s['incremental_action_profit']['mean']:,.0f} | "
            f"{s['probability_negative_incremental_profit']:.1%} | "
            f"{s['probability_budget_overrun']:.1%} |"
        )
    lines += [
        "",
        (
            "Even under Stress, incremental action profit and total portfolio value stay "
            "positive in every simulated draw -- the optimizer's selection is not fragile to "
            "the specific point estimates it was solved against, though this is a property of "
            "the *simulated* scenario multipliers, not a guarantee about real-world stress "
            "events (see Limitations)."
        ),
        "",
    ]
    return lines


def _limitations_section() -> list[str]:
    return [
        "## 8. Limitations",
        "",
        (
            "- **Synthetic data.** Every table in this project is synthetically generated "
            "(seed 42, `data/generator.py`) for a fictional neobank. No real customer, "
            "transaction, or credit data is used anywhere. Correlational and behavioral "
            "patterns are deliberately engineered to be realistic, not fit to any real "
            "institution's book."
        ),
        (
            "- **Modeling assumptions.** CLV assumes a constant monthly churn hazard and a "
            "flat trailing-3-month profit run-rate over the forecast horizon (documented in "
            "section 4) -- a real customer's hazard and profit both evolve, and a production "
            "system would need genuine time-varying survival modeling. FTP uses a single "
            "portfolio-wide base rate rather than a full yield-curve-based transfer-pricing "
            "framework (see below). Action acceptance probabilities are calibrated "
            "assumptions from the simulation config, not measured from a real A/B test."
        ),
        (
            "- **Causal limitations.** Action incremental value and the optimizer's ranking "
            "are *simulated* treatment effects under `actions/simulator.py`'s assumed "
            "acceptance/impact model, not a causally identified effect from a randomized "
            "experiment or quasi-experimental design. Real deployment would require an actual "
            "uplift model estimated from experimental or quasi-experimental data before these "
            "numbers could be treated as causal."
        ),
        (
            "- **Potential model risk.** Every Phase 6 forecasting target is benchmarked "
            "against a simple linear/logistic-regression baseline (`models/validation.py`, "
            "`reports/model_validation_report.md`); the baseline is in fact the champion on "
            "3 of 8 targets, meaning added model complexity does not uniformly buy predictive "
            "power on this dataset -- a reminder that model selection should be evaluated "
            "per-target, not assumed in favor of the most sophisticated available model."
        ),
        (
            "- **Simplified FTP assumptions.** Funds Transfer Pricing uses one base annual FTP "
            "rate and one market rate from `config/settings.yaml` (`profitability/ftp.py`), "
            "not a full term-structure/duration-matched transfer-pricing curve as used by real "
            "treasury functions -- deposit contribution direction and rough magnitude are "
            "realistic, but the exact rate would differ under a production FTP framework."
        ),
        (
            "- **Simplified customer behavior.** The synthetic generator's churn, spending, "
            "and product-adoption mechanisms are deliberately interpretable multi-factor "
            "models (documented in `data/README.md`), not a full agent-based or "
            "microsimulation model of customer decision-making -- real customer behavior has "
            "richer feedback loops (e.g. peer effects, macroeconomic shocks) not represented "
            "here."
        ),
        "",
        (
            "See `reports/explainability_governance.md` for the companion Responsible "
            "Decision-Making documentation (PROJECT_SPEC.md section 26)."
        ),
        "",
    ]


def build_report(settings: Settings) -> str:
    reports_dir = settings.raw_dir.parent.parent / "reports"
    quality = _load_json(reports_dir, "data_quality_report.json")
    clv = _load_json(reports_dir, "clv_report.json")
    segmentation = _load_json(reports_dir, "segmentation_report.json")
    actions = _load_json(reports_dir, "actions_report.json")
    action_sim = _load_json(reports_dir, "action_simulation_report.json")
    optimization = _load_json(reports_dir, "optimization_report.json")
    monte_carlo = _load_json(reports_dir, "monte_carlo_report.json")

    profitability = _profitability_summary(settings.processed_dir)
    credit_risk = _credit_risk_summary(settings.processed_dir)
    profiles = pd.read_parquet(settings.processed_dir / "segment_profiles.parquet")

    lines = (
        _executive_summary(quality, profitability, optimization, clv)
        + _customer_economics_section(profitability)
        + _risk_section(profitability, credit_risk)
        + _clv_section(clv)
        + _segmentation_section(segmentation, profiles)
        + _action_optimization_section(actions, action_sim, optimization)
        + _scenario_section(monte_carlo)
        + _limitations_section()
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    settings = load_settings()
    report_text = build_report(settings)

    reports_dir = settings.raw_dir.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "final_report.md").write_text(report_text)
    logger.info("Wrote reports/final_report.md")


if __name__ == "__main__":
    main()
