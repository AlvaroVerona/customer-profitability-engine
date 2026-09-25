"""Phase 14 orchestrator: assembles the Explainability & Governance report.

Per PROJECT_SPEC.md 14, this documents *how* every stage's output can be
interpreted (profitability, CLV, ML, optimization) and, per its
"Customer-Level Decision Explanation" mockup, generates worked examples --
using `actions.explanation` and `optimization.explanation` against real
pipeline output, not hypothetical numbers, so the examples in this
document can't drift out of sync with what the pipeline actually produces.
"""

from __future__ import annotations

import json

import pandas as pd

from customer_profitability.actions.explanation import (
    explain_customer_recommendation,
    format_explanation_text,
)
from customer_profitability.optimization.constraints import constraints_from_config
from customer_profitability.optimization.explanation import explain_selection
from customer_profitability.optimization.solver import optimize
from customer_profitability.utils.config import Settings, load_settings
from customer_profitability.utils.logging import get_logger

logger = get_logger(__name__)


def _pick_worked_examples(incremental_value: pd.DataFrame, clv: pd.DataFrame) -> dict[str, str]:
    real_actions = incremental_value[incremental_value["action_type"] != "NO_ACTION"]
    best_per_customer = real_actions.groupby("customer_id")["incremental_profit"].max()

    high_value_id = best_per_customer.idxmax()

    no_action_ids = best_per_customer[best_per_customer <= 0]
    no_action_id = no_action_ids.index[0] if len(no_action_ids) else None

    active_ids = clv.loc[clv["is_active"], "customer_id"]
    typical_pool = best_per_customer[
        (best_per_customer > 0) & (best_per_customer < best_per_customer.median() * 1.5) & best_per_customer.index.isin(active_ids)
    ]
    typical_id = typical_pool.index[len(typical_pool) // 2] if len(typical_pool) else high_value_id

    return {"high_value": high_value_id, "typical": typical_id, "no_action": no_action_id}


def _churn_shap_lines(ml_report: dict) -> list[str]:
    importance = ml_report["explainability"]["churn"].get("global_importance", {})
    champion = ml_report["explainability"]["churn"].get("champion", "the champion model")
    lines = [f"Top SHAP drivers for the {champion} churn model (mean |SHAP|, portfolio-wide):", ""]
    for feature, value in list(importance.items())[:5]:
        lines.append(f"- `{feature}`: {value:.4f}")
    return lines


def build_report(settings: Settings) -> str:
    incremental_value = pd.read_parquet(settings.processed_dir / "action_incremental_value.parquet")
    clv = pd.read_parquet(settings.processed_dir / "clv.parquet")
    reports_dir = settings.raw_dir.parent.parent / "reports"
    ml_report = json.loads((reports_dir / "ml_report.json").read_text())

    examples = _pick_worked_examples(incremental_value, clv)
    constraints = constraints_from_config(settings.optimization)
    optimization_result = optimize(incremental_value, constraints)

    high_value_explanation = explain_customer_recommendation(examples["high_value"], incremental_value, clv, settings)
    typical_explanation = explain_customer_recommendation(examples["typical"], incremental_value, clv, settings)
    high_value_selection = explain_selection(examples["high_value"], optimization_result)
    typical_selection = explain_selection(examples["typical"], optimization_result)

    lines = [
        "# Explainability & Governance",
        "",
        (
            "PROJECT_SPEC.md Phase 14: how every stage's output can be interpreted, and a worked "
            "customer-level decision explanation for a real customer from the current pipeline run -- "
            "not a hypothetical example, so this section cannot silently drift out of sync with the code."
        ),
        "",
        "## How economic profit is calculated (Profitability, Phase 4)",
        "",
        (
            "Every customer-month's economic profit follows one waterfall (`profitability/engine.py`), "
            "traceable back to the raw transactional data at each step:"
        ),
        "",
        "```",
        "gross_contribution         = total_revenue + deposit_contribution - operating_cost",
        "risk_adjusted_contribution = gross_contribution - expected_loss",
        "economic_profit             = risk_adjusted_contribution - acquisition_cost",
        "```",
        "",
        (
            "- `total_revenue` = interest revenue (`AvgLoanBalance x LoanRate`) + interchange revenue "
            "(`CardVolume x InterchangeRate`) + two documented synthetic fee streams (low-balance fee, "
            "Premium subscription) -- see `profitability/revenue.py`."
        ),
        (
            "- `deposit_contribution` = `AvgDepositBalance x (FTPRate - CustomerDepositRate) / 12` -- "
            "positive when the bank's internal funding value exceeds what it pays the customer "
            "(`profitability/ftp.py`'s sign convention)."
        ),
        "- `expected_loss` = `PD x LGD x EAD`, summed over every loan the customer holds that month.",
        (
            "- `operating_cost` = support cost (`contacts x cost_per_contact`) + card payment-processing "
            "cost + a flat per-account servicing cost."
        ),
        (
            "- `acquisition_cost` is recognized exactly once, in the customer's acquisition month, and "
            "only if that month falls inside the observed data window (a back-book customer's CAC was "
            "already spent in the past, so it is not re-charged)."
        ),
        "",
        "Every one of these formulas has a dedicated unit test with a hand-calculated example (`tests/test_profitability.py`, `tests/test_ftp.py`, `tests/test_risk_cost.py`).",
        "",
        "## How future value is forecast (CLV, Phase 7)",
        "",
        (
            "`CLV_i = HistoricalEconomicProfit_i + sum_t [P(Survival_t) x ExpectedProfit_t] / (1+r)^t` "
            "(`clv/calculator.py`), kept as three separate columns -- `historical_economic_profit`, "
            "`future_clv`, `total_customer_economic_value` -- so a reader is never left guessing which "
            "one a plain \"CLV\" figure means."
        ),
        (
            "- `P(Survival_t)` comes from Phase 6's *calibrated* churn model, evaluated once on the "
            "customer's latest snapshot and held as a **constant monthly hazard** over the horizon -- a "
            "documented simplification, not a month-by-month re-forecast of the customer's evolving state."
        ),
        (
            "- `ExpectedProfit` is the customer's own trailing 3-month run-rate across the Phase 4 "
            "waterfall, held **flat** over the horizon -- consistent with the flat-hazard assumption above."
        ),
        (
            "- Phase 7.5's Monte Carlo band (`clv_p5`/`clv_p50`/`clv_p95`) exists specifically so CLV is "
            "never presented as one exact number: it simulates stochastic churn timing (geometric "
            "survival draw) and profit-level uncertainty (log-normal multiplier, sized by each "
            "customer's own historical profit volatility)."
        ),
        "",
        "## Why a model produces a particular prediction (ML, Phase 6)",
        "",
        *_churn_shap_lines(ml_report),
        "",
        (
            "Every classification model's raw probabilities are additionally **calibrated** (Platt "
            "scaling, `models/evaluation.calibrate_probabilities`) before being used downstream, because "
            "the class-rebalancing needed for usable ranking on a rare event (~2%/month churn) distorts "
            "`predict_proba` away from true probabilities -- a raw-vs-calibrated Brier score comparison "
            "is in `reports/ml_report.md`."
        ),
        (
            "Per-customer SHAP explanations (not just the global importance above) are available via "
            "`models.evaluation.explain_customer`, applied identically for any of the eight Phase 6 "
            "targets' XGBoost/random-forest champions."
        ),
        "",
        "## Why a particular customer/action combination was selected (Optimization, Phase 11)",
        "",
        (
            f"Solver status on the current run: **{optimization_result['status']}**, "
            f"{optimization_result['n_selected']:,} of {optimization_result['n_candidates']:,} candidates selected."
        ),
        "",
        "Two worked examples (`optimization/explanation.py`, a rank-based approximation -- see its module docstring for why this is not a literal trace of the CP-SAT solver's internal reasoning):",
        "",
        f"- `{examples['high_value']}`: {high_value_selection.reason}",
        f"- `{examples['typical']}`: {typical_selection.reason}",
        "",
        "## Customer-Level Decision Explanations (worked examples)",
        "",
        "### A high-value customer",
        "",
        "```",
        format_explanation_text(high_value_explanation),
        "```",
        "",
        "### A typical (unremarkable) customer",
        "",
        "```",
        format_explanation_text(typical_explanation),
        "```",
        "",
    ]

    if examples["no_action"]:
        no_action_explanation = explain_customer_recommendation(examples["no_action"], incremental_value, clv, settings)
        lines += [
            "### A customer for whom no action is recommended",
            "",
            "```",
            format_explanation_text(no_action_explanation),
            "```",
            "",
        ]

    lines += [
        "## Responsible decision-making (PROJECT_SPEC.md section 26)",
        "",
        "- The entire dataset is synthetic (`data/README.md`); no real customer, transaction, or credit data is used anywhere in this project.",
        "- Every profitability, CLV, and action-value figure is model-based and carries the assumptions documented in this report and in each phase's own module docstrings -- none of it is a verified real-world business outcome.",
        "- ML predictions carry quantified uncertainty (calibration, Monte Carlo bands); Phase 6/7 explicitly avoid presenting a point estimate as if it were certain.",
        "- The optimization engine's output is a decision-support ranking under Phase 10's simulated effects, not a causal guarantee -- `optimization/solver.py`'s own docstring says so explicitly, and repeats it in every generated report.",
        "- No sensitive personal characteristic (e.g. protected demographic attributes) is used as an eligibility or treatment criterion anywhere in `actions/definitions.py`; eligibility rules are behavioral/financial (product ownership, income, employment status for credit risk).",
        "- Any real-world deployment of a system like this would need additional regulatory, fair-lending/fair-treatment, legal, and model-risk-management controls well beyond what this portfolio project implements.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    settings = load_settings()
    report_text = build_report(settings)

    reports_dir = settings.raw_dir.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "explainability_governance.md").write_text(report_text)
    logger.info("Wrote reports/explainability_governance.md")


if __name__ == "__main__":
    main()
