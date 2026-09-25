"""Phase 15 -- Model Validation.

Consolidates the two things PROJECT_SPEC.md's "PHASE 15" section asks for,
across all eight Phase 6 targets in one place, using the metrics already
computed and persisted by `models.report` (`reports/ml_report.json`) -- no
model is refit here:

1. **Baseline comparison**: how much (if at all) a tree ensemble improves
   on the simple linear/logistic-regression baseline on the held-out test
   set. This is reported honestly in both directions -- three of the eight
   targets in this project are in fact won by the baseline, which is a
   real finding about this dataset, not a result to paper over.
2. **Time-based validation**: the actual calendar-month train/validation/
   test ranges used for each target (captured by `models.report`'s
   `_split_summary`), since each target dataset drops a different set of
   rows and therefore does not necessarily span identical months.
"""

from __future__ import annotations

import json

import pandas as pd

from customer_profitability.utils.config import Settings, load_settings
from customer_profitability.utils.logging import get_logger

logger = get_logger(__name__)

# (metric, higher_is_better, baseline_model_name) -- mirrors the champion
# metric each target is actually selected on in `models.report`.
_CLASSIFICATION = ("roc_auc", True, "logistic_regression")
_REGRESSION = ("rmse", False, "linear_regression")

TARGET_METRIC = {
    "churn": _CLASSIFICATION,
    "revenue": _REGRESSION,
    "deposit_balance": _REGRESSION,
    "loan_balance": _REGRESSION,
    "transaction_volume": _REGRESSION,
    "adoption_savings_account": _CLASSIFICATION,
    "adoption_consumer_loan": _CLASSIFICATION,
    "adoption_investment_account": _CLASSIFICATION,
}


def _pct_improvement(baseline_value: float, champion_value: float, higher_is_better: bool) -> float:
    if baseline_value == 0:
        return float("nan")
    if higher_is_better:
        return 100.0 * (champion_value - baseline_value) / abs(baseline_value)
    return 100.0 * (baseline_value - champion_value) / abs(baseline_value)


def champion_vs_baseline_table(ml_report: dict) -> pd.DataFrame:
    rows = []
    for target, (metric, higher_is_better, baseline_name) in TARGET_METRIC.items():
        result = ml_report[target]
        champion_name = result["champion"]
        baseline_value = result["models"][baseline_name]["test_metrics"][metric]
        champion_value = result["models"][champion_name]["test_metrics"][metric]
        improvement = _pct_improvement(baseline_value, champion_value, higher_is_better)
        rows.append(
            {
                "target": target,
                "metric": metric,
                "baseline_model": baseline_name,
                "baseline_value": baseline_value,
                "champion_model": champion_name,
                "champion_value": champion_value,
                "pct_improvement_over_baseline": improvement,
                "champion_beats_baseline": champion_name != baseline_name,
            }
        )
    return pd.DataFrame(rows)


def _uplift_table_markdown(table: pd.DataFrame) -> str:
    lines = [
        "| target | metric | baseline (logistic/linear) | champion | champion value | improvement vs. baseline |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in table.iterrows():
        improvement = row["pct_improvement_over_baseline"]
        improvement_str = f"{improvement:+.1f}%" if pd.notna(improvement) else "n/a"
        champion_label = row["champion_model"] if row["champion_beats_baseline"] else f"{row['champion_model']} (= baseline)"
        lines.append(
            f"| {row['target']} | {row['metric']} | {row['baseline_value']:.4g} | {champion_label} | "
            f"{row['champion_value']:.4g} | {improvement_str} |"
        )
    return "\n".join(lines)


def _split_cell(split: dict) -> str:
    return f"{split['start']}..{split['end']} (n={split['n']:,})"


def _split_summary_markdown(ml_report: dict) -> str:
    lines = ["| target | train | validation | test |", "|---|---|---|---|"]
    for target in TARGET_METRIC:
        split = ml_report[target]["split"]
        lines.append(
            f"| {target} | {_split_cell(split['train'])} | {_split_cell(split['validation'])} | {_split_cell(split['test'])} |"
        )
    return "\n".join(lines)


def build_report(ml_report: dict) -> str:
    table = champion_vs_baseline_table(ml_report)
    n_baseline_wins = int((~table["champion_beats_baseline"]).sum())

    lines = [
        "# Model Validation Report",
        "",
        (
            "PROJECT_SPEC.md PHASE 15: for every Phase 6 target, compare the sophisticated "
            "models (random forest, XGBoost) against a simple linear/logistic-regression "
            "baseline on a held-out, *future* time slice (never a random split, since every "
            "target here is a forecast of month t+1) -- reusing the metrics already computed "
            "by `make models` (`reports/ml_report.json`), not refitting anything."
        ),
        "",
        "## Baseline vs. champion (test set)",
        "",
        _uplift_table_markdown(table),
        "",
        (
            f"The baseline (linear/logistic regression) is itself the champion on "
            f"{n_baseline_wins} of {len(table)} targets. This is a genuine finding about "
            "this dataset, not an oversight: `loan_balance`, `adoption_consumer_loan`, and "
            "`adoption_investment_account` all have close-to-linear, low-noise generating "
            "mechanisms in the synthetic data generator, so a linear model's extra bias "
            "(relative to a tree ensemble's flexibility) buys nothing, and its lower "
            "variance wins on held-out data. This is exactly why PROJECT_SPEC.md's modeling "
            "philosophy (section 25) ranks predictive performance below economic coherence "
            "and interpretability -- a simpler, equally accurate model is preferred, not "
            "treated as a consolation prize."
        ),
        "",
        "## Time-based validation: actual split ranges used per target",
        "",
        (
            "Train/validation/test are split by calendar month (never by row), and the "
            "split point is computed independently per target dataset because each target "
            "drops a different subset of customer-months (e.g. product-adoption datasets "
            "only keep rows where the customer doesn't yet own the product) -- see "
            "`models.evaluation.time_based_split` and `models.report._split_summary`. "
            f"Configured split sizes: test = last {ml_report['_config']['test_months']} months, "
            f"validation = the {ml_report['_config']['validation_months']} months before that."
        ),
        "",
        _split_summary_markdown(ml_report),
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    settings: Settings = load_settings()
    reports_dir = settings.raw_dir.parent.parent / "reports"
    ml_report = json.loads((reports_dir / "ml_report.json").read_text())
    ml_report["_config"] = {
        "test_months": settings.models.test_months,
        "validation_months": settings.models.validation_months,
    }

    report_text = build_report(ml_report)
    (reports_dir / "model_validation_report.md").write_text(report_text)
    logger.info("Wrote reports/model_validation_report.md")


if __name__ == "__main__":
    main()
