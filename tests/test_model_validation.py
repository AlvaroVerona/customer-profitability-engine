"""Phase 15 -- model validation report tests: baseline-vs-champion uplift
direction handling and split-summary rendering, against a hand-built
ml_report.json-shaped fixture (no model fitting -- validation.py is pure
post-processing of already-computed metrics)."""

from __future__ import annotations

from customer_profitability.models.validation import (
    TARGET_METRIC,
    _pct_improvement,
    build_report,
    champion_vs_baseline_table,
)


def _split(n_train: int = 100, n_val: int = 20, n_test: int = 30) -> dict:
    return {
        "train": {"n": n_train, "start": "2023-01", "end": "2024-06"},
        "validation": {"n": n_val, "start": "2024-07", "end": "2024-09"},
        "test": {"n": n_test, "start": "2024-10", "end": "2024-12"},
    }


def _target_result(champion: str, baseline_name: str, metric: str, baseline_value: float, champion_value: float) -> dict:
    models = {baseline_name: {"test_metrics": {metric: baseline_value}}}
    models[champion] = {"test_metrics": {metric: champion_value}}
    return {"champion": champion, "split": _split(), "models": models}


def _full_ml_report(overrides: dict | None = None) -> dict:
    """A minimal but complete ml_report dict: every key TARGET_METRIC expects,
    each with its baseline as its own champion (0% improvement) unless overridden."""
    report = {}
    for target, (metric, higher_is_better, baseline_name) in TARGET_METRIC.items():
        base_value = 0.8 if higher_is_better else 100.0
        report[target] = _target_result(baseline_name, baseline_name, metric, base_value, base_value)
    if overrides:
        report.update(overrides)
    report["_config"] = {"test_months": 6, "validation_months": 3}
    return report


def test_pct_improvement_higher_is_better() -> None:
    assert _pct_improvement(100.0, 110.0, higher_is_better=True) == 10.0
    assert _pct_improvement(100.0, 90.0, higher_is_better=True) == -10.0


def test_pct_improvement_lower_is_better() -> None:
    # rmse: champion lower than baseline is an improvement -> positive.
    assert _pct_improvement(100.0, 90.0, higher_is_better=False) == 10.0
    assert _pct_improvement(100.0, 110.0, higher_is_better=False) == -10.0


def test_pct_improvement_zero_baseline_is_nan() -> None:
    import math

    assert math.isnan(_pct_improvement(0.0, 5.0, higher_is_better=True))


def test_champion_vs_baseline_table_classification_champion_wins() -> None:
    report = _full_ml_report(
        overrides={"churn": _target_result("random_forest", "logistic_regression", "roc_auc", 0.70, 0.75)}
    )
    table = champion_vs_baseline_table(report)
    row = table[table["target"] == "churn"].iloc[0]

    assert row["champion_model"] == "random_forest"
    assert row["champion_beats_baseline"]
    assert row["pct_improvement_over_baseline"] > 0


def test_champion_vs_baseline_table_regression_baseline_wins() -> None:
    # linear_regression is its own champion: no lift, flagged as baseline win.
    report = _full_ml_report()
    table = champion_vs_baseline_table(report)
    row = table[table["target"] == "loan_balance"].iloc[0]

    assert row["champion_model"] == "linear_regression"
    assert not row["champion_beats_baseline"]
    assert row["pct_improvement_over_baseline"] == 0.0


def test_champion_vs_baseline_table_covers_every_target() -> None:
    report = _full_ml_report()
    table = champion_vs_baseline_table(report)
    assert set(table["target"]) == set(TARGET_METRIC)


def test_build_report_renders_all_targets_and_split_ranges() -> None:
    report = _full_ml_report(
        overrides={"revenue": _target_result("xgboost", "linear_regression", "rmse", 50.0, 40.0)}
    )
    text = build_report(report)

    assert "Model Validation Report" in text
    assert "Baseline vs. champion" in text
    for target in TARGET_METRIC:
        assert target in text
    assert "2023-01..2024-06" in text  # train range from the fixture
    assert "+20.0%" in text  # revenue's hand-set rmse improvement: (50-40)/50 * 100
