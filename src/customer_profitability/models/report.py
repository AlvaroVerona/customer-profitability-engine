"""Phase 6 orchestrator: builds every model's dataset, fits baseline + both
tree ensembles on a time-based split, and writes a comparison report.

Per Phase 15 ("time-based validation... adjust based on the generated
dataset"), the train/validation/test split is computed *per target
dataset* (not once globally) because each target drops a different set of
rows (e.g. product-adoption datasets only keep customer-months where the
customer doesn't yet own the product), so the set of calendar months
actually present can differ slightly between targets.

SHAP explainability (Phase 6.5) is computed in full (global importance +
one individual example) for the two headline models -- churn and revenue,
which is what Phase 7's CLV and Phase 9's action framework consume
directly -- and is a documented scope decision, not an oversight: the same
`evaluation.global_feature_importance` / `explain_customer` helpers apply
identically to the balance and product-adoption models' XGBoost champions.
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
import shap

from customer_profitability.models import balance, churn, product_adoption, revenue
from customer_profitability.models.evaluation import (
    explain_customer,
    fit_and_evaluate_classification,
    global_feature_importance,
    time_based_split,
)
from customer_profitability.utils.config import Settings, load_settings
from customer_profitability.utils.logging import get_logger

logger = get_logger(__name__)

SHAP_SAMPLE_SIZE = 1000


def _best_model_name(models: dict, metric: str, higher_is_better: bool) -> str:
    scored = {name: res["test_metrics"][metric] for name, res in models.items()}
    return max(scored, key=scored.get) if higher_is_better else min(scored, key=scored.get)


def _split_summary(df: pd.DataFrame, train: pd.Series, val: pd.Series, test: pd.Series, month_col: str = "month") -> dict:
    """Per Phase 15's time-based validation: the actual calendar-month range
    and row count of each split, for the dataset actually used -- not the
    spec's illustrative "Months 1-18/19-21/22-24", which assumes every
    target dataset spans the same months (it doesn't; see module docstring)."""
    months = pd.to_datetime(df[month_col])

    def _range(mask: pd.Series) -> dict:
        subset = months[mask]
        return {
            "n": int(mask.sum()),
            "start": subset.min().strftime("%Y-%m") if len(subset) else None,
            "end": subset.max().strftime("%Y-%m") if len(subset) else None,
        }

    return {"train": _range(train), "validation": _range(val), "test": _range(test)}


def _run_classification_target(name: str, df: pd.DataFrame, settings: Settings) -> dict:
    train, val, test = time_based_split(df, settings.models)
    logger.info(
        "%s: n=%d (train=%d val=%d test=%d), positive rate=%.4f",
        name,
        len(df),
        train.sum(),
        val.sum(),
        test.sum(),
        df["target"].mean(),
    )
    out = fit_and_evaluate_classification(df, train, val, test)
    champion = _best_model_name(out["models"], "roc_auc", higher_is_better=True)
    return {"name": name, "champion": champion, "split": _split_summary(df, train, val, test), **out}


def _run_regression_target(name: str, df: pd.DataFrame, settings: Settings, fit_fn) -> dict:
    train, val, test = time_based_split(df, settings.models)
    logger.info("%s: n=%d (train=%d val=%d test=%d)", name, len(df), train.sum(), val.sum(), test.sum())
    out = fit_fn(df, train, val, test)
    champion = _best_model_name(out["models"], "rmse", higher_is_better=False)
    return {"name": name, "champion": champion, "split": _split_summary(df, train, val, test), **out}


def run_all(settings: Settings) -> dict:
    customer_360 = pd.read_parquet(settings.features_dir / "customer_360.parquet")
    profitability_monthly = pd.read_parquet(settings.processed_dir / "profitability_monthly.parquet")
    accounts = pd.read_parquet(settings.processed_dir / "accounts_validated.parquet")

    results = {}

    results["churn"] = _run_classification_target("churn", churn.build_churn_dataset(customer_360), settings)

    results["revenue"] = _run_regression_target(
        "revenue",
        revenue.build_revenue_dataset(customer_360, profitability_monthly),
        settings,
        revenue.fit_and_evaluate,
    )

    for target_name in balance.BALANCE_TARGETS:
        df = balance.build_balance_dataset(customer_360, target_name)
        results[target_name] = _run_regression_target(target_name, df, settings, balance.fit_and_evaluate)

    for product in product_adoption.ADOPTION_PRODUCTS:
        df = product_adoption.build_adoption_dataset(customer_360, accounts, product)
        key = f"adoption_{product}"
        results[key] = _run_classification_target(key, df, settings)

    return results


_TREE_MODELS = {"random_forest", "xgboost"}


def _explain(result: dict) -> dict:
    champion_name = result["champion"]
    if champion_name not in _TREE_MODELS:
        return {"note": f"champion model ({champion_name}) is not tree-based; SHAP (TreeExplainer) skipped for this target."}
    model = result["models"][champion_name]["model"]
    X_test = result["X_test"]
    sample = X_test.sample(n=min(SHAP_SAMPLE_SIZE, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):  # some sklearn versions: [neg_class, pos_class]
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:  # some sklearn versions: (n_samples, n_features, n_classes)
        shap_values = shap_values[:, :, 1]
    importance = global_feature_importance(shap_values, list(sample.columns))
    example = explain_customer(shap_values, sample.reset_index(drop=True), 0)
    return {
        "champion": champion_name,
        "global_importance": importance.head(10).to_dict(),
        "example_customer_explanation": example.head(10).to_dict(),
    }


def _calibration_section(result: dict) -> list[str]:
    champion = result["champion"]
    m = result["models"][champion]
    raw_brier = m["test_metrics"]["brier_score"]
    cal_brier = m["test_metrics_calibrated"]["brier_score"]
    lines = [
        (
            f"**Calibration ({champion}, test set):** raw Brier score = {raw_brier:.4f}, "
            f"post-hoc Platt-scaled Brier score = {cal_brier:.4f}. Class rebalancing "
            "(needed for usable ranking on a rare event) distorts predict_proba away "
            "from true probabilities; the calibrated version, not the raw model, is "
            "what should feed Phase 7's CLV survival term. Deciles of predicted "
            "probability vs. actual event rate, calibrated model:"
        ),
        "",
        m["test_calibration_calibrated"].round(4).to_string(index=False),
    ]
    return lines


def _metrics_markdown(results: dict, target: str) -> str:
    result = results[target]
    lines = [f"### {target} (champion: {result['champion']})", ""]
    metric_keys = list(next(iter(result["models"].values()))["test_metrics"].keys())
    lines.append("| model | " + " | ".join(metric_keys) + " |")
    lines.append("|---|" + "---|" * len(metric_keys))
    for model_name, res in result["models"].items():
        m = res["test_metrics"]
        lines.append(f"| {model_name} | " + " | ".join(f"{m[k]:.4g}" for k in metric_keys) + " |")
    return "\n".join(lines)


def _explanation_markdown(explanation: dict) -> str:
    if "note" in explanation:
        return explanation["note"]
    return "\n".join(f"- `{k}`: {v:.4f}" for k, v in explanation["global_importance"].items())


def to_markdown(results: dict, explanations: dict) -> str:
    lines = [
        "# Machine Learning Model Report",
        "",
        (
            "All targets use a time-based train/validation/test split (see "
            "`models.evaluation.time_based_split`) and predict month t+1 "
            "from features observed as of month t -- no target uses "
            "same-month information about itself."
        ),
        "",
        "## 6.1 Churn",
        "",
        _metrics_markdown(results, "churn"),
        "",
        *_calibration_section(results["churn"]),
        "",
        "## 6.2 Future revenue",
        "",
        _metrics_markdown(results, "revenue"),
        "",
        "## 6.3 Future balances / activity",
        "",
        _metrics_markdown(results, "deposit_balance"),
        "",
        (
            "Note: linear regression's negative R^2 on deposit balance is a "
            "real finding, not a bug -- monetary balances are heavy-tailed, "
            "and a handful of high-balance customers dominate squared error "
            "for a model that can't flex to outliers the way a tree ensemble can."
        ),
        "",
        _metrics_markdown(results, "loan_balance"),
        "",
        _metrics_markdown(results, "transaction_volume"),
        "",
        "## 6.4 Product adoption",
        "",
        _metrics_markdown(results, "adoption_savings_account"),
        "",
        _metrics_markdown(results, "adoption_consumer_loan"),
        "",
        _metrics_markdown(results, "adoption_investment_account"),
        "",
        "## 6.5 Explainability (SHAP)",
        "",
        "### Churn -- global feature importance (mean |SHAP|, top 10)",
        "",
        _explanation_markdown(explanations["churn"]),
        "",
        "### Revenue -- global feature importance (mean |SHAP|, top 10)",
        "",
        _explanation_markdown(explanations["revenue"]),
        "",
    ]
    return "\n".join(lines) + "\n"


def _json_safe(results: dict, explanations: dict) -> dict:
    out = {}
    for name, result in results.items():
        models_out = {}
        for model_name, res in result["models"].items():
            entry = {"test_metrics": res["test_metrics"], "validation_metrics": res["validation_metrics"]}
            if "test_metrics_calibrated" in res:
                entry["test_metrics_calibrated"] = res["test_metrics_calibrated"]
            models_out[model_name] = entry
        out[name] = {
            "champion": result["champion"],
            "n_features": len(result["feature_columns"]),
            "split": result["split"],
            "models": models_out,
        }
    out["explainability"] = explanations
    return out


# Champion models persisted for reuse as production forecasters in later
# phases (Phase 7 CLV needs the calibrated churn model specifically --
# PROJECT_SPEC.md 6.1 is explicit that its probabilities feed the CLV
# survival term). Churn is saved *calibrated* (see evaluation.calibrate_probabilities);
# the regression targets don't have that concept, so their raw champion is saved.
_PERSISTED_TARGETS = ["churn", "revenue", "deposit_balance", "loan_balance"]


def _persist_champion_models(results: dict, models_dir) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for target in _PERSISTED_TARGETS:
        result = results[target]
        champion = result["champion"]
        model_entry = result["models"][champion]
        model = model_entry.get("calibrated_model", model_entry["model"])
        joblib.dump(model, models_dir / f"{target}_champion.joblib")
        (models_dir / f"{target}_feature_columns.json").write_text(json.dumps(result["feature_columns"]))
    logger.info("Persisted champion models for %s to %s", _PERSISTED_TARGETS, models_dir)


def main() -> None:
    settings = load_settings()
    results = run_all(settings)

    logger.info("Computing SHAP explainability for churn and revenue champions")
    explanations = {"churn": _explain(results["churn"]), "revenue": _explain(results["revenue"])}

    reports_dir = settings.raw_dir.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "ml_report.md").write_text(to_markdown(results, explanations))
    (reports_dir / "ml_report.json").write_text(json.dumps(_json_safe(results, explanations), indent=2, default=str))
    logger.info("Wrote reports/ml_report.{md,json}")

    _persist_champion_models(results, settings.processed_dir / "models")


if __name__ == "__main__":
    main()
