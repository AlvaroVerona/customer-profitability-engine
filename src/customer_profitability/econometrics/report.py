"""Phase 5 orchestrator: fits the revenue and churn econometric models,
runs their diagnostics, and writes a report.

Not a notebook per Development Principle #7 ("keep business logic outside
notebooks; notebooks call functions from src/") -- `notebooks/04_econometrics.ipynb`
is expected to call `revenue_models.py` / `churn_models.py` / `diagnostics.py`
directly for interactive exploration; this module is the reproducible,
scriptable path (`make econometrics`) that regenerates the same report.
"""

from __future__ import annotations

import json

import pandas as pd

from customer_profitability.econometrics.churn_models import (
    CHURN_DRIVERS,
    build_churn_dataset,
    fit_churn_model,
    interpret_churn_model,
)
from customer_profitability.econometrics.diagnostics import (
    add_constant,
    breusch_pagan_test,
    coefficient_table,
    compute_vif,
    flag_economic_significance,
    goodness_of_fit_logit,
    goodness_of_fit_ols,
    residual_normality,
)
from customer_profitability.econometrics.revenue_models import (
    REVENUE_DRIVERS,
    build_revenue_dataset,
    fit_revenue_model,
    interpret_revenue_model,
)
from customer_profitability.utils.config import Settings, load_settings
from customer_profitability.utils.logging import get_logger

logger = get_logger(__name__)

# Effect sizes the analyst considers practically meaningful, used to keep
# "statistically significant" and "economically significant" as separate
# questions (Development Principle #13 / PROJECT_SPEC.md 5.3).
REVENUE_ECONOMIC_THRESHOLDS = {
    "average_balance": 0.01,  # 1 cent of monthly revenue per euro of balance
    "transaction_count": 0.5,  # 0.5 EUR of monthly revenue per transaction
    "product_count": 1.0,  # 1 EUR of monthly revenue per product held
    "income": 0.0005,  # half a cent of monthly revenue per euro of income
}
CHURN_ECONOMIC_THRESHOLDS = {
    "rate_gap": 0.05,  # per 1pp of rate gap; log-odds 0.05 =~ a 5% odds change
    "activity": 0.02,
    "tenure": 0.01,
    "profitability": 0.001,
}


def run_revenue_analysis(customer_360: pd.DataFrame, profitability_monthly: pd.DataFrame) -> dict:
    df = build_revenue_dataset(customer_360, profitability_monthly)
    results = fit_revenue_model(df)
    coefs = flag_economic_significance(coefficient_table(results), REVENUE_ECONOMIC_THRESHOLDS)
    vif = compute_vif(add_constant(df[REVENUE_DRIVERS]))
    return {
        "n_obs": len(df),
        "coefficients": coefs,
        "vif": vif,
        "goodness_of_fit": goodness_of_fit_ols(results),
        "residual_normality": residual_normality(results),
        "breusch_pagan": breusch_pagan_test(results),
        "interpretation": interpret_revenue_model(results),
    }


def run_churn_analysis(customer_360: pd.DataFrame, profitability_monthly: pd.DataFrame) -> dict:
    df = build_churn_dataset(customer_360, profitability_monthly)
    results = fit_churn_model(df)
    coefs = flag_economic_significance(coefficient_table(results), CHURN_ECONOMIC_THRESHOLDS)
    vif = compute_vif(add_constant(df[CHURN_DRIVERS]))
    return {
        "n_obs": len(df),
        "churn_rate": float(df["churn"].mean()),
        "coefficients": coefs,
        "vif": vif,
        "goodness_of_fit": goodness_of_fit_logit(results),
        "interpretation": interpret_churn_model(results),
    }


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal DataFrame -> GitHub-flavoured markdown table, written by hand
    to avoid an extra dependency (`tabulate`) for what `pandas.to_markdown`
    would otherwise need."""
    df = df.reset_index().round(4)
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, separator, *rows])


def to_markdown(revenue: dict, churn: dict) -> str:
    lines = [
        "# Econometric Analysis Report",
        "",
        (
            "Explanatory (same-month) models -- see module docstrings in "
            "`src/customer_profitability/econometrics/` for why these are not "
            "point-in-time forecasters. Statistical significance (p < 0.05) and "
            "economic significance (effect size vs. an analyst-set threshold) "
            "are reported as separate columns; do not conflate them."
        ),
        "",
        "## 5.1 Revenue drivers (OLS)",
        "",
        (
            f"n = {revenue['n_obs']}, R² = {revenue['goodness_of_fit']['r_squared']:.4f}, "
            f"adjusted R² = {revenue['goodness_of_fit']['adj_r_squared']:.4f}"
        ),
        "",
        _df_to_markdown(revenue["coefficients"]),
        "",
        "**VIF (multicollinearity):**",
        "",
        _df_to_markdown(revenue["vif"].to_frame()),
        "",
        (
            f"**Breusch-Pagan heteroskedasticity test:** LM p-value = "
            f"{revenue['breusch_pagan']['lm_pvalue']:.4g} "
            f"({'heteroskedastic' if revenue['breusch_pagan']['heteroskedastic_p05'] else 'homoskedastic'} at 5%) "
            "-- the model is fit with HC3 heteroskedasticity-robust standard errors "
            "for exactly this reason (see `revenue_models.fit_revenue_model` docstring); "
            "the coefficients/p-values above already reflect that correction."
        ),
        "",
        f"**Residual normality (Jarque-Bera):** p-value = {revenue['residual_normality']['jarque_bera_pvalue']:.4g}",
        "",
        "**Interpretation:**",
        "",
        *[f"- {line}" for line in revenue["interpretation"]],
        "",
        "## 5.2 Churn drivers (Logistic regression)",
        "",
        (
            f"n = {churn['n_obs']}, churn rate = {churn['churn_rate']:.4f}, "
            f"McFadden pseudo-R² = {churn['goodness_of_fit']['pseudo_r_squared_mcfadden']:.4f}"
        ),
        "",
        _df_to_markdown(churn["coefficients"]),
        "",
        "**VIF (multicollinearity):**",
        "",
        _df_to_markdown(churn["vif"].to_frame()),
        "",
        "**Interpretation:**",
        "",
        *[f"- {line}" for line in churn["interpretation"]],
        "",
    ]
    return "\n".join(lines) + "\n"


def _read_inputs(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    customer_360 = pd.read_parquet(settings.features_dir / "customer_360.parquet")
    profitability_monthly = pd.read_parquet(settings.processed_dir / "profitability_monthly.parquet")
    return customer_360, profitability_monthly


def main() -> None:
    settings = load_settings()
    customer_360, profitability_monthly = _read_inputs(settings)

    logger.info("Fitting revenue drivers model (OLS)")
    revenue = run_revenue_analysis(customer_360, profitability_monthly)
    logger.info(
        "Revenue model: n=%d R2=%.4f adj_R2=%.4f",
        revenue["n_obs"],
        revenue["goodness_of_fit"]["r_squared"],
        revenue["goodness_of_fit"]["adj_r_squared"],
    )

    logger.info("Fitting churn drivers model (Logistic regression)")
    churn = run_churn_analysis(customer_360, profitability_monthly)
    logger.info(
        "Churn model: n=%d churn_rate=%.4f pseudo_R2=%.4f",
        churn["n_obs"],
        churn["churn_rate"],
        churn["goodness_of_fit"]["pseudo_r_squared_mcfadden"],
    )

    reports_dir = settings.raw_dir.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "econometrics_report.md").write_text(to_markdown(revenue, churn))

    json_payload = {
        "revenue": {
            "n_obs": revenue["n_obs"],
            "goodness_of_fit": revenue["goodness_of_fit"],
            "breusch_pagan": revenue["breusch_pagan"],
            "residual_normality": revenue["residual_normality"],
            "coefficients": json.loads(revenue["coefficients"].to_json(orient="index")),
            "vif": json.loads(revenue["vif"].to_json()),
        },
        "churn": {
            "n_obs": churn["n_obs"],
            "churn_rate": churn["churn_rate"],
            "goodness_of_fit": churn["goodness_of_fit"],
            "coefficients": json.loads(churn["coefficients"].to_json(orient="index")),
            "vif": json.loads(churn["vif"].to_json()),
        },
    }
    (reports_dir / "econometrics_report.json").write_text(json.dumps(json_payload, indent=2))
    logger.info("Wrote reports/econometrics_report.{md,json}")


if __name__ == "__main__":
    main()
