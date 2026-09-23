"""Phase 5.1 -- Revenue drivers.

    Revenue_{i,t} = b0 + b1*Balance_{i,t} + b2*Transactions_{i,t}
                  + b3*Products_{i,t} + b4*Income_i + e_{i,t}

Fit by OLS on the customer-month panel (Customer 360 features x Phase 4
`total_revenue`). This is an *explanatory* model (Phase 5's job is
understanding relationships, not maximizing predictive accuracy -- see
PROJECT_SPEC.md section 25), so it deliberately uses the same-month
regressors rather than lagged ones; Phase 6's ML models are the ones that
have to be point-in-time-correct for forecasting.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from customer_profitability.econometrics.diagnostics import add_constant

REVENUE_DRIVERS = ["average_balance", "transaction_count", "product_count", "income"]


def build_revenue_dataset(customer_360: pd.DataFrame, profitability_monthly: pd.DataFrame) -> pd.DataFrame:
    df = customer_360[["customer_id", "month", *REVENUE_DRIVERS]].merge(
        profitability_monthly[["customer_id", "month", "total_revenue"]],
        on=["customer_id", "month"],
        how="inner",
    )
    return df.dropna(subset=[*REVENUE_DRIVERS, "total_revenue"])


def fit_revenue_model(df: pd.DataFrame):
    """Fit with HC3 heteroskedasticity-robust standard errors rather than
    the OLS default. Financial revenue data is essentially never
    homoskedastic (higher-balance customers have higher-variance revenue),
    which `diagnostics.breusch_pagan_test` confirms on this data -- so using
    the non-robust default would understate standard errors and overstate
    significance. Point estimates (the coefficients themselves) are
    unaffected; only their standard errors, t-stats, p-values, and
    confidence intervals change.
    """
    X = add_constant(df[REVENUE_DRIVERS])
    y = df["total_revenue"]
    return sm.OLS(y, X).fit(cov_type="HC3")


def interpret_revenue_model(results) -> list[str]:
    """Plain-language sign/magnitude interpretation, one line per driver
    (excludes the constant, which has no standalone economic meaning)."""
    lines = []
    for var in REVENUE_DRIVERS:
        coef = results.params[var]
        pvalue = results.pvalues[var]
        direction = "increases" if coef > 0 else "decreases"
        sig = "statistically significant" if pvalue < 0.05 else "not statistically significant at 5%"
        lines.append(
            f"A one-unit increase in `{var}` is associated with a {coef:+.4f} change in monthly "
            f"revenue ({direction} revenue), holding the other drivers fixed; {sig} (p={pvalue:.4g})."
        )
    return lines
