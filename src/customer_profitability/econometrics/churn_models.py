"""Phase 5.2 -- Churn drivers.

    P(Churn_i=1) = sigma(b0 + b1*RateGap_i + b2*Activity_i + b3*Tenure_i + b4*Profitability_i)

Fit by logistic regression on the customer-month panel. Same-month
regressors are used deliberately (not lagged): the synthetic churn
generating mechanism itself (see data/README.md) evaluates a logistic
hazard from that same month's satisfaction/engagement/profitability before
drawing churn, so this model's job is to *recover* that same-month
association -- it is an explanatory model, not the point-in-time forecaster
Phase 6 builds.

    RateGap        = (market_rate - deposit_rate) x 100, in percentage
                     points (positive = customer is paid below market, i.e.
                     potentially underpriced). Rates in the underlying data
                     are small decimals (~0-5%), so without this rescaling
                     a "one-unit" (100 percentage point) move would be
                     economically meaningless -- exactly the kind of
                     statistically-real-but-uninterpretable effect size
                     PROJECT_SPEC.md 5.3 warns against.
    Activity       = login_frequency (engagement proxy)
    Tenure         = tenure_months
    Profitability  = that month's economic_profit (Phase 4 output)
"""

from __future__ import annotations

import math

import pandas as pd
import statsmodels.api as sm

from customer_profitability.econometrics.diagnostics import add_constant

CHURN_DRIVERS = ["rate_gap", "activity", "tenure", "profitability"]

DRIVER_UNITS = {
    "rate_gap": "1 percentage point of (market_rate - deposit_rate)",
    "activity": "1 login/month",
    "tenure": "1 month of tenure",
    "profitability": "1 EUR of monthly economic profit",
}


def build_churn_dataset(customer_360: pd.DataFrame, profitability_monthly: pd.DataFrame) -> pd.DataFrame:
    df = customer_360[
        ["customer_id", "month", "rate_differential", "login_frequency", "tenure_months", "churned_this_month"]
    ].merge(
        profitability_monthly[["customer_id", "month", "economic_profit"]],
        on=["customer_id", "month"],
        how="inner",
    )
    df = df.rename(
        columns={
            "login_frequency": "activity",
            "tenure_months": "tenure",
            "economic_profit": "profitability",
        }
    )
    df["rate_gap"] = -df["rate_differential"] * 100
    df["churn"] = df["churned_this_month"].astype(int)
    return df.dropna(subset=[*CHURN_DRIVERS, "churn"])[["customer_id", "month", *CHURN_DRIVERS, "churn"]]


def fit_churn_model(df: pd.DataFrame):
    X = add_constant(df[CHURN_DRIVERS])
    y = df["churn"]
    return sm.Logit(y, X).fit(disp=0)


def interpret_churn_model(results) -> list[str]:
    """Plain-language sign/odds-ratio interpretation, one line per driver."""
    lines = []
    for var in CHURN_DRIVERS:
        coef = results.params[var]
        pvalue = results.pvalues[var]
        odds_ratio = math.exp(coef)
        direction = "increases" if coef > 0 else "decreases"
        sig = "statistically significant" if pvalue < 0.05 else "not statistically significant at 5%"
        lines.append(
            f"A one-unit increase in `{var}` ({DRIVER_UNITS[var]}) {direction} the odds of churn "
            f"by a factor of {odds_ratio:.4f} (log-odds coef={coef:+.4f}), holding the other "
            f"drivers fixed; {sig} (p={pvalue:.4g})."
        )
    return lines
