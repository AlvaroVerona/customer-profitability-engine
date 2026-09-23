"""Phase 5 -- Econometric analysis tests.

Not in PROJECT_SPEC.md's explicit test list (same rationale as
test_features.py: every important calculation gets a test). Uses simulated
data with a *known* true relationship so model recovery can be checked
against ground truth, not just "runs without error".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from customer_profitability.econometrics.churn_models import (
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
)
from customer_profitability.econometrics.revenue_models import (
    build_revenue_dataset,
    fit_revenue_model,
    interpret_revenue_model,
)

RNG = np.random.default_rng(42)


def test_revenue_model_recovers_known_linear_relationship() -> None:
    n = 5000
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(n)],
            "month": pd.Timestamp("2023-01-01"),
            "average_balance": RNG.uniform(0, 10000, n),
            "transaction_count": RNG.uniform(0, 50, n),
            "product_count": RNG.integers(1, 5, n),
            "income": RNG.uniform(10000, 100000, n),
        }
    )
    true_coefs = {"average_balance": 0.01, "transaction_count": 2.0, "product_count": 5.0, "income": 0.001}
    noise = RNG.normal(0, 1.0, n)
    df["total_revenue"] = (
        20.0
        + true_coefs["average_balance"] * df["average_balance"]
        + true_coefs["transaction_count"] * df["transaction_count"]
        + true_coefs["product_count"] * df["product_count"]
        + true_coefs["income"] * df["income"]
        + noise
    )
    results = fit_revenue_model(df)
    for var, true_coef in true_coefs.items():
        assert results.params[var] == pytest.approx(true_coef, abs=0.01)
    assert results.rsquared > 0.99  # near-noiseless by construction


def test_build_revenue_dataset_merges_and_drops_missing() -> None:
    customer_360 = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "month": pd.to_datetime(["2023-01-01", "2023-01-01"]),
            "average_balance": [100.0, np.nan],
            "transaction_count": [5, 3],
            "product_count": [2, 1],
            "income": [30000.0, 40000.0],
        }
    )
    profitability = pd.DataFrame(
        {"customer_id": ["C1", "C2"], "month": pd.to_datetime(["2023-01-01", "2023-01-01"]), "total_revenue": [50.0, 20.0]}
    )
    out = build_revenue_dataset(customer_360, profitability)
    assert len(out) == 1
    assert out.iloc[0]["customer_id"] == "C1"


def test_churn_model_recovers_expected_signs() -> None:
    n = 20000
    rate_gap = RNG.uniform(0, 3, n)
    activity = RNG.uniform(0, 30, n)
    tenure = RNG.uniform(0, 60, n)
    profitability = RNG.uniform(-50, 200, n)
    logit = -3.0 + 0.4 * rate_gap - 0.05 * activity - 0.02 * tenure - 0.005 * profitability
    p = 1 / (1 + np.exp(-logit))
    churn = (RNG.random(n) < p).astype(int)
    df = pd.DataFrame(
        {"rate_gap": rate_gap, "activity": activity, "tenure": tenure, "profitability": profitability, "churn": churn}
    )
    results = fit_churn_model(df)
    assert results.params["rate_gap"] > 0
    assert results.params["activity"] < 0
    assert results.params["tenure"] < 0
    assert results.params["profitability"] < 0
    # Recovers the true coefficients reasonably well at this sample size.
    assert results.params["rate_gap"] == pytest.approx(0.4, abs=0.05)


def test_build_churn_dataset_rate_gap_sign_and_scale() -> None:
    customer_360 = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "month": pd.to_datetime(["2023-01-01"]),
            "rate_differential": [-0.02],  # deposit_rate 2pp below market
            "login_frequency": [10.0],
            "tenure_months": [12],
            "churned_this_month": [False],
        }
    )
    profitability = pd.DataFrame(
        {"customer_id": ["C1"], "month": pd.to_datetime(["2023-01-01"]), "economic_profit": [100.0]}
    )
    out = build_churn_dataset(customer_360, profitability)
    # rate_differential=-0.02 -> rate_gap should be +2.0 percentage points, not +0.02.
    assert out.iloc[0]["rate_gap"] == pytest.approx(2.0)


def test_coefficient_table_columns() -> None:
    X = add_constant(pd.DataFrame({"x": RNG.normal(0, 1, 200)}))
    y = 3.0 + 2.0 * X["x"] + RNG.normal(0, 0.1, 200)
    results = sm.OLS(y, X).fit()
    table = coefficient_table(results)
    assert set(table.columns) == {"coef", "std_err", "statistic", "p_value", "ci_lower", "ci_upper", "significant_p05"}
    assert table.loc["x", "significant_p05"]


def test_flag_economic_significance_separates_from_statistical() -> None:
    X = add_constant(pd.DataFrame({"x": RNG.normal(0, 1, 100000)}))
    y = 3.0 + 0.001 * X["x"] + RNG.normal(0, 0.01, 100000)  # tiny but very precisely estimated effect
    results = sm.OLS(y, X).fit()
    table = coefficient_table(results)
    flagged = flag_economic_significance(table, {"x": 1.0})
    assert flagged.loc["x", "significant_p05"]  # huge n -> statistically significant
    assert not flagged.loc["x", "economically_significant"]  # but tiny vs. threshold


def test_compute_vif_high_for_collinear_columns() -> None:
    n = 2000
    x1 = RNG.normal(0, 1, n)
    x2 = x1 + RNG.normal(0, 0.01, n)  # near-perfect collinearity
    x3 = RNG.normal(0, 1, n)  # independent
    X = add_constant(pd.DataFrame({"x1": x1, "x2": x2, "x3": x3}))
    vif = compute_vif(X)
    assert vif["x1"] > 20
    assert vif["x2"] > 20
    assert vif["x3"] < 5


def test_breusch_pagan_detects_heteroskedasticity() -> None:
    n = 5000
    x = RNG.uniform(1, 100, n)
    y_homo = 2.0 * x + RNG.normal(0, 1.0, n)
    y_hetero = 2.0 * x + RNG.normal(0, 1.0, n) * x  # variance grows with x

    X = add_constant(pd.DataFrame({"x": x}))
    homo_results = sm.OLS(y_homo, X).fit()
    hetero_results = sm.OLS(y_hetero, X).fit()

    assert not breusch_pagan_test(homo_results)["heteroskedastic_p05"]
    assert breusch_pagan_test(hetero_results)["heteroskedastic_p05"]


def test_goodness_of_fit_helpers_return_expected_keys() -> None:
    X = add_constant(pd.DataFrame({"x": RNG.normal(0, 1, 500)}))
    y = 1.0 + X["x"] + RNG.normal(0, 0.1, 500)
    ols_results = sm.OLS(y, X).fit()
    ols_fit = goodness_of_fit_ols(ols_results)
    assert 0 <= ols_fit["r_squared"] <= 1

    y_bin = (RNG.random(500) < 1 / (1 + np.exp(-X["x"]))).astype(int)
    logit_results = sm.Logit(y_bin, X).fit(disp=0)
    logit_fit = goodness_of_fit_logit(logit_results)
    assert logit_fit["pseudo_r_squared_mcfadden"] <= 1


def test_interpretation_helpers_return_one_line_per_driver() -> None:
    n = 2000
    df = pd.DataFrame(
        {
            "average_balance": RNG.uniform(0, 1000, n),
            "transaction_count": RNG.uniform(0, 20, n),
            "product_count": RNG.integers(1, 4, n),
            "income": RNG.uniform(10000, 80000, n),
        }
    )
    df["total_revenue"] = 10 + 0.01 * df["average_balance"] + RNG.normal(0, 1, n)
    results = fit_revenue_model(df)
    lines = interpret_revenue_model(results)
    assert len(lines) == 4

    churn_df = pd.DataFrame(
        {
            "rate_gap": RNG.uniform(0, 3, n),
            "activity": RNG.uniform(0, 30, n),
            "tenure": RNG.uniform(0, 60, n),
            "profitability": RNG.uniform(-50, 200, n),
            "churn": RNG.integers(0, 2, n),
        }
    )
    churn_results = fit_churn_model(churn_df)
    churn_lines = interpret_churn_model(churn_results)
    assert len(churn_lines) == 4
