"""Phase 7 -- Customer Lifetime Value tests.

Includes a direct cross-check between the closed-form survival-weighted
annuity factor and a Monte Carlo simulation of the same quantity: this is
exactly the check that caught a real off-by-one bug in the survival
formula (S(t) = (1-p)^t vs. the correct (1-p)^(t-1)) while building this
phase -- see discounting.py's docstring.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from customer_profitability.clv.calculator import compute_base_clv, monte_carlo_clv
from customer_profitability.clv.discounting import (
    annuity_factor,
    monthly_discount_rate,
    survival_weighted_annuity_factor,
)
from customer_profitability.clv.forecasting import (
    build_expected_profit,
    compute_run_rate,
    estimate_profit_volatility,
    latest_snapshot,
    predict_churn_probability,
)

from .conftest import small_settings


def test_monthly_discount_rate_hand_calculated() -> None:
    # (1.08)^(1/12) - 1 ~= 0.006434
    assert monthly_discount_rate(0.08) == pytest.approx(0.0064340301, abs=1e-8)


def test_annuity_factor_hand_calculated() -> None:
    # 3 months at monthly rate 0.01: 1/1.01 + 1/1.01^2 + 1/1.01^3
    expected = 1 / 1.01 + 1 / 1.01**2 + 1 / 1.01**3
    assert annuity_factor(0.01, 3) == pytest.approx(expected)


def test_annuity_factor_zero_months_is_zero() -> None:
    assert annuity_factor(0.01, 0) == 0.0


def test_survival_weighted_factor_no_churn_equals_plain_annuity() -> None:
    # p=0 -> customer never churns -> should reduce to the plain annuity factor.
    plain = annuity_factor(0.01, 12)
    survival = survival_weighted_annuity_factor(0.01, np.array([0.0]), 12)[0]
    assert survival == pytest.approx(plain)


def test_survival_weighted_factor_certain_churn_only_first_month() -> None:
    # p=1 -> churns for certain after month 1 -> only month 1's discounted cashflow counts.
    survival = survival_weighted_annuity_factor(0.01, np.array([1.0]), 12)[0]
    assert survival == pytest.approx(1 / 1.01)


def test_survival_weighted_factor_matches_monte_carlo_expectation() -> None:
    """Cross-checks the closed-form formula against a large-n Monte Carlo
    simulation of the same survival process -- the check that caught the
    original off-by-one bug."""
    rng = np.random.default_rng(0)
    monthly_rate = 0.006
    horizon = 24
    p = 0.05
    n_sim = 50000

    k = rng.geometric(p, size=n_sim)
    months_active = np.minimum(k, horizon)
    simulated_factor = annuity_factor(monthly_rate, months_active).mean()

    analytical_factor = survival_weighted_annuity_factor(monthly_rate, np.array([p]), horizon)[0]
    assert simulated_factor == pytest.approx(analytical_factor, rel=0.01)


def test_compute_base_clv_hand_calculated() -> None:
    df = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "historical_economic_profit": [500.0],
            "expected_profit": [100.0],
            "p_churn_monthly": [0.05],
        }
    )
    settings = small_settings()
    out = compute_base_clv(df, settings.clv)
    monthly_rate = monthly_discount_rate(settings.clv.annual_discount_rate)
    expected_factor = survival_weighted_annuity_factor(monthly_rate, np.array([0.05]), settings.clv.horizon_months)[0]
    assert out.loc[0, "future_clv"] == pytest.approx(100.0 * expected_factor)
    assert out.loc[0, "total_customer_economic_value"] == pytest.approx(500.0 + 100.0 * expected_factor)


def test_monte_carlo_clv_percentiles_are_ordered_and_reasonable() -> None:
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "historical_economic_profit": [500.0, 0.0],
            "expected_profit": [100.0, 50.0],
            "p_churn_monthly": [0.05, 0.2],
        }
    )
    settings = small_settings()
    sigma = pd.Series([0.3, 0.5], index=["C1", "C2"], name="profit_sigma")
    mc = monte_carlo_clv(df, settings.clv, sigma, n_simulations=500, seed=1)
    assert set(mc["customer_id"]) == {"C1", "C2"}
    assert (mc["clv_p5"] <= mc["clv_p50"]).all()
    assert (mc["clv_p50"] <= mc["clv_p95"]).all()


def test_monte_carlo_clv_zero_churn_prob_uses_full_horizon() -> None:
    settings = small_settings()
    df = pd.DataFrame(
        {"customer_id": ["C1"], "historical_economic_profit": [0.0], "expected_profit": [10.0], "p_churn_monthly": [1e-9]}
    )
    sigma = pd.Series([0.01], index=["C1"], name="profit_sigma")  # near-zero noise for a tight check
    mc = monte_carlo_clv(df, settings.clv, sigma, n_simulations=2000, seed=1)
    monthly_rate = monthly_discount_rate(settings.clv.annual_discount_rate)
    expected = 10.0 * annuity_factor(monthly_rate, settings.clv.horizon_months)
    assert mc.loc[0, "clv_p50"] == pytest.approx(expected, rel=0.05)


def test_latest_snapshot_picks_most_recent_row_per_customer() -> None:
    customer_360 = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C2"],
            "month": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-01-01"]),
            "income": [10.0, 20.0, 30.0],
        }
    )
    out = latest_snapshot(customer_360)
    c1 = out[out["customer_id"] == "C1"].iloc[0]
    assert c1["month"] == pd.Timestamp("2023-02-01")
    assert c1["income"] == 20.0


def test_compute_run_rate_averages_last_n_months() -> None:
    profitability_monthly = pd.DataFrame(
        {
            "customer_id": ["C1"] * 4,
            "month": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"]),
            "total_revenue": [10.0, 20.0, 30.0, 40.0],
            "deposit_contribution": [1.0, 1.0, 1.0, 1.0],
            "expected_loss": [0.0, 0.0, 0.0, 0.0],
            "operating_cost": [2.0, 2.0, 2.0, 2.0],
        }
    )
    out = compute_run_rate(profitability_monthly, window_months=3)
    # last 3 months of total_revenue: 20, 30, 40 -> mean 30
    assert out.loc[0, "expected_revenue"] == pytest.approx(30.0)


def test_build_expected_profit_matches_waterfall() -> None:
    run_rate = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "expected_revenue": [100.0],
            "expected_deposit_contribution": [5.0],
            "expected_loss": [10.0],
            "expected_operating_cost": [8.0],
        }
    )
    out = build_expected_profit(run_rate)
    assert out.loc[0, "expected_profit"] == pytest.approx(100.0 + 5.0 - 10.0 - 8.0)


def test_estimate_profit_volatility_defaults_for_short_history() -> None:
    profitability_monthly = pd.DataFrame(
        {"customer_id": ["C1", "C1"], "economic_profit": [10.0, 20.0]}  # only 2 months
    )
    sigma = estimate_profit_volatility(profitability_monthly, default_sigma=0.42)
    assert sigma["C1"] == 0.42


def test_estimate_profit_volatility_computed_and_clipped_for_longer_history() -> None:
    profitability_monthly = pd.DataFrame(
        {
            "customer_id": ["C1"] * 6,
            "economic_profit": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],  # zero volatility
        }
    )
    sigma = estimate_profit_volatility(profitability_monthly, min_sigma=0.15, max_sigma=1.5)
    assert sigma["C1"] == 0.15  # clipped up from a true CV of 0


def test_predict_churn_probability_aligns_persisted_feature_columns(tmp_path) -> None:
    """The persisted feature_columns list may not match prepare_feature_matrix's
    output 1:1 (e.g. a customer_segment level absent from the snapshot) --
    reindexing to the persisted columns (fill 0) must keep predictions aligned."""
    X_train = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0], "b": [1.0, 0.0, 1.0, 0.0]})
    y_train = [0, 1, 0, 1]
    model = LogisticRegression().fit(X_train, y_train)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(model, models_dir / "churn_champion.joblib")
    (models_dir / "churn_feature_columns.json").write_text(json.dumps(["a", "b"]))

    snapshot = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "age": [30],
            "income": [40000.0],
            "tenure_months": [12],
            "login_frequency": [5.0],
            "product_count": [2],
            "support_interactions": [0],
            "average_balance": [1000.0],
            "balance_volatility_3m": [10.0],
            "rate_differential": [-0.01],
            "transaction_count": [5],
            "transaction_volume": [200.0],
            "average_transaction_value": [40.0],
            "atm_usage_share": [0.1],
            "outstanding_balance": [0.0],
            "utilization": [np.nan],
            "expected_loss": [0.0],
            "balance_growth_1m": [0.0],
            "spending_growth_1m": [0.0],
            "activity_volatility_3m": [0.0],
            "recent_activity_ratio": [1.0],
            "customer_segment": ["Mass"],
        }
    )
    out = predict_churn_probability(snapshot, models_dir)
    assert list(out.index) == ["C1"]
    assert 0 <= out.iloc[0] <= 1
