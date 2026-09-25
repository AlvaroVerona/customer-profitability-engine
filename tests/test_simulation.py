"""Phase 12 -- Monte Carlo scenario simulation tests.

Not in PROJECT_SPEC.md's explicit test list (same rationale as
test_features.py/test_econometrics.py: every important calculation gets a
test). Includes the check that caught a real modeling bug while building
this phase: dividing Phase 10's already-blended `incremental_profit` back
out by `acceptance_probability` does *not* recover the correct
conditional-on-acceptance value, because the CLV survival formula blends
accept/reject nonlinearly -- see monte_carlo.py's module docstring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from customer_profitability.clv.discounting import (
    clv_config_monthly_rate,
    survival_weighted_annuity_factor,
)
from customer_profitability.simulation.monte_carlo import (
    run_scenario,
    simulate_action_outcomes,
    simulate_baseline_portfolio,
    summarize_samples,
)
from customer_profitability.simulation.scenarios import SCENARIOS

from .conftest import small_settings


def test_base_scenario_is_neutral() -> None:
    base = SCENARIOS["base"]
    assert base.churn_multiplier == 1.0
    assert base.profit_multiplier == 1.0


def test_downside_and_stress_are_worse_than_base() -> None:
    base, downside, stress = SCENARIOS["base"], SCENARIOS["downside"], SCENARIOS["stress"]
    assert downside.churn_multiplier > base.churn_multiplier
    assert downside.profit_multiplier < base.profit_multiplier
    assert stress.churn_multiplier > downside.churn_multiplier
    assert stress.profit_multiplier < downside.profit_multiplier


def test_upside_is_better_than_base() -> None:
    base, upside = SCENARIOS["base"], SCENARIOS["upside"]
    assert upside.churn_multiplier < base.churn_multiplier
    assert upside.profit_multiplier > base.profit_multiplier


def test_summarize_samples_known_array() -> None:
    samples = np.arange(1, 101, dtype=float)  # 1..100
    out = summarize_samples(samples)
    assert out["mean"] == pytest.approx(50.5)
    assert out["p50"] == pytest.approx(50.5, abs=1.0)
    assert out["p5"] < out["p50"] < out["p95"]


def test_baseline_portfolio_churned_customer_contributes_only_historical() -> None:
    settings = small_settings()
    clv_df = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "is_active": [False],
            "p_churn_monthly": [0.5],  # irrelevant: churned, no future simulated
            "expected_profit": [999.0],
            "historical_economic_profit": [500.0],
        }
    )
    sigma = pd.Series([0.3], index=pd.Index(["C1"], name="customer_id"), name="profit_sigma")
    samples = simulate_baseline_portfolio(clv_df, sigma, SCENARIOS["base"], settings.clv, n_simulations=100, seed=1)
    assert np.all(samples == 500.0)  # deterministic: no variance from a churned customer


def test_baseline_portfolio_active_customer_mean_matches_closed_form() -> None:
    settings = small_settings()
    clv_df = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "is_active": [True],
            "p_churn_monthly": [0.05],
            "expected_profit": [100.0],
            "historical_economic_profit": [200.0],
        }
    )
    # Zero profit-level noise isolates the survival-timing simulation from the closed form.
    sigma = pd.Series([1e-9], index=pd.Index(["C1"], name="customer_id"), name="profit_sigma")
    samples = simulate_baseline_portfolio(
        clv_df, sigma, SCENARIOS["base"], settings.clv, n_simulations=20000, seed=1
    )
    monthly_rate = clv_config_monthly_rate(settings.clv)
    factor = survival_weighted_annuity_factor(monthly_rate, np.array([0.05]), settings.clv.horizon_months)[0]
    expected_mean = 200.0 + 100.0 * factor
    assert samples.mean() == pytest.approx(expected_mean, rel=0.02)


def test_baseline_portfolio_scenario_multipliers_shift_distribution() -> None:
    settings = small_settings()
    clv_df = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "is_active": [True],
            "p_churn_monthly": [0.05],
            "expected_profit": [100.0],
            "historical_economic_profit": [200.0],
        }
    )
    sigma = pd.Series([0.2], index=pd.Index(["C1"], name="customer_id"), name="profit_sigma")
    base_samples = simulate_baseline_portfolio(clv_df, sigma, SCENARIOS["base"], settings.clv, 5000, seed=1)
    stress_samples = simulate_baseline_portfolio(clv_df, sigma, SCENARIOS["stress"], settings.clv, 5000, seed=1)
    assert stress_samples.mean() < base_samples.mean()  # higher churn + lower profit -> lower value


def _one_selected_action() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1"],
            "action_type": ["SAVINGS_CROSS_SELL"],
            "p_churn_monthly": [0.05],
            "expected_profit": [50.0],
            "future_clv": [900.0],
            "churn_reduction_pct": [0.1],
            "additional_revenue_monthly": [10.0],
            "additional_risk_monthly": [0.0],
            "operational_cost_monthly": [1.0],
            "acceptance_probability": [1.0],  # certain acceptance -> no Bernoulli variance
            "incentive_cost": [5.0],
        }
    )


def test_action_outcomes_certain_acceptance_matches_hand_calculated_delta() -> None:
    settings = small_settings()
    selected = _one_selected_action()
    profit_samples, cost_samples = simulate_action_outcomes(
        selected, SCENARIOS["base"], settings.clv, n_simulations=500, seed=1
    )
    monthly_rate = clv_config_monthly_rate(settings.clv)
    horizon = settings.clv.horizon_months

    p_churn_accept = 0.05 * (1 - 0.1)
    profit_accept = 50.0 + 10.0 - 0.0 - 1.0
    factor_accept = survival_weighted_annuity_factor(monthly_rate, np.array([p_churn_accept]), horizon)[0]
    future_clv_accept = profit_accept * factor_accept

    factor_reject = survival_weighted_annuity_factor(monthly_rate, np.array([0.05]), horizon)[0]
    future_clv_reject = 50.0 * factor_reject

    expected_delta = future_clv_accept - future_clv_reject - 5.0

    assert np.all(profit_samples == pytest.approx(expected_delta))
    assert np.all(cost_samples == 5.0)  # certain acceptance -> cost always incurred


def test_action_outcomes_zero_acceptance_never_pays_or_gains() -> None:
    settings = small_settings()
    selected = _one_selected_action()
    selected["acceptance_probability"] = 0.0
    profit_samples, cost_samples = simulate_action_outcomes(
        selected, SCENARIOS["base"], settings.clv, n_simulations=500, seed=1
    )
    assert np.all(profit_samples == 0.0)
    assert np.all(cost_samples == 0.0)


def test_action_outcomes_empty_selection_returns_zeros() -> None:
    settings = small_settings()
    empty = pd.DataFrame(
        columns=[
            "customer_id",
            "action_type",
            "p_churn_monthly",
            "expected_profit",
            "future_clv",
            "churn_reduction_pct",
            "additional_revenue_monthly",
            "additional_risk_monthly",
            "operational_cost_monthly",
            "acceptance_probability",
            "incentive_cost",
        ]
    )
    profit_samples, cost_samples = simulate_action_outcomes(empty, SCENARIOS["base"], settings.clv, 100, seed=1)
    assert np.all(profit_samples == 0.0)
    assert np.all(cost_samples == 0.0)


def test_action_outcomes_bernoulli_mean_converges_to_expectation() -> None:
    settings = small_settings()
    selected = _one_selected_action()
    selected["acceptance_probability"] = 0.4
    profit_samples, cost_samples = simulate_action_outcomes(
        selected, SCENARIOS["base"], settings.clv, n_simulations=30000, seed=1
    )
    # E[cost] = acceptance_probability x incentive_cost = 0.4 x 5.0 = 2.0
    assert cost_samples.mean() == pytest.approx(2.0, rel=0.1)
    assert not np.all(profit_samples == profit_samples[0])  # genuine Bernoulli variance present


def test_run_scenario_probability_of_budget_overrun_responds_to_budget() -> None:
    settings = small_settings()
    clv_df = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "is_active": [True],
            "p_churn_monthly": [0.05],
            "expected_profit": [100.0],
            "historical_economic_profit": [200.0],
        }
    )
    selected = _one_selected_action()
    sigma = pd.Series([0.2], index=pd.Index(["C1"], name="customer_id"), name="profit_sigma")

    tiny_budget_result = run_scenario(clv_df, selected, sigma, SCENARIOS["base"], settings.clv, budget=0.01, n_simulations=2000, seed=1)
    huge_budget_result = run_scenario(clv_df, selected, sigma, SCENARIOS["base"], settings.clv, budget=1e9, n_simulations=2000, seed=1)

    assert tiny_budget_result["probability_budget_overrun"] > huge_budget_result["probability_budget_overrun"]
    assert huge_budget_result["probability_budget_overrun"] == 0.0


def test_run_scenario_returns_expected_keys() -> None:
    settings = small_settings()
    clv_df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "is_active": [True, False],
            "p_churn_monthly": [0.05, 0.0],
            "expected_profit": [100.0, 0.0],
            "historical_economic_profit": [200.0, 50.0],
        }
    )
    sigma = pd.Series([0.2, 0.2], index=pd.Index(["C1", "C2"], name="customer_id"), name="profit_sigma")
    result = run_scenario(clv_df, _one_selected_action(), sigma, SCENARIOS["base"], settings.clv, budget=1000, n_simulations=500, seed=1)
    assert result["scenario"] == "Base"
    for key in ["total_portfolio_value", "incremental_action_profit", "realized_action_cost"]:
        assert set(result[key].keys()) == {"mean", "p5", "p50", "p95", "std"}
    assert 0.0 <= result["probability_negative_incremental_profit"] <= 1.0
    assert 0.0 <= result["probability_budget_overrun"] <= 1.0
