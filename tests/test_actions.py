"""Phase 9 -- Customer Action Framework, and Phase 10 -- Action Impact
Simulation tests (one file per PROJECT_SPEC.md's tests/test_actions.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from customer_profitability.actions.definitions import (
    ActionType,
    build_action_catalog,
    eligible_actions_for_customer,
    evaluate_eligibility,
)
from customer_profitability.actions.incremental_value import (
    compute_incremental_value,
    recommend_action,
)
from customer_profitability.actions.simulator import simulate_action_effects
from customer_profitability.clv.discounting import (
    clv_config_monthly_rate,
    survival_weighted_annuity_factor,
)

from .conftest import small_settings


def test_build_action_catalog_always_includes_no_action_at_zero_cost() -> None:
    settings = small_settings()
    catalog = build_action_catalog(settings.actions)
    no_action = next(a for a in catalog if a.action_type == ActionType.NO_ACTION)
    assert no_action.incentive_cost == 0.0
    assert len(catalog) == 6  # NO_ACTION + the 5 named interventions


def test_build_action_catalog_costs_come_from_config() -> None:
    settings = small_settings()
    catalog = build_action_catalog(settings.actions)
    by_type = {a.action_type: a for a in catalog}
    assert by_type[ActionType.RETENTION_INCENTIVE].incentive_cost == settings.actions.retention_incentive_cost
    assert by_type[ActionType.SAVINGS_CROSS_SELL].incentive_cost == settings.actions.savings_cross_sell_cost
    assert by_type[ActionType.CREDIT_PRODUCT].incentive_cost == settings.actions.credit_product_cost


def _sample_customers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4"],
            "customer_segment": ["Mass", "Premium", "Mass", "Mass"],
            "income": [40000.0, 60000.0, 5000.0, 40000.0],  # C3 below credit_product_min_income
            "employment_status": ["employed", "employed", "employed", "unemployed"],  # C4 unemployed
        }
    )


def _sample_accounts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "product": ["savings_account", "consumer_loan"],
            "account_open_date": pd.to_datetime(["2022-01-01", "2022-01-01"]),
            "account_close_date": pd.to_datetime([pd.NaT, pd.NaT]),
        }
    )


def _sample_snapshot() -> pd.DataFrame:
    return pd.DataFrame({"customer_id": ["C1", "C2", "C3", "C4"], "month": pd.to_datetime(["2023-06-01"] * 4)})


def test_evaluate_eligibility_savings_cross_sell() -> None:
    settings = small_settings()
    out = evaluate_eligibility(_sample_customers(), _sample_accounts(), _sample_snapshot(), settings.actions).set_index(
        "customer_id"
    )
    assert not out.loc["C1", "SAVINGS_CROSS_SELL"]  # C1 already has savings
    assert out.loc["C2", "SAVINGS_CROSS_SELL"]
    assert out.loc["C3", "SAVINGS_CROSS_SELL"]


def test_evaluate_eligibility_credit_product_risk_gate() -> None:
    settings = small_settings()
    out = evaluate_eligibility(_sample_customers(), _sample_accounts(), _sample_snapshot(), settings.actions).set_index(
        "customer_id"
    )
    assert not out.loc["C2", "CREDIT_PRODUCT"]  # already has a loan
    assert out.loc["C1", "CREDIT_PRODUCT"]  # no loan, sufficient income, employed
    assert not out.loc["C3", "CREDIT_PRODUCT"]  # income below floor
    assert not out.loc["C4", "CREDIT_PRODUCT"]  # unemployed


def test_evaluate_eligibility_premium_subscription() -> None:
    settings = small_settings()
    out = evaluate_eligibility(_sample_customers(), _sample_accounts(), _sample_snapshot(), settings.actions).set_index(
        "customer_id"
    )
    assert not out.loc["C2", "PREMIUM_SUBSCRIPTION"]  # already Premium
    assert out.loc["C1", "PREMIUM_SUBSCRIPTION"]


def test_no_action_and_retention_always_eligible_for_every_customer() -> None:
    settings = small_settings()
    out = evaluate_eligibility(_sample_customers(), _sample_accounts(), _sample_snapshot(), settings.actions)
    assert out["NO_ACTION"].all()
    assert out["RETENTION_INCENTIVE"].all()


def test_eligible_actions_for_customer_matches_row() -> None:
    settings = small_settings()
    eligibility = evaluate_eligibility(_sample_customers(), _sample_accounts(), _sample_snapshot(), settings.actions)
    actions = eligible_actions_for_customer(eligibility, "C2")
    assert ActionType.NO_ACTION in actions
    assert ActionType.RETENTION_INCENTIVE in actions
    assert ActionType.SAVINGS_CROSS_SELL in actions  # C2 has no savings
    assert ActionType.CREDIT_PRODUCT not in actions  # C2 already has a loan
    assert ActionType.PREMIUM_SUBSCRIPTION not in actions  # C2 already Premium


# --- Phase 10: action impact simulation ------------------------------------


def _sim_customer_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "p_churn_monthly": [0.02, 0.10],  # C2 much higher churn risk
            "income": [40000.0, 80000.0],
            "average_balance_avg_3m": [1000.0, 2000.0],
        }
    )


def _sim_eligibility_all_true() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "NO_ACTION": [True, True],
            "RETENTION_INCENTIVE": [True, True],
            "SAVINGS_CROSS_SELL": [True, True],
            "CREDIT_PRODUCT": [True, True],
            "INVESTMENT_PRODUCT": [True, True],
            "PREMIUM_SUBSCRIPTION": [True, True],
        }
    )


def test_simulate_no_action_is_all_zero_except_acceptance() -> None:
    settings = small_settings()
    effects = simulate_action_effects(_sim_customer_df(), _sim_eligibility_all_true(), settings)
    no_action = effects[effects["action_type"] == "NO_ACTION"]
    assert (no_action["acceptance_probability"] == 1.0).all()
    for col in ["churn_reduction_pct", "additional_revenue_monthly", "additional_balance", "additional_risk_monthly", "operational_cost_monthly"]:
        assert (no_action[col] == 0.0).all()


def test_simulate_retention_acceptance_increases_with_churn_risk() -> None:
    settings = small_settings()
    effects = simulate_action_effects(_sim_customer_df(), _sim_eligibility_all_true(), settings)
    retention = effects[effects["action_type"] == "RETENTION_INCENTIVE"].set_index("customer_id")
    params = settings.action_simulation.params["retention_incentive"]
    expected_c1 = params["acceptance_base"] + params["acceptance_churn_sensitivity"] * 0.02
    expected_c2 = params["acceptance_base"] + params["acceptance_churn_sensitivity"] * 0.10
    assert retention.loc["C1", "acceptance_probability"] == pytest.approx(expected_c1)
    assert retention.loc["C2", "acceptance_probability"] == pytest.approx(expected_c2)
    assert retention.loc["C2", "acceptance_probability"] > retention.loc["C1", "acceptance_probability"]


def test_simulate_savings_cross_sell_hand_calculated() -> None:
    settings = small_settings()
    effects = simulate_action_effects(_sim_customer_df(), _sim_eligibility_all_true(), settings)
    row = effects[(effects["action_type"] == "SAVINGS_CROSS_SELL") & (effects["customer_id"] == "C1")].iloc[0]
    params = settings.action_simulation.params["savings_cross_sell"]
    expected_balance = 1000.0 * params["balance_uplift_pct"]
    assert row["additional_balance"] == pytest.approx(expected_balance)
    monthly_spread = (settings.ftp.base_ftp_rate_annual - params["assumed_savings_rate_annual"]) / 12
    assert row["additional_revenue_monthly"] == pytest.approx(expected_balance * monthly_spread)


def test_simulate_credit_product_hand_calculated() -> None:
    settings = small_settings()
    effects = simulate_action_effects(_sim_customer_df(), _sim_eligibility_all_true(), settings)
    row = effects[(effects["action_type"] == "CREDIT_PRODUCT") & (effects["customer_id"] == "C1")].iloc[0]
    params = settings.action_simulation.params["credit_product"]
    expected_limit = 40000.0 * params["credit_limit_income_multiple"]
    expected_balance = expected_limit * params["expected_utilization"]
    assert row["additional_balance"] == pytest.approx(expected_balance)
    assert row["additional_revenue_monthly"] == pytest.approx(expected_balance * params["interest_rate_annual"] / 12)
    assert row["additional_risk_monthly"] == pytest.approx(
        params["expected_pd_monthly"] * params["expected_lgd"] * expected_balance
    )


def test_simulate_premium_subscription_uses_configured_fee() -> None:
    settings = small_settings()
    effects = simulate_action_effects(_sim_customer_df(), _sim_eligibility_all_true(), settings)
    row = effects[(effects["action_type"] == "PREMIUM_SUBSCRIPTION") & (effects["customer_id"] == "C1")].iloc[0]
    assert row["additional_revenue_monthly"] == settings.revenue.premium_monthly_fee


def test_simulate_respects_eligibility_filtering() -> None:
    settings = small_settings()
    eligibility = _sim_eligibility_all_true()
    eligibility.loc[eligibility["customer_id"] == "C1", "CREDIT_PRODUCT"] = False
    effects = simulate_action_effects(_sim_customer_df(), eligibility, settings)
    credit_rows = effects[effects["action_type"] == "CREDIT_PRODUCT"]
    assert set(credit_rows["customer_id"]) == {"C2"}


def _baseline_for_incremental(settings) -> pd.DataFrame:
    # future_clv must come from the same closed-form formula NO_ACTION's
    # simulated value will be recomputed with -- in production both trace
    # back to the same Phase 7 compute_base_clv call on clv.parquet, so a
    # baseline with an arbitrary/inconsistent future_clv isn't a real scenario.
    monthly_rate = clv_config_monthly_rate(settings.clv)
    p_churn = 0.05
    expected_profit = 50.0
    factor = survival_weighted_annuity_factor(monthly_rate, np.array([p_churn]), settings.clv.horizon_months)[0]
    return pd.DataFrame(
        {
            "customer_id": ["C1"],
            "p_churn_monthly": [p_churn],
            "expected_profit": [expected_profit],
            "future_clv": [expected_profit * factor],
        }
    )


def test_incremental_value_no_action_nets_to_zero() -> None:
    settings = small_settings()
    baseline = _baseline_for_incremental(settings)
    effects = simulate_action_effects(
        pd.DataFrame({"customer_id": ["C1"], "p_churn_monthly": [0.05], "income": [40000.0], "average_balance_avg_3m": [1000.0]}),
        pd.DataFrame({"customer_id": ["C1"], **{a.value: [True] for a in ActionType}}),
        settings,
    )
    out = compute_incremental_value(baseline, effects, settings)
    no_action = out[out["action_type"] == "NO_ACTION"].iloc[0]
    assert no_action["incremental_profit"] == pytest.approx(0.0, abs=1e-9)
    assert no_action["delta_clv"] == pytest.approx(0.0, abs=1e-9)
    assert no_action["expected_action_cost"] == 0.0


def test_incremental_value_hand_calculated_example() -> None:
    settings = small_settings()
    baseline = pd.DataFrame(
        {"customer_id": ["C1"], "p_churn_monthly": [0.05], "expected_profit": [50.0], "future_clv": [900.0]}
    )
    effects = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "action_type": ["SAVINGS_CROSS_SELL"],
            "acceptance_probability": [0.5],
            "churn_reduction_pct": [0.2],
            "additional_revenue_monthly": [10.0],
            "additional_balance": [100.0],
            "additional_risk_monthly": [0.0],
            "operational_cost_monthly": [1.0],
        }
    )
    out = compute_incremental_value(baseline, effects, settings)
    row = out.iloc[0]

    monthly_rate = clv_config_monthly_rate(settings.clv)
    p_churn_with_action = 0.05 * (1 - 0.5 * 0.2)
    expected_profit_with_action = 50.0 + 0.5 * (10.0 - 0.0 - 1.0)
    factor = survival_weighted_annuity_factor(monthly_rate, np.array([p_churn_with_action]), settings.clv.horizon_months)[0]
    expected_simulated_clv = expected_profit_with_action * factor
    expected_delta_clv = expected_simulated_clv - 900.0
    expected_cost = 0.5 * settings.actions.savings_cross_sell_cost
    expected_incremental = expected_delta_clv - expected_cost

    assert row["delta_clv"] == pytest.approx(expected_delta_clv)
    assert row["expected_action_cost"] == pytest.approx(expected_cost)
    assert row["incremental_profit"] == pytest.approx(expected_incremental)


def test_recommend_action_sorted_descending_and_includes_no_action() -> None:
    settings = small_settings()
    incremental_value = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C1"],
            "action_type": ["NO_ACTION", "RETENTION_INCENTIVE", "SAVINGS_CROSS_SELL"],
            "incremental_profit": [0.0, 15.0, 50.0],
            "delta_clv": [0.0, 20.0, 55.0],
            "expected_action_cost": [0.0, 5.0, 5.0],
        }
    )
    out = recommend_action(incremental_value, "C1", settings)
    assert list(out["action_type"]) == ["SAVINGS_CROSS_SELL", "RETENTION_INCENTIVE", "NO_ACTION"]
    assert "Savings Cross-Sell" in out["action_name"].to_numpy()
