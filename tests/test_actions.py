"""Phase 9 -- Customer Action Framework tests."""

from __future__ import annotations

import pandas as pd

from customer_profitability.actions.definitions import (
    ActionType,
    build_action_catalog,
    eligible_actions_for_customer,
    evaluate_eligibility,
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
