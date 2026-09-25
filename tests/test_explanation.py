"""Phase 14 -- customer-level decision explanation and optimization
selection explanation tests."""

from __future__ import annotations

import pandas as pd

from customer_profitability.actions.explanation import (
    explain_customer_recommendation,
    format_explanation_text,
)
from customer_profitability.optimization.explanation import explain_selection

from .conftest import small_settings


def _incremental_value_for(customer_id: str, action_type: str, incremental_profit: float, delta_clv: float) -> dict:
    return {
        "customer_id": customer_id,
        "action_type": action_type,
        "acceptance_probability": 0.3,
        "incremental_profit": incremental_profit,
        "delta_clv": delta_clv,
        "expected_action_cost": 3.0,
    }


def _clv_row(customer_id: str, total_value: float, historical: float, p_churn: float, is_active: bool = True) -> dict:
    return {
        "customer_id": customer_id,
        "total_customer_economic_value": total_value,
        "historical_economic_profit": historical,
        "p_churn_monthly": p_churn,
        "is_active": is_active,
    }


def _population_clv() -> pd.DataFrame:
    # 20 unremarkable customers to form the percentile benchmark population.
    return pd.DataFrame([_clv_row(f"BASE_{i}", 100.0 + i, 50.0 + i, 0.02) for i in range(20)])


def test_explain_customer_recommendation_high_percentile_bullets() -> None:
    settings = small_settings()
    customer_id = "C_STAR"
    incremental_value = pd.DataFrame(
        [
            _incremental_value_for(customer_id, "NO_ACTION", 0.0, 0.0),
            _incremental_value_for(customer_id, "SAVINGS_CROSS_SELL", 500.0, 520.0),
        ]
    )
    clv = pd.concat(
        [_population_clv(), pd.DataFrame([_clv_row(customer_id, 10000.0, 5000.0, 0.5)])], ignore_index=True
    )

    explanation = explain_customer_recommendation(customer_id, incremental_value, clv, settings)

    assert explanation.recommended_action == "Savings Cross-Sell"
    assert any("High expected CLV" in reason for reason in explanation.why)
    assert any("Strong historical profitability" in reason for reason in explanation.why)
    assert any("Elevated churn probability" in reason for reason in explanation.why)
    assert any("does not currently hold a savings account" in reason for reason in explanation.why)
    assert any("Positive incremental value" in reason for reason in explanation.why)
    assert explanation.incremental_profit == 500.0
    assert explanation.clv_impact == 520.0
    assert explanation.cost == settings.actions.savings_cross_sell_cost


def test_explain_customer_recommendation_unremarkable_customer_still_gets_a_reason() -> None:
    settings = small_settings()
    customer_id = "C_AVERAGE"
    incremental_value = pd.DataFrame(
        [
            _incremental_value_for(customer_id, "NO_ACTION", 0.0, 0.0),
            _incremental_value_for(customer_id, "INVESTMENT_PRODUCT", 10.0, 12.0),
        ]
    )
    # Below (not tied with) the base population's values on every dimension,
    # so none of the percentile-triggered bullets fire.
    clv = pd.concat(
        [_population_clv(), pd.DataFrame([_clv_row(customer_id, 50.0, 25.0, 0.005)])], ignore_index=True
    )

    explanation = explain_customer_recommendation(customer_id, incremental_value, clv, settings)
    assert not any("High expected CLV" in r for r in explanation.why)
    assert not any("Strong historical profitability" in r for r in explanation.why)
    assert not any("Elevated churn probability" in r for r in explanation.why)
    assert any("Positive incremental value" in r for r in explanation.why)  # never an empty explanation


def test_explain_customer_recommendation_no_action_case() -> None:
    settings = small_settings()
    customer_id = "C_NONE"
    incremental_value = pd.DataFrame(
        [
            _incremental_value_for(customer_id, "NO_ACTION", 0.0, 0.0),
            _incremental_value_for(customer_id, "RETENTION_INCENTIVE", -5.0, -2.0),
        ]
    )
    clv = pd.concat([_population_clv(), pd.DataFrame([_clv_row(customer_id, 100.0, 50.0, 0.02)])], ignore_index=True)

    explanation = explain_customer_recommendation(customer_id, incremental_value, clv, settings)
    assert explanation.recommended_action == "No Action"
    assert explanation.cost == 0.0
    assert explanation.incremental_profit == 0.0
    assert explanation.clv_impact == 0.0
    assert "No eligible action" in explanation.why[0]


def test_format_explanation_text_structure() -> None:
    settings = small_settings()
    customer_id = "C_STAR"
    incremental_value = pd.DataFrame(
        [
            _incremental_value_for(customer_id, "NO_ACTION", 0.0, 0.0),
            _incremental_value_for(customer_id, "SAVINGS_CROSS_SELL", 500.0, 520.0),
        ]
    )
    clv = pd.concat([_population_clv(), pd.DataFrame([_clv_row(customer_id, 100.0, 50.0, 0.02)])], ignore_index=True)
    explanation = explain_customer_recommendation(customer_id, incremental_value, clv, settings)
    text = format_explanation_text(explanation)

    assert text.startswith(f"Customer {customer_id}")
    assert "Recommended Action:" in text
    assert "Why:" in text
    assert "Expected impact:" in text
    assert "- Cost: EUR" in text
    assert "- Incremental profit: EUR" in text
    assert "- CLV impact: EUR" in text


def _optimize_result_fixture() -> dict:
    candidates = pd.DataFrame(
        [
            {"customer_id": "WINNER", "action_type": "CREDIT_PRODUCT", "incremental_profit": 500.0},
            {"customer_id": "LOSER", "action_type": "PREMIUM_SUBSCRIPTION", "incremental_profit": 10.0},
            {"customer_id": "LOSER", "action_type": "INVESTMENT_PRODUCT", "incremental_profit": 5.0},
        ]
        + [{"customer_id": f"FILLER_{i}", "action_type": "CREDIT_PRODUCT", "incremental_profit": 400.0 - i} for i in range(50)]
    )
    selected = candidates[candidates["customer_id"].isin(["WINNER", *[f"FILLER_{i}" for i in range(50)]])]
    return {"candidates": candidates, "selected": selected}


def test_explain_selection_when_selected() -> None:
    result = _optimize_result_fixture()
    explanation = explain_selection("WINNER", result)
    assert explanation.was_selected
    assert explanation.action_type == "CREDIT_PRODUCT"
    assert "ranks #1" in explanation.reason


def test_explain_selection_when_not_selected_but_had_candidates() -> None:
    result = _optimize_result_fixture()
    explanation = explain_selection("LOSER", result)
    assert not explanation.was_selected
    assert explanation.action_type is None
    assert "PREMIUM_SUBSCRIPTION" in explanation.reason  # the better of LOSER's two candidates
    assert "not evidence" in explanation.reason


def test_explain_selection_when_no_candidates_at_all() -> None:
    result = _optimize_result_fixture()
    explanation = explain_selection("GHOST", result)
    assert not explanation.was_selected
    assert "ROI pre-filter" in explanation.reason
