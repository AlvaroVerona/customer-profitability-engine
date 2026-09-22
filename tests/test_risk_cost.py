"""Phase 4.3 -- Expected credit loss tests, with hand-calculated examples."""

from __future__ import annotations

import pandas as pd
import pytest

from customer_profitability.profitability.risk_cost import calculate_expected_loss


def _loan_row(loan_id: str, customer_id: str, pd_: float, lgd: float, ead: float) -> dict:
    return {
        "loan_id": loan_id,
        "customer_id": customer_id,
        "month": pd.Timestamp("2023-01-01"),
        "origination_date": pd.Timestamp("2022-06-01"),
        "outstanding_balance": ead,
        "credit_limit": ead * 2,
        "utilization": 0.5,
        "interest_rate": 0.12,
        "interest_income": ead * 0.12 / 12,
        "pd": pd_,
        "lgd": lgd,
        "ead": ead,
    }


def test_expected_loss_hand_calculated_example() -> None:
    # EL = PD x LGD x EAD = 0.02 x 0.45 x 5000 = 45.0
    loans = pd.DataFrame([_loan_row("L1", "C1", 0.02, 0.45, 5000.0)])
    out = calculate_expected_loss(loans)
    assert out.loc[0, "expected_loss"] == pytest.approx(45.0)


def test_expected_loss_sums_across_multiple_loans_same_customer_month() -> None:
    loans = pd.DataFrame(
        [
            _loan_row("L1", "C1", 0.02, 0.45, 5000.0),  # EL = 45.0
            _loan_row("L2", "C1", 0.10, 0.60, 1000.0),  # EL = 60.0
        ]
    )
    out = calculate_expected_loss(loans)
    assert out.loc[0, "expected_loss"] == 105.0


def test_expected_loss_zero_when_pd_zero() -> None:
    loans = pd.DataFrame([_loan_row("L1", "C1", 0.0, 0.45, 5000.0)])
    out = calculate_expected_loss(loans)
    assert out.loc[0, "expected_loss"] == 0.0
