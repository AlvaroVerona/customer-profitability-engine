"""Phase 4 -- Historical Customer Profitability Engine tests.

Covers revenue, operating cost, acquisition cost, and the full economic
profit waterfall (revenue.py, operating_cost.py, engine.py), with
hand-calculated examples per the Phase 16 testing requirement. FTP and
expected-loss are covered separately in test_ftp.py / test_risk_cost.py.
"""

from __future__ import annotations

import pandas as pd

from customer_profitability.profitability.engine import (
    aggregate_profit,
    build_customer_profitability_summary,
    build_monthly_profitability,
    calculate_acquisition_cost,
)
from customer_profitability.profitability.operating_cost import calculate_operating_cost
from customer_profitability.profitability.revenue import (
    calculate_fee_revenue,
    calculate_interchange_revenue,
    calculate_interest_revenue,
)

from .conftest import small_settings


def test_interest_revenue_hand_calculated_example() -> None:
    # InterestRevenue = AverageLoanBalance x LoanRate / 12 = 5000 x 0.12 / 12 = 50.0
    loans = pd.DataFrame(
        {
            "loan_id": ["L1"],
            "customer_id": ["C1"],
            "month": pd.to_datetime(["2023-01-01"]),
            "origination_date": pd.to_datetime(["2022-06-01"]),
            "outstanding_balance": [5000.0],
            "credit_limit": [10000.0],
            "utilization": [0.5],
            "interest_rate": [0.12],
            "interest_income": [5000.0 * 0.12 / 12],
            "pd": [0.02],
            "lgd": [0.45],
            "ead": [5000.0],
        }
    )
    out = calculate_interest_revenue(loans)
    assert out.loc[0, "interest_revenue"] == 50.0


def test_interchange_revenue_hand_calculated_example() -> None:
    # InterchangeRevenue = CardTransactionVolume x InterchangeRate = 2000 x 0.007 = 14.0
    cards = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "month": pd.to_datetime(["2023-01-01"]),
            "transaction_count": [40],
            "transaction_volume": [2000.0],
            "interchange_revenue": [999.0],  # deliberately wrong to prove we recompute, not pass through
            "atm_withdrawal_volume": [100.0],
            "international_transaction_volume": [50.0],
            "payment_processing_cost": [3.0],
        }
    )
    settings = small_settings()
    out = calculate_interchange_revenue(cards, settings.revenue)
    assert out.loc[0, "interchange_revenue"] == 2000.0 * settings.revenue.interchange_rate == 14.0


def test_fee_revenue_low_balance_and_premium_subscription() -> None:
    customers = pd.DataFrame(
        {"customer_id": ["C1", "C2", "C3"], "customer_segment": ["Mass", "Premium", "Mass"]}
    )
    deposits = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "month": pd.to_datetime(["2023-01-01"] * 3),
            "product": ["current_account"] * 3,
            "opening_balance": [100.0, 5000.0, 5000.0],
            "closing_balance": [100.0, 5000.0, 5000.0],
            "average_balance": [100.0, 5000.0, 5000.0],  # C1 below threshold, C2/C3 above
            "inflows": [0.0, 0.0, 0.0],
            "outflows": [0.0, 0.0, 0.0],
            "deposit_rate": [0.02, 0.02, 0.02],
            "market_rate": [0.035, 0.035, 0.035],
        }
    )
    settings = small_settings()
    out = calculate_fee_revenue(deposits, customers, settings.revenue).set_index("customer_id")

    assert out.loc["C1", "account_fee"] == settings.revenue.low_balance_fee
    assert out.loc["C1", "premium_subscription_fee"] == 0.0
    assert out.loc["C2", "account_fee"] == 0.0
    assert out.loc["C2", "premium_subscription_fee"] == settings.revenue.premium_monthly_fee
    assert out.loc["C3", "fee_revenue"] == 0.0  # Mass, above threshold: no fee at all


def test_operating_cost_sums_support_processing_and_account_servicing() -> None:
    settings = small_settings()
    active = pd.DataFrame({"customer_id": ["C1"], "month": pd.to_datetime(["2023-01-01"])})
    service = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "month": pd.to_datetime(["2023-01-01"]),
            "support_contacts": [2],
            "average_handling_time": [8.0],
            "support_channel": ["chat"],
            "estimated_service_cost": [16.0],
        }
    )
    cards = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "month": pd.to_datetime(["2023-01-01"]),
            "transaction_count": [10],
            "transaction_volume": [500.0],
            "interchange_revenue": [3.5],
            "atm_withdrawal_volume": [0.0],
            "international_transaction_volume": [0.0],
            "payment_processing_cost": [1.75],
        }
    )
    out = calculate_operating_cost(service, cards, active, settings.operating)
    expected = 16.0 + 1.75 + settings.operating.account_servicing_cost_monthly
    assert out.loc[0, "operating_cost"] == expected


def test_acquisition_cost_recognized_only_in_acquisition_month_within_window() -> None:
    customers = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "acquisition_channel": ["paid_search", "organic"],
            # C1 acquired inside the window, C2 acquired before it (back book).
            "acquisition_date": pd.to_datetime(["2023-02-01", "2020-01-01"]),
        }
    )
    panel_months = pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"])
    settings = small_settings()
    out = calculate_acquisition_cost(customers, panel_months, settings.risk.cac_by_channel)

    assert len(out) == 1
    assert out.iloc[0]["customer_id"] == "C1"
    assert out.iloc[0]["month"] == pd.Timestamp("2023-02-01")
    assert out.iloc[0]["acquisition_cost"] == settings.risk.cac_by_channel["paid_search"]


def _minimal_tables_for_engine() -> dict[str, pd.DataFrame]:
    month = pd.Timestamp("2023-01-01")
    customers = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "customer_segment": ["Mass"],
            "acquisition_channel": ["organic"],
            "acquisition_date": pd.to_datetime(["2022-01-01"]),  # before the window: no CAC recognized
        }
    )
    customer_monthly = pd.DataFrame({"customer_id": ["C1"], "month": [month]})
    deposits = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "month": [month],
            "product": ["current_account"],
            "opening_balance": [1000.0],
            "closing_balance": [1000.0],
            "average_balance": [1000.0],
            "inflows": [50.0],
            "outflows": [20.0],
            "deposit_rate": [0.02],
            "market_rate": [0.035],
        }
    )
    cards = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "month": [month],
            "transaction_count": [10],
            "transaction_volume": [500.0],
            "interchange_revenue": [3.5],
            "atm_withdrawal_volume": [0.0],
            "international_transaction_volume": [0.0],
            "payment_processing_cost": [1.0],
        }
    )
    loans = pd.DataFrame(
        {
            "loan_id": ["L1"],
            "customer_id": ["C1"],
            "month": [month],
            "origination_date": pd.to_datetime(["2022-06-01"]),
            "outstanding_balance": [2000.0],
            "credit_limit": [4000.0],
            "utilization": [0.5],
            "interest_rate": [0.12],
            "interest_income": [2000.0 * 0.12 / 12],
            "pd": [0.02],
            "lgd": [0.45],
            "ead": [2000.0],
        }
    )
    service = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "month": [month],
            "support_contacts": [1],
            "average_handling_time": [5.0],
            "support_channel": ["chat"],
            "estimated_service_cost": [8.0],
        }
    )
    return {
        "customers": customers,
        "customer_monthly": customer_monthly,
        "deposits": deposits,
        "cards": cards,
        "loans": loans,
        "customer_service": service,
    }


def test_economic_profit_matches_spec_waterfall_formula() -> None:
    settings = small_settings()
    tables = _minimal_tables_for_engine()
    monthly = build_monthly_profitability(tables, settings)
    row = monthly.iloc[0]

    expected_economic_profit = (
        row["total_revenue"]
        + row["deposit_contribution"]
        - row["expected_loss"]
        - row["operating_cost"]
        - row["acquisition_cost"]
    )
    assert row["economic_profit"] == expected_economic_profit

    # And the intermediate waterfall stages must chain to the same total.
    assert row["gross_contribution"] == row["total_revenue"] + row["deposit_contribution"] - row["operating_cost"]
    assert row["risk_adjusted_contribution"] == row["gross_contribution"] - row["expected_loss"]
    assert row["economic_profit"] == row["risk_adjusted_contribution"] - row["acquisition_cost"]


def test_customer_summary_metrics() -> None:
    settings = small_settings()
    tables = _minimal_tables_for_engine()
    monthly = build_monthly_profitability(tables, settings)
    summary = build_customer_profitability_summary(monthly)
    row = summary.iloc[0]

    assert row["months_observed"] == 1
    assert row["revenue_per_month"] == row["total_revenue"]
    assert row["total_costs"] == row["total_expected_loss"] + row["total_operating_cost"] + row["total_acquisition_cost"]
    assert row["profit_margin"] == row["economic_profit"] / row["total_revenue"]


def test_aggregate_profit_quarterly_sums_months() -> None:
    settings = small_settings()
    tables = _minimal_tables_for_engine()
    monthly = build_monthly_profitability(tables, settings)
    two_months = pd.concat([monthly, monthly.assign(month=pd.Timestamp("2023-02-01"))], ignore_index=True)
    quarterly = aggregate_profit(two_months, freq="Q")
    assert len(quarterly) == 1  # both months fall in the same quarter
    assert quarterly.loc[0, "economic_profit"] == monthly["economic_profit"].iloc[0] * 2
