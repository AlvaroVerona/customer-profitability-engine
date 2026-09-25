"""Phase 2 acceptance tests for the data quality engine.

Builds small, hand-crafted tables with known, deliberate defects (rather than
reusing the generator) so each check can be verified against an exact
expected count.
"""

from __future__ import annotations

import pandas as pd
import pytest

from customer_profitability.data.quality import run_quality_checks, split_validated_quarantined


@pytest.fixture
def customers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4"],
            "age": [30, 40, 200, 50],  # C3: impossible age
            "income": [30000, 40000, 50000, 60000],
            "country": ["ES", "PT", "IT", "FR"],
            "acquisition_channel": ["organic", "organic", "organic", "organic"],
            "acquisition_date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-01", "2023-01-01"]),
            "customer_segment": ["Mass", "Mass", "Mass", "Mass"],
            "employment_status": ["employed", "employed", "employed", "employed"],
            "rate_sensitivity": [0.5, 0.5, 0.5, 0.5],
            "satisfaction_baseline": [0.7, 0.7, 0.7, 0.7],
            "churn_flag": [False, False, False, True],
            "churn_date": pd.to_datetime([pd.NaT, pd.NaT, pd.NaT, "2022-06-01"]),  # C4: churn before acquisition
        }
    )


@pytest.fixture
def deposits() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C2", "C3", "C9"],  # C9: not a real customer
            "month": pd.to_datetime(
                ["2023-01-01", "2023-02-01", "2022-12-01", "2023-01-01", "2023-01-01"]
            ),  # C2: month before its own acquisition
            "product": ["current_account"] * 5,
            "opening_balance": [100.0, 100.0, 100.0, -50.0, 100.0],  # C3: negative balance
            "closing_balance": [100.0, 100.0, 100.0, 100.0, 100.0],
            "average_balance": [100.0, 100.0, 100.0, 100.0, 100.0],
            "inflows": [10.0, 10.0, 10.0, 10.0, 10.0],
            "outflows": [5.0, 5.0, 5.0, 5.0, 5.0],
            "deposit_rate": [0.02, 0.02, 0.02, 0.02, 0.02],
            "market_rate": [0.03, 0.03, 0.03, 0.03, 0.03],
        }
    )
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate the first row


@pytest.fixture
def tables(customers: pd.DataFrame, deposits: pd.DataFrame) -> dict[str, pd.DataFrame]:
    empty = lambda cols: pd.DataFrame(columns=cols)
    return {
        "customers": customers,
        "accounts": empty(["customer_id", "product", "account_open_date", "account_close_date", "is_active"]),
        "deposits": deposits,
        "cards": empty(
            [
                "customer_id",
                "month",
                "transaction_count",
                "transaction_volume",
                "interchange_revenue",
                "atm_withdrawal_volume",
                "international_transaction_volume",
                "payment_processing_cost",
            ]
        ),
        "loans": empty(
            [
                "loan_id",
                "customer_id",
                "month",
                "origination_date",
                "outstanding_balance",
                "credit_limit",
                "utilization",
                "interest_rate",
                "interest_income",
                "pd",
                "lgd",
                "ead",
            ]
        ),
        "customer_service": empty(
            ["customer_id", "month", "support_contacts", "average_handling_time", "support_channel", "estimated_service_cost"]
        ),
        "customer_monthly": empty(
            [
                "customer_id",
                "month",
                "tenure_months",
                "is_active",
                "product_count",
                "total_deposit_balance",
                "total_loan_balance",
                "transaction_count",
                "transaction_volume",
                "support_contacts",
                "login_frequency",
                "satisfaction_proxy",
                "churned_this_month",
            ]
        ),
    }


def test_detects_impossible_age_and_churn_before_acquisition(tables: dict[str, pd.DataFrame]) -> None:
    report, _ = run_quality_checks(tables)
    r = report.tables["customers"]
    assert r.invalid_value_counts["age_out_of_range"] == 1
    assert r.invalid_value_counts["churn_before_acquisition"] == 1


def test_detects_negative_balance(tables: dict[str, pd.DataFrame]) -> None:
    report, _ = run_quality_checks(tables)
    assert report.tables["deposits"].invalid_value_counts["opening_balance_negative"] == 1


def test_detects_duplicate_rows(tables: dict[str, pd.DataFrame]) -> None:
    report, _ = run_quality_checks(tables)
    assert report.tables["deposits"].duplicate_count == 1


def test_detects_observation_before_acquisition(tables: dict[str, pd.DataFrame]) -> None:
    report, _ = run_quality_checks(tables)
    assert report.tables["deposits"].temporal_violations["before_acquisition"] == 1


def test_detects_referential_integrity_violation(tables: dict[str, pd.DataFrame]) -> None:
    report, _ = run_quality_checks(tables)
    assert report.tables["deposits"].referential_violations["customer_id_not_in_customers"] == 1


def test_no_rows_are_dropped_only_split(tables: dict[str, pd.DataFrame]) -> None:
    _, masks = run_quality_checks(tables)
    validated, quarantined = split_validated_quarantined(tables, masks)
    for name, df in tables.items():
        assert len(validated[name]) + len(quarantined[name]) == len(df)


def test_clean_row_is_validated_not_quarantined(tables: dict[str, pd.DataFrame]) -> None:
    _, masks = run_quality_checks(tables)
    validated, _quarantined = split_validated_quarantined(tables, masks)
    # The clean C1/2023-01 deposit row must survive as validated.
    assert ((validated["deposits"]["customer_id"] == "C1") & (validated["deposits"]["month"] == "2023-01-01")).any()


def test_detects_missing_and_extra_columns(tables: dict[str, pd.DataFrame]) -> None:
    # Phase 16 "schema" coverage: _check_structural is computed and reported
    # for every table but was previously untested. `market_rate` is dropped
    # (rather than a column referenced by the financial/temporal checks)
    # because those checks assume the documented schema and are not meant
    # to be resilient to an arbitrarily malformed table -- that contract is
    # exactly what this structural check itself exists to report.
    broken = dict(tables)
    broken["deposits"] = tables["deposits"].drop(columns=["market_rate"]).assign(unexpected_col=0)
    report, _ = run_quality_checks(broken)
    r = report.tables["deposits"]
    assert r.missing_columns == ["market_rate"]
    assert r.extra_columns == ["unexpected_col"]


def test_missing_primary_key_component_is_quarantined(tables: dict[str, pd.DataFrame]) -> None:
    # Phase 16 "schema" coverage: a row missing a primary-key component
    # must be quarantined even if every other check passes.
    broken = dict(tables)
    deposits = tables["deposits"].copy()
    deposits.loc[0, "product"] = None
    broken["deposits"] = deposits
    report, masks = run_quality_checks(broken)
    assert report.tables["deposits"].missing_primary_key_count == 1
    assert masks["deposits"].iloc[0]


def test_expected_missing_churn_date_is_not_penalized(customers: pd.DataFrame) -> None:
    tables = {
        "customers": customers,
        "accounts": pd.DataFrame(columns=["customer_id", "product", "account_open_date", "account_close_date", "is_active"]),
        "deposits": pd.DataFrame(
            columns=["customer_id", "month", "product", "opening_balance", "closing_balance", "average_balance", "inflows", "outflows", "deposit_rate", "market_rate"]
        ),
        "cards": pd.DataFrame(
            columns=["customer_id", "month", "transaction_count", "transaction_volume", "interchange_revenue", "atm_withdrawal_volume", "international_transaction_volume", "payment_processing_cost"]
        ),
        "loans": pd.DataFrame(
            columns=["loan_id", "customer_id", "month", "origination_date", "outstanding_balance", "credit_limit", "utilization", "interest_rate", "interest_income", "pd", "lgd", "ead"]
        ),
        "customer_service": pd.DataFrame(
            columns=["customer_id", "month", "support_contacts", "average_handling_time", "support_channel", "estimated_service_cost"]
        ),
        "customer_monthly": pd.DataFrame(
            columns=["customer_id", "month", "tenure_months", "is_active", "product_count", "total_deposit_balance", "total_loan_balance", "transaction_count", "transaction_volume", "support_contacts", "login_frequency", "satisfaction_proxy", "churned_this_month"]
        ),
    }
    report, masks = run_quality_checks(tables)
    r = report.tables["customers"]
    churn_date_report = next(m for m in r.missingness if m.column == "churn_date")
    assert churn_date_report.severity == "expected"
    # C1 and C2 have no other defects; their missing churn_date alone must not
    # trigger quarantine (C3 fails the age check, C4 fails churn-before-acquisition).
    idx_by_id = customers.set_index("customer_id").index
    clean_mask = masks["customers"].set_axis(idx_by_id)
    assert not clean_mask.loc[["C1", "C2"]].any()
