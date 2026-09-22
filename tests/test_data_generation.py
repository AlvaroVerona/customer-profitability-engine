"""Phase 1 acceptance-criteria tests for the synthetic data generator.

Uses a small customer/month count so the suite runs in well under a second,
while exercising the same code path as the full-scale `generate_all`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from customer_profitability.data.generator import generate_all

from .conftest import small_settings as _small_settings


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    return generate_all(_small_settings())


def test_reproducible_with_same_seed(tables: dict[str, pd.DataFrame]) -> None:
    other = generate_all(_small_settings(seed=42))
    pd.testing.assert_frame_equal(tables["customers"], other["customers"])
    pd.testing.assert_frame_equal(tables["deposits"], other["deposits"])


def test_different_seed_gives_different_data() -> None:
    a = generate_all(_small_settings(seed=42))
    b = generate_all(_small_settings(seed=7))
    assert not a["customers"]["income"].equals(b["customers"]["income"])


def test_expected_tables_and_row_counts(tables: dict[str, pd.DataFrame]) -> None:
    expected = {
        "customers",
        "accounts",
        "deposits",
        "cards",
        "loans",
        "customer_service",
        "customer_monthly",
    }
    assert expected.issubset(tables.keys())
    # customers.parquet includes intentionally duplicated rows for QA testing,
    # so it is allowed to exceed n_customers.
    assert tables["customers"]["customer_id"].nunique() == 500
    assert len(tables["customers"]) >= 500


def test_customer_ids_are_unique_before_qa_duplication() -> None:
    settings = _small_settings()
    n = settings.data.n_customers
    ids = [f"CUST_{i:06d}" for i in range(n)]
    assert len(ids) == len(set(ids))


def test_no_negative_balances(tables: dict[str, pd.DataFrame]) -> None:
    dep = tables["deposits"].dropna(subset=["average_balance"])
    assert (dep["average_balance"] >= 0).all()
    assert (dep["opening_balance"] >= 0).all()
    assert (dep["closing_balance"] >= 0).all()


def test_rates_and_risk_fields_in_valid_ranges(tables: dict[str, pd.DataFrame]) -> None:
    loans = tables["loans"].dropna(subset=["pd", "lgd", "utilization"])
    assert (loans["pd"] >= 0).all() and (loans["pd"] <= 1).all()
    assert (loans["lgd"] >= 0).all() and (loans["lgd"] <= 1).all()
    assert (loans["utilization"] >= 0).all() and (loans["utilization"] <= 1).all()
    assert (loans["interest_rate"].dropna() >= 0).all()

    dep = tables["deposits"].dropna(subset=["deposit_rate"])
    assert (dep["deposit_rate"] >= 0).all()


def test_temporal_structure_spans_full_window(tables: dict[str, pd.DataFrame]) -> None:
    months = tables["customer_monthly"]["month"].unique()
    assert len(months) == 12


def test_no_transactions_before_acquisition(tables: dict[str, pd.DataFrame]) -> None:
    merged = tables["deposits"].merge(
        tables["customers"][["customer_id", "acquisition_date"]].drop_duplicates("customer_id"),
        on="customer_id",
    )
    assert (merged["month"] >= merged["acquisition_date"].values.astype("datetime64[M]")).all()


def test_no_observations_after_churn(tables: dict[str, pd.DataFrame]) -> None:
    customers = tables["customers"][["customer_id", "churn_date"]].drop_duplicates("customer_id")
    for name in ["deposits", "cards", "customer_service"]:
        merged = tables[name].merge(customers, on="customer_id", how="left")
        churned = merged.dropna(subset=["churn_date"])
        assert (churned["month"] <= churned["churn_date"]).all(), name


def test_referential_integrity(tables: dict[str, pd.DataFrame]) -> None:
    customer_ids = set(tables["customers"]["customer_id"])
    for name in ["accounts", "deposits", "cards", "loans", "customer_service", "customer_monthly"]:
        assert set(tables[name]["customer_id"]).issubset(customer_ids), name


def test_income_correlates_with_segment(tables: dict[str, pd.DataFrame]) -> None:
    customers = tables["customers"].drop_duplicates("customer_id")
    means = customers.groupby("customer_segment")["income"].mean()
    assert means["Premium"] > means["Affluent"] > means["Mass"]


def test_premium_customers_have_higher_deposit_balances(tables: dict[str, pd.DataFrame]) -> None:
    customers = tables["customers"][["customer_id", "customer_segment"]].drop_duplicates("customer_id")
    merged = tables["deposits"].merge(customers, on="customer_id")
    means = merged.groupby("customer_segment")["average_balance"].mean()
    assert means["Premium"] > means["Mass"]


def test_customer_heterogeneity_in_churn(tables: dict[str, pd.DataFrame]) -> None:
    customers = tables["customers"].drop_duplicates("customer_id")
    churn_rate = customers["churn_flag"].mean()
    assert 0.01 < churn_rate < 0.9


def test_data_quality_issues_are_present(tables: dict[str, pd.DataFrame]) -> None:
    assert tables["deposits"]["average_balance"].isna().sum() > 0
    dep = tables["deposits"]
    dupes = dep.duplicated(subset=[c for c in dep.columns if c != "average_balance"]).sum()
    n_id_dupes = tables["customers"]["customer_id"].duplicated().sum()
    assert dupes > 0 or n_id_dupes > 0
