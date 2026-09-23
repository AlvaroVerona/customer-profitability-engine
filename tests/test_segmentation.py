"""Phase 8 -- economic customer segmentation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from customer_profitability.segmentation.clustering import (
    FEATURE_COLUMNS,
    build_segmentation_features,
    evaluate_k_range,
    fit_kmeans,
    standardize_features,
)
from customer_profitability.segmentation.profiling import (
    assign_segments,
    label_segments,
    profile_segments,
)


def _synthetic_customer_360() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C2", "C3"],
            "month": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-02-01", "2023-02-01"]),
            "login_frequency": [10.0, 12.0, 20.0, 5.0],
            "transaction_volume_avg_3m": [200.0, 250.0, 1000.0, 100.0],
            "product_count": [2, 2, 4, 1],
            "average_balance_avg_3m": [1000.0, 1200.0, 8000.0, 300.0],
            "utilization": [0.1, 0.1, np.nan, 0.5],
        }
    )


def _synthetic_profitability_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4"],
            "total_revenue": [50.0, 500.0, 10.0, 20.0],
            "total_deposit_contribution": [5.0, 50.0, 1.0, 2.0],
            "total_expected_loss": [1.0, 5.0, 0.0, 0.5],
            "total_operating_cost": [10.0, 20.0, 5.0, 8.0],
        }
    )


def _synthetic_clv() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4"],
            "historical_economic_profit": [40.0, 400.0, 5.0, 10.0],
            "total_customer_economic_value": [80.0, 900.0, 8.0, 15.0],
            "p_churn_monthly": [0.02, 0.01, 0.03, 0.5],
            "is_active": [True, True, True, False],  # C4 already churned -> must be excluded
        }
    )


def test_build_segmentation_features_excludes_churned_customers() -> None:
    df = build_segmentation_features(_synthetic_profitability_summary(), _synthetic_clv(), _synthetic_customer_360())
    assert "C4" not in set(df["customer_id"])
    assert set(df["customer_id"]) == {"C1", "C2", "C3"}


def test_build_segmentation_features_uses_latest_month_and_fills_utilization() -> None:
    df = build_segmentation_features(_synthetic_profitability_summary(), _synthetic_clv(), _synthetic_customer_360())
    c1 = df[df["customer_id"] == "C1"].iloc[0]
    assert c1["login_frequency"] == 12.0  # from month 2023-02, not 2023-01
    c2 = df[df["customer_id"] == "C2"].iloc[0]
    assert c2["utilization"] == 0.0  # NaN (no loan) filled to 0, not dropped


def test_standardize_features_produces_zero_mean_unit_variance() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({col: rng.normal(rng.uniform(10, 100), rng.uniform(1, 20), 500) for col in FEATURE_COLUMNS})
    X, _ = standardize_features(df)
    assert np.allclose(X.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(X.std(axis=0), 1, atol=1e-8)


def test_evaluate_k_range_returns_scores_in_valid_range() -> None:
    rng = np.random.default_rng(0)
    # Three well-separated blobs so clustering has real structure to find.
    blobs = [rng.normal(center, 0.3, size=(60, 3)) for center in ([0, 0, 0], [10, 10, 10], [-10, 10, -10])]
    X = np.vstack(blobs)
    result = evaluate_k_range(X, range(2, 5), sample_size=180)
    assert set(result["k"]) == {2, 3, 4}
    assert (result["silhouette_score"] >= -1).all() and (result["silhouette_score"] <= 1).all()
    assert result.loc[result["k"] == 3, "silhouette_score"].iloc[0] > 0.5  # true structure is 3 clusters


def test_fit_kmeans_is_deterministic_and_correct_cluster_count() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal([0, 0], 0.2, size=(50, 2)), rng.normal([10, 10], 0.2, size=(50, 2))])
    labels_a, _ = fit_kmeans(X, 2)
    labels_b, _ = fit_kmeans(X, 2)
    assert len(set(labels_a)) == 2
    assert (labels_a == labels_b).all()


def test_label_segments_high_value() -> None:
    # historical_economic_profit, total_customer_economic_value both high; everything else neutral.
    centers = np.zeros((1, len(FEATURE_COLUMNS)))
    idx = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
    centers[0, idx["historical_economic_profit"]] = 2.0
    centers[0, idx["total_customer_economic_value"]] = 2.0
    labels = label_segments(centers)
    assert labels[0] == "High Value"


def test_label_segments_at_risk_high_value_beats_high_value() -> None:
    centers = np.zeros((1, len(FEATURE_COLUMNS)))
    idx = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
    centers[0, idx["historical_economic_profit"]] = 2.0
    centers[0, idx["total_customer_economic_value"]] = 2.0
    centers[0, idx["p_churn_monthly"]] = 2.0  # also high churn -> should take priority
    labels = label_segments(centers)
    assert labels[0] == "At-Risk High Value"


def test_label_segments_deposit_funders() -> None:
    centers = np.zeros((1, len(FEATURE_COLUMNS)))
    idx = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
    centers[0, idx["total_deposit_contribution"]] = 2.0
    labels = label_segments(centers)
    assert labels[0] == "Deposit Funders"


def test_label_segments_credit_driven() -> None:
    centers = np.zeros((1, len(FEATURE_COLUMNS)))
    idx = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
    centers[0, idx["utilization"]] = 2.0
    labels = label_segments(centers)
    assert labels[0] == "Credit-Driven"


def test_label_segments_low_profitability() -> None:
    centers = np.zeros((1, len(FEATURE_COLUMNS)))
    idx = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
    centers[0, idx["historical_economic_profit"]] = -2.0
    centers[0, idx["total_customer_economic_value"]] = -2.0
    labels = label_segments(centers)
    assert labels[0] == "Low Profitability"


def test_label_segments_fallback_for_undifferentiated_cluster() -> None:
    centers = np.zeros((1, len(FEATURE_COLUMNS)))  # perfectly average on every dimension
    labels = label_segments(centers)
    assert labels[0] == "Balanced Segment 0"


def test_label_segments_disambiguates_duplicate_names() -> None:
    idx = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
    centers = np.zeros((2, len(FEATURE_COLUMNS)))
    # Both clusters trigger "High Value"; cluster 0 has the higher profit.
    for row, profit in enumerate([3.0, 1.5]):
        centers[row, idx["historical_economic_profit"]] = profit
        centers[row, idx["total_customer_economic_value"]] = profit
    labels = label_segments(centers)
    assert labels[0] == "High Value I"  # highest profit ranked first
    assert labels[1] == "High Value II"


def test_profile_segments_and_assign_segments_shapes() -> None:
    df = build_segmentation_features(_synthetic_profitability_summary(), _synthetic_clv(), _synthetic_customer_360())
    X, _ = standardize_features(df)
    labels, model = fit_kmeans(X, 2)
    names = label_segments(model.cluster_centers_)

    profile = profile_segments(df, labels, names)
    assert set(profile["n_customers"]) == {pd.Series(labels).value_counts()[i] for i in range(2)}
    assert profile["n_customers"].sum() == len(df)

    assignments = assign_segments(df, labels, names)
    assert set(assignments["customer_id"]) == set(df["customer_id"])
    assert assignments["segment_name"].isin(names.values()).all()
