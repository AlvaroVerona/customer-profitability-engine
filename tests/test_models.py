"""Phase 6 -- machine learning model tests.

Focuses on the parts that are easy to get subtly wrong and hard to notice
from aggregate metrics alone: the forward-shift target construction (no
leakage), the calendar-based train/val/test split, product-ownership
derivation, and calibration -- rather than re-testing sklearn/XGBoost/SHAP
themselves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from customer_profitability.models.balance import build_balance_dataset
from customer_profitability.models.churn import build_churn_dataset
from customer_profitability.models.evaluation import (
    calibrate_probabilities,
    classification_metrics,
    fit_and_evaluate_classification,
    fit_and_evaluate_regression,
    global_feature_importance,
    make_classification_models,
    regression_metrics,
    time_based_split,
)
from customer_profitability.models.product_adoption import build_adoption_dataset
from customer_profitability.models.revenue import build_revenue_dataset

from .conftest import small_settings

RNG = np.random.default_rng(42)


def _months(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2023-01-01", periods=n, freq="MS")


def test_churn_target_is_next_month_not_current_month() -> None:
    customer_360 = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C1"],
            "month": _months(3),
            "churned_this_month": [False, False, True],
        }
    )
    out = build_churn_dataset(customer_360)
    # Row for month 1 should be labeled with month 2's outcome (False), not its own.
    row1 = out[out["month"] == pd.Timestamp("2023-01-01")].iloc[0]
    assert row1["target"] == 0.0
    # Row for month 2 should be labeled with month 3's outcome (True: churns then).
    row2 = out[out["month"] == pd.Timestamp("2023-02-01")].iloc[0]
    assert row2["target"] == 1.0
    # Month 3 (the churn month itself) has no month 4 to read a label from -> dropped.
    assert (out["month"] == pd.Timestamp("2023-03-01")).sum() == 0


def test_churn_dataset_drops_customers_last_observed_month() -> None:
    # A customer who never churns but whose panel simply ends: their last
    # row also has no known future outcome and must be dropped too.
    customer_360 = pd.DataFrame(
        {"customer_id": ["C1", "C1"], "month": _months(2), "churned_this_month": [False, False]}
    )
    out = build_churn_dataset(customer_360)
    assert len(out) == 1
    assert out.iloc[0]["month"] == pd.Timestamp("2023-01-01")


def test_revenue_dataset_target_is_next_month_revenue() -> None:
    customer_360 = pd.DataFrame({"customer_id": ["C1", "C1"], "month": _months(2)})
    profitability = pd.DataFrame(
        {"customer_id": ["C1", "C1"], "month": _months(2), "total_revenue": [50.0, 80.0]}
    )
    out = build_revenue_dataset(customer_360, profitability)
    assert len(out) == 1
    assert out.iloc[0]["target"] == 80.0  # month 1's row is labeled with month 2's revenue


def test_balance_dataset_restricts_loan_balance_to_active_loans() -> None:
    customer_360 = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C2", "C2"],
            "month": list(_months(2)) * 2,
            "outstanding_balance": [1000.0, 1200.0, 0.0, 0.0],  # C1 has a loan, C2 never does
        }
    )
    out = build_balance_dataset(customer_360, "loan_balance")
    assert len(out) == 1
    assert out.iloc[0]["customer_id"] == "C1"
    assert out.iloc[0]["target"] == 1200.0


def test_balance_dataset_deposit_balance_keeps_all_customers() -> None:
    customer_360 = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C2", "C2"],
            "month": list(_months(2)) * 2,
            "average_balance": [100.0, 150.0, 0.0, 0.0],
        }
    )
    out = build_balance_dataset(customer_360, "deposit_balance")
    assert len(out) == 2  # both customers' first row kept, even C2 with a 0 balance


def test_adoption_dataset_ownership_and_eligibility() -> None:
    customer_360 = pd.DataFrame(
        {"customer_id": ["C1", "C1", "C1", "C2", "C2"], "month": list(_months(3)) + list(_months(2))}
    )
    accounts = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "product": ["savings_account"],
            "account_open_date": pd.to_datetime(["2023-02-01"]),  # C1 adopts in month 2
            "account_close_date": pd.to_datetime([pd.NaT]),
        }
    )
    out = build_adoption_dataset(customer_360, accounts, "savings_account")
    # C1 month 1: doesn't have it yet, adopts next month -> eligible, target=1.
    c1_month1 = out[(out["customer_id"] == "C1") & (out["month"] == pd.Timestamp("2023-01-01"))]
    assert len(c1_month1) == 1
    assert c1_month1.iloc[0]["target"] == 1
    # C1 month 2: already owns it -> not eligible (excluded from the dataset).
    assert not ((out["customer_id"] == "C1") & (out["month"] == pd.Timestamp("2023-02-01"))).any()
    # C2 never adopts and has no known future beyond month 2 -> only month 1 is eligible+labeled.
    c2_rows = out[out["customer_id"] == "C2"]
    assert len(c2_rows) == 1
    assert c2_rows.iloc[0]["target"] == 0


def test_time_based_split_respects_config_boundaries() -> None:
    settings = small_settings()  # test_months=3, validation_months=2, 12 months total
    df = pd.DataFrame({"month": _months(12)})
    train, val, test = time_based_split(df, settings.models)
    assert test.sum() == 3
    assert val.sum() == 2
    assert train.sum() == 7
    assert not (train & val).any()
    assert not (val & test).any()
    # Test months must be strictly later than validation months, which must be
    # strictly later than train months.
    assert df.loc[test, "month"].min() > df.loc[val, "month"].max()
    assert df.loc[val, "month"].min() > df.loc[train, "month"].max()


def test_classification_metrics_perfect_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.01, 0.02, 0.98, 0.99])
    m = classification_metrics(y_true, y_prob)
    assert m["roc_auc"] == 1.0
    assert m["precision_at_0.5"] == 1.0
    assert m["recall_at_0.5"] == 1.0
    assert m["brier_score"] < 0.01


def test_regression_metrics_zero_error() -> None:
    y = np.array([10.0, 20.0, 30.0])
    m = regression_metrics(y, y)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["r_squared"] == 1.0


def test_calibration_improves_brier_score_after_rebalanced_training() -> None:
    n = 8000
    X = pd.DataFrame({"x": RNG.normal(0, 1, n)})
    true_p = 1 / (1 + np.exp(-(2.0 * X["x"] - 3.0)))  # rare positive class
    y = (RNG.random(n) < true_p).astype(int)

    split = n // 2
    X_train, y_train = X.iloc[:split], y[:split]
    X_val, y_val = X.iloc[split:], y[split:]

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(class_weight="balanced").fit(X_train, y_train)
    raw_prob = model.predict_proba(X_val)[:, 1]
    raw_brier = classification_metrics(y_val, raw_prob)["brier_score"]

    calibrated = calibrate_probabilities(model, X_val, y_val)
    cal_prob = calibrated.predict_proba(X_val)[:, 1]
    cal_brier = classification_metrics(y_val, cal_prob)["brier_score"]

    assert cal_brier < raw_brier  # class-balanced training miscalibrates; Platt scaling fixes it


def test_fit_and_evaluate_classification_end_to_end() -> None:
    n = 3000
    df = pd.DataFrame(
        {
            "month": pd.to_datetime(RNG.choice(_months(10), n)),
            "age": RNG.uniform(20, 70, n),
            "income": RNG.uniform(10000, 100000, n),
            "tenure_months": RNG.uniform(0, 60, n),
            "login_frequency": RNG.uniform(0, 30, n),
            "product_count": RNG.integers(1, 5, n),
            "support_interactions": RNG.integers(0, 5, n),
            "average_balance": RNG.uniform(0, 5000, n),
            "balance_volatility_3m": RNG.uniform(0, 500, n),
            "rate_differential": RNG.uniform(-0.02, 0.02, n),
            "transaction_count": RNG.uniform(0, 30, n),
            "transaction_volume": RNG.uniform(0, 2000, n),
            "average_transaction_value": RNG.uniform(10, 100, n),
            "atm_usage_share": RNG.uniform(0, 1, n),
            "outstanding_balance": RNG.uniform(0, 3000, n),
            "utilization": RNG.uniform(0, 1, n),
            "expected_loss": RNG.uniform(0, 50, n),
            "balance_growth_1m": RNG.normal(0, 0.1, n),
            "spending_growth_1m": RNG.normal(0, 0.1, n),
            "activity_volatility_3m": RNG.uniform(0, 5, n),
            "recent_activity_ratio": RNG.uniform(0, 2, n),
            "customer_segment": RNG.choice(["Mass", "Affluent", "Premium"], n),
            "target": RNG.integers(0, 2, n),
        }
    )
    settings = small_settings(n_months=10)
    train, val, test = time_based_split(df, settings.models)
    out = fit_and_evaluate_classification(df, train, val, test)

    assert set(out["models"].keys()) == {"logistic_regression", "random_forest", "xgboost"}
    for res in out["models"].values():
        assert 0 <= res["test_metrics"]["roc_auc"] <= 1
        assert "test_metrics_calibrated" in res


def test_fit_and_evaluate_regression_end_to_end() -> None:
    n = 3000
    df = pd.DataFrame(
        {
            "month": pd.to_datetime(RNG.choice(_months(10), n)),
            "age": RNG.uniform(20, 70, n),
            "income": RNG.uniform(10000, 100000, n),
            "tenure_months": RNG.uniform(0, 60, n),
            "login_frequency": RNG.uniform(0, 30, n),
            "product_count": RNG.integers(1, 5, n),
            "support_interactions": RNG.integers(0, 5, n),
            "average_balance": RNG.uniform(0, 5000, n),
            "balance_volatility_3m": RNG.uniform(0, 500, n),
            "rate_differential": RNG.uniform(-0.02, 0.02, n),
            "transaction_count": RNG.uniform(0, 30, n),
            "transaction_volume": RNG.uniform(0, 2000, n),
            "average_transaction_value": RNG.uniform(10, 100, n),
            "atm_usage_share": RNG.uniform(0, 1, n),
            "outstanding_balance": RNG.uniform(0, 3000, n),
            "utilization": RNG.uniform(0, 1, n),
            "expected_loss": RNG.uniform(0, 50, n),
            "balance_growth_1m": RNG.normal(0, 0.1, n),
            "spending_growth_1m": RNG.normal(0, 0.1, n),
            "activity_volatility_3m": RNG.uniform(0, 5, n),
            "recent_activity_ratio": RNG.uniform(0, 2, n),
            "customer_segment": RNG.choice(["Mass", "Affluent", "Premium"], n),
        }
    )
    df["target"] = 10 + 0.01 * df["income"] + RNG.normal(0, 5, n)
    settings = small_settings(n_months=10)
    train, val, test = time_based_split(df, settings.models)
    out = fit_and_evaluate_regression(df, train, val, test)

    assert set(out["models"].keys()) == {"linear_regression", "random_forest", "xgboost"}
    for res in out["models"].values():
        assert res["test_metrics"]["mae"] >= 0


def test_global_feature_importance_orders_by_magnitude() -> None:
    shap_values = np.array([[0.1, -0.9], [0.2, 0.8], [-0.15, 0.85]])
    importance = global_feature_importance(shap_values, ["low_impact", "high_impact"])
    assert importance.index[0] == "high_impact"
    assert importance.iloc[0] > importance.iloc[1]


def test_make_classification_models_returns_all_three() -> None:
    models = make_classification_models(scale_pos_weight=5.0)
    assert set(models.keys()) == {"logistic_regression", "random_forest", "xgboost"}
