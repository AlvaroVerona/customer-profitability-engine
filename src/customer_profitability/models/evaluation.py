"""Phase 6 -- shared feature preparation, time-based splitting, evaluation
metrics, and SHAP explainability used by every model in this package.

Feature/target construction lives in each model's own module (churn.py,
revenue.py, ...) since the target differs; everything that is genuinely
shared -- the feature matrix, the train/validation/test split, and how a
fitted model gets scored/explained -- lives here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from customer_profitability.utils.config import ModelsConfig
from customer_profitability.utils.metrics import safe_divide

RANDOM_STATE = 42

# A curated subset of Customer 360's 60 columns: interpretable, low-redundancy
# drivers rather than every rolling window at every horizon (which would mostly
# just re-encode the same signal 3x and inflate multicollinearity for no gain).
NUMERIC_FEATURES = [
    "age",
    "income",
    "tenure_months",
    "login_frequency",
    "product_count",
    "support_interactions",
    "average_balance",
    "balance_volatility_3m",
    "rate_differential",
    "transaction_count",
    "transaction_volume",
    "average_transaction_value",
    "atm_usage_share",
    "outstanding_balance",
    "utilization",
    "expected_loss",
    "balance_growth_1m",
    "spending_growth_1m",
    "activity_volatility_3m",
    "recent_activity_ratio",
]
CATEGORICAL_FEATURES = ["customer_segment"]


def prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """NaN in a ratio/growth column (e.g. `utilization` for a customer with
    no loan) means "not applicable", which for a model that cannot accept
    NaN natively (scikit-learn) is encoded as 0 -- the neutral/no-signal
    value for every feature in this list. XGBoost could accept the NaNs
    directly, but using the same matrix for all model families keeps the
    comparison apples-to-apples."""
    numeric = df[NUMERIC_FEATURES].fillna(0.0).reset_index(drop=True)
    dummies = pd.get_dummies(df[CATEGORICAL_FEATURES].reset_index(drop=True), drop_first=True)
    return pd.concat([numeric, dummies], axis=1)


def time_based_split(
    df: pd.DataFrame, models_config: ModelsConfig, month_col: str = "month"
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Boolean masks for train / validation / test, split by calendar month
    (never by row), so no customer's future observation leaks into training
    through a different customer-month of the same calendar period.

        test:       the last `test_months` calendar months
        validation: the `validation_months` immediately before that
        train:      everything earlier
    """
    months = pd.to_datetime(df[month_col])
    unique_months = sorted(months.unique())
    n = len(unique_months)
    test_start = unique_months[max(n - models_config.test_months, 0)]
    val_start = unique_months[max(n - models_config.test_months - models_config.validation_months, 0)]

    is_test = months >= test_start
    is_val = (months >= val_start) & (months < test_start)
    is_train = months < val_start
    return is_train, is_val, is_test


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision_at_0.5": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_0.5": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "base_rate": float(np.mean(y_true)),
        "n": len(y_true),
    }


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Predicted-probability deciles vs. actual event rate -- a well
    calibrated model has `predicted_mean` ~= `actual_rate` in every bin,
    which matters here because churn probabilities feed directly into
    Phase 7's CLV survival adjustment."""
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df["bin"] = pd.qcut(df["y_prob"], q=n_bins, duplicates="drop")
    out = df.groupby("bin", observed=True).agg(
        predicted_mean=("y_prob", "mean"), actual_rate=("y_true", "mean"), n=("y_true", "size")
    )
    return out.reset_index(drop=True)


def calibrate_probabilities(fitted_model, X_val: pd.DataFrame, y_val: pd.Series):
    """Post-hoc Platt scaling (sigmoid) of an already-fitted classifier's
    probabilities, fit on the validation set.

    Every classifier in `make_classification_models` is trained with class
    rebalancing (`class_weight="balanced"` / `scale_pos_weight`) to get
    usable ranking performance on a rare event (churn ~2%/month, product
    adoption ~1-2%/month) -- but rebalancing systematically shifts
    `predict_proba` away from true probabilities (it optimizes ranking, not
    calibration). That is fine for ROC-AUC/PR-AUC, but PROJECT_SPEC.md 6.1
    is explicit that churn probabilities feed Phase 7's CLV survival term,
    where the *absolute* probability value matters, not just the ranking --
    so the champion model used downstream should be this calibrated
    wrapper, not the raw rebalanced classifier.
    """
    calibrated = CalibratedClassifierCV(FrozenEstimator(fitted_model), method="sigmoid")
    calibrated.fit(X_val, y_val)
    return calibrated


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    nonzero = y_true != 0
    mape = float(np.mean(np.abs(safe_divide(y_true - y_pred, y_true)[nonzero]))) if nonzero.any() else float("nan")
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "mape": mape, "r_squared": r2, "n": len(y_true)}


def make_classification_models(scale_pos_weight: float = 1.0) -> dict:
    """Baseline (logistic regression) vs. two tree ensembles, per Phase 15's
    "compare sophisticated models against simple baselines". `scale_pos_weight`
    should be set to (n_negative / n_positive) for XGBoost on an imbalanced
    target such as churn or product adoption; logistic regression and random
    forest use `class_weight="balanced"` for the same purpose."""
    return {
        # Scaled: unlike the tree ensembles, gradient-based LogisticRegression
        # needs comparable feature scales (income ~1e4-1e5 vs. e.g.
        # activity_volatility_3m ~1e0) to converge in a reasonable number of
        # iterations.
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=50,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        ),
    }


def make_regression_models() -> dict:
    """Baseline (linear regression) vs. two tree ensembles, per Phase 15.
    `max_samples=0.3` (standard bagging subsampling) keeps the random
    forest's runtime reasonable on the ~250k-row training sets in this
    project without materially hurting accuracy -- each of the 200 trees
    still sees a large, independently-drawn sample."""
    return {
        "linear_regression": make_pipeline(StandardScaler(), LinearRegression()),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=20,
            max_samples=0.3,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE),
    }


def fit_and_evaluate_classification(
    df: pd.DataFrame, train_mask: pd.Series, val_mask: pd.Series, test_mask: pd.Series, target_col: str = "target"
) -> dict:
    """Shared fit/evaluate loop for every binary classification model in
    Phase 6 (churn, product adoption): fits baseline + both tree ensembles
    on the same train split, scores all three on validation and (held-out,
    future) test."""
    X = prepare_feature_matrix(df)
    y = df[target_col].astype(int)
    train_np, val_np, test_np = train_mask.to_numpy(), val_mask.to_numpy(), test_mask.to_numpy()
    X_train, y_train = X[train_np], y[train_np]
    X_val, y_val = X[val_np], y[val_np]
    X_test, y_test = X[test_np], y[test_np]

    n_pos = max(int(y_train.sum()), 1)
    scale_pos_weight = (len(y_train) - n_pos) / n_pos
    models = make_classification_models(scale_pos_weight=scale_pos_weight)

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]

        calibrated = calibrate_probabilities(model, X_val, y_val)
        calibrated_test_prob = calibrated.predict_proba(X_test)[:, 1]

        results[name] = {
            "model": model,
            "calibrated_model": calibrated,
            "validation_metrics": classification_metrics(y_val.to_numpy(), val_prob),
            "test_metrics": classification_metrics(y_test.to_numpy(), test_prob),
            "test_calibration": calibration_table(y_test.to_numpy(), test_prob),
            "test_metrics_calibrated": classification_metrics(y_test.to_numpy(), calibrated_test_prob),
            "test_calibration_calibrated": calibration_table(y_test.to_numpy(), calibrated_test_prob),
        }
    return {
        "feature_columns": list(X.columns),
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "models": results,
    }


def fit_and_evaluate_regression(
    df: pd.DataFrame, train_mask: pd.Series, val_mask: pd.Series, test_mask: pd.Series, target_col: str = "target"
) -> dict:
    """Shared fit/evaluate loop for every regression model in Phase 6
    (revenue, balances): fits baseline + both tree ensembles on the same
    train split, scores all three on validation and test."""
    X = prepare_feature_matrix(df)
    y = df[target_col].astype(float)
    train_np, val_np, test_np = train_mask.to_numpy(), val_mask.to_numpy(), test_mask.to_numpy()
    X_train, y_train = X[train_np], y[train_np]
    X_val, y_val = X[val_np], y[val_np]
    X_test, y_test = X[test_np], y[test_np]

    models = make_regression_models()
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        results[name] = {
            "model": model,
            "validation_metrics": regression_metrics(y_val.to_numpy(), val_pred),
            "test_metrics": regression_metrics(y_test.to_numpy(), test_pred),
        }
    return {"feature_columns": list(X.columns), "X_test": X_test, "y_test": y_test, "models": results}


def global_feature_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.Series:
    """Mean |SHAP value| per feature -- the standard SHAP "global
    importance" summary (Phase 6.5)."""
    importance = np.abs(shap_values).mean(axis=0)
    return pd.Series(importance, index=feature_names, name="mean_abs_shap").sort_values(ascending=False)


def explain_customer(shap_values: np.ndarray, X: pd.DataFrame, row_index: int) -> pd.Series:
    """Per-feature SHAP contribution for a single customer's prediction
    (Phase 6.5 "individual customer explanation"), sorted by magnitude."""
    contributions = pd.Series(shap_values[row_index], index=X.columns, name="shap_value")
    return contributions.reindex(contributions.abs().sort_values(ascending=False).index)
