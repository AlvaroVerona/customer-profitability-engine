"""Phase 7.1 / 7.2 -- expected future profit and churn-adjusted survival.

Design (documented here because it drives every number in this phase):

- **Churn probability** comes from Phase 6's persisted, *calibrated*
  champion churn model (`data/processed/models/churn_champion.joblib`) --
  PROJECT_SPEC.md 6.1/7.2 are explicit that this is exactly what that
  calibration step was for. It is evaluated once, on each still-active
  customer's most recently observed feature row, and held constant as a
  flat monthly hazard for the whole forecast horizon (see
  `discounting.survival_weighted_annuity_factor`) -- re-forecasting a
  customer's evolving churn risk month by month over a 3-year horizon
  would require chaining every Phase 6 model forward recursively, which is
  out of scope for this phase and is called out as a simplification in the
  final report.
- **Expected profit** (revenue, deposit contribution, expected loss,
  operating cost) is each customer's own trailing 3-month average from the
  Phase 4 profitability panel, held flat over the horizon -- consistent
  with the flat-hazard assumption above (there is no more reason to
  believe next month's profit differs from this month's than there is to
  believe next month's churn risk does), and it is a materially more
  stable base than a single most-recent month, without requiring a second
  chain of forward-looking regression models.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from customer_profitability.models.evaluation import prepare_feature_matrix

RUN_RATE_WINDOW_MONTHS = 3


def latest_snapshot(customer_360: pd.DataFrame) -> pd.DataFrame:
    """Each still-active customer's most recently observed feature row."""
    return (
        customer_360.sort_values(["customer_id", "month"])
        .groupby("customer_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def _load_champion(models_dir: Path, target: str):
    model = joblib.load(models_dir / f"{target}_champion.joblib")
    feature_columns = json.loads((models_dir / f"{target}_feature_columns.json").read_text())
    return model, feature_columns


def predict_churn_probability(snapshot: pd.DataFrame, models_dir: Path) -> pd.Series:
    model, feature_columns = _load_champion(models_dir, "churn")
    X = prepare_feature_matrix(snapshot).reindex(columns=feature_columns, fill_value=0.0)
    p = model.predict_proba(X)[:, 1]
    return pd.Series(p, index=snapshot["customer_id"], name="p_churn_monthly")


def compute_run_rate(profitability_monthly: pd.DataFrame, window_months: int = RUN_RATE_WINDOW_MONTHS) -> pd.DataFrame:
    """Each customer's trailing `window_months`-month average of every
    profit-waterfall component, computed from their most recently observed
    months (not a fixed calendar window, since customers churn/join at
    different times)."""
    df = profitability_monthly.sort_values(["customer_id", "month"])
    recent = df.groupby("customer_id", as_index=False).tail(window_months)
    return recent.groupby("customer_id", as_index=False).agg(
        expected_revenue=("total_revenue", "mean"),
        expected_deposit_contribution=("deposit_contribution", "mean"),
        expected_loss=("expected_loss", "mean"),
        expected_operating_cost=("operating_cost", "mean"),
    )


def build_expected_profit(run_rate: pd.DataFrame) -> pd.DataFrame:
    df = run_rate.copy()
    df["expected_profit"] = (
        df["expected_revenue"] + df["expected_deposit_contribution"] - df["expected_loss"] - df["expected_operating_cost"]
    )
    return df


def estimate_profit_volatility(
    profitability_monthly: pd.DataFrame, default_sigma: float = 0.5, min_sigma: float = 0.15, max_sigma: float = 1.5
) -> pd.Series:
    """Per-customer coefficient of variation of historical monthly economic
    profit, used as the profit-level uncertainty parameter in Phase 7.5's
    Monte Carlo. Customers with fewer than 3 observed months (too little
    history to estimate their own volatility) get `default_sigma`; the
    clip keeps a handful of near-zero-mean customers from producing
    absurd (near-infinite or near-zero) CV values.
    """
    g = profitability_monthly.groupby("customer_id")["economic_profit"]
    n = g.size()
    mean = g.mean()
    std = g.std()
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = (std / mean.abs()).replace([np.inf, -np.inf], np.nan)
    cv = cv.where(n >= 3, default_sigma).fillna(default_sigma)
    return cv.clip(min_sigma, max_sigma).rename("profit_sigma")
