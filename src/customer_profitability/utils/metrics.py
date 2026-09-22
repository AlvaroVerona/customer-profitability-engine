"""Small numeric helpers shared across profitability, model evaluation, and QA code."""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Elementwise division that returns `fill` wherever the denominator is zero."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator != 0, numerator / denominator, fill)
    return result


def nansum(s: pd.Series) -> float:
    """`sum` with pandas' default skipna silently turns an all-NaN group
    into 0.0, which would erase a genuinely missing value instead of
    propagating it (e.g. a customer-month with exactly one deposit row whose
    `average_balance` is NaN). `min_count=1` keeps the group NaN instead."""
    return s.sum(min_count=1)


def rolling_trailing_stats(
    df: pd.DataFrame,
    value_cols: list[str],
    windows: tuple[int, ...] = (3, 6, 12),
    group_col: str = "customer_id",
    time_col: str = "month",
    stats: tuple[str, ...] = ("mean",),
) -> pd.DataFrame:
    """Trailing rolling stats per group, keyed by (group_col, time_col).

    A window of size `w` ending at row t covers t and the w-1 rows before it
    -- never rows after t. This is what makes features built from this
    function safe to use for point-in-time prediction (Phase 3's "no data
    leakage" requirement): a feature for month t never depends on month t+1.
    `df` must be pre-sorted, or unsorted -- this function sorts a copy by
    (group_col, time_col) itself.
    """
    df = df.sort_values([group_col, time_col])
    out = df[[group_col, time_col]].copy()
    grouped = df.groupby(group_col, sort=False)
    for col in value_cols:
        for w in windows:
            roll = grouped[col].rolling(window=w, min_periods=1)
            if "mean" in stats:
                out[f"{col}_avg_{w}m"] = roll.mean().reset_index(level=0, drop=True)
            if "std" in stats:
                out[f"{col}_std_{w}m"] = roll.std().reset_index(level=0, drop=True)
    return out
