"""Small numeric helpers shared across profitability, model evaluation, and QA code."""

from __future__ import annotations

import numpy as np


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Elementwise division that returns `fill` wherever the denominator is zero."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator != 0, numerator / denominator, fill)
    return result
