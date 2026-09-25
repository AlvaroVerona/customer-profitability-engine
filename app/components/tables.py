"""Shared KPI/table formatting helpers."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    """`items`: list of (label, value, delta). `delta` may be None."""
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items, strict=True):
        col.metric(label, value, delta)


def eur(value: float) -> str:
    return f"€{value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.1%}"


def format_currency_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].map(lambda v: eur(v) if pd.notna(v) else "")
    return out
