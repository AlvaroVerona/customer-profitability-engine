"""Shared sidebar filter widgets (Page 3's "filter by segment, product,
acquisition channel, profitability band")."""

from __future__ import annotations

import pandas as pd
import streamlit as st

PROFITABILITY_BANDS = {
    "All": (-float("inf"), float("inf")),
    "Loss-making (< 0)": (-float("inf"), 0),
    "Low (0 - 200)": (0, 200),
    "Mid (200 - 1,000)": (200, 1000),
    "High (>= 1,000)": (1000, float("inf")),
}


def profitability_filters(df: pd.DataFrame, key_prefix: str = "") -> pd.DataFrame:
    st.sidebar.subheader("Filters")

    segments = ["All"] + sorted(df["segment_name"].dropna().unique().tolist())
    segment = st.sidebar.selectbox("Segment", segments, key=f"{key_prefix}segment")

    channels = ["All"] + sorted(df["acquisition_channel"].dropna().unique().tolist())
    channel = st.sidebar.selectbox("Acquisition channel", channels, key=f"{key_prefix}channel")

    band = st.sidebar.selectbox("Profitability band", list(PROFITABILITY_BANDS.keys()), key=f"{key_prefix}band")

    out = df
    if segment != "All":
        out = out[out["segment_name"] == segment]
    if channel != "All":
        out = out[out["acquisition_channel"] == channel]
    lo, hi = PROFITABILITY_BANDS[band]
    out = out[(out["economic_profit"] >= lo) & (out["economic_profit"] < hi)]
    return out


def product_filter(accounts: pd.DataFrame, customer_ids: pd.Series, key_prefix: str = "") -> pd.Series:
    """Returns the subset of `customer_ids` who hold the selected product
    ("All" returns every id unchanged)."""
    products = ["All"] + sorted(accounts["product"].dropna().unique().tolist())
    product = st.sidebar.selectbox("Product", products, key=f"{key_prefix}product")
    if product == "All":
        return customer_ids
    holders = set(accounts.loc[accounts["product"] == product, "customer_id"])
    return customer_ids[customer_ids.isin(holders)]
