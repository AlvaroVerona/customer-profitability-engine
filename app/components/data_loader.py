"""Cached loaders for every artifact the dashboard reads.

Centralized here so every page reads the same paths the same way, and so
Streamlit's cache is shared across pages (switching pages doesn't re-read
parquet files already loaded this session). Every loader is a thin,
read-only wrapper around a Phase 1-12 output -- the dashboard computes
nothing new; it visualizes what the pipeline already produced.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from customer_profitability.utils.config import Settings, load_settings
from customer_profitability.utils.demo_sample import DEMO_DIR

DEMO_META_PATH = DEMO_DIR / "demo_meta.json"


def _resolve(path):
    """The full pipeline output if it exists, else the same table from `demo_data/`
    (the small sample the hosted demo runs on; see `utils/demo_sample.py`)."""
    return path if path.exists() else DEMO_DIR / path.name


def demo_meta() -> dict | None:
    """Sampling info when the dashboard is running on the demo sample, else None."""
    full = get_settings().processed_dir / "customers_validated.parquet"
    if full.exists() or not DEMO_META_PATH.exists():
        return None
    return json.loads(DEMO_META_PATH.read_text())


def demo_scale() -> float:
    """Fraction of the population in the data (1.0 on the full pipeline output). Population-level
    constraints (budget, capacity, risk cap) are scaled by it so they stay binding on a sample."""
    meta = demo_meta()
    return meta["fraction"] if meta else 1.0


@st.cache_resource
def get_settings() -> Settings:
    return load_settings()


@st.cache_data
def load_customers() -> pd.DataFrame:
    settings = get_settings()
    return pd.read_parquet(_resolve(settings.processed_dir / "customers_validated.parquet")).drop_duplicates("customer_id")


@st.cache_data
def load_accounts() -> pd.DataFrame:
    settings = get_settings()
    return pd.read_parquet(_resolve(settings.processed_dir / "accounts_validated.parquet"))


@st.cache_data
def load_customer_360() -> pd.DataFrame:
    settings = get_settings()
    return pd.read_parquet(_resolve(settings.features_dir / "customer_360.parquet"))


@st.cache_data
def load_profitability_monthly() -> pd.DataFrame:
    settings = get_settings()
    return pd.read_parquet(_resolve(settings.processed_dir / "profitability_monthly.parquet"))


@st.cache_data
def load_profitability_summary() -> pd.DataFrame:
    settings = get_settings()
    return pd.read_parquet(_resolve(settings.processed_dir / "profitability_customer_summary.parquet"))


@st.cache_data
def load_clv() -> pd.DataFrame:
    settings = get_settings()
    return pd.read_parquet(_resolve(settings.processed_dir / "clv.parquet"))


@st.cache_data
def load_segments() -> pd.DataFrame:
    settings = get_settings()
    return pd.read_parquet(_resolve(settings.processed_dir / "customer_segments.parquet"))


@st.cache_data
def load_segment_profiles() -> pd.DataFrame:
    settings = get_settings()
    return pd.read_parquet(_resolve(settings.processed_dir / "segment_profiles.parquet"))


@st.cache_data
def load_action_eligibility() -> pd.DataFrame:
    settings = get_settings()
    return pd.read_parquet(_resolve(settings.processed_dir / "action_eligibility.parquet"))


@st.cache_data
def load_incremental_value() -> pd.DataFrame:
    settings = get_settings()
    return pd.read_parquet(_resolve(settings.processed_dir / "action_incremental_value.parquet"))


@st.cache_data
def load_optimized_plan() -> pd.DataFrame:
    settings = get_settings()
    return pd.read_parquet(_resolve(settings.processed_dir / "optimized_action_plan.parquet"))


@st.cache_data
def load_monte_carlo_report() -> dict:
    settings = get_settings()
    reports_dir = settings.raw_dir.parent.parent / "reports"
    return json.loads((reports_dir / "monte_carlo_report.json").read_text())


@st.cache_data
def load_customer_360_latest() -> pd.DataFrame:
    """Each customer's most recently observed Customer 360 row -- the
    "current state" snapshot most pages need, not the full monthly panel."""
    df = load_customer_360()
    return df.sort_values(["customer_id", "month"]).groupby("customer_id", as_index=False).tail(1)


@st.cache_data
def merged_customer_overview() -> pd.DataFrame:
    """One row per customer: profitability summary + CLV + segment + latest
    Customer 360 snapshot -- the base table Pages 1, 3, and 5 all filter/
    aggregate from.

    `customer_360` carries its own copy of `customer_segment` (joined in
    back at Phase 3 from the same source `customers` table); dropped here
    before merging so `customers_validated`'s copy is the single,
    unambiguous `customer_segment` column downstream, rather than pandas
    silently suffixing both into `customer_segment_x`/`_y`.
    """
    summary = load_profitability_summary()
    clv = load_clv()
    segments = load_segments()
    latest = load_customer_360_latest().drop(columns=["customer_segment"])
    customers = load_customers()[["customer_id", "customer_segment", "acquisition_channel", "employment_status"]]

    return (
        summary.merge(clv, on="customer_id", how="left")
        .merge(segments[["customer_id", "segment_name"]], on="customer_id", how="left")
        .merge(latest, on="customer_id", how="left")
        .merge(customers, on="customer_id", how="left")
    )
