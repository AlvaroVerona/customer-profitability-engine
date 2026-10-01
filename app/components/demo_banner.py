"""Banner shown on every page when the dashboard runs on the demo sample."""

from __future__ import annotations

import streamlit as st

from components.data_loader import demo_meta


def demo_banner() -> None:
    meta = demo_meta()
    if meta is None:
        return
    st.info(
        f"**Demo sample**: {meta['n_customers_sample']:,} of {meta['n_customers_total']:,} customers "
        "(stratified by segment). Totals here are therefore smaller than the full-run figures in the "
        "README and `reports/`; the Scenario Analysis page shows the full-population simulation. "
        "Run the pipeline locally for the complete data."
    )
