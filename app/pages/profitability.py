"""Page 3 -- Profitability."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.charts import bar_by_category, polarity_histogram
from components.data_loader import (
    load_accounts,
    load_profitability_monthly,
    merged_customer_overview,
)
from components.filters import product_filter, profitability_filters
from components.tables import eur, kpi_row

st.title("Profitability")
st.caption("Revenue decomposition, funding contribution, risk cost, operating cost, and acquisition cost -- Phase 4's economic profit waterfall.")

overview = merged_customer_overview()
filtered = profitability_filters(overview, key_prefix="prof_")
accounts = load_accounts()
filtered_ids = product_filter(accounts, filtered["customer_id"], key_prefix="prof_")
filtered = filtered[filtered["customer_id"].isin(filtered_ids)]

st.write(f"**{len(filtered):,}** customers match the current filters (of {len(overview):,} total).")

monthly = load_profitability_monthly()
scoped = monthly[monthly["customer_id"].isin(filtered["customer_id"])]

revenue_components = ["interest_revenue", "interchange_revenue", "fee_revenue"]
waterfall_components = ["total_revenue", "deposit_contribution", "expected_loss", "operating_cost", "acquisition_cost"]
totals = scoped[[*revenue_components, *waterfall_components, "economic_profit"]].sum()

kpi_row(
    [
        ("Total revenue", eur(totals["total_revenue"]), None),
        ("Deposit contribution", eur(totals["deposit_contribution"]), None),
        ("Expected loss (cost)", eur(totals["expected_loss"]), None),
        ("Operating cost", eur(totals["operating_cost"]), None),
    ]
)
kpi_row(
    [
        ("Acquisition cost", eur(totals["acquisition_cost"]), None),
        ("Economic profit", eur(totals["economic_profit"]), None),
        ("Interest revenue", eur(totals["interest_revenue"]), None),
        ("Interchange + fee revenue", eur(totals["interchange_revenue"] + totals["fee_revenue"]), None),
    ]
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    revenue_df = pd.DataFrame(
        {
            "Component": ["Interest", "Interchange", "Fee"],
            "Revenue": [totals["interest_revenue"], totals["interchange_revenue"], totals["fee_revenue"]],
        }
    )
    st.plotly_chart(bar_by_category(revenue_df, "Component", "Revenue", "Revenue Decomposition"), use_container_width=True)
with col2:
    waterfall_df = pd.DataFrame(
        {
            "Stage": ["Revenue", "+Deposit Contrib.", "-Expected Loss", "-Operating Cost", "-Acquisition Cost"],
            "Amount": [
                totals["total_revenue"],
                totals["deposit_contribution"],
                -totals["expected_loss"],
                -totals["operating_cost"],
                -totals["acquisition_cost"],
            ],
        }
    )
    st.plotly_chart(bar_by_category(waterfall_df, "Stage", "Amount", "Economic Profit Waterfall"), use_container_width=True)

st.plotly_chart(polarity_histogram(filtered, "economic_profit", "Economic Profit Distribution (filtered)"), use_container_width=True)
