"""Page 5 -- Economic Segmentation."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.charts import bar_by_category, scatter_2d
from components.data_loader import load_segment_profiles
from components.tables import format_currency_columns

st.title("Economic Segmentation")
st.caption("K-Means on 12 standardized economic variables (Phase 8) -- active customers only. Segment names come from a rule over each cluster's standardized centroid, not hand-picked.")

profiles = load_segment_profiles().sort_values("n_customers", ascending=False)
order = profiles["segment_name"].tolist()

st.plotly_chart(
    bar_by_category(profiles, "segment_name", "n_customers", "Segment Sizes", category_order=order), use_container_width=True
)

st.divider()
st.subheader("Segments on the Profitability x CLV plane")
st.plotly_chart(
    scatter_2d(
        profiles,
        "total_customer_economic_value",
        "historical_economic_profit",
        "Profitability vs. CLV (bubble size = segment size)",
        size="n_customers",
        color="segment_name",
        text="segment_name",
    ),
    use_container_width=True,
)

st.divider()
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(bar_by_category(profiles, "segment_name", "p_churn_monthly", "Monthly Churn Probability by Segment", category_order=order), use_container_width=True)
with col2:
    st.plotly_chart(bar_by_category(profiles, "segment_name", "utilization", "Credit Utilization by Segment", category_order=order), use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(bar_by_category(profiles, "segment_name", "product_count", "Avg. Product Count by Segment", category_order=order), use_container_width=True)
with col4:
    st.plotly_chart(bar_by_category(profiles, "segment_name", "transaction_volume_avg_3m", "Avg. Transaction Volume by Segment", category_order=order), use_container_width=True)

st.divider()
st.subheader("Full segment profile")
currency_cols = [
    "historical_economic_profit",
    "total_customer_economic_value",
    "total_revenue",
    "total_deposit_contribution",
    "total_expected_loss",
    "total_operating_cost",
    "average_balance_avg_3m",
]
st.dataframe(format_currency_columns(profiles.drop(columns=["segment_id"]), currency_cols), use_container_width=True)
