"""Page 1 -- Executive Overview."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.charts import bar_by_category, polarity_histogram, revenue_vs_cost_scatter
from components.data_loader import merged_customer_overview
from components.demo_banner import demo_banner
from components.tables import eur, kpi_row, pct

st.title("Executive Overview")
demo_banner()
st.caption("Synthetic neobank portfolio -- Phase 4 profitability + Phase 7 CLV, as of the observation window end.")

df = merged_customer_overview()
n_customers = len(df)
total_profit = df["economic_profit"].sum()
avg_profit = df["economic_profit"].mean()
total_clv = df["total_customer_economic_value"].sum()
avg_clv = df["total_customer_economic_value"].mean()
pct_profitable = (df["economic_profit"] > 0).mean()
pct_loss = (df["economic_profit"] <= 0).mean()

kpi_row(
    [
        ("Total Customers", f"{n_customers:,}", None),
        ("Total Economic Profit", eur(total_profit), None),
        ("Average Customer Profit", eur(avg_profit), None),
        ("Total CLV", eur(total_clv), None),
    ]
)
kpi_row(
    [
        ("Average CLV", eur(avg_clv), None),
        ("% Profitable Customers", pct(pct_profitable), None),
        ("% Loss-Making Customers", pct(pct_loss), None),
        ("Active Customers", f"{int(df['is_active'].sum()):,}", None),
    ]
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(polarity_histogram(df, "economic_profit", "Profitability Distribution"), use_container_width=True)
with col2:
    st.plotly_chart(
        polarity_histogram(df, "total_customer_economic_value", "CLV Distribution"), use_container_width=True
    )

col3, col4 = st.columns(2)
with col3:
    by_segment = df.groupby("segment_name", as_index=False)["economic_profit"].mean().sort_values(
        "economic_profit", ascending=False
    )
    st.plotly_chart(bar_by_category(by_segment, "segment_name", "economic_profit", "Avg. Profit by Segment"), use_container_width=True)
with col4:
    clv_by_segment = df.groupby("segment_name", as_index=False)["total_customer_economic_value"].mean().sort_values(
        "total_customer_economic_value", ascending=False
    )
    st.plotly_chart(
        bar_by_category(clv_by_segment, "segment_name", "total_customer_economic_value", "Avg. CLV by Segment"),
        use_container_width=True,
    )

st.plotly_chart(revenue_vs_cost_scatter(df), use_container_width=True)
st.caption("Points above the diagonal have costs exceeding revenue; color = economic profit (orange positive, blue negative).")
