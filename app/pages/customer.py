"""Page 2 -- Customer 360."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.charts import line_series
from components.data_loader import (
    load_accounts,
    load_clv,
    load_customer_360_latest,
    load_customers,
    load_incremental_value,
    load_profitability_monthly,
    load_profitability_summary,
    load_segments,
)
from components.tables import eur, pct

st.title("Customer 360")

customers = load_customers()
customer_id = st.selectbox("Customer", sorted(customers["customer_id"]))

customer = customers[customers["customer_id"] == customer_id].iloc[0]
latest = load_customer_360_latest()
c360_rows = latest[latest["customer_id"] == customer_id]
c360 = c360_rows.iloc[0] if len(c360_rows) else None
summary_rows = load_profitability_summary()[load_profitability_summary()["customer_id"] == customer_id]
summary = summary_rows.iloc[0] if len(summary_rows) else None
segment_rows = load_segments()[load_segments()["customer_id"] == customer_id]
segment_name = segment_rows.iloc[0]["segment_name"] if len(segment_rows) else "Not segmented (churned)"

st.subheader("Profile")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Age", int(customer["age"]))
c2.metric("Income", eur(customer["income"]))
c3.metric("Segment", segment_name)
c4.metric("Employment", customer["employment_status"])
c5, c6, c7, c8 = st.columns(4)
c5.metric("Acquisition channel", customer["acquisition_channel"])
c6.metric("Acquisition date", str(customer["acquisition_date"])[:10])
c7.metric("Status", "Churned" if customer["churn_flag"] else "Active")
c8.metric("Churn date", str(customer["churn_date"])[:10] if customer["churn_flag"] else "-")

st.divider()
st.subheader("Products")
accounts = load_accounts()
own = accounts[accounts["customer_id"] == customer_id]
if len(own):
    st.dataframe(
        own[["product", "account_open_date", "account_close_date", "is_active"]].reset_index(drop=True),
        use_container_width=True,
    )
else:
    st.info("No product records found.")

st.divider()
st.subheader("Balances & Risk")
if c360 is not None:
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Deposit balance", eur(c360["average_balance"]))
    b2.metric("Loan balance", eur(c360["outstanding_balance"]))
    b3.metric("Credit utilization", pct(c360["utilization"]) if c360["utilization"] == c360["utilization"] else "n/a")
    b4.metric("Expected loss (monthly)", eur(c360["expected_loss"]))
else:
    st.info("No Customer 360 snapshot found (never had an active month in the panel).")

st.divider()
st.subheader("Revenue, Cost & Economic Profit (lifetime, observed)")
if summary is not None:
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Total revenue", eur(summary["total_revenue"]))
    r2.metric("Total costs", eur(summary["total_costs"]))
    r3.metric("Economic profit", eur(summary["economic_profit"]))
    r4.metric("Profit margin", pct(summary["profit_margin"]) if summary["profit_margin"] == summary["profit_margin"] else "n/a")
else:
    st.info("No profitability history found.")

st.divider()
st.subheader("Churn Probability & CLV")
clv_df = load_clv()
clv_rows = clv_df[clv_df["customer_id"] == customer_id]
if len(clv_rows):
    clv = clv_rows.iloc[0]
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Monthly churn probability", pct(clv["p_churn_monthly"]) if clv["is_active"] else "n/a (churned)")
    v2.metric("Historical economic profit", eur(clv["historical_economic_profit"]))
    v3.metric("Future CLV", eur(clv["future_clv"]))
    v4.metric("Total customer value", eur(clv["total_customer_economic_value"]))
    st.caption(f"Monte Carlo band (P5 / P50 / P95): {eur(clv['clv_p5'])} / {eur(clv['clv_p50'])} / {eur(clv['clv_p95'])}")

st.divider()
st.subheader("Recommended actions (top 3 by incremental profit)")
iv = load_incremental_value()
customer_actions = iv[iv["customer_id"] == customer_id].sort_values("incremental_profit", ascending=False).head(3)
if len(customer_actions):
    st.dataframe(
        customer_actions[["action_type", "acceptance_probability", "expected_action_cost", "delta_clv", "incremental_profit"]],
        use_container_width=True,
    )
else:
    st.info("No simulated actions for this customer (likely churned).")

st.divider()
st.subheader("Historical profitability timeline")
monthly = load_profitability_monthly()
history = monthly[monthly["customer_id"] == customer_id].sort_values("month")
if len(history):
    st.plotly_chart(line_series(history, "month", "economic_profit", "Monthly Economic Profit"), use_container_width=True)
else:
    st.info("No monthly profitability history found.")
