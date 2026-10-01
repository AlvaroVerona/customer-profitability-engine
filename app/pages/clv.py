"""Page 4 -- Customer Lifetime Value."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.charts import line_series, polarity_histogram, scatter_2d
from components.data_loader import get_settings, load_clv
from components.demo_banner import demo_banner
from components.tables import eur, kpi_row

from customer_profitability.clv.discounting import clv_config_monthly_rate

st.title("Customer Lifetime Value")
demo_banner()

clv = load_clv()
active = clv[clv["is_active"]]

kpi_row(
    [
        ("Active customers", f"{len(active):,}", None),
        ("Total historical profit", eur(clv["historical_economic_profit"].sum()), None),
        ("Total future CLV", eur(active["future_clv"].sum()), None),
        ("Total customer value", eur(clv["total_customer_economic_value"].sum()), None),
    ]
)

st.divider()
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(polarity_histogram(active, "total_customer_economic_value", "CLV Distribution (active customers)"), use_container_width=True)
with col2:
    st.plotly_chart(
        scatter_2d(
            active.sample(min(2000, len(active)), random_state=42),
            "historical_economic_profit",
            "future_clv",
            "Historical vs. Future Value (sample of 2,000)",
        ),
        use_container_width=True,
    )

st.divider()
st.subheader("Uncertainty band (Phase 7.5 Monte Carlo)")
band = active[["clv_p5", "clv_p50", "clv_p95"]].mean()
kpi_row(
    [
        ("P5 (portfolio avg.)", eur(band["clv_p5"]), None),
        ("P50 (portfolio avg.)", eur(band["clv_p50"]), None),
        ("P95 (portfolio avg.)", eur(band["clv_p95"]), None),
        ("Avg. expected monthly profit", eur(active["expected_profit"].mean()), None),
    ]
)

st.divider()
st.subheader("Survival curve")
st.caption("P(still active) under a constant monthly hazard, for a selected customer or the portfolio average.")

settings = get_settings()
options = ["Portfolio average"] + sorted(active["customer_id"].sample(min(200, len(active)), random_state=1).tolist())
choice = st.selectbox("Customer", options)

p_churn = active["p_churn_monthly"].mean() if choice == "Portfolio average" else active.loc[active["customer_id"] == choice, "p_churn_monthly"].iloc[0]
horizon = settings.clv.horizon_months
months = np.arange(1, horizon + 1)
survival = (1 - p_churn) ** (months - 1)
survival_df = pd.DataFrame({"Month": months, "Survival probability": survival})
st.plotly_chart(line_series(survival_df, "Month", "Survival probability", f"Survival Curve (monthly churn hazard = {p_churn:.2%})"), use_container_width=True)
st.caption(f"Effective monthly discount rate: {clv_config_monthly_rate(settings.clv):.4%} (from {settings.clv.annual_discount_rate:.0%} annual).")
