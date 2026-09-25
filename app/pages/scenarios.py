"""Page 8 -- Scenario Analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.charts import bar_by_category
from components.data_loader import load_monte_carlo_report
from components.tables import eur, pct

from customer_profitability.simulation.scenarios import SCENARIOS

st.title("Scenario Analysis")
st.caption("Base / Downside / Upside / Stress Monte Carlo (Phase 12), 10,000 simulations per scenario against Phase 11's optimized action plan.")

report = load_monte_carlo_report()
scenario_key = st.selectbox("Scenario", list(SCENARIOS.keys()), format_func=lambda k: SCENARIOS[k].name)
scenario = SCENARIOS[scenario_key]
result = report[scenario_key]

st.subheader(f"{scenario.name} assumptions")
c1, c2 = st.columns(2)
c1.metric("Churn multiplier", f"{scenario.churn_multiplier:.2f}x")
c2.metric("Net profit multiplier", f"{scenario.profit_multiplier:.2f}x")
st.caption(
    "Two composite multipliers rather than separately-tracked balance/activity/credit-loss shocks -- "
    "`expected_profit` is already netted by Phase 7, so there's no separate component left to shock "
    "independently at this stage of the pipeline (documented in `simulation/scenarios.py`)."
)

st.divider()
st.subheader("Total portfolio value (CLV distribution)")
tv = result["total_portfolio_value"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean", eur(tv["mean"]))
c2.metric("P5", eur(tv["p5"]))
c3.metric("P50", eur(tv["p50"]))
c4.metric("P95", eur(tv["p95"]))

st.divider()
st.subheader("Incremental action value")
ap = result["incremental_action_profit"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean", eur(ap["mean"]))
c2.metric("P5", eur(ap["p5"]))
c3.metric("P50", eur(ap["p50"]))
c4.metric("P95", eur(ap["p95"]))

c5, c6 = st.columns(2)
c5.metric("P(negative incremental profit)", pct(result["probability_negative_incremental_profit"]))
c6.metric("P(budget overrun)", pct(result["probability_budget_overrun"]))

st.divider()
st.subheader("Compare all scenarios")
compare_df = pd.DataFrame(
    [
        {
            "Scenario": report[key]["scenario"],
            "Total value (P50)": report[key]["total_portfolio_value"]["p50"],
            "Incremental profit (P50)": report[key]["incremental_action_profit"]["p50"],
        }
        for key in SCENARIOS
    ]
)
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(
        bar_by_category(compare_df, "Scenario", "Total value (P50)", "Total Portfolio Value by Scenario", category_order=list(compare_df["Scenario"])),
        use_container_width=True,
    )
with col2:
    st.plotly_chart(
        bar_by_category(compare_df, "Scenario", "Incremental profit (P50)", "Incremental Action Profit by Scenario", category_order=list(compare_df["Scenario"])),
        use_container_width=True,
    )

st.caption(
    "Expected credit loss is not decomposed separately at the scenario level -- it is folded into the "
    "net profit multiplier above, a documented simplification (see `simulation/scenarios.py`), not an omission."
)
