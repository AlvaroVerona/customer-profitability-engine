"""Page 6 -- Actions."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.data_loader import get_settings, load_clv, load_incremental_value
from components.demo_banner import demo_banner
from components.tables import eur, pct

from customer_profitability.actions.explanation import explain_customer_recommendation
from customer_profitability.actions.incremental_value import recommend_action

st.title("Actions")
demo_banner()
st.caption(
    "Recommended actions for a selected customer, ranked by incremental profit against NO_ACTION "
    "(Phase 10's action impact simulation). Positive values are not a guarantee of acceptance -- "
    "they are expected values that already blend in each action's acceptance probability."
)

settings = get_settings()
incremental_value = load_incremental_value()
customer_ids = sorted(incremental_value["customer_id"].unique())
customer_id = st.selectbox("Customer", customer_ids)

recommendation = recommend_action(incremental_value, customer_id, settings)

display = recommendation.rename(
    columns={
        "action_name": "Action",
        "expected_action_cost": "Cost",
        "delta_clv": "ΔCLV",
        "incremental_profit": "ΔProfit",
        "acceptance_probability": "P(accept)",
    }
)[["Action", "Cost", "P(accept)", "ΔCLV", "ΔProfit"]]

st.dataframe(
    display.style.format({"Cost": "€{:.2f}", "P(accept)": "{:.0%}", "ΔCLV": "€{:.2f}", "ΔProfit": "€{:.2f}"}),
    use_container_width=True,
)

best = recommendation.iloc[0]
if best["action_type"] != "NO_ACTION" and best["incremental_profit"] > 0:
    st.success(
        f"Best action: **{best['action_name']}** -- expected incremental profit {eur(best['incremental_profit'])}, "
        f"cost {eur(best['expected_action_cost'])}, acceptance probability {pct(best['acceptance_probability'])}."
    )
else:
    st.info("No action beats NO_ACTION for this customer under current simulation assumptions.")

st.divider()
st.subheader("Decision explanation")
st.caption(
    "Phase 14's customer-level decision explanation -- every 'Why' line is a computed condition "
    "against portfolio benchmarks (CLV/profit/churn percentile among active customers), not a "
    "canned line shown for everyone."
)
clv = load_clv()
explanation = explain_customer_recommendation(customer_id, incremental_value, clv, settings)
st.markdown(f"**Recommended action:** {explanation.recommended_action}")
st.markdown("**Why:**")
for reason in explanation.why:
    st.markdown(f"- {reason}")
st.markdown("**Expected impact:**")
st.markdown(
    f"- Cost: {eur(explanation.cost)}\n"
    f"- Incremental profit: {eur(explanation.incremental_profit)}\n"
    f"- CLV impact: {eur(explanation.clv_impact)}"
)
