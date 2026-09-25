"""Page 6 -- Actions."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.data_loader import get_settings, load_incremental_value
from components.tables import eur, pct

from customer_profitability.actions.incremental_value import recommend_action

st.title("Actions")
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
