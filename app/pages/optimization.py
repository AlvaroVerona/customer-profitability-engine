"""Page 7 -- Optimization.

Re-solves Phase 11's ILP against user-adjusted constraints on Phase 10's
already-simulated candidate pool (`action_incremental_value.parquet`) --
no re-simulation, just a fresh OR-Tools solve (seconds).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.charts import bar_by_category
from components.data_loader import get_settings, load_incremental_value, load_segments
from components.tables import eur, kpi_row

from customer_profitability.optimization.constraints import OptimizationConstraints
from customer_profitability.optimization.solver import counterfactual_summary, optimize

st.title("Optimization")
st.caption(
    "Selects the customer/action portfolio maximizing total incremental profit under your constraints "
    "(exact OR-Tools CP-SAT solve). Optimal under Phase 10's simulated effects -- a decision-support "
    "ranking, not a causal guarantee."
)

settings = get_settings()

with st.form("optimization_constraints"):
    c1, c2 = st.columns(2)
    budget = c1.number_input("Budget (€)", min_value=0.0, value=float(settings.optimization.default_budget), step=1000.0)
    capacity = c2.number_input("Operational capacity (# actions)", min_value=1, value=int(settings.optimization.default_capacity), step=100)
    c3, c4 = st.columns(2)
    max_risk = c3.number_input(
        "Max incremental monthly credit risk (€)", min_value=0.0, value=float(settings.optimization.default_max_incremental_risk_monthly), step=1000.0
    )
    min_roi = c4.number_input("Min gross ROI (delta_clv / cost)", value=float(settings.optimization.default_min_expected_roi), step=0.1)
    submitted = st.form_submit_button("Run Optimization", type="primary")

if submitted:
    constraints = OptimizationConstraints(
        budget=budget, capacity=int(capacity), max_incremental_risk_monthly=max_risk, min_expected_roi=min_roi
    )
    with st.spinner("Solving..."):
        incremental_value = load_incremental_value()
        result = optimize(incremental_value, constraints)
    st.session_state["optimization_result"] = result

result = st.session_state.get("optimization_result")

if result is None:
    st.info("Set your constraints and click **Run Optimization**.")
else:
    st.subheader(f"Solver status: {result['status']}")
    kpi_row(
        [
            ("Candidates considered", f"{result['n_candidates']:,}", None),
            ("Actions selected", f"{result['n_selected']:,}", None),
            ("Total expected cost", eur(result["total_cost"]), None),
            ("Remaining budget", eur(result["remaining_budget"]), None),
        ]
    )
    kpi_row(
        [
            ("Total incremental profit", eur(result["total_incremental_profit"]), None),
            ("Total ΔCLV", eur(result["total_delta_clv"]), None),
            ("Total incremental monthly risk", eur(result["total_expected_incremental_risk_monthly"]), None),
            ("Expected monthly churns averted", f"{result['expected_monthly_churns_averted']:.1f}", None),
        ]
    )

    counterfactual = counterfactual_summary(result)
    st.caption(
        f"Counterfactual: no optimization = {eur(counterfactual['no_optimization_incremental_profit'])} incremental profit "
        f"vs. optimized allocation = {eur(counterfactual['optimized_incremental_profit'])} "
        f"(value added: {eur(counterfactual['value_added_by_optimization'])})."
    )

    st.divider()
    selected = result["selected"]
    col1, col2 = st.columns(2)
    with col1:
        dist = selected["action_type"].value_counts().rename_axis("action_type").reset_index(name="count")
        st.plotly_chart(bar_by_category(dist, "action_type", "count", "Action Distribution (selected)"), use_container_width=True)
    with col2:
        segments = load_segments()
        targeted = selected.merge(segments[["customer_id", "segment_name"]], on="customer_id", how="left")
        seg_dist = targeted["segment_name"].value_counts().rename_axis("segment_name").reset_index(name="count")
        st.plotly_chart(bar_by_category(seg_dist, "segment_name", "count", "Segment Distribution (targeted customers)"), use_container_width=True)

    st.subheader(f"Selected actions ({len(selected):,})")
    st.dataframe(
        selected[["customer_id", "action_type", "acceptance_probability", "expected_action_cost", "delta_clv", "incremental_profit"]].reset_index(drop=True),
        use_container_width=True,
    )
