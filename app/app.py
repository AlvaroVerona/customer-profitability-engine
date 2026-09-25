"""Phase 13 -- Streamlit Decision Intelligence Dashboard entry point.

Every page reads already-computed outputs from `data/processed/`,
`data/features/`, and `reports/` (via `components/data_loader.py`) -- the
dashboard visualizes and lets an analyst interact with Phase 1-12's
results; it does not run new pipeline stages, except Page 7 (Optimization),
which re-solves the Phase 11 ILP against user-adjusted constraints on
Phase 10's already-simulated candidate pool (seconds, not a full
re-simulation).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Running via `streamlit run app/app.py`, the src/ package layout isn't on
# sys.path by default (unlike `uv run python -m ...`, which pythonpath
# handles via pyproject.toml) -- add it explicitly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

st.set_page_config(
    page_title="Customer Profitability & Action Optimization Engine",
    page_icon="\U0001f4b0",
    layout="wide",
)

PAGES_DIR = Path(__file__).parent / "pages"

pages = [
    st.Page(PAGES_DIR / "overview.py", title="Executive Overview", icon="\U0001f4ca", default=True),
    st.Page(PAGES_DIR / "customer.py", title="Customer 360", icon="\U0001f464"),
    st.Page(PAGES_DIR / "profitability.py", title="Profitability", icon="\U0001f4b5"),
    st.Page(PAGES_DIR / "clv.py", title="CLV", icon="\U0001f4c8"),
    st.Page(PAGES_DIR / "segmentation.py", title="Segmentation", icon="\U0001f9e9"),
    st.Page(PAGES_DIR / "actions.py", title="Actions", icon="\U0001f3af"),
    st.Page(PAGES_DIR / "optimization.py", title="Optimization", icon="\U0001f9ee"),
    st.Page(PAGES_DIR / "scenarios.py", title="Scenario Analysis", icon="\U0001f3b2"),
]

nav = st.navigation(pages)

st.sidebar.caption(
    "Synthetic neobank portfolio -- every number here is model output on "
    "generated data (seed 42), not a real business finding. See "
    "`PROJECT_SPEC.md` and `README.md`."
)

nav.run()
